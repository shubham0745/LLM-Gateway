# Deploying the demo on AWS (one EC2 host)

The whole stack runs on one instance with Docker Compose. Caddy terminates TLS
and exposes only the API, `/healthz` and read-only Grafana; the admin API,
Prometheus and the mock listen on `127.0.0.1` and are reached over SSH.

Everything here needs your AWS account and provider keys, so it is a list of
steps rather than a script that runs on its own.

## 1. Before you start

- A domain (or subdomain) you can point at the instance, e.g. `gw.example.com`.
- Provider API keys. Set a **hard monthly spend limit in each provider's
  console** as well: the gateway caps each tenant, and the provider limit is
  the backstop if something is misconfigured.
  - OpenAI: Settings → Limits → monthly budget.
  - Anthropic: Console → Settings → Limits → spend limit.

## 2. Launch the instance

| Setting | Value |
|---|---|
| AMI | Ubuntu Server 24.04 LTS (x86_64) |
| Type | `t3.medium` (2 vCPU, 4 GB) is enough for the demo. Use `c7i.xlarge` or larger to rerun the load benchmarks |
| Disk | 30 GB gp3 |
| Security group | 22 from **your IP only**; 80 and 443 from anywhere |
| Elastic IP | allocate one and associate it, so DNS survives restarts |

Point an `A` record for your domain at the Elastic IP.

Rough cost: a `t3.medium` on demand is about $30/month, plus the disk and the
Elastic IP. Provider spend is capped by the demo tenant's budget (below).

## 3. Install Docker and the code

```bash
ssh ubuntu@<elastic-ip>
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu && exit     # log in again for the group change
ssh ubuntu@<elastic-ip>
git clone https://github.com/shubham0745/LLM-Gateway.git && cd LLM-Gateway
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=...            # leave empty to skip a provider
ANTHROPIC_API_KEY=...
GATEWAY_ADMIN_TOKEN=$(openssl rand -hex 32)   # paste the output, not the command
GATEWAY_KEY_PEPPER=$(openssl rand -hex 32)    # never change it after keys exist
DOMAIN=gw.example.com
GRAFANA_ADMIN_PASSWORD=...
```

## 4. Start it

```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml up -d --build
docker compose ps          # everything "running", gateway "healthy"
curl https://gw.example.com/healthz
```

The first build downloads the embedding model into the image (about 90 MB).
Caddy gets a Let's Encrypt certificate on the first HTTPS request.

## 5. Create the demo tenant and key

The admin API is only on localhost. From your laptop:

```bash
ssh -N -L 8080:localhost:8080 ubuntu@<elastic-ip> &
GATEWAY_ADMIN_TOKEN=<token from .env> python scripts/bootstrap.py --gateway http://localhost:8080
```

This creates the `demo` tenant (budget **$2.00/month**, 30 requests/minute,
60k tokens/minute) and writes its key to `.gateway-keys.json`. When the budget
is spent, the gateway answers `429 insufficient_quota` until the next month;
nothing reaches the providers. To change the cap:

```bash
curl -X PATCH localhost:8080/admin/tenants/demo -H "authorization: Bearer $GATEWAY_ADMIN_TOKEN" \
  -H 'content-type: application/json' -d '{"monthly_budget_usd": 5}'
```

Share only the `demo` key. Revoke it any time with
`DELETE /admin/keys/{id}` (ids from `GET /admin/tenants/demo/keys`).

## 6. Check it

```bash
curl https://gw.example.com/v1/chat/completions \
  -H "authorization: Bearer <demo key>" -H 'content-type: application/json' \
  -d '{"model": "fast", "messages": [{"role": "user", "content": "Say hi"}]}'
```

Dashboards: `https://gw.example.com/grafana/` (read-only without logging in).

## Operations

- **Update:** `git pull && docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml up -d --build`.
- **Routing changes without a restart:** `PUT /admin/config` over the tunnel.
- **Backups:** `docker compose exec postgres pg_dump -U gateway gateway | gzip > backup.sql.gz`.
  Redis holds rate-limit buckets, breakers and the exact cache; budgets are
  rebuilt from Postgres if Redis is lost.
- **Logs:** `docker compose logs -f gateway worker` (one JSON line per request).
