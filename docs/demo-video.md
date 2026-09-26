# Failover demo: 2-minute video script

The point to show: a provider goes down in the middle of real traffic and no
client request fails. Everything runs locally against the mock provider, so it
can be recorded without API keys.

## Setup (before recording)

```bash
docker compose up -d --build
python scripts/bootstrap.py
```

Arrange three windows side by side:

1. **Grafana** at http://localhost:3000, dashboard "LLM Gateway: live", time
   range "Last 5 minutes", refresh 5s. Panels to keep visible: requests by
   provider, failovers, circuit state, errors.
2. **Terminal A** for traffic.
3. **Terminal B** for breaking things.

Warm up for a minute so the graphs have a baseline:

```bash
# Terminal A: 20 simulated users streaming continuously (Ctrl-C to stop);
# Locust prints a running failure count, which should stay at 0.
locust -f bench/locustfile.py --host http://localhost:8080 --headless -u 20 -r 5
```

## Script

| Time | On screen | Say |
|---|---|---|
| 0:00 | Grafana, traffic flowing, all blue (primary) | "This is an OpenAI-compatible gateway in front of two providers. Twenty clients are streaming completions through it right now." |
| 0:15 | Terminal A: one `curl -N` streaming request, tokens printing | "Clients use the normal OpenAI SDK; they only changed the base URL." |
| 0:30 | Terminal B: `curl -X PUT localhost:9000/control/primary -d '{"mode":"outage"}' -H 'content-type: application/json'` | "Now I take the primary provider down. Every request to it returns 503." |
| 0:40 | Grafana: failovers spike, circuit goes open, traffic turns orange (backup), error panel stays at zero | "The gateway retries once, fails over to the backup before the first token, and after five failures the circuit breaker opens, so the primary isn't even tried." |
| 1:05 | Terminal A: Locust's stats, failures still 0 | "Clients saw nothing. Zero failed requests." |
| 1:15 | Terminal B: `curl -X PUT localhost:9000/control/primary -d '{"mode":"ok"}' -H 'content-type: application/json'` | "Bring the primary back." |
| 1:25 | Grafana: after the cooldown one probe goes through, circuit closes, traffic returns to blue | "After a 15-second cooldown the breaker lets one probe request through; it succeeds and traffic moves back." |
| 1:45 | `docs/img/failover_timeline.png` | "The same experiment, scripted: `python -m bench.chaos`. Six failure types, numbers in the README." |
| 2:00 | end | |

## Recording tips

- 1920x1080, browser zoom 90% so all Grafana panels fit.
- Increase terminal font size; clear the screen between commands.
- If the cooldown wait feels long on camera, lower it first:
  `PUT /admin/config` with `breaker.cooldown_s: 5` (and say so).
