# Benchmarks

Everything here runs against the local stack and the mock provider, so it
costs nothing and anyone can reproduce it:

```bash
docker compose up -d --build
pip install -r requirements-dev.txt
python scripts/bootstrap.py            # creates the "bench" tenant and key
```

| Script | What it measures | Output |
|---|---|---|
| `python -m bench.overhead` | Latency the gateway adds at 10/100/500/1000 concurrent streams, and its CPU and memory | `docs/results/overhead*.json`, `docs/img/overhead*.png` |
| `python -m bench.chaos` | What clients see when the primary provider fails six different ways; breaker detection and recovery time | `docs/results/chaos.json`, `docs/img/chaos_timelines.png`, `docs/img/failover_timeline.png` |
| `python -m bench.cache_eval` | Semantic cache accuracy per similarity threshold (QQP + hand-written pairs); picks the threshold | `docs/results/cache_eval.json`, `docs/img/semantic_threshold.png` |
| `python -m bench.cost_replay` | Spend with caching off, exact-only and exact + semantic on the same request stream | `docs/results/cost_replay.json` |
| `locust -f bench/locustfile.py --host http://localhost:8080` | Interactive load test in a browser UI | Locust UI |

Results are written into `docs/`, and `docs/benchmarks.md` explains them.

## Quora Question Pairs

`cache_eval` and `cost_replay` need the QQP file (about 58 MB), which is not
in the repository:

```bash
mkdir -p bench/data/raw
curl -L -o bench/data/raw/qqp.tsv \
  https://raw.githubusercontent.com/MLDroid/quora_duplicate_challenge/master/data/quora_duplicate_questions.tsv
# Original source (same file): http://qim.fb.com/data/quora_duplicate_questions.tsv
wc -l bench/data/raw/qqp.tsv   # 404,302 lines including the header
```

The hand-written pairs are in `bench/data/handwritten_pairs.jsonl`
(regenerate with `python bench/data/make_handwritten_pairs.py`) and the
held-out pairs, written after the guard was frozen, in
`bench/data/heldout_pairs.jsonl`.

## Things to know before trusting a number

- **The load generator, the mock and the gateway share the machine.** On a
  small box the mock or the load generator saturates before the gateway
  does; `overhead.json` records the direct-to-mock baseline at every level so
  you can see when that happens. For clean numbers at 500+ streams, run the
  load generator on a second machine (`GATEWAY_URL=http://<host>:8080`).
- Overhead is reported as *gateway percentile minus direct percentile*. At
  saturation the two distributions have different shapes, so p99 overhead can
  come out lower than p95; the raw distributions are in the JSON.
- The gateway process count is `GATEWAY_WORKERS` (empty = one per CPU, at most 4).
  Record it with the result.
- `chaos` changes the live routing config (it shortens the TTFT timeout for
  the hang scenario) and restores it at the end. Do not run it against a
  gateway serving real traffic.
