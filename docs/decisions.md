# Decisions

Short records of the choices that shaped the gateway: what was picked, what
was rejected, and why.

## 1. OpenAI's format is the internal standard

Every request and response inside the gateway is an OpenAI chat completion.
Adapters translate at the edge (Anthropic's Messages API is the only real
translation today). **Why:** every SDK and most tools already speak it, so
clients change only `base_url`. **Cost:** features with no OpenAI equivalent
(Anthropic thinking blocks, citations) are dropped in v1, and tool calls are
not translated for Anthropic yet; a request that needs them fails over to an
OpenAI-compatible target instead of being silently degraded.

## 2. Raw httpx adapters, no LiteLLM or LangChain

The adapters are a few hundred lines of plain HTTP. **Why:** the point of the
project is to own the failure handling (timeouts per phase, what counts as
retryable, closing upstream on disconnect), and wrappers hide exactly those
details. **Cost:** each new provider is real work. **Lesson:** httpx's
connection pool checks every idle connection when handing one out, so a
single pool holding hundreds of keep-alive connections became the gateway's
largest CPU cost under load; each provider now gets 16 smaller pools used
round-robin (`gateway/providers/registry.py`).

## 3. Hold the stream until the first token

The router does not return a stream to the client until the provider has
produced real output. Any failure before that (connection refused, 5xx, 429,
a hang caught by the time-to-first-token timeout) is retried or failed over
invisibly. **Why:** that is the window where failover is free, and it covers
almost every outage in practice. **Cost:** the first token reaches the client
a few milliseconds later, and a stream that fails after output has started
cannot be rescued, so it ends with an explicit error event rather than a
truncated answer that looks complete.

## 4. Circuit breakers in Redis, not in memory

Breaker state (closed, open, half-open with a limited number of probes) is a
Lua script over a Redis hash. **Why:** with several processes or hosts, an
in-memory breaker would let each one rediscover the outage separately and
would probe a sick provider N times as often. **Cost:** one extra Redis
round trip per attempt.

## 5. Retry once, honour Retry-After, but never wait long

One retry with full-jitter backoff, then the next target. A `Retry-After`
above `max_retry_after` (2 s) means "fail over now" rather than "sleep".
**Why:** the client is waiting; another provider is usually faster than a
rate-limited one.

## 6. Reserve the worst case, settle the truth

Rate limits and budgets are checked before the call, when the real token
count is unknown. The gateway reserves prompt estimate + `max_tokens` (or a
default completion estimate) against tokens-per-minute and the worst-case
cost against the budget, then corrects both when usage is known. **Why:**
checking after the fact lets a burst of concurrent requests overspend.
**Cost:** a tenant close to its limit can be refused a request that would
actually have fit. Budget counters live in Redis for speed and are rebuilt
from `request_logs` if Redis loses them.

## 7. Config in Postgres, versioned, hot-reloaded

One validated YAML/JSON document holds routing, prices, timeouts and cache
settings. Every change is a new version; processes swap it in on a pub/sub
message (with polling as a fallback) and can roll back. **Why:** changing a
fallback chain during an incident should not need a deploy, and "what config
was live at 14:02?" should have an answer.

## 8. Request logging off the hot path

Handlers put one event per request on an in-memory queue; a background task
batches them into a Redis stream; a separate worker (consumer group, with
reclaim of stuck messages) writes them to Postgres. **Why:** a Postgres
insert per request would add latency and couple availability to the
database. **Cost:** logs lag by up to a second, and a gateway crash can lose
the unflushed batch; metrics and budgets are unaffected.

## 9. Semantic cache: strict threshold, measured, with a guard

The threshold is not a guess. `bench/cache_eval.py` replays 20,000 Quora
question pairs as a retrieval workload and picks the lowest threshold that
keeps wrong answers at or under 2 per 1,000 lookups: **0.99** for
all-MiniLM-L6-v2. At 0.95, the value often suggested, about 1 in 10 hits was
wrong. Embeddings also ignore word order ("convert string to integer" vs
"integer to string" scores 0.996), so a cheap lexical guard rejects matches
that swap direction around to/from/into, change a number, or swap a common
opposite. Only single-turn prompts are cached semantically, and the system
prompt and sampling parameters must match exactly. **Cost:** the semantic
cache catches only near-verbatim rephrasings (about 6% of QQP lookups hit),
and on the held-out near-miss set the guard still lets roughly 1 in 6 hits
through wrongly. And it is not free: every miss pays for an embedding and a
vector search (about 55 ms under load in the cost replay), while on a
realistic FAQ workload it added only 0.4 points of hit rate over the exact
cache. So it ships **off by default**; turn it on (globally or per alias)
where prompts repeat with small wording changes and answers need not be exact.

## 10. The model runs in ONNX, on CPU, inside the gateway

MiniLM (384 dimensions) runs with onnxruntime and tokenizers, not PyTorch:
the image stays small and one prompt embeds in about 8 ms on a single thread.
**Why:** a separate embedding service would add a network hop to every
lookup. **Cost:** the embedding competes with request handling for CPU.

## 11. Several processes, not threads

One Python process relays roughly one CPU core of streaming. The gateway runs
`GATEWAY_WORKERS` uvicorn processes (default: one per CPU, at most 4) and
Prometheus runs in multiprocess mode. This works because all shared state is
in Redis and Postgres already.

## 12. Scope

Not built, on purpose: a custom frontend (Grafana covers it), more than two
paid providers, Kubernetes (one host with Compose is enough for the
workload), price-based routing (the router follows the configured chain;
cheapest-first routing would be the next step).
