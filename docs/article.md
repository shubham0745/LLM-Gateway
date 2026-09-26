# What I learned building an LLM gateway, with the numbers

*Draft for a blog post. A short LinkedIn version is at the end.*

Every team that ships an LLM feature ends up writing the same wrapper: retry
when the provider has a bad minute, switch to another provider when it has a
bad hour, stop one customer from spending the month's budget in an afternoon,
and answer "who spent how much, on which model, yesterday?". I built that
wrapper as a standalone, self-hosted gateway with an OpenAI-compatible API,
then tried to break it and measured what happened. The code, the benchmark
scripts and all raw results are in the repository; this post is about what
the numbers taught me.

## The shape of it

Clients use the normal OpenAI SDK and change only `base_url`. Behind that,
each request goes through five steps: authenticate the API key, check the
tenant's rate limits and monthly budget, look in the cache, route along a
fallback chain (for example OpenAI, then Anthropic, then a small local model
on Ollama), and record cost and latency. It is FastAPI with hand-written
HTTP adapters (no LiteLLM, because the failure handling was the whole point),
Redis for anything shared between processes, Postgres with pgvector, and
Prometheus and Grafana for the dashboards.

To test failure without paying for it, the repository includes a mock
provider that can be told, while traffic is flowing, to return 503s, answer
429 with a long `Retry-After`, accept connections and never answer, or drop
the connection halfway through a stream.

## 1. Failover is free until the first token, and impossible after it

With streaming, the client starts rendering the moment the first token
arrives. Before that moment the gateway can retry or switch providers and the
client never knows. After it, switching would mean the user sees half an
answer from one model continued by another.

So the gateway holds each stream until the provider has produced real output.
With a primary provider returning 503 on every request for 30 seconds under
steady load, 2,559 requests went through and none failed; time to first
token during the outage was the same as when healthy (88 ms median against
the mock). The same was true for a flood of 429s and for a provider failing
30% of requests at random.

A provider that dies mid-stream is different: 21 of 2,668 clients had
already received tokens when the connection dropped. The gateway ends those
streams with an explicit error event. The alternative, closing the stream
quietly, produces an answer that looks complete and isn't, which is worse
than an error.

## 2. My circuit breaker had a bug that only chaos testing found

The circuit breaker stops the gateway from sending traffic to a provider
that keeps failing: after five consecutive failures it opens, waits 15
seconds, then lets one probe request through.

The first run of the mid-stream scenario showed 281 client errors instead of
the handful I expected. The breaker was recording a stream as a success as
soon as its first token arrived. A provider that sent a few tokens and then
dropped every connection therefore kept resetting its own failure count, and
every probe closed the breaker and let the full load through again. The fix
was to record a stream's outcome when it ends. Same scenario afterwards: 21
errors. No unit test I had written would have caught this; a timeline plot
of requests per second by provider made it obvious.

## 3. "What threshold should the semantic cache use?" has a measurable answer

A semantic cache returns a stored answer when a new question is similar
enough to an old one. The usual advice is a cosine similarity around 0.95.
I replayed 20,000 Quora question pairs as a cache and counted how often the
cached answer belonged to a different question. At 0.95 about 1 hit in 10
was wrong. Picking the lowest threshold with at most 2 wrong answers per
1,000 lookups gave 0.99.

Embeddings also ignore word order. "How do I convert a string to an integer
in JavaScript?" and "...an integer to a string..." score 0.996, which clears
any threshold. A small lexical check for swapped direction, changed numbers
and common opposites blocks those, but on a held-out set written after the
check was frozen it still let through about 1 bad hit in 6.

## 4. The cache that saved money was the boring one

I replayed 3,000 requests of an FAQ-style workload (popular questions asked
in different real phrasings, plus one-offs) with caching off, exact-match
only, and exact plus semantic. The exact cache cut spend by 53%. The semantic
cache added 0.4 percentage points (29 hits, none wrong), while making every
cache miss about 55 ms slower because each one pays for an embedding and a
vector search. On a workload with no repeats, both saved nothing.

So the semantic cache ships switched off, with a one-line setting to turn it
on where prompts repeat with small wording changes. The measurement changed
the default.

## 5. Where the overhead goes

Below saturation the gateway adds 15 ms at the median with 10 concurrent
streams and 37 ms with 100 (realistic 2-second streams). The first
throughput test was much worse: about 0.7 seconds added at 100 streams. A
profiler showed two surprises in the HTTP client. Responses were closed
before their last bytes were read, so every request opened a new connection;
and once connections were reused, the connection pool checked every idle
connection each time it handed one out, which at a few hundred connections
was the single biggest CPU cost. Draining responses, splitting the pool into
16 small pools, batching tokens per network read and running several worker
processes brought it to the numbers above.

The honest limit: on a 4-core machine that also runs the load generator and
the mock, two gateway processes top out around 100–140 new streams per
second. At 500 and 1,000 concurrent streams the benchmark measures queueing,
and says so. The gateway keeps no state in memory, so more capacity means
more processes or hosts.

## What I would do next

Route by price and latency instead of a fixed order; merge the rate-limit and
budget checks into one Redis call; translate tool calls for Anthropic; and
rerun the load test with the load generator on a separate machine.

---

## LinkedIn version

I built a self-hosted LLM gateway (OpenAI-compatible API in front of OpenAI,
Anthropic and a local model) and then tried to break it. Four things the
measurements taught me:

1. Failover is free until the first token. With the primary returning 503s
   for 30 s under load: 2,559 requests, 0 failed.
2. Chaos testing found a real bug: my circuit breaker counted a stream as
   healthy at its first token, so a provider that died mid-stream kept
   closing its own breaker. 281 errors before the fix, 21 after.
3. The usual semantic-cache threshold (0.95) was wrong about 1 time in 10 on
   Quora question pairs. Measured choice: 0.99.
4. The exact-match cache saved 53% on an FAQ-style workload. The semantic
   cache added 0.4 points and ~55 ms to every miss, so it ships off.

Code, benchmark scripts and raw results: https://github.com/shubham0745/LLM-Gateway
