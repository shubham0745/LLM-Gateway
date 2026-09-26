"""Phase 4: exact and semantic caching."""

from __future__ import annotations

import asyncio
import time

import httpx
import openai
import pytest

from tests.conftest import DB_URL, REDIS_URL, ROOT

ADMIN = {"authorization": "Bearer test-admin"}
HAS_MODEL = (ROOT / "models" / "all-MiniLM-L6-v2" / "model.onnx").exists()


@pytest.fixture
def cclient(gateway_server, make_key, request):
    key = make_key(f"t-cache-{request.node.name[:40]}")
    return openai.OpenAI(base_url=f"{gateway_server.url}/v1", api_key=key, max_retries=0, timeout=15)


def ask(c, content="What is a circuit breaker?", model="mock", system=None, stream=False, headers=None, **kw):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    if stream:
        with c.chat.completions.with_streaming_response.create(model=model, messages=msgs, stream=True,
                                                               extra_headers=headers, **kw) as r:
            text = ""
            for chunk in r.parse():
                if chunk.choices:
                    text += chunk.choices[0].delta.content or ""
            return text, r.headers
    raw = c.chat.completions.with_raw_response.create(model=model, messages=msgs, extra_headers=headers, **kw)
    return raw.parse().choices[0].message.content, raw.headers


def drain_worker():
    from redis.asyncio import Redis

    from gateway import db
    from worker.main import Writer

    async def go():
        pool = await db.create_pool(DB_URL, min_size=1, max_size=2)
        redis = Redis.from_url(REDIS_URL)
        w = Writer(pool, redis, "gw:events", "test-worker")
        await w.ensure_group()
        while await w.run_once(block_ms=200):
            pass
        await redis.aclose()
        await pool.close()

    time.sleep(0.3)  # let the gateway flush its event batch
    asyncio.run(go())


def wait_for_exact_store():
    time.sleep(0.1)  # exact entries are written by a background task


def test_exact_hit(cclient, mock_stats):
    t1, h1 = ask(cclient)
    wait_for_exact_store()
    t2, h2 = ask(cclient)
    assert h1["x-gateway-cache"] == "miss" and h2["x-gateway-cache"] == "hit-exact"
    assert t1 == t2
    assert h2["x-gateway-provider"] == "cache" and h2["x-gateway-cached-from"] == "mock-primary"
    assert float(h2["x-gateway-cost-usd"]) == 0
    assert mock_stats("primary")["requests"] == 1


def test_stream_is_cached_and_replayed_as_stream(cclient, mock_stats):
    t1, h1 = ask(cclient, stream=True)
    wait_for_exact_store()
    t2, h2 = ask(cclient, stream=True)
    t3, h3 = ask(cclient)  # the non-streaming form shares the entry
    assert h2["x-gateway-cache"] == "hit-exact" and h3["x-gateway-cache"] == "hit-exact"
    assert t1 == t2 == t3
    assert mock_stats("primary")["requests"] == 1


def test_replayed_stream_is_chunked_like_a_real_one(cclient):
    ask(cclient, stream=True)
    wait_for_exact_store()
    chunks = list(cclient.chat.completions.create(model="mock", messages=[{"role": "user", "content": "What is a circuit breaker?"}],
                                                  stream=True, stream_options={"include_usage": True}))
    content_chunks = [c for c in chunks if c.choices and c.choices[0].delta.content]
    assert len(content_chunks) > 5
    assert content_chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-2].choices[0].finish_reason == "stop"
    assert chunks[-1].usage.completion_tokens == 40


def test_different_parameters_miss(cclient):
    ask(cclient, temperature=0.2)
    wait_for_exact_store()
    _, h = ask(cclient, temperature=0.9)
    assert h["x-gateway-cache"] == "miss"


def test_cache_is_scoped_per_tenant(gateway_server, make_key, cclient):
    ask(cclient, content="tenant scoped question")
    wait_for_exact_store()
    other = openai.OpenAI(base_url=f"{gateway_server.url}/v1", api_key=make_key("t-cache-other-tenant"), max_retries=0)
    _, h = ask(other, content="tenant scoped question")
    assert h["x-gateway-cache"] == "miss"


def test_bypass_headers(cclient, mock_stats):
    ask(cclient, headers={"x-gateway-cache": "no-store"})
    wait_for_exact_store()
    _, h = ask(cclient)
    assert h["x-gateway-cache"] == "miss"  # no-store wrote nothing
    wait_for_exact_store()
    _, h = ask(cclient, headers={"x-gateway-cache": "no-cache"})
    assert h["x-gateway-cache"] == "bypass"  # entry exists, but the caller asked for a fresh answer
    assert mock_stats("primary")["requests"] == 3


def test_disabled_alias_never_cached(cclient):
    ask(cclient, model="primary-only")
    wait_for_exact_store()
    _, h = ask(cclient, model="primary-only")
    assert h["x-gateway-cache"] == "miss"


def test_truncated_stream_not_cached(cclient, mock_control):
    mock_control("primary", mode="midstream_disconnect", disconnect_after_tokens=3)
    with pytest.raises(openai.APIError):
        ask(cclient, stream=True)
    mock_control("primary", mode="ok")
    wait_for_exact_store()
    _, h = ask(cclient)
    assert h["x-gateway-cache"] == "miss"


def test_cache_hit_saves_money_and_is_logged(gateway_server, cclient):
    ask(cclient, content="savings question")
    wait_for_exact_store()
    _, h = ask(cclient, content="savings question")
    drain_worker()
    row = httpx.get(f"{gateway_server.url}/admin/requests/{h['x-request-id']}", headers=ADMIN).json()
    assert row["cache_status"] == "hit-exact" and row["cost_usd"] == 0 and row["saved_usd"] > 0
    assert "gateway_cache_saved_usd_total" in httpx.get(f"{gateway_server.url}/metrics").text


@pytest.mark.skipif(not HAS_MODEL, reason="embedding model not downloaded")
def test_semantic_hit_on_paraphrase(cclient, mock_stats):
    t1, _ = ask(cclient, content="What is the capital of France?")
    drain_worker()  # semantic entries are inserted by the worker
    t2, h = ask(cclient, content="what's the capital of france")
    assert h["x-gateway-cache"] == "hit-semantic", h.get("x-gateway-cache")
    assert float(h["x-gateway-cache-similarity"]) >= 0.9
    assert t1 == t2
    assert mock_stats("primary")["requests"] == 1


@pytest.mark.skipif(not HAS_MODEL, reason="embedding model not downloaded")
def test_semantic_rejects_unrelated_prompt(cclient):
    ask(cclient, content="What is the capital of France?")
    drain_worker()
    _, h = ask(cclient, content="What is the capital of Australia?")
    assert h["x-gateway-cache"] == "miss"


@pytest.mark.skipif(not HAS_MODEL, reason="embedding model not downloaded")
def test_semantic_scope_includes_system_prompt(cclient):
    ask(cclient, content="Explain recursion", system="Answer like a pirate.")
    drain_worker()
    _, h = ask(cclient, content="Explain recursion please", system="Answer formally.")
    assert h["x-gateway-cache"] == "miss"
    _, h = ask(cclient, content="Explain recursion please", system="Answer like a pirate.")
    assert h["x-gateway-cache"] == "hit-semantic"


@pytest.mark.skipif(not HAS_MODEL, reason="embedding model not downloaded")
def test_semantic_skips_multi_turn(cclient):
    ask(cclient, content="What is the capital of France?")
    drain_worker()
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "what's the capital of france"}]
    raw = cclient.chat.completions.with_raw_response.create(model="mock", messages=msgs)
    assert raw.headers["x-gateway-cache"] == "miss"


@pytest.mark.skipif(not HAS_MODEL, reason="embedding model not downloaded")
def test_semantic_scoped_per_tenant(gateway_server, make_key, cclient):
    ask(cclient, content="How do vaccines train the immune system?")
    drain_worker()
    other = openai.OpenAI(base_url=f"{gateway_server.url}/v1", api_key=make_key("t-cache-sem-other"), max_retries=0)
    _, h = ask(other, content="How do vaccines train the immune system?")
    assert h["x-gateway-cache"] == "miss"


def test_normalization_ignores_irrelevant_fields(cclient):
    ask(cclient, content="  normalize me  ", user="alice")
    wait_for_exact_store()
    _, h = ask(cclient, content="normalize me", user="bob", stream=False)
    assert h["x-gateway-cache"] == "hit-exact"


@pytest.mark.skipif(not HAS_MODEL, reason="embedding model not downloaded")
def test_semantic_guard_blocks_direction_swap(cclient):
    ask(cclient, content="How do I convert a string to an integer in JavaScript?")
    drain_worker()
    # Similarity is ~0.996 here: only the guard stops a wrong answer.
    _, h = ask(cclient, content="How do I convert an integer to a string in JavaScript?")
    assert h["x-gateway-cache"] == "miss"


def test_guard_rules():
    from gateway.cache.guard import compatible

    assert compatible("What is the capital of France?", "what's the capital city of france")[0]
    assert not compatible("Convert 5 km to miles", "Convert 5 miles to km")[0]
    assert not compatible("What is 15% of 200?", "What is 20% of 150?")[0]
    assert not compatible("Sort ascending in Python", "Sort descending in Python")[0]
    assert compatible("In JavaScript, how do I turn a string into an integer?", "How do I convert a string to an integer in JavaScript?")[0]
