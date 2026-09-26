"""A fake LLM API for load and chaos testing.

Speaks OpenAI's ``/v1/chat/completions`` (streaming and not). Several named
instances live in one process so a single container can play "primary" and
"backup": ``/{instance}/v1/chat/completions`` (``/v1/...`` is instance
``default``).

Behaviour is controlled per instance at runtime, so a test can flip a provider
into an outage while traffic is flowing::

    PUT  /control/{instance}   {"mode": "outage"}        # every request 503s
    PUT  /control/{instance}   {"error_rate": 0.2, "error_status": 500}
    PUT  /control/{instance}   {"mode": "hang"}          # accept, never answer
    PUT  /control/{instance}   {"mode": "midstream_disconnect", "disconnect_after_tokens": 5}
    PUT  /control/{instance}   {"rate_limit_rate": 0.5, "retry_after": 1}
    POST /control/{instance}/reset
    GET  /stats                                          # per-instance counters

A single request can also override behaviour with headers
(``x-mock-mode``, ``x-mock-ttft-ms``, ``x-mock-tokens``, ...), which is how the
unit tests exercise each failure without shared state.

Responses are deterministic for a given prompt (seeded by its hash) so cache
tests are repeatable, and usage is reported like OpenAI does.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

WORDS = (
    "the gateway routes each request through authentication limits cache and a provider chain "
    "latency matters because streaming connections stay open for many seconds while tokens arrive "
    "a circuit breaker opens after repeated failures and probes again after a cooldown period "
    "caching saves money only when the hit rate is high enough to pay for the lookup cost"
).split()


@dataclass
class Behavior:
    mode: str = "ok"  # ok | outage | hang | midstream_disconnect | slow
    ttft_ms: float = 50.0
    tokens_per_s: float = 200.0
    response_tokens: int = 40
    error_rate: float = 0.0
    error_status: int = 500
    rate_limit_rate: float = 0.0
    retry_after: float | None = 1.0
    disconnect_after_tokens: int = 5
    model_name: str | None = None
    # Fail exactly the next N requests with fail_next_status, then behave normally.
    fail_next: int = 0
    fail_next_status: int = 500


@dataclass
class Stats:
    requests: int = 0
    streams_started: int = 0
    streams_completed: int = 0
    streams_cancelled: int = 0
    tokens_generated: int = 0
    errors_injected: int = 0
    rate_limited: int = 0
    by_status: dict[str, int] = field(default_factory=dict)


app = FastAPI(title="Mock LLM provider")
_behaviors: dict[str, Behavior] = {}
_stats: dict[str, Stats] = {}


def _b(instance: str) -> Behavior:
    return _behaviors.setdefault(instance, Behavior())


def _s(instance: str) -> Stats:
    return _stats.setdefault(instance, Stats())


def _count(instance: str, status: int) -> None:
    st = _s(instance)
    st.by_status[str(status)] = st.by_status.get(str(status), 0) + 1


def _effective(instance: str, request: Request) -> Behavior:
    b = Behavior(**asdict(_b(instance)))
    h = request.headers
    if "x-mock-mode" in h:
        b.mode = h["x-mock-mode"]
    for name, attr, cast in (
        ("x-mock-ttft-ms", "ttft_ms", float),
        ("x-mock-tokens-per-s", "tokens_per_s", float),
        ("x-mock-tokens", "response_tokens", int),
        ("x-mock-error-rate", "error_rate", float),
        ("x-mock-error-status", "error_status", int),
        ("x-mock-rate-limit-rate", "rate_limit_rate", float),
        ("x-mock-retry-after", "retry_after", float),
        ("x-mock-disconnect-after", "disconnect_after_tokens", int),
    ):
        if name in h:
            setattr(b, attr, cast(h[name]))
    return b


def _prompt_text(body: dict) -> str:
    parts = []
    for m in body.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(p.get("text", "") for p in c if isinstance(p, dict))
    return "\n".join(parts)


def _tokens_for(body: dict, n: int) -> list[str]:
    seed = int(hashlib.sha256(_prompt_text(body).encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    if max_tokens:
        n = min(n, int(max_tokens))
    return [(" " if i else "") + rng.choice(WORDS) for i in range(n)]


def _usage(body: dict, completion_tokens: int) -> dict:
    prompt_tokens = max(1, len(_prompt_text(body)) // 4) + 3 * len(body.get("messages") or [])
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}


def _error(status: int, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "mock_error", "code": status}}, status_code=status, headers=headers)


@app.get("/stats")
async def stats() -> dict:
    return {k: asdict(v) for k, v in _stats.items()}


@app.get("/control")
async def get_all_controls() -> dict:
    return {k: asdict(v) for k, v in _behaviors.items()}


@app.put("/control/{instance}")
async def set_control(instance: str, patch: dict[str, Any]) -> dict:
    b = _b(instance)
    for k, v in patch.items():
        if not hasattr(b, k):
            return _error(400, f"unknown field {k}")  # type: ignore[return-value]
        setattr(b, k, v)
    return asdict(b)


@app.post("/control/{instance}/reset")
async def reset_control(instance: str) -> dict:
    _behaviors[instance] = Behavior()
    _stats[instance] = Stats()
    return asdict(_behaviors[instance])


@app.post("/control/reset-all")
async def reset_all() -> dict:
    _behaviors.clear()
    _stats.clear()
    return {"ok": True}


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_default(request: Request) -> Response:
    return await _chat("default", request)


@app.post("/{instance}/v1/chat/completions")
async def chat_instance(instance: str, request: Request) -> Response:
    return await _chat(instance, request)


async def _chat(instance: str, request: Request) -> Response:
    body = await request.json()
    b = _effective(instance, request)
    st = _s(instance)
    st.requests += 1

    if b.mode == "outage":
        _count(instance, 503)
        st.errors_injected += 1
        return _error(503, "mock outage")
    if b.mode == "hang":
        # Accept the connection and never answer; the gateway's timeouts must save us.
        await asyncio.sleep(3600)
    if _b(instance).fail_next > 0:
        _b(instance).fail_next -= 1
        _count(instance, b.fail_next_status)
        st.errors_injected += 1
        headers = {"retry-after": str(b.retry_after)} if b.fail_next_status == 429 and b.retry_after is not None else None
        return _error(b.fail_next_status, "mock scripted failure", headers)
    if b.rate_limit_rate and random.random() < b.rate_limit_rate:
        _count(instance, 429)
        st.rate_limited += 1
        headers = {"retry-after": str(b.retry_after)} if b.retry_after is not None else None
        return _error(429, "mock rate limit", headers)
    if b.error_rate and random.random() < b.error_rate:
        _count(instance, b.error_status)
        st.errors_injected += 1
        return _error(b.error_status, "mock injected error")
    if body.get("messages") is None:
        _count(instance, 400)
        return _error(400, "messages is required")

    model = b.model_name or body.get("model", "mock")
    tokens = _tokens_for(body, b.response_tokens)
    cid = "chatcmpl-mock-" + uuid.uuid4().hex[:12]
    created = int(time.time())
    _count(instance, 200)

    if not body.get("stream"):
        await asyncio.sleep(b.ttft_ms / 1000 + len(tokens) / max(b.tokens_per_s, 1e-6))
        st.tokens_generated += len(tokens)
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(tokens)},
                             "finish_reason": "stop", "logprobs": None}],
                "usage": _usage(body, len(tokens)),
            }
        )

    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    async def gen():
        st.streams_started += 1
        sent = 0
        try:
            await asyncio.sleep(b.ttft_ms / 1000)
            first = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
            yield f"data: {json.dumps(first)}\n\n"
            delay = 1.0 / max(b.tokens_per_s, 1e-6)
            for i, tok in enumerate(tokens):
                if b.mode == "midstream_disconnect" and i >= b.disconnect_after_tokens:
                    # Abruptly end the body without a finish chunk or [DONE].
                    st.tokens_generated += sent
                    return
                if i:
                    await asyncio.sleep(delay)
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                sent += 1
            last = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(last)}\n\n"
            if include_usage:
                u = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                     "choices": [], "usage": _usage(body, len(tokens))}
                yield f"data: {json.dumps(u)}\n\n"
            yield "data: [DONE]\n\n"
            st.streams_completed += 1
            st.tokens_generated += sent
        except (asyncio.CancelledError, GeneratorExit):
            # The gateway closed the connection: we stop "generating" right here.
            st.streams_cancelled += 1
            st.tokens_generated += sent
            raise

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Anthropic Messages API imitation, so the Anthropic adapter's translation
# (including its streaming event conversion) is tested without a real key.
# ---------------------------------------------------------------------------


@app.post("/{instance}/v1/messages")
async def anthropic_messages(instance: str, request: Request) -> Response:
    body = await request.json()
    b = _effective(instance, request)
    st = _s(instance)
    st.requests += 1
    if request.headers.get("anthropic-version") is None:
        _count(instance, 400)
        return _error(400, "anthropic-version header is required")
    if b.mode == "outage":
        _count(instance, 529)
        st.errors_injected += 1
        return JSONResponse({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}, status_code=529)
    if b.error_rate and random.random() < b.error_rate:
        _count(instance, b.error_status)
        st.errors_injected += 1
        return _error(b.error_status, "mock injected error")
    if "max_tokens" not in body:
        _count(instance, 400)
        return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: Field required"}}, status_code=400)

    oa_like = {"messages": [{"content": body.get("system", "")}] + [
        {"content": "".join(p.get("text", "") for p in m["content"]) if isinstance(m["content"], list) else m["content"]}
        for m in body.get("messages", [])
    ]}
    tokens = _tokens_for({**oa_like, "max_tokens": body["max_tokens"]}, b.response_tokens)
    usage_in = _usage(oa_like, 0)["prompt_tokens"]
    mid = "msg_mock_" + uuid.uuid4().hex[:12]
    model = body.get("model", "mock")
    _count(instance, 200)
    stop_reason = "max_tokens" if len(tokens) >= body["max_tokens"] and len(tokens) < b.response_tokens else "end_turn"

    if not body.get("stream"):
        await asyncio.sleep(b.ttft_ms / 1000)
        st.tokens_generated += len(tokens)
        return JSONResponse({
            "id": mid, "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": "".join(tokens)}],
            "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": usage_in, "output_tokens": len(tokens)},
        })

    def ev(name: str, data: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n"

    async def gen():
        st.streams_started += 1
        sent = 0
        try:
            await asyncio.sleep(b.ttft_ms / 1000)
            yield ev("message_start", {"type": "message_start", "message": {
                "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
                "stop_reason": None, "usage": {"input_tokens": usage_in, "output_tokens": 1}}})
            yield ev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
            yield ev("ping", {"type": "ping"})
            for i, tok in enumerate(tokens):
                if b.mode == "midstream_disconnect" and i >= b.disconnect_after_tokens:
                    return
                if i:
                    await asyncio.sleep(1.0 / max(b.tokens_per_s, 1e-6))
                yield ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": tok}})
                sent += 1
            yield ev("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                       "usage": {"output_tokens": len(tokens)}})
            yield ev("message_stop", {"type": "message_stop"})
            st.streams_completed += 1
        except (asyncio.CancelledError, GeneratorExit):
            st.streams_cancelled += 1
            raise
        finally:
            st.tokens_generated += sent

    return StreamingResponse(gen(), media_type="text/event-stream")
