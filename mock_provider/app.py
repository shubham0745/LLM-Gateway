"""A small stand-in for an OpenAI-style API, for tests, CI and local dev.

It answers ``POST /v1/chat/completions`` (streaming and not) and
``GET /v1/models`` with deterministic text: "Mock reply to: <last user message>".
One word is one token.

Behaviour is set by environment variables, and each can be overridden per
request with a header, so a single mock serves every test:

======================  ============================  =================================
Env var                 Header                        Meaning
======================  ============================  =================================
MOCK_LATENCY_MS         x-mock-latency-ms             Delay before the response/first token
MOCK_TOKENS_PER_SECOND  x-mock-tokens-per-second      Streaming speed; 0 means no delay
(none)                  x-mock-status                 Reply with this HTTP error status
======================  ============================  =================================

The last received request is kept at ``app.state.last_request`` for tests.

Run with::

    uvicorn mock_provider.app:app --port 9000
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MODELS = ["mock-small", "mock-large"]


def _setting(request: Request, header: str, env: str, default: float) -> float:
    raw = request.headers.get(header) or os.environ.get(env)
    return float(raw) if raw else default


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _reply_tokens(messages: list[dict[str, Any]]) -> list[str]:
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    prompt = _text_of(last_user.get("content")) if last_user else ""
    words = f"Mock reply to: {prompt}".split()
    # Leading spaces on every token after the first, as real tokenizers do.
    return [words[0], *(f" {w}" for w in words[1:])]


def _usage(messages: list[dict[str, Any]], completion_tokens: int) -> dict[str, int]:
    prompt_tokens = sum(len(_text_of(m.get("content")).split()) for m in messages)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "mock_error", "code": str(status)}},
        status_code=status,
    )


app = FastAPI(title="Mock LLM provider")
app.state.last_request = None


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "created": 0, "owned_by": "mock"} for m in MODELS],
    }


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    body = await request.json()
    app.state.last_request = {"headers": dict(request.headers), "json": body}

    if status := request.headers.get("x-mock-status"):
        return _error(int(status), f"mock failure with status {status}")
    if not request.headers.get("authorization", "").startswith("Bearer "):
        return _error(401, "missing bearer token")
    model = body.get("model")
    if model not in MODELS:
        return _error(404, f"model {model!r} does not exist")

    latency = _setting(request, "x-mock-latency-ms", "MOCK_LATENCY_MS", 0) / 1000
    tps = _setting(request, "x-mock-tokens-per-second", "MOCK_TOKENS_PER_SECOND", 50)
    messages = body.get("messages", [])
    tokens = _reply_tokens(messages)
    completion_id = f"chatcmpl-mock-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if latency:
        await asyncio.sleep(latency)

    if not body.get("stream"):
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "".join(tokens)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage(messages, len(tokens)),
            }
        )

    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
        payload: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if include_usage:
            payload["usage"] = None
        return f"data: {json.dumps(payload)}\n\n".encode()

    async def stream() -> AsyncIterator[bytes]:
        yield chunk({"role": "assistant", "content": ""})
        for token in tokens:
            if tps > 0:
                await asyncio.sleep(1 / tps)
            yield chunk({"content": token})
        yield chunk({}, finish_reason="stop")
        if include_usage:
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": _usage(messages, len(tokens)),
            }
            yield f"data: {json.dumps(final)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
