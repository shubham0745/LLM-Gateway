"""OpenAI-compatible endpoints: /v1/chat/completions and /v1/models."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from gateway.accounting.tokens import estimate_prompt_tokens, estimate_text_tokens
from gateway.errors import GatewayError, ProviderError
from gateway.routing.router import StreamHandle
from gateway.services import Services
from gateway.telemetry.record import RequestRecord

router = APIRouter()
logger = logging.getLogger(__name__)


class ClosingStreamingResponse(StreamingResponse):
    """Guarantees the body generator is closed however the response ends.

    If the client disconnects while the generator is suspended at a ``yield``,
    Starlette abandons it without closing it. Closing it here runs its
    ``finally`` block, which closes the upstream HTTP stream so the provider
    stops generating tokens nobody will read.
    """

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[override]
        try:
            await super().__call__(scope, receive, send)
        finally:
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                await aclose()


def _services(request: Request) -> Services:
    return request.app.state.services


async def _parse_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise GatewayError(400, "Request body must be valid JSON.") from None
    if not isinstance(body, dict):
        raise GatewayError(400, "Request body must be a JSON object.")
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise GatewayError(400, "'model' is required.", code="missing_model")
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) and "role" in m for m in msgs):
        raise GatewayError(400, "'messages' must be a non-empty list of {role, content} objects.", code="invalid_messages")
    return body


def _gateway_headers(record: RequestRecord) -> dict[str, str]:
    h = {"x-request-id": record.request_id, "x-gateway-cache": record.cache_status}
    if record.provider:
        h["x-gateway-provider"] = record.provider
        h["x-gateway-model"] = record.model or ""
    h["x-gateway-attempts"] = str(sum(1 for a in record.attempts if a.outcome in ("ok", "error")))
    return h


def _content_of_response(resp: dict) -> str:
    try:
        return resp["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def _apply_usage(record: RequestRecord, usage: dict | None, body: dict, completion_text: str) -> None:
    if usage and usage.get("prompt_tokens") is not None:
        record.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        record.completion_tokens = int(usage.get("completion_tokens") or 0)
    else:
        record.prompt_tokens = estimate_prompt_tokens(body)
        record.completion_tokens = estimate_text_tokens(completion_text)
        record.usage_estimated = True


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    svc = _services(request)
    await svc.keys.authenticate(request.headers.get("authorization"))
    now = int(time.time())
    data = [{"id": alias, "object": "model", "created": now, "owned_by": "gateway"} for alias in sorted(svc.config.aliases)]
    return JSONResponse({"object": "list", "data": data})


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    svc = _services(request)
    record = RequestRecord()
    try:
        principal = await svc.keys.authenticate(request.headers.get("authorization"))
        record.tenant_id, record.key_id = principal.tenant_id, principal.key_id
        body = await _parse_body(request)
        record.alias = body["model"]
        record.stream = bool(body.get("stream"))
        chain = svc.router.resolve(body["model"])

        if record.stream:
            handle = await svc.router.open_stream(body, chain, record)
            record.provider, record.model = handle.target.provider, handle.target.model
            gen = _relay_stream(svc, body, handle, record)
            return ClosingStreamingResponse(
                gen, media_type="text/event-stream", headers={**_gateway_headers(record), "cache-control": "no-cache"}
            )

        target, resp = await svc.router.complete(body, chain, record)
        record.provider, record.model = target.provider, target.model
        resp["id"] = "chatcmpl-" + record.request_id[4:]
        _apply_usage(record, resp.get("usage"), body, _content_of_response(resp))
        resp["usage"] = {
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "total_tokens": record.prompt_tokens + record.completion_tokens,
        }
        record.finish()
        svc.telemetry.emit(record)
        return JSONResponse(resp, headers=_gateway_headers(record))
    except GatewayError as err:
        record.status, record.http_status, record.error = "error", err.status, err.message
        record.finish()
        svc.telemetry.emit(record)
        return err.response(_gateway_headers(record))


async def _relay_stream(svc: Services, body: dict, handle: StreamHandle, record: RequestRecord) -> AsyncIterator[bytes]:
    completion_id = "chatcmpl-" + record.request_id[4:]
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    text_parts: list[str] = []
    usage: dict | None = None
    completed = False
    try:
        async for chunk in svc.router.iterate(handle, record):
            if chunk.get("usage"):
                usage = chunk["usage"]
            if not chunk.get("choices"):
                continue  # usage-only chunk; re-emitted below if the caller asked for it
            chunk["id"] = completion_id
            chunk["model"] = record.model
            chunk.pop("usage", None)
            for choice in chunk["choices"]:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    text_parts.append(piece)
            yield b"data: " + json.dumps(chunk, separators=(",", ":")).encode() + b"\n\n"
        _apply_usage(record, usage, body, "".join(text_parts))
        if include_usage:
            u = {
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "total_tokens": record.prompt_tokens + record.completion_tokens,
            }
            final = {"id": completion_id, "object": "chat.completion.chunk", "created": int(time.time()),
                     "model": record.model, "choices": [], "usage": u}
            yield b"data: " + json.dumps(final, separators=(",", ":")).encode() + b"\n\n"
        yield b"data: [DONE]\n\n"
        completed = True
    except (TimeoutError, ProviderError) as exc:
        # After the first token we cannot fail over invisibly: the client has
        # already shown partial output. We end the stream with an explicit
        # error event (the OpenAI SDK raises APIError on it) rather than a
        # silent truncation that looks like a complete answer.
        kind = exc.kind if isinstance(exc, ProviderError) else "stream_timeout"
        record.status, record.http_status, record.error = "error", 502, f"mid-stream failure: {kind}"
        _apply_usage(record, None, body, "".join(text_parts))
        err = {"error": {"message": f"Upstream stream failed after output started ({kind}).",
                         "type": "api_error", "code": "upstream_stream_error", "param": None}}
        yield b"data: " + json.dumps(err).encode() + b"\n\n"
        completed = True
    finally:
        if not completed:
            # Client went away (cancellation or generator close). Bill what was generated.
            record.status, record.http_status = "client_disconnected", 499
            _apply_usage(record, usage, body, "".join(text_parts))
        await handle.close()
        record.finish()
        svc.telemetry.emit(record)
