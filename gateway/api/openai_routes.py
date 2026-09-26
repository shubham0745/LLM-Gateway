"""OpenAI-compatible endpoints: /v1/chat/completions and /v1/models."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import orjson
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from gateway.accounting.pricing import cost_usd
from gateway.accounting.tokens import estimate_prompt_tokens, estimate_text_tokens
from gateway.cache.layer import CachedAnswer, Lookup, cache_directive
from gateway.errors import GatewayError, ProviderError
from gateway.limits.guard import Reservation
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
    stops generating tokens nobody will read. ``on_unstarted`` covers the
    corner where the client vanished before the first byte, so the generator
    never ran at all (closing an unstarted generator skips its ``finally``).
    """

    def __init__(self, content, *, on_unstarted=None, started=None, **kw):
        super().__init__(content, **kw)
        self._on_unstarted = on_unstarted
        self._started = started

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[override]
        try:
            await super().__call__(scope, receive, send)
        finally:
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                await aclose()
            if self._on_unstarted is not None and self._started is not None and not self._started.get("yes"):
                self._on_unstarted()


def _services(request: Request) -> Services:
    return request.app.state.services


async def _parse_body(request: Request) -> dict[str, Any]:
    try:
        body = orjson.loads(await request.body())
    except orjson.JSONDecodeError:
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
    if not record.stream:
        h["x-gateway-cost-usd"] = f"{record.cost_usd:.8f}"
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
    reservation: Reservation | None = None
    handed_off = False  # a streaming response now owns finalization
    svc.telemetry.metrics.inflight.inc()
    try:
        # authenticate -> check limits -> (cache) -> pick a route -> call the provider -> record
        principal = await svc.keys.authenticate(request.headers.get("authorization"))
        record.tenant_id, record.key_id = principal.tenant_id, principal.key_id
        body = await _parse_body(request)
        record.alias = body["model"]
        record.stream = bool(body.get("stream"))
        chain = svc.router.resolve(body["model"])
        reservation = await svc.limits.admit(principal, body, chain, svc.config)

        lookup: Lookup | None = None
        if svc.cache is not None:
            directive = cache_directive(request.headers)
            cached, lookup = await svc.cache.lookup(principal.tenant_id, record.alias, body, svc.config.cache, directive)
            record.cache_status = "bypass" if directive else "miss"
            if cached is not None:
                handed_off = record.stream
                return _serve_cached(svc, body, record, reservation, cached)

        if record.stream:
            handle = await svc.router.open_stream(body, chain, record)
            record.provider, record.model = handle.target.provider, handle.target.model
            started: dict[str, bool] = {}
            gen = _relay_stream(svc, body, handle, record, reservation, started, lookup)

            def never_started() -> None:
                record.status, record.http_status = "client_disconnected", 499
                svc.spawn(handle.close())
                _finalize(svc, record, reservation)

            handed_off = True
            return ClosingStreamingResponse(
                gen,
                on_unstarted=never_started,
                started=started,
                media_type="text/event-stream",
                headers={**reservation.headers, **_gateway_headers(record), "cache-control": "no-cache"},
            )

        target, resp = await svc.router.complete(body, chain, record)
        record.provider, record.model = target.provider, target.model
        resp["id"] = "chatcmpl-" + record.request_id[4:]
        _apply_usage(record, resp.get("usage"), body, _content_of_response(resp))
        _maybe_store(svc, record, lookup, _content_of_response(resp), _finish_reason(resp), resp)
        resp["usage"] = {
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "total_tokens": record.prompt_tokens + record.completion_tokens,
        }
        _finalize(svc, record, reservation)
        return JSONResponse(resp, headers={**reservation.headers, **_gateway_headers(record)})
    except GatewayError as err:
        record.status, record.http_status, record.error = "error", err.status, err.message
        if err.code in ("rate_limit_exceeded", "request_too_large", "budget_exceeded"):
            record.status = "rejected"
        _finalize(svc, record, reservation)
        return err.response(_gateway_headers(record))
    finally:
        if not handed_off:
            svc.telemetry.metrics.inflight.dec()


def _finish_reason(resp: dict) -> str | None:
    try:
        return resp["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def _maybe_store(svc: Services, record: RequestRecord, lookup: Lookup | None, content: str, finish: str | None, resp: dict | None) -> None:
    """Cache complete, plain-text, single-choice answers only."""
    if svc.cache is None or lookup is None or not lookup.store or not content or finish not in ("stop", "length"):
        return
    if resp is not None:
        choices = resp.get("choices") or []
        if len(choices) != 1 or (choices[0].get("message") or {}).get("tool_calls"):
            return
    answer = CachedAnswer(content, finish, record.model or "", record.provider or "", record.prompt_tokens,
                          record.completion_tokens, kind="").to_json()
    svc.spawn(svc.cache.store(record.tenant_id or "", lookup, answer, svc.config.cache, svc.telemetry.enqueue_event))


def _serve_cached(svc: Services, body: dict, record: RequestRecord, reservation: Reservation, hit: CachedAnswer) -> Response:
    """Answer from the cache. Streams are replayed chunk by chunk so clients can't tell the difference."""
    record.cache_status = hit.kind
    record.provider, record.model = "cache", hit.model
    # The caller pays nothing; we record what the answer would have cost.
    record.saved_usd = round(cost_usd(svc.config.pricing, hit.model, hit.prompt_tokens, hit.completion_tokens), 8)
    headers = {**reservation.headers, **_gateway_headers(record), "x-gateway-cached-from": hit.provider}
    if hit.similarity is not None:
        headers["x-gateway-cache-similarity"] = str(hit.similarity)
    completion_id = "chatcmpl-" + record.request_id[4:]
    usage = {"prompt_tokens": hit.prompt_tokens, "completion_tokens": hit.completion_tokens,
             "total_tokens": hit.prompt_tokens + hit.completion_tokens}
    if not record.stream:
        _finalize(svc, record, reservation)
        return JSONResponse({
            "id": completion_id, "object": "chat.completion", "created": int(time.time()), "model": hit.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": hit.content},
                         "finish_reason": hit.finish_reason, "logprobs": None}],
            "usage": usage,
        }, headers=headers)

    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    record.mark_first_token()

    started: dict[str, bool] = {}

    async def replay() -> AsyncIterator[bytes]:
        started["yes"] = True
        try:
            created = int(time.time())
            pieces = _split_for_replay(hit.content)
            for i, piece in enumerate(pieces):
                delta = {"role": "assistant", "content": piece} if i == 0 else {"content": piece}
                yield _sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": hit.model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None, "logprobs": None}]})
            yield _sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": hit.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": hit.finish_reason, "logprobs": None}]})
            if include_usage:
                yield _sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": hit.model,
                            "choices": [], "usage": usage})
            yield b"data: [DONE]\n\n"
        finally:
            _finalize(svc, record, reservation)

    return ClosingStreamingResponse(replay(), started=started, on_unstarted=lambda: _finalize(svc, record, reservation),
                                    media_type="text/event-stream",
                                    headers={**headers, "cache-control": "no-cache"})


def _split_for_replay(text: str, words_per_chunk: int = 3) -> list[str]:
    import re

    tokens = re.findall(r"\s*\S+", text) or [text]
    return ["".join(tokens[i:i + words_per_chunk]) for i in range(0, len(tokens), words_per_chunk)]


def _sse(obj: dict) -> bytes:
    return b"data: " + orjson.dumps(obj) + b"\n\n"


def _finalize(svc: Services, record: RequestRecord, reservation: Reservation | None) -> None:
    """Price the request, emit its record, and settle its reservation (in the background)."""
    record.cost_usd = round(cost_usd(svc.config.pricing, record.model, record.prompt_tokens, record.completion_tokens), 8)
    record.finish()
    svc.telemetry.emit(record)
    if record.stream and record.provider:
        svc.telemetry.metrics.inflight.dec()
    if reservation is not None:
        svc.spawn(svc.limits.settle(reservation, record.prompt_tokens + record.completion_tokens, record.cost_usd))


async def _relay_stream(
    svc: Services, body: dict, handle: StreamHandle, record: RequestRecord, reservation: Reservation,
    started: dict[str, bool], lookup: Lookup | None = None,
) -> AsyncIterator[bytes]:
    started["yes"] = True
    completion_id = "chatcmpl-" + record.request_id[4:]
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    text_parts: list[str] = []
    usage: dict | None = None
    completed = False
    finish_reason: str | None = None
    tool_calls = False
    try:
        async for batch in svc.router.iterate(handle, record):
            out: list[bytes] = []
            for chunk in batch:
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if not chunk.get("choices"):
                    continue  # usage-only chunk; re-emitted below if the caller asked for it
                chunk["id"] = completion_id
                chunk["model"] = record.model
                chunk.pop("usage", None)
                for choice in chunk["choices"]:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        text_parts.append(delta["content"])
                    if delta.get("tool_calls"):
                        tool_calls = True
                    finish_reason = choice.get("finish_reason") or finish_reason
                out.append(_sse(chunk))
            if out:
                yield b"".join(out)  # one write per upstream read, not per token
        _apply_usage(record, usage, body, "".join(text_parts))
        if not tool_calls:
            _maybe_store(svc, record, lookup, "".join(text_parts), finish_reason, None)
        if include_usage:
            u = {
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "total_tokens": record.prompt_tokens + record.completion_tokens,
            }
            final = {"id": completion_id, "object": "chat.completion.chunk", "created": int(time.time()),
                     "model": record.model, "choices": [], "usage": u}
            yield _sse(final)
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
        yield _sse(err)
        completed = True
    finally:
        if not completed:
            # Client went away (cancellation or generator close). Bill what was generated.
            record.status, record.http_status = "client_disconnected", 499
            _apply_usage(record, usage, body, "".join(text_parts))
        _finalize(svc, record, reservation)
        await handle.close()
