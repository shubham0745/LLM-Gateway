"""OpenAI-compatible HTTP routes.

This layer knows about aliases, adapters and the upstream client, and nothing
about any provider's wire format.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from gateway import upstream
from gateway.config import GatewayConfig
from gateway.errors import GatewayError
from gateway.providers.base import ProviderAdapter, StreamTranslationError
from gateway.schemas import ChatCompletionRequest, ChatCompletionResponse, ModelCard, ModelList
from gateway.streaming.sse import DONE, encode_sse, parse_sse

log = logging.getLogger("gateway")

router = APIRouter(prefix="/v1")


def _resolve(request: Request, alias: str) -> tuple[ProviderAdapter, str]:
    config: GatewayConfig = request.app.state.config
    route = config.models.get(alias)
    if route is None:
        raise GatewayError(
            404,
            f"model {alias!r} is not configured; see GET /v1/models",
            type="invalid_request_error",
            code="model_not_found",
        )
    return request.app.state.adapters[route.provider], route.model


@router.get("/models")
async def list_models(request: Request) -> ModelList:
    config: GatewayConfig = request.app.state.config
    return ModelList(
        data=[
            ModelCard(id=alias, owned_by=route.provider) for alias, route in config.models.items()
        ]
    )


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest, request: Request
) -> ChatCompletionResponse | StreamingResponse:
    adapter, upstream_model = _resolve(request, body.model)
    client: httpx.AsyncClient = request.app.state.http_client
    request_id: str = request.state.request_id
    upstream_req = adapter.build_request(body, upstream_model)
    started = time.perf_counter()

    if not body.stream:
        response = await upstream.send(client, adapter, upstream_req, request_id)
        try:
            payload = response.json()
        except ValueError as exc:
            raise GatewayError(502, f"provider {adapter.name!r} returned invalid JSON") from exc
        result = adapter.translate_response(payload)
        log.info(
            "request_id=%s model=%s provider=%s upstream_model=%s stream=false ms=%.0f usage=%s",
            request_id,
            body.model,
            adapter.name,
            upstream_model,
            (time.perf_counter() - started) * 1000,
            result.usage,
        )
        return result

    # Open the upstream stream before answering, so a provider that fails
    # up front still gets a proper HTTP error status instead of a 200 stream.
    stack = AsyncExitStack()
    upstream_resp = await stack.enter_async_context(
        upstream.open_stream(client, adapter, upstream_req, request_id)
    )

    async def body_iter() -> AsyncIterator[bytes]:
        usage = None
        try:
            events = parse_sse(upstream_resp.aiter_bytes())
            async for chunk in adapter.translate_stream(events):
                usage = chunk.usage or usage
                yield encode_sse(chunk.model_dump_json())
            yield encode_sse(DONE)
        except (httpx.HTTPError, StreamTranslationError) as exc:
            # Headers are already sent, so the error has to travel in-band.
            err = GatewayError(502, f"stream from provider {adapter.name!r} failed: {exc}")
            log.warning("request_id=%s stream failed: %r", request_id, exc)
            yield encode_sse(json.dumps(err.body(request_id)))
        finally:
            await stack.aclose()
            log.info(
                "request_id=%s model=%s provider=%s upstream_model=%s stream=true ms=%.0f usage=%s",
                request_id,
                body.model,
                adapter.name,
                upstream_model,
                (time.perf_counter() - started) * 1000,
                usage,
            )

    return StreamingResponse(
        body_iter(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache"},
        # Runs even if the client disconnects before the body is iterated.
        background=BackgroundTask(stack.aclose),
    )
