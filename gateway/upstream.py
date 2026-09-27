"""Sending adapter-built requests to providers, and mapping failures.

Every provider call goes through here, so this is where retries and the
circuit breaker will go in a later PR. Today it makes one attempt.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from gateway.errors import GatewayError
from gateway.providers.base import ProviderAdapter, UpstreamRequest
from gateway.request_id import HEADER as REQUEST_ID_HEADER


def _timeout(adapter: ProviderAdapter) -> httpx.Timeout:
    return httpx.Timeout(adapter.config.timeout_seconds, connect=5.0)


def _build(
    client: httpx.AsyncClient, adapter: ProviderAdapter, req: UpstreamRequest, request_id: str
) -> httpx.Request:
    headers = {**req.headers, REQUEST_ID_HEADER: request_id}
    return client.build_request(
        req.method, req.url, json=req.json, headers=headers, timeout=_timeout(adapter)
    )


def _status_error(adapter: ProviderAdapter, response: httpx.Response) -> GatewayError:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    detail = adapter.error_message(payload) or response.text[:200] or response.reason_phrase
    status = response.status_code
    prefix = f"provider {adapter.name!r} returned {status}"
    if status in (401, 403):
        # The client didn't fail auth; the gateway's provider credentials did.
        return GatewayError(502, f"{prefix}: {detail}", code="upstream_auth_failed")
    if status == 429:
        return GatewayError(
            429, f"{prefix}: {detail}", type="rate_limit_error", code="upstream_rate_limited"
        )
    if 400 <= status < 500:
        return GatewayError(
            status, f"{prefix}: {detail}", type="invalid_request_error", code="upstream_rejected"
        )
    return GatewayError(502, f"{prefix}: {detail}", code="upstream_error")


def _transport_error(adapter: ProviderAdapter, exc: httpx.HTTPError) -> GatewayError:
    if isinstance(exc, httpx.TimeoutException):
        return GatewayError(504, f"provider {adapter.name!r} timed out", code="upstream_timeout")
    return GatewayError(
        502, f"could not reach provider {adapter.name!r}: {exc!r}", code="upstream_unreachable"
    )


async def send(
    client: httpx.AsyncClient, adapter: ProviderAdapter, req: UpstreamRequest, request_id: str
) -> httpx.Response:
    """Send a non-streaming request; return the 2xx response or raise GatewayError."""
    try:
        response = await client.send(_build(client, adapter, req, request_id))
    except httpx.HTTPError as exc:
        raise _transport_error(adapter, exc) from exc
    if response.is_error:
        raise _status_error(adapter, response)
    return response


@asynccontextmanager
async def open_stream(
    client: httpx.AsyncClient, adapter: ProviderAdapter, req: UpstreamRequest, request_id: str
) -> AsyncIterator[httpx.Response]:
    """Open a streaming request. Errors before the first byte raise GatewayError."""
    try:
        response = await client.send(_build(client, adapter, req, request_id), stream=True)
    except httpx.HTTPError as exc:
        raise _transport_error(adapter, exc) from exc
    try:
        if response.is_error:
            await response.aread()
            raise _status_error(adapter, response)
        yield response
    finally:
        await response.aclose()
