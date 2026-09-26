"""Provider adapter interface.

OpenAI's chat completions format is the gateway's internal standard. Every
adapter takes an OpenAI-format request body and returns OpenAI-format
responses; for streaming it yields OpenAI ``chat.completion.chunk`` dicts.

Contract for ``stream``:
  * yields non-empty *lists* of chunk dicts, one list per upstream network
    read (see ``aiter_sse_batches``), whose ``choices[0].delta`` carries
    role/content/tool_calls
  * the final item carries ``usage`` when the provider reported it (with
    ``choices: []``, the same shape OpenAI uses for ``include_usage``)
  * raises ``ProviderError`` on any failure, including a stream that ends
    without a proper finish
  * closing the generator (``aclose``) must close the upstream HTTP response,
    so a client disconnect stops token generation upstream
"""

from __future__ import annotations

import abc
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import orjson

from gateway.config import ProviderConfig
from gateway.errors import ProviderError, classify_status, parse_retry_after


class Provider(abc.ABC):
    def __init__(self, name: str, cfg: ProviderConfig, client: Any, api_key: str | None):
        self.name = name
        self.cfg = cfg
        self._client = client  # an httpx.AsyncClient or a registry.ClientPool
        self.api_key = api_key

    @property
    def client(self) -> httpx.AsyncClient:
        pick = getattr(self._client, "pick", None)
        return pick() if pick else self._client

    @property
    def enabled(self) -> bool:
        return self.cfg.api_key_env is None or bool(self.api_key)

    @abc.abstractmethod
    async def complete(self, body: dict[str, Any], model: str, connect_timeout: float, read_timeout: float) -> dict:
        """Non-streaming call. Returns an OpenAI ``chat.completion`` dict."""

    @abc.abstractmethod
    def stream(
        self, body: dict[str, Any], model: str, connect_timeout: float, read_timeout: float
    ) -> AsyncIterator[list[dict]]:
        """Streaming call. See module docstring for the contract."""

    # -- helpers shared by adapters ---------------------------------------

    def _timeout(self, connect: float, read: float) -> httpx.Timeout:
        return httpx.Timeout(connect=connect, read=read, write=read, pool=connect)

    async def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            body = (await resp.aread()).decode(errors="replace")
        except httpx.HTTPError:
            body = ""
        raise classify_status(self.name, resp.status_code, body, parse_retry_after(resp.headers.get("retry-after")))

    def _wrap_transport_error(self, exc: Exception) -> ProviderError:
        if isinstance(exc, httpx.ConnectTimeout):
            return ProviderError(self.name, "connect_timeout", str(exc) or "connect timeout", retryable=True)
        if isinstance(exc, httpx.PoolTimeout):
            return ProviderError(self.name, "pool_timeout", "connection pool exhausted", retryable=True)
        if isinstance(exc, httpx.TimeoutException):
            return ProviderError(self.name, "read_timeout", str(exc) or "read timeout", retryable=True)
        if isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)):
            return ProviderError(self.name, "connection", f"{type(exc).__name__}: {exc}", retryable=True)
        return ProviderError(self.name, "transport", f"{type(exc).__name__}: {exc}", retryable=True)


async def close_response(resp: httpx.Response) -> None:
    """Close an upstream response even while our task is being cancelled.

    A client disconnect cancels the request task, and AnyIO keeps re-delivering
    that cancellation at every await. Shielding the close lets it finish, so
    the upstream connection is really torn down and the provider stops
    generating tokens.
    """
    import asyncio

    task = asyncio.ensure_future(resp.aclose())
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        raise  # the close task keeps running to completion on its own
    except Exception:  # noqa: BLE001 - closing is best effort
        pass


async def drain(events: AsyncIterator, limit_s: float = 0.5) -> None:
    """Read whatever follows the terminal event, so the connection returns to the pool.

    Closing a response whose body has not been fully read makes httpx drop the
    TCP connection, and the next request pays for a new handshake (a large
    part of the gateway's overhead under load before this existed). The tail
    after ``[DONE]`` / ``message_stop`` is normally just the end of the
    chunked body; if it does not arrive quickly we give up and close.
    """
    import asyncio

    try:
        async with asyncio.timeout(limit_s):
            async for _ in events:
                pass
    except (TimeoutError, httpx.HTTPError):
        pass


async def aiter_sse_batches(resp: httpx.Response) -> AsyncIterator[list[tuple[str | None, str]]]:
    """Parse a text/event-stream body into batches of (event, data) pairs.

    One batch per network read. When the gateway keeps up, a read holds one
    event; when it falls behind under load, reads hold several, and handling
    them as one batch (one generator step, one write to the client) makes the
    per-token cost shrink exactly when it matters.

    Works on bytes (``aiter_bytes`` still undoes any content-encoding) rather
    than ``aiter_lines``, which stacks a text decoder and a line splitter on
    top. Splitting on newline before decoding is safe for UTF-8, since no
    multi-byte character contains a newline byte.
    """
    event: str | None = None
    data_lines: list[str] = []
    buf = b""
    async for piece in resp.aiter_bytes():
        buf += piece
        if b"\n" not in buf:
            continue
        *lines, buf = buf.split(b"\n")
        batch: list[tuple[str | None, str]] = []
        for raw in lines:
            line = raw.decode("utf-8", "replace").rstrip("\r")
            if line == "":
                if data_lines:
                    batch.append((event, "\n".join(data_lines)))
                event, data_lines = None, []
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "event":
                event = value
            elif field == "data":
                data_lines.append(value)
        if batch:
            yield batch
    if buf.strip():
        field, _, value = buf.decode("utf-8", "replace").rstrip("\r").partition(":")
        if field == "data":
            data_lines.append(value[1:] if value.startswith(" ") else value)
    if data_lines:
        yield [(event, "\n".join(data_lines))]


async def aiter_sse(resp: httpx.Response) -> AsyncIterator[tuple[str | None, str]]:
    """Parse a text/event-stream body into (event, data) pairs."""
    async for batch in aiter_sse_batches(resp):
        for item in batch:
            yield item


def loads_or_error(provider: str, data: str) -> dict:
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError as exc:
        raise ProviderError(provider, "bad_stream", f"invalid JSON in stream: {exc}", retryable=True) from exc


def make_chunk(
    completion_id: str,
    model: str,
    created: int | None = None,
    *,
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
) -> dict:
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason, "logprobs": None}],
    }


def usage_chunk(completion_id: str, model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def chunk_has_output(chunk: dict) -> bool:
    """True once a chunk carries something the user would see (or a finish)."""
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("tool_calls") or delta.get("refusal"):
            return True
        if choice.get("finish_reason"):
            return True
    return False
