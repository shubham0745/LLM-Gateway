"""Alias resolution and provider calls."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from gateway.config import GatewayConfig, Target
from gateway.errors import GatewayError, ProviderError
from gateway.providers.base import chunk_has_output
from gateway.providers.registry import ProviderRegistry
from gateway.telemetry.record import Attempt, RequestRecord


@dataclass
class StreamHandle:
    """An upstream stream that has already produced its first token."""

    target: Target
    prelude: list[dict] = field(default_factory=list)
    rest: AsyncIterator[dict] | None = None
    idle_timeout: float = 20.0
    deadline: float = 0.0  # time.monotonic() value

    async def close(self) -> None:
        if self.rest is not None:
            await self.rest.aclose()  # type: ignore[attr-defined]


class Router:
    def __init__(self, registry: ProviderRegistry, config: GatewayConfig):
        self.registry = registry
        self.config = config

    def resolve(self, model: str) -> list[Target]:
        """``fast`` -> its chain; ``provider/model`` -> that single target."""
        chain = self.config.aliases.get(model)
        if chain:
            return list(chain)
        provider, sep, raw_model = model.partition("/")
        if sep and provider in self.config.providers and raw_model:
            return [Target(provider=provider, model=raw_model)]
        raise GatewayError(
            404,
            f"Unknown model {model!r}. Use one of: {', '.join(sorted(self.config.aliases))} or 'provider/model'.",
            code="model_not_found",
        )

    def _first_enabled(self, chain: list[Target], record: RequestRecord) -> Target:
        for t in chain:
            p = self.registry.get(t.provider)
            if p and p.enabled:
                return t
            record.attempts.append(Attempt(t.provider, t.model, "skipped_disabled"))
        raise GatewayError(503, "No configured provider is available for this model.", "api_error", "no_provider")

    async def complete(self, body: dict, chain: list[Target], record: RequestRecord) -> tuple[Target, dict]:
        target = self._first_enabled(chain, record)
        provider = self.registry.get(target.provider)
        to = self.config.timeouts_for(target)
        started = time.perf_counter()
        try:
            async with asyncio.timeout(to.total):
                resp = await provider.complete(body, target.model, to.connect, to.total)
        except TimeoutError:
            err = ProviderError(target.provider, "total_timeout", f"no response within {to.total}s", retryable=True)
            record.attempts.append(Attempt(target.provider, target.model, "error", _ms(started), err.kind, err.message))
            raise _to_gateway_error(err) from None
        except ProviderError as err:
            record.attempts.append(
                Attempt(target.provider, target.model, "error", _ms(started), err.kind, err.message, err.status)
            )
            raise _to_gateway_error(err) from err
        record.attempts.append(Attempt(target.provider, target.model, "ok", _ms(started)))
        return target, resp

    async def open_stream(self, body: dict, chain: list[Target], record: RequestRecord) -> StreamHandle:
        target = self._first_enabled(chain, record)
        provider = self.registry.get(target.provider)
        to = self.config.timeouts_for(target)
        started = time.perf_counter()
        it = provider.stream(body, target.model, to.connect, to.idle).__aiter__()
        prelude: list[dict] = []
        try:
            while True:
                async with asyncio.timeout(to.ttft):
                    chunk = await it.__anext__()
                prelude.append(chunk)
                if chunk_has_output(chunk):
                    break
        except (TimeoutError, ProviderError, StopAsyncIteration) as exc:
            await it.aclose()
            err = exc if isinstance(exc, ProviderError) else ProviderError(target.provider, "ttft_timeout", "no first token", retryable=True)
            record.attempts.append(Attempt(target.provider, target.model, "error", _ms(started), err.kind, err.message, err.status))
            raise _to_gateway_error(err) from None
        record.attempts.append(Attempt(target.provider, target.model, "ok", _ms(started), ttft_ms=_ms(started)))
        return StreamHandle(target, prelude, it, to.idle, time.monotonic() + to.total)

    async def iterate(self, handle: StreamHandle, record: RequestRecord) -> AsyncIterator[dict]:
        """Yield the buffered prelude, then the rest of the stream under idle/total timeouts."""
        for chunk in handle.prelude:
            if chunk_has_output(chunk):
                record.mark_first_token()
            yield chunk
        assert handle.rest is not None
        while True:
            remaining = handle.deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderError(handle.target.provider, "total_timeout", "stream exceeded total timeout")
            try:
                async with asyncio.timeout(min(handle.idle_timeout, remaining)):
                    chunk = await handle.rest.__anext__()
            except StopAsyncIteration:
                return
            except TimeoutError:
                raise ProviderError(handle.target.provider, "idle_timeout", "stream stalled") from None
            yield chunk


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _to_gateway_error(err: ProviderError) -> GatewayError:
    if not err.failover and not err.retryable and err.status and 400 <= err.status < 500:
        # The request itself was rejected (e.g. invalid parameters): pass it through.
        return GatewayError(err.status, f"Upstream rejected the request: {err.message}", code="upstream_invalid_request")
    return GatewayError(502, f"All providers failed; last error from {err.provider}: {err.kind}", "api_error", "upstream_error")
