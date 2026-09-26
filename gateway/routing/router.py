"""Alias resolution, fallback chains, retries, timeouts and circuit breakers.

For each target in an alias's chain, in order:

1. Skip it if its provider has no API key configured or its circuit is open.
2. Call it under three timeouts: connect (httpx), time to first token
   (streaming only) and total.
3. On a retryable error (429, 5xx, timeouts, connection resets) retry the same
   target with exponential backoff and full jitter, honouring Retry-After. A
   Retry-After longer than we are willing to wait means "fail over now".
4. On a non-retryable provider-side error (bad key, unknown model, feature the
   adapter doesn't support) move to the next target without retrying.
5. On a client error (400 and friends) stop: every provider would reject it.

Failover is only invisible before the first token. Streams are therefore
opened and held until the first chunk with real output arrives; only then do
we commit to that provider and start sending bytes to the client. What
happens after that is handled by the caller (see ``openai_routes``).
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from gateway.config import GatewayConfig, RetryPolicy, Target, Timeouts
from gateway.errors import GatewayError, ProviderError
from gateway.providers.base import Provider, chunk_has_output
from gateway.providers.registry import ProviderRegistry
from gateway.routing.breaker import CircuitBreakers
from gateway.telemetry.record import Attempt, RequestRecord

T = TypeVar("T")


class RouterHooks:
    """Metrics callbacks; the default does nothing (overridden by telemetry)."""

    def attempt(self, provider: str, outcome: str, error_kind: str | None, latency_s: float) -> None: ...

    def failover(self, alias: str, from_provider: str, to_provider: str) -> None: ...

    def circuit(self, provider: str, state: str) -> None: ...


@dataclass
class StreamHandle:
    """An upstream stream that has already produced its first token."""

    target: Target
    prelude: list[dict] = field(default_factory=list)
    rest: AsyncIterator[dict] | None = None
    idle_timeout: float = 20.0
    deadline: float = 0.0  # time.monotonic() value
    # The breaker outcome of a stream is known only when it ends: a provider
    # that sends a few tokens and then drops every connection is not healthy.
    # Until then the attempt holds its breaker slot (a half-open probe stays a
    # probe), and exactly one of success/failure/release settles it.
    breakers: Any = None
    breaker_pending: bool = False

    async def settle_breaker(self, outcome: str) -> str | None:
        if not self.breaker_pending or self.breakers is None:
            return None
        self.breaker_pending = False
        provider = self.target.provider
        if outcome == "success":
            state, _ = await asyncio.shield(self.breakers.success(provider))
            return state
        if outcome == "failure":
            state, _ = await asyncio.shield(self.breakers.failure(provider))
            return state
        await asyncio.shield(self.breakers.release(provider))
        return None

    async def close(self) -> None:
        try:
            if self.rest is not None:
                await self.rest.aclose()  # type: ignore[attr-defined]
        finally:
            await self.settle_breaker("release")


def backoff_delay(attempt: int, policy: RetryPolicy, retry_after: float | None = None) -> float | None:
    """Delay before retry number ``attempt`` (0-based), or None to fail over instead.

    Full jitter (a uniform draw in [0, base * 2^attempt]) spreads retries from
    many clients so they don't hit a recovering provider in lockstep.
    """
    delay = random.uniform(0, min(policy.max_delay, policy.base_delay * (2**attempt)))
    if retry_after is not None:
        if retry_after > policy.max_retry_after:
            return None
        delay = max(delay, retry_after)
    return delay


class Router:
    def __init__(
        self,
        registry: ProviderRegistry,
        config: GatewayConfig,
        breakers: CircuitBreakers | None = None,
        hooks: RouterHooks | None = None,
    ):
        self.registry = registry
        self.config = config
        self.breakers = breakers
        self.hooks = hooks or RouterHooks()

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

    # -- public entry points ------------------------------------------------

    async def complete(self, body: dict, chain: list[Target], record: RequestRecord) -> tuple[Target, dict]:
        async def call(provider: Provider, target: Target, to: Timeouts) -> dict:
            try:
                async with asyncio.timeout(to.total):
                    return await provider.complete(body, target.model, to.connect, to.total)
            except TimeoutError:
                raise ProviderError(provider.name, "total_timeout", f"no response within {to.total}s", retryable=True) from None

        return await self._run_chain(chain, record, call)

    async def open_stream(self, body: dict, chain: list[Target], record: RequestRecord) -> StreamHandle:
        async def call(provider: Provider, target: Target, to: Timeouts) -> StreamHandle:
            it = provider.stream(body, target.model, to.connect, to.idle).__aiter__()
            deadline = time.monotonic() + to.total
            prelude: list[dict] = []
            try:
                async with asyncio.timeout(min(to.ttft, to.total)):
                    while True:
                        batch = await it.__anext__()
                        prelude.extend(batch)
                        if any(chunk_has_output(c) for c in batch):
                            break
            except TimeoutError:
                await _safe_aclose(it)
                raise ProviderError(provider.name, "ttft_timeout", f"no first token within {to.ttft}s", retryable=True) from None
            except StopAsyncIteration:
                await _safe_aclose(it)
                raise ProviderError(provider.name, "empty_stream", "stream ended before any output", retryable=True) from None
            except BaseException:
                await _safe_aclose(it)
                raise
            return StreamHandle(target, prelude, it, to.idle, deadline)

        _, handle = await self._run_chain(chain, record, call)
        return handle

    async def iterate(self, handle: StreamHandle, record: RequestRecord) -> AsyncIterator[list[dict]]:
        """Yield the buffered prelude, then the rest of the stream (in batches) under idle/total timeouts."""
        if any(chunk_has_output(c) for c in handle.prelude):
            record.mark_first_token()
        if handle.prelude:
            yield handle.prelude
        assert handle.rest is not None
        provider = handle.target.provider
        try:
            while True:
                remaining = handle.deadline - time.monotonic()
                if remaining <= 0:
                    raise ProviderError(provider, "total_timeout", "stream exceeded total timeout", retryable=True)
                try:
                    if remaining > handle.idle_timeout:
                        # The adapter's httpx read timeout is the idle timeout, so a
                        # stall already raises; a per-read asyncio timer would only
                        # add overhead. It is needed only near the total deadline.
                        batch = await handle.rest.__anext__()
                    else:
                        async with asyncio.timeout(remaining):
                            batch = await handle.rest.__anext__()
                except StopAsyncIteration:
                    state = await handle.settle_breaker("success")
                    if state:
                        self.hooks.circuit(provider, state)
                    return
                except TimeoutError:
                    raise ProviderError(provider, "idle_timeout", "stream stalled", retryable=True) from None
                yield batch
        except ProviderError as err:
            # A provider that dies mid-stream is unhealthy too.
            state = await handle.settle_breaker("failure" if err.counts_as_provider_failure else "release")
            if state:
                self.hooks.circuit(provider, state)
            raise
        finally:
            # Client went away (or a bug): the stream told us nothing about the provider.
            await handle.settle_breaker("release")

    # -- the chain loop -------------------------------------------------------

    async def _run_chain(
        self,
        chain: list[Target],
        record: RequestRecord,
        call: Callable[[Provider, Target, Timeouts], Awaitable[T]],
    ) -> tuple[Target, T]:
        last_err: ProviderError | None = None
        previous_provider: str | None = None
        for target in chain:
            provider = self.registry.get(target.provider)
            if provider is None or not provider.enabled:
                record.attempts.append(Attempt(target.provider, target.model, "skipped_disabled"))
                continue
            policy = self.config.retry_for(target)
            to = self.config.timeouts_for(target)
            for attempt_no in range(policy.max_retries + 1):
                if self.breakers:
                    decision = await self.breakers.acquire(target.provider)
                    self.hooks.circuit(target.provider, decision.state)
                    if not decision.allowed:
                        record.attempts.append(Attempt(target.provider, target.model, "skipped_circuit_open"))
                        break
                if previous_provider and previous_provider != target.provider and record.alias:
                    self.hooks.failover(record.alias, previous_provider, target.provider)
                previous_provider = target.provider
                started = time.perf_counter()
                try:
                    result = await call(provider, target, to)
                except ProviderError as err:
                    elapsed = time.perf_counter() - started
                    last_err = err
                    record.attempts.append(
                        Attempt(target.provider, target.model, "error", round(elapsed * 1000, 2), err.kind, err.message[:300], err.status)
                    )
                    self.hooks.attempt(target.provider, "error", err.kind, elapsed)
                    if self.breakers:
                        if err.counts_as_provider_failure:
                            await self._breaker_failure(target.provider)
                        else:
                            await self.breakers.release(target.provider)
                    if not err.failover and not err.retryable:
                        raise _client_error(err) from None
                    if not err.retryable or attempt_no == policy.max_retries:
                        break
                    delay = backoff_delay(attempt_no, policy, err.retry_after)
                    if delay is None:
                        break  # Retry-After too long: next target is faster
                    await asyncio.sleep(delay)
                    continue
                except BaseException:
                    # Cancelled (client went away) or a bug: say nothing about provider health.
                    if self.breakers:
                        await asyncio.shield(self.breakers.release(target.provider))
                    raise
                elapsed = time.perf_counter() - started
                ttft = round(elapsed * 1000, 2) if isinstance(result, StreamHandle) else None
                record.attempts.append(Attempt(target.provider, target.model, "ok", round(elapsed * 1000, 2), ttft_ms=ttft))
                self.hooks.attempt(target.provider, "ok", None, elapsed)
                if self.breakers:
                    if isinstance(result, StreamHandle):
                        result.breakers, result.breaker_pending = self.breakers, True
                    else:
                        state, _ = await self.breakers.success(target.provider)
                        self.hooks.circuit(target.provider, state)
                return target, result
        raise _exhausted_error(last_err, record)

    async def _breaker_failure(self, provider: str) -> None:
        assert self.breakers is not None
        state, _ = await self.breakers.failure(provider)
        self.hooks.circuit(provider, state)


async def _safe_aclose(it: Any) -> None:
    try:
        await it.aclose()
    except Exception:  # noqa: BLE001 - closing is best effort
        pass


def _client_error(err: ProviderError) -> GatewayError:
    status = err.status if err.status and 400 <= err.status < 500 else 400
    return GatewayError(status, f"Upstream rejected the request: {err.message}", code="upstream_invalid_request")


def _exhausted_error(last: ProviderError | None, record: RequestRecord) -> GatewayError:
    if last is None:
        if any(a.outcome == "skipped_circuit_open" for a in record.attempts):
            return GatewayError(503, "Every provider for this model is temporarily unavailable (circuit open).", "api_error", "circuit_open")
        return GatewayError(503, "No configured provider is available for this model.", "api_error", "no_provider")
    if last.kind == "rate_limited":
        headers = {"retry-after": str(int(last.retry_after or 1))}
        return GatewayError(429, "All providers are rate limiting this request.", "rate_limit_error", "upstream_rate_limited", headers)
    return GatewayError(502, f"All providers failed; last error from {last.provider}: {last.kind}.", "api_error", "upstream_error")
