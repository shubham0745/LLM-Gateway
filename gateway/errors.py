"""Error types and OpenAI-shaped error responses."""

from __future__ import annotations

from fastapi.responses import JSONResponse


class GatewayError(Exception):
    """An error returned to the caller in OpenAI's error format."""

    def __init__(
        self,
        status: int,
        message: str,
        type_: str = "invalid_request_error",
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type_
        self.code = code
        self.headers = headers or {}

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "param": None, "code": self.code}}

    def response(self, extra_headers: dict[str, str] | None = None) -> JSONResponse:
        return JSONResponse(self.body(), status_code=self.status, headers={**self.headers, **(extra_headers or {})})


class ProviderError(Exception):
    """A failed call to an upstream provider.

    ``retryable``: worth retrying on the same provider (429, 5xx, timeouts, resets).
    ``failover``: worth trying the next target in the chain. A 400 is neither: the
    request itself is bad and every provider would reject it. A 401 from a
    provider is our misconfiguration, so it is not retryable but we do fail over.
    """

    def __init__(
        self,
        provider: str,
        kind: str,
        message: str,
        status: int | None = None,
        retryable: bool = False,
        failover: bool = True,
        retry_after: float | None = None,
    ):
        super().__init__(f"{provider}: {kind}: {message}")
        self.provider = provider
        self.kind = kind
        self.message = message
        self.status = status
        self.retryable = retryable
        self.failover = failover
        self.retry_after = retry_after

    # Whether this failure should count against the provider's circuit breaker.
    # Client-caused errors (400s) and our own "unsupported feature" skips say
    # nothing about the provider's health.
    @property
    def counts_as_provider_failure(self) -> bool:
        return self.retryable


def classify_status(provider: str, status: int, body: str, retry_after: float | None) -> ProviderError:
    snippet = body[:500]
    if status == 429:
        return ProviderError(provider, "rate_limited", snippet, status, retryable=True, retry_after=retry_after)
    if status in (408, 409) or status >= 500:
        # 529 is Anthropic's "overloaded".
        return ProviderError(provider, f"http_{status}", snippet, status, retryable=True, retry_after=retry_after)
    if status in (401, 403):
        return ProviderError(provider, "auth", snippet, status, retryable=False, failover=True)
    if status == 404:
        # Usually a model name the provider does not know: config error, try the next target.
        return ProviderError(provider, "not_found", snippet, status, retryable=False, failover=True)
    return ProviderError(provider, f"http_{status}", snippet, status, retryable=False, failover=False)


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # HTTP-date form: rare from LLM APIs, treat as absent
