"""The provider adapter interface.

An adapter is a pure translator between the gateway's OpenAI-shaped contract
(gateway.schemas) and one provider's wire format. It does no I/O: it builds
the upstream request, and it converts what comes back. Sending, timeouts,
error mapping and (later) retries live in gateway.upstream, once, for every
provider.

That split is what lets a new provider land without touching the API layer,
and lets each translation be tested with plain data and no HTTP.

The four jobs, in the order a request goes through them:

1. ``build_request``      ChatCompletionRequest   -> UpstreamRequest
2. ``translate_response`` provider JSON body      -> ChatCompletionResponse
3. ``translate_stream``   provider SSE events     -> ChatCompletionChunk stream
4. ``extract_usage``      provider JSON (body or stream event) -> Usage | None
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from gateway.config import ProviderConfig
from gateway.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)
from gateway.streaming.sse import SSEEvent


@dataclass
class UpstreamRequest:
    """Everything needed to make the HTTP call, and nothing about how to make it."""

    url: str
    json: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    method: str = "POST"


class StreamTranslationError(Exception):
    """The upstream stream contained something the adapter can't translate."""


class ProviderAdapter(ABC):
    def __init__(self, name: str, config: ProviderConfig) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    def build_request(self, request: ChatCompletionRequest, upstream_model: str) -> UpstreamRequest:
        """Request translation. ``upstream_model`` replaces the client's alias."""

    @abstractmethod
    def translate_response(self, payload: dict[str, Any]) -> ChatCompletionResponse:
        """Response translation for a non-streaming 2xx body."""

    @abstractmethod
    def translate_stream(
        self, events: AsyncIterator[SSEEvent]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream translation.

        Consume parsed SSE events and yield OpenAI chunks. Stop at the
        provider's end-of-stream marker. Raise StreamTranslationError on
        input it can't make sense of, or an error event sent mid-stream.
        The API layer adds the final ``data: [DONE]`` itself.
        """

    @abstractmethod
    def extract_usage(self, payload: dict[str, Any]) -> Usage | None:
        """Token usage from a response body or stream event, if it carries any."""

    def error_message(self, payload: Any) -> str | None:
        """Pull a human-readable message out of a provider's error body.

        OpenAI and Anthropic both use ``{"error": {"message": ...}}``; override
        for providers that don't.
        """
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and isinstance(err.get("message"), str):
                return err["message"]
            if isinstance(err, str):
                return err
        return None
