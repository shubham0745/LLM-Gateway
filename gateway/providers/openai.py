"""OpenAI (and OpenAI-compatible) Chat Completions adapter.

The gateway's contract is already OpenAI-shaped, so request and response
translation are close to the identity: swap in the upstream model name, add
auth, and validate what comes back.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from gateway.providers.base import ProviderAdapter, UpstreamRequest
from gateway.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)
from gateway.streaming.sse import SSEEvent


class OpenAIAdapter(ProviderAdapter):
    def build_request(self, request: ChatCompletionRequest, upstream_model: str) -> UpstreamRequest:
        body = request.model_dump(exclude_none=True)
        body["model"] = upstream_model
        headers = {"content-type": "application/json"}
        if self.config.api_key:
            headers["authorization"] = f"Bearer {self.config.api_key}"
        if request.stream:
            headers["accept"] = "text/event-stream"
        return UpstreamRequest(
            url=f"{self.config.base_url.rstrip('/')}/chat/completions",
            json=body,
            headers=headers,
        )

    def translate_response(self, payload: dict[str, Any]) -> ChatCompletionResponse:
        response = ChatCompletionResponse.model_validate(payload)
        response.usage = self.extract_usage(payload)
        return response

    async def translate_stream(
        self, events: AsyncIterator[SSEEvent]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Turn OpenAI SSE events into ChatCompletionChunk objects.

        Rules this must follow (see tests/test_openai_stream.py):

        - Each event's ``data`` is one JSON chunk; yield it as a
          ChatCompletionChunk, keeping extra fields.
        - Stop at ``data: [DONE]`` without yielding anything for it, and
          ignore anything after it.
        - The last chunk may have ``choices: []`` and a ``usage`` object
          (when the client set stream_options.include_usage); yield it, with
          usage normalised through ``extract_usage``.
        - Raise StreamTranslationError for data that isn't valid JSON, and
          for an ``{"error": ...}`` event (use ``error_message`` for the text).
        - A stream that ends without ``[DONE]`` just ends; don't raise.

        TODO(core): implement by hand.
        """
        raise NotImplementedError(
            "OpenAIAdapter.translate_stream is a stub: implement it to make "
            "tests/test_openai_stream.py pass"
        )
        yield  # unreachable; makes this method an async generator

    def extract_usage(self, payload: dict[str, Any]) -> Usage | None:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        return Usage.model_validate(usage)
