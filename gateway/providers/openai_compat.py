"""Adapter for OpenAI and anything that speaks its API (Ollama, the mock provider)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from gateway.errors import ProviderError
from gateway.providers.base import Provider, aiter_sse_batches, close_response, drain, loads_or_error

# Fields the gateway consumes itself and never forwards upstream.
_GATEWAY_ONLY_FIELDS = {"stream", "stream_options", "model"}


class OpenAICompatProvider(Provider):
    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json", **self.cfg.headers}
        if self.api_key:
            h["authorization"] = f"Bearer {self.api_key}"
        return h

    def _body(self, body: dict[str, Any], model: str, stream: bool) -> dict[str, Any]:
        out = {k: v for k, v in body.items() if k not in _GATEWAY_ONLY_FIELDS}
        out["model"] = model
        if stream:
            out["stream"] = True
            # Always ask for usage: we need it for billing even if the caller didn't.
            out["stream_options"] = {"include_usage": True}
        return out

    @property
    def _url(self) -> str:
        return self.cfg.base_url.rstrip("/") + "/chat/completions"

    async def complete(self, body: dict[str, Any], model: str, connect_timeout: float, read_timeout: float) -> dict:
        try:
            resp = await self.client.post(
                self._url,
                json=self._body(body, model, stream=False),
                headers=self._headers(),
                timeout=self._timeout(connect_timeout, read_timeout),
            )
        except httpx.HTTPError as exc:
            raise self._wrap_transport_error(exc) from exc
        await self._raise_for_status(resp)
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(self.name, "bad_response", "response was not JSON", retryable=True) from exc
        if "choices" not in data:
            raise ProviderError(self.name, "bad_response", f"no choices in response: {str(data)[:200]}", retryable=True)
        return data

    async def stream(
        self, body: dict[str, Any], model: str, connect_timeout: float, read_timeout: float
    ) -> AsyncIterator[list[dict]]:
        client = self.client
        req = client.build_request(
            "POST",
            self._url,
            json=self._body(body, model, stream=True),
            headers=self._headers(),
            timeout=self._timeout(connect_timeout, read_timeout),
        )
        try:
            resp = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            raise self._wrap_transport_error(exc) from exc
        try:
            await self._raise_for_status(resp)
            finished = done = False
            batches = aiter_sse_batches(resp)
            async for batch in batches:
                out: list[dict] = []
                for _event, data in batch:
                    if data == "[DONE]":
                        finished = done = True
                        break
                    chunk = loads_or_error(self.name, data)
                    if "error" in chunk:
                        if out:
                            yield out
                        err = chunk["error"] or {}
                        raise ProviderError(self.name, "stream_error", str(err.get("message", err))[:300], retryable=True)
                    for choice in chunk.get("choices") or []:
                        if choice.get("finish_reason"):
                            finished = True
                    out.append(chunk)
                if out:
                    yield out
                if done:
                    await drain(batches)
                    break
            if not finished:
                raise ProviderError(self.name, "stream_truncated", "stream ended before a finish", retryable=True)
        except httpx.HTTPError as exc:
            raise self._wrap_transport_error(exc) from exc
        finally:
            # Runs on normal end, on error, and when the consumer closes us
            # (client disconnect): closing the response aborts the upstream call.
            await close_response(resp)
