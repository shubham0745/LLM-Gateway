"""Adapter for Anthropic's Messages API, written against raw HTTP.

Translation rules (OpenAI -> Anthropic):
  * ``system``/``developer`` messages become the top-level ``system`` string
  * consecutive messages with the same role are merged (Anthropic wants turns
    to alternate)
  * ``max_tokens``/``max_completion_tokens`` -> ``max_tokens`` (required upstream)
  * ``stop`` -> ``stop_sequences``; ``temperature`` is clamped to [0, 1]
  * text and image parts are translated; tools, tool messages and ``n > 1`` are
    not supported in v1 and raise a failover error so the router can try an
    OpenAI-compatible target instead

Streaming events are converted chunk by chunk (and yielded in per-read batches):
  message_start        -> remembers input tokens (usage)
  content_block_delta  -> one OpenAI chunk per text delta
  message_delta        -> stop_reason and output tokens
  message_stop         -> final chunk with finish_reason, then a usage chunk
  error                -> ProviderError (overloaded_error is retryable)
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from gateway.errors import ProviderError
from gateway.providers.base import Provider, aiter_sse_batches, close_response, drain, loads_or_error, make_chunk, usage_chunk

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024

STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
    "pause_turn": "stop",
}


def _unsupported(provider: str, what: str) -> ProviderError:
    return ProviderError(provider, "unsupported", f"{what} is not supported by this adapter", retryable=False)


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def _convert_part(provider: str, part: dict) -> dict:
    ptype = part.get("type")
    if ptype == "text":
        return {"type": "text", "text": part.get("text", "")}
    if ptype == "image_url":
        url = (part.get("image_url") or {}).get("url", "")
        if url.startswith("data:"):
            header, _, data = url.partition(",")
            media_type = header[5:].split(";")[0] or "image/png"
            return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
        return {"type": "image", "source": {"type": "url", "url": url}}
    raise _unsupported(provider, f"content part type {ptype!r}")


def to_anthropic_request(provider: str, body: dict[str, Any], model: str, stream: bool) -> dict[str, Any]:
    if body.get("tools") or body.get("functions"):
        raise _unsupported(provider, "tool calling")
    if (body.get("n") or 1) != 1:
        raise _unsupported(provider, "n > 1")

    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for m in body.get("messages") or []:
        role = m.get("role")
        content = m.get("content")
        if role in ("system", "developer"):
            system_parts.append(_text_of(content))
            continue
        if role not in ("user", "assistant"):
            raise _unsupported(provider, f"message role {role!r}")
        if isinstance(content, list):
            blocks = [_convert_part(provider, p) for p in content]
        else:
            blocks = [{"type": "text", "text": content or ""}]
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    out: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS,
    }
    if system_parts:
        out["system"] = "\n\n".join(p for p in system_parts if p)
    if body.get("temperature") is not None:
        out["temperature"] = max(0.0, min(1.0, float(body["temperature"])))
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if stream:
        out["stream"] = True
    return out


def _input_tokens(usage: dict) -> int:
    return (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
    )


def from_anthropic_response(data: dict, completion_id: str) -> dict:
    text = "".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text")
    usage = data.get("usage") or {}
    prompt, completion = _input_tokens(usage), int(usage.get("output_tokens") or 0)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": STOP_REASON_MAP.get(data.get("stop_reason") or "", "stop"),
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    }


class AnthropicProvider(Provider):
    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "x-api-key": self.api_key or "",
            **self.cfg.headers,
        }

    @property
    def _url(self) -> str:
        return self.cfg.base_url.rstrip("/") + "/v1/messages"

    async def complete(self, body: dict[str, Any], model: str, connect_timeout: float, read_timeout: float) -> dict:
        payload = to_anthropic_request(self.name, body, model, stream=False)
        try:
            resp = await self.client.post(
                self._url, json=payload, headers=self._headers(), timeout=self._timeout(connect_timeout, read_timeout)
            )
        except httpx.HTTPError as exc:
            raise self._wrap_transport_error(exc) from exc
        await self._raise_for_status(resp)
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(self.name, "bad_response", "response was not JSON", retryable=True) from exc
        return from_anthropic_response(data, completion_id=data.get("id", ""))

    async def stream(
        self, body: dict[str, Any], model: str, connect_timeout: float, read_timeout: float
    ) -> AsyncIterator[list[dict]]:
        payload = to_anthropic_request(self.name, body, model, stream=True)
        client = self.client
        req = client.build_request(
            "POST", self._url, json=payload, headers=self._headers(), timeout=self._timeout(connect_timeout, read_timeout)
        )
        try:
            resp = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            raise self._wrap_transport_error(exc) from exc

        completion_id, served_model, created = "", model, int(time.time())
        input_tokens = output_tokens = 0
        stop_reason: str | None = None
        sent_role = False
        try:
            await self._raise_for_status(resp)
            batches = aiter_sse_batches(resp)
            async for batch in batches:
                out: list[dict] = []
                for event, data in batch:
                    msg = loads_or_error(self.name, data)
                    etype = msg.get("type") or event
                    if etype == "message_start":
                        m = msg.get("message") or {}
                        completion_id = m.get("id", "")
                        served_model = m.get("model", model)
                        input_tokens = _input_tokens(m.get("usage") or {})
                        output_tokens = int((m.get("usage") or {}).get("output_tokens") or 0)
                    elif etype == "content_block_delta":
                        delta = msg.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            out.append(make_chunk(
                                completion_id,
                                served_model,
                                created,
                                role=None if sent_role else "assistant",
                                content=delta["text"],
                            ))
                            sent_role = True
                        # thinking / signature / input_json deltas are not surfaced in v1
                    elif etype == "message_delta":
                        stop_reason = (msg.get("delta") or {}).get("stop_reason") or stop_reason
                        usage = msg.get("usage") or {}
                        if usage.get("output_tokens") is not None:
                            output_tokens = int(usage["output_tokens"])
                        if usage.get("input_tokens") is not None:
                            input_tokens = _input_tokens(usage)
                    elif etype == "message_stop":
                        out.append(make_chunk(
                            completion_id,
                            served_model,
                            created,
                            role=None if sent_role else "assistant",
                            finish_reason=STOP_REASON_MAP.get(stop_reason or "", "stop"),
                        ))
                        out.append(usage_chunk(completion_id, served_model, input_tokens, output_tokens))
                        yield out
                        await drain(batches)
                        return
                    elif etype == "error":
                        if out:
                            yield out
                        err = msg.get("error") or {}
                        kind = err.get("type", "stream_error")
                        raise ProviderError(
                            self.name,
                            kind,
                            str(err.get("message", ""))[:300],
                            retryable=kind in ("overloaded_error", "api_error", "rate_limit_error"),
                        )
                    # ping, content_block_start, content_block_stop: nothing to emit
                if out:
                    yield out
            raise ProviderError(self.name, "stream_truncated", "stream ended before message_stop", retryable=True)
        except httpx.HTTPError as exc:
            raise self._wrap_transport_error(exc) from exc
        finally:
            await close_response(resp)
