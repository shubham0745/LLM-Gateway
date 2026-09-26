"""Adapter translation against recorded real-format provider payloads."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from gateway.config import ProviderConfig
from gateway.errors import ProviderError
from gateway.providers.anthropic import AnthropicProvider, from_anthropic_response, to_anthropic_request
from gateway.providers.openai_compat import OpenAICompatProvider

ANTHROPIC_SSE = (
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant",'
    '"model":"claude-haiku-4-5","content":[],"stop_reason":null,"usage":{"input_tokens":12,"cache_read_input_tokens":3,"output_tokens":1}}}\n\n'
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    'event: ping\ndata: {"type": "ping"}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n\n'
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":7}}\n\n'
    'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)

OPENAI_SSE = (
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],"usage":null}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}],"usage":null}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":1,"total_tokens":10}}\n\n'
    "data: [DONE]\n\n"
)


def _provider(cls, handler, type_="openai"):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return cls("p", ProviderConfig(type=type_, base_url="https://example.test/v1"), client, "sk-test")


async def _collect(agen):
    return [c async for c in agen]


def test_anthropic_request_translation():
    body = {
        "messages": [
            {"role": "system", "content": "sys one"},
            {"role": "developer", "content": "sys two"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": [{"type": "text", "text": "b"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
            {"role": "assistant", "content": "c"},
            {"role": "user", "content": "d"},
        ],
        "temperature": 1.7,
        "stop": "END",
        "max_completion_tokens": 50,
    }
    out = to_anthropic_request("p", body, "claude-haiku-4-5", stream=True)
    assert out["system"] == "sys one\n\nsys two"
    assert [m["role"] for m in out["messages"]] == ["user", "assistant", "user"]
    assert len(out["messages"][0]["content"]) == 3  # merged consecutive user turns
    assert out["messages"][0]["content"][2]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}
    assert out["temperature"] == 1.0
    assert out["stop_sequences"] == ["END"]
    assert out["max_tokens"] == 50
    assert out["stream"] is True


@pytest.mark.parametrize("body", [
    {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function"}]},
    {"messages": [{"role": "user", "content": "x"}], "n": 2},
    {"messages": [{"role": "tool", "content": "x"}]},
])
def test_anthropic_unsupported_features_fail_over(body):
    with pytest.raises(ProviderError) as ei:
        to_anthropic_request("p", body, "m", stream=False)
    assert ei.value.kind == "unsupported" and ei.value.failover and not ei.value.retryable


def test_anthropic_response_translation():
    out = from_anthropic_response(
        {"id": "msg", "model": "claude-haiku-4-5", "content": [{"type": "text", "text": "Hi"}], "stop_reason": "max_tokens",
         "usage": {"input_tokens": 10, "cache_creation_input_tokens": 2, "output_tokens": 4}},
        "cid",
    )
    assert out["choices"][0]["message"]["content"] == "Hi"
    assert out["choices"][0]["finish_reason"] == "length"
    assert out["usage"] == {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}


def test_anthropic_stream_conversion():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["headers"] = req.headers
        return httpx.Response(200, text=ANTHROPIC_SSE, headers={"content-type": "text/event-stream"})

    p = _provider(AnthropicProvider, handler, "anthropic")
    chunks = asyncio.run(_collect(p.stream({"messages": [{"role": "user", "content": "x"}]}, "claude-haiku-4-5", 1, 1)))
    assert seen["headers"]["x-api-key"] == "sk-test"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    texts = [c["choices"][0]["delta"].get("content") for c in chunks if c["choices"]]
    assert texts == ["Hello", " world", None]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"] == {"prompt_tokens": 15, "completion_tokens": 7, "total_tokens": 22}


def test_anthropic_stream_error_event_is_retryable():
    sse = ANTHROPIC_SSE.split("event: content_block_stop")[0] + 'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n'
    p = _provider(AnthropicProvider, lambda r: httpx.Response(200, text=sse), "anthropic")
    with pytest.raises(ProviderError) as ei:
        asyncio.run(_collect(p.stream({"messages": [{"role": "user", "content": "x"}]}, "m", 1, 1)))
    assert ei.value.retryable


def test_openai_stream_passthrough_and_usage_requested():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(req.content)
        return httpx.Response(200, text=OPENAI_SSE)

    p = _provider(OpenAICompatProvider, handler)
    chunks = asyncio.run(_collect(p.stream({"model": "fast", "messages": [], "stream": True}, "gpt-4o-mini", 1, 1)))
    assert seen["body"]["model"] == "gpt-4o-mini"
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert chunks[-1]["usage"]["prompt_tokens"] == 9


def test_openai_truncated_stream_raises():
    body = OPENAI_SSE.split('data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}')[0]
    p = _provider(OpenAICompatProvider, lambda r: httpx.Response(200, text=body))
    with pytest.raises(ProviderError) as ei:
        asyncio.run(_collect(p.stream({"messages": []}, "m", 1, 1)))
    assert ei.value.kind == "stream_truncated"


@pytest.mark.parametrize("status,retryable,failover", [(400, False, False), (401, False, True), (404, False, True),
                                                       (429, True, True), (500, True, True), (529, True, True)])
def test_status_classification(status, retryable, failover):
    p = _provider(OpenAICompatProvider, lambda r: httpx.Response(status, json={"error": {"message": "x"}}, headers={"retry-after": "3"}))
    with pytest.raises(ProviderError) as ei:
        asyncio.run(p.complete({"messages": []}, "m", 1, 1))
    assert (ei.value.retryable, ei.value.failover) == (retryable, failover)
    if status == 429:
        assert ei.value.retry_after == 3.0
