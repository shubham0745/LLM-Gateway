"""Tests for OpenAIAdapter.translate_stream, left as a stub to implement by hand.

Marked core_todo: xfail while the stub raises NotImplementedError. Implement
it in gateway/providers/openai.py, then remove the marker.
"""

import json
from collections.abc import AsyncIterator

import pytest

from gateway.config import ProviderConfig
from gateway.providers.base import StreamTranslationError
from gateway.providers.openai import OpenAIAdapter
from gateway.streaming.sse import SSEEvent

pytestmark = pytest.mark.core_todo

ADAPTER = OpenAIAdapter("openai", ProviderConfig(type="openai", base_url="http://x"))


def chunk(content: str | None = None, finish: str | None = None, **extra) -> str:
    delta = {} if content is None else {"content": content}
    return json.dumps(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            **extra,
        }
    )


async def events(*data: str) -> AsyncIterator[SSEEvent]:
    for d in data:
        yield SSEEvent(data=d)


async def translate(*data: str):
    return [c async for c in ADAPTER.translate_stream(events(*data))]


async def test_yields_one_chunk_per_event_and_stops_at_done():
    chunks = await translate(chunk("Hel"), chunk("lo"), chunk(finish="stop"), "[DONE]")
    assert [c.choices[0].delta.content for c in chunks] == ["Hel", "lo", None]
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[0].model == "gpt-4o-mini"


async def test_ignores_events_after_done():
    chunks = await translate(chunk("a"), "[DONE]", chunk("never"))
    assert len(chunks) == 1


async def test_final_usage_chunk():
    usage_chunk = json.dumps(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }
    )
    chunks = await translate(chunk("a", usage=None), usage_chunk, "[DONE]")
    assert chunks[0].usage is None
    assert chunks[-1].choices == []
    assert chunks[-1].usage is not None and chunks[-1].usage.total_tokens == 5


async def test_keeps_extra_fields():
    chunks = await translate(chunk("a", system_fingerprint="fp_9"), "[DONE]")
    assert chunks[0].model_dump()["system_fingerprint"] == "fp_9"


async def test_invalid_json_raises():
    with pytest.raises(StreamTranslationError):
        await translate(chunk("a"), "{not json", "[DONE]")


async def test_error_event_raises_with_provider_message():
    with pytest.raises(StreamTranslationError, match="overloaded"):
        await translate(chunk("a"), json.dumps({"error": {"message": "overloaded"}}))


async def test_stream_without_done_just_ends():
    chunks = await translate(chunk("a"), chunk("b"))
    assert len(chunks) == 2
