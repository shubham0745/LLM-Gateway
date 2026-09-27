"""Tests for parse_sse, which is left as a stub to implement by hand.

Every test here is marked core_todo, so while parse_sse raises
NotImplementedError they show as xfail. Implement gateway/streaming/sse.py
until they all XPASS, then remove the marker.
"""

from collections.abc import AsyncIterator

import pytest

from gateway.streaming.sse import SSEEvent, parse_sse

pytestmark = pytest.mark.core_todo


async def feed(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def parse(*chunks: bytes) -> list[SSEEvent]:
    return [event async for event in parse_sse(feed(*chunks))]


async def test_single_event():
    assert await parse(b'data: {"a": 1}\n\n') == [SSEEvent(data='{"a": 1}')]


async def test_several_events_in_one_chunk():
    events = await parse(b"data: one\n\ndata: two\n\ndata: [DONE]\n\n")
    assert [e.data for e in events] == ["one", "two", "[DONE]"]


async def test_event_split_across_chunks():
    events = await parse(b"da", b"ta: hel", b"lo\n", b"\n", b"data: again\n\n")
    assert [e.data for e in events] == ["hello", "again"]


async def test_utf8_character_split_across_chunks():
    encoded = "data: café ☕\n\n".encode()
    cut = encoded.index("☕".encode()) + 1  # inside the 3-byte character
    events = await parse(encoded[:cut], encoded[cut:])
    assert [e.data for e in events] == ["café ☕"]


@pytest.mark.parametrize("eol", [b"\n", b"\r\n", b"\r"])
async def test_line_endings(eol: bytes):
    events = await parse(b"data: x" + eol + eol + b"data: y" + eol + eol)
    assert [e.data for e in events] == ["x", "y"]


async def test_crlf_split_between_chunks():
    # A lone \r at the end of a chunk must not be read as a blank line
    # when the \n arrives in the next chunk.
    events = await parse(b"data: x\r", b"\n\r\n", b"data: y\r\n\r\n")
    assert [e.data for e in events] == ["x", "y"]


async def test_multiline_data_joins_with_newline():
    assert await parse(b"data: line1\ndata: line2\n\n") == [SSEEvent(data="line1\nline2")]


async def test_only_one_leading_space_is_stripped():
    events = await parse(b"data:no-space\n\ndata:  two-spaces\n\n")
    assert [e.data for e in events] == ["no-space", " two-spaces"]


async def test_event_and_id_fields():
    events = await parse(b"event: message_start\nid: 7\ndata: {}\n\n")
    assert events == [SSEEvent(data="{}", event="message_start", id="7")]


async def test_comments_and_unknown_fields_are_ignored():
    events = await parse(b": keep-alive\n\nretry: 100\nfoo: bar\ndata: x\n\n")
    assert events == [SSEEvent(data="x")]


async def test_block_without_data_yields_nothing():
    assert await parse(b"event: ping\n\n") == []


async def test_incomplete_trailing_event_is_dropped():
    events = await parse(b"data: done\n\ndata: cut off")
    assert [e.data for e in events] == ["done"]


async def test_yields_before_stream_ends():
    # The parser must be incremental: the first event is available while
    # the upstream is still sending.
    async def slow() -> AsyncIterator[bytes]:
        yield b"data: first\n\n"
        raise AssertionError("parser read past the first complete event")

    events = parse_sse(slow())
    first = await anext(events)
    assert first.data == "first"
