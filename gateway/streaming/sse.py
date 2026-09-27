"""Server-sent events: parsing upstream streams and encoding our own.

Wire format (https://html.spec.whatwg.org/multipage/server-sent-events.html)::

    event: message_start\n
    data: {"a": 1}\n
    \n

An event is a block of ``field: value`` lines ended by a blank line. Lines
may end in ``\n``, ``\r\n`` or ``\r``. Several ``data:`` lines in one event
join with ``\n``. Lines starting with ``:`` are comments (keep-alives).
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass

# OpenAI's end-of-stream sentinel, sent as the data of the last event.
DONE = "[DONE]"


@dataclass(frozen=True)
class SSEEvent:
    data: str
    event: str | None = None
    id: str | None = None


async def parse_sse(chunks: AsyncIterable[bytes]) -> AsyncIterator[SSEEvent]:
    """Turn raw bytes from the network into complete SSE events.

    ``chunks`` arrive however the network split them: one chunk can hold
    several events, and one event (or one UTF-8 character) can be split
    across chunks. Yield each event as soon as its terminating blank line
    arrives; never wait for the whole stream.

    Rules this must follow (see tests/test_sse.py):

    - Handle ``\\n``, ``\\r\\n`` and ``\\r`` line endings, including a ``\\r\\n``
      split across two chunks.
    - ``field: value`` and ``field:value`` both work: strip one leading
      space from the value only.
    - Join multiple ``data`` lines with ``\\n``.
    - Skip comment lines (starting with ``:``) and unknown fields.
    - A block with no ``data`` line yields nothing.
    - An incomplete event left over when the stream ends is discarded.
    - Pass ``[DONE]`` through as an ordinary event; stopping is the
      adapter's job, not the parser's.

    TODO(core): implement by hand.
    """
    raise NotImplementedError("parse_sse is a stub: implement it to make tests/test_sse.py pass")
    yield  # unreachable; makes this function an async generator


def encode_sse(data: str) -> bytes:
    """Encode one ``data:`` event. ``data`` must not contain newlines (JSON is fine)."""
    return f"data: {data}\n\n".encode()
