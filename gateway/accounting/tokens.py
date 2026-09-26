"""Token estimation.

Exact counts come from provider usage fields. Estimates are needed in two
places: reserving rate-limit/budget capacity before a call (the response length
is unknown), and billing a stream that died before the provider reported
usage. ~4 characters per token is the standard rule of thumb for English with
BPE tokenizers; it is deliberately a slight over-estimate for reservations,
which are settled against the real numbers afterwards.
"""

from __future__ import annotations

from typing import Any

CHARS_PER_TOKEN = 4
PER_MESSAGE_OVERHEAD = 4


def estimate_text_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict):
            if p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif p.get("type") == "image_url":
                parts.append("x" * 85 * CHARS_PER_TOKEN)  # low-detail image cost in OpenAI's accounting
    return "".join(parts)


def estimate_prompt_tokens(body: dict[str, Any]) -> int:
    total = 3
    for m in body.get("messages") or []:
        total += PER_MESSAGE_OVERHEAD + estimate_text_tokens(_content_text(m.get("content")))
    return total


def max_completion_tokens(body: dict[str, Any], default: int) -> int:
    return int(body.get("max_completion_tokens") or body.get("max_tokens") or default)
