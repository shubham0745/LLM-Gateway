"""A lexical safety check applied after the similarity threshold.

Measured in bench/cache_eval.py: mean-pooled sentence embeddings barely see
word order or small words. "convert a string to an integer" and "convert an
integer to a string" score 0.996, higher than most true paraphrases, so no
threshold can separate them. This guard rejects the three patterns that
caused most wrong hits, without a second model:

1. Different numbers ("15% of 200" vs "20% of 150").
2. A direction swap around to/into/from/than/vs ("km to miles" vs
   "miles to km").
3. An opposite pair where each prompt uses a different side
   ("ascending" vs "descending", "install" vs "uninstall").

It is deliberately conservative: it only rejects on positive evidence, so
it costs few true hits.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_DIRECTIONAL = {"to", "into", "from", "than", "vs", "versus", "onto"}
_STOP = {
    "a", "an", "the", "of", "in", "on", "for", "with", "and", "or", "is", "are", "was", "be", "do", "does", "did", "i",
    "you", "my", "me", "can", "could", "how", "what", "what's", "whats", "why", "which", "who", "when", "where", "it",
    "this", "that", "please", "way", "using", "use", "some", "any", "there", "should", "would", "will", "turn", "convert",
}
# Generic antonyms only. Pairs that exist just to fix a specific example in
# the hand-written set would inflate its score, so they are not allowed here;
# bench/data/heldout_pairs.jsonl was written after this list was frozen.
_OPPOSITES = [
    ("ascending", "descending"), ("best", "worst"), ("good", "bad"), ("soft", "hard"), ("install", "uninstall"),
    ("read", "write"), ("largest", "smallest"), ("biggest", "smallest"), ("tallest", "shortest"), ("start", "end"),
    ("begin", "end"), ("accept", "decline"), ("accepting", "declining"), ("min", "max"), ("minimum", "maximum"),
    ("increase", "decrease"), ("before", "after"), ("add", "remove"), ("encode", "decode"), ("encrypt", "decrypt"),
    ("enable", "disable"), ("open", "close"), ("upload", "download"), ("import", "export"), ("push", "pull"),
    ("advantages", "disadvantages"), ("pros", "cons"), ("buy", "sell"), ("hot", "cold"), ("fast", "slow"),
    ("faster", "slower"), ("high", "low"), ("higher", "lower"), ("more", "less"), ("positive", "negative"),
    ("true", "false"), ("win", "lose"), ("left", "right"), ("north", "south"), ("east", "west"), ("male", "female"),
    ("men", "women"), ("first", "last"), ("plus", "minus"), ("create", "delete"), ("insert", "delete"),
    ("love", "hate"), ("safe", "dangerous"), ("legal", "illegal"), ("possible", "impossible"),
]
_OPP: dict[str, set[str]] = {}
for _a, _b in _OPPOSITES:
    _OPP.setdefault(_a, set()).add(_b)
    _OPP.setdefault(_b, set()).add(_a)


def _stem(w: str) -> str:
    for suffix in ("ing", "ers", "er", "es", "s"):
        if len(w) > 4 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _directed_pairs(words: list[str]) -> set[tuple[str, str]]:
    """(left, right) content words around each directional word."""
    pairs = set()
    for i, w in enumerate(words):
        if w not in _DIRECTIONAL:
            continue
        left = next((_stem(x) for x in reversed(words[:i]) if x not in _STOP and x not in _DIRECTIONAL), None)
        right = next((_stem(x) for x in words[i + 1:] if x not in _STOP and x not in _DIRECTIONAL), None)
        if left and right and left != right:
            pairs.add((left, right))
    return pairs


def compatible(cached_prompt: str, new_prompt: str) -> tuple[bool, str]:
    """(True, "") when a cached answer may be reused, else (False, reason)."""
    if sorted(_NUMBER.findall(cached_prompt)) != sorted(_NUMBER.findall(new_prompt)):
        return False, "numbers differ"
    a_words, b_words = _words(cached_prompt), _words(new_prompt)
    a_pairs, b_pairs = _directed_pairs(a_words), _directed_pairs(b_words)
    if any((r, left) in b_pairs for left, r in a_pairs):
        return False, "direction swapped"
    a_set, b_set = set(a_words), set(b_words)
    for w in a_set - b_set:
        if _OPP.get(w, set()) & (b_set - a_set):
            return False, f"opposite terms ({w})"
    return True, ""
