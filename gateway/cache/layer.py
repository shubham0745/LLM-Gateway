"""Exact and semantic response caches.

Rules that apply to both:
  * Every entry is scoped to a tenant: one customer never receives another's
    cached answer, even for an identical prompt.
  * Only complete, successful answers are stored (finish_reason stop/length,
    plain text, n=1, no tool calls).
  * Callers can opt out per request with ``x-gateway-cache: no-cache`` (skip
    the lookup but refresh the entry) or ``no-store`` (skip both), or the
    standard ``Cache-Control`` equivalents.

Exact cache: Redis, key = SHA-256 of the canonical JSON of every field that
changes the answer (alias, messages, sampling parameters, tools, format).

Semantic cache: embeds the prompt and looks for a similar earlier prompt in
pgvector, above a similarity threshold measured in ``bench/cache_eval.py``,
and then passes the candidate through a lexical guard (``guard.py``) that
catches what embeddings miss: swapped direction, changed numbers, opposites.
Version one only considers single-turn prompts (optional system message plus
one user message): with conversation history, two requests can end in the
same question yet need different answers, and similarity says nothing about
that. The system prompt and sampling parameters must match exactly; they go
into a "scope hash" rather than the embedding.

Lookups cost latency on every miss, so each lookup is timed and exported
(``gateway_cache_lookup_duration_seconds``) and the semantic cache can be
turned off per alias.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import asyncpg
import numpy as np
from redis.asyncio import Redis

from gateway.cache.guard import compatible
from gateway.config import CacheConfig
from gateway.telemetry.metrics import Metrics

logger = logging.getLogger(__name__)

# Request fields that can change the answer. Everything else (stream,
# stream_options, user, metadata...) is ignored for cache keys.
_ANSWER_FIELDS = (
    "model", "messages", "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "tools", "tool_choice",
    "functions", "function_call", "response_format", "seed", "n", "frequency_penalty", "presence_penalty", "logit_bias",
    "logprobs", "top_logprobs", "reasoning_effort",
)
_SCOPE_FIELDS = tuple(f for f in _ANSWER_FIELDS if f != "messages")


@dataclass
class CachedAnswer:
    content: str
    finish_reason: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    kind: str  # "hit-exact" | "hit-semantic"
    similarity: float | None = None

    def to_json(self) -> dict:
        return {"content": self.content, "finish_reason": self.finish_reason, "model": self.model, "provider": self.provider,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


@dataclass
class Lookup:
    """What we learned on the way in; reused to store the answer on the way out."""

    exact_key: str | None = None
    semantic_scope: str | None = None
    prompt_text: str | None = None
    embedding: np.ndarray | None = None
    store: bool = True


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _norm_content(content: Any) -> Any:
    if isinstance(content, str):
        return content.strip()
    return content


def exact_key(tenant: str, body: dict) -> str:
    fields = {k: body[k] for k in _ANSWER_FIELDS if body.get(k) is not None}
    fields["messages"] = [
        {"role": m.get("role"), "content": _norm_content(m.get("content")), **({"name": m["name"]} if m.get("name") else {})}
        for m in body.get("messages") or []
    ]
    digest = hashlib.sha256(canonical(fields).encode()).hexdigest()
    return f"gw:cache:exact:{tenant}:{digest}"


def single_turn_prompt(body: dict) -> tuple[str, str] | None:
    """(system prompt, user prompt) when the request is single-turn plain text, else None."""
    if body.get("tools") or body.get("functions") or (body.get("n") or 1) != 1:
        return None
    system, user = [], []
    for m in body.get("messages") or []:
        role, content = m.get("role"), m.get("content")
        if not isinstance(content, str):
            return None  # images and multi-part content: not in v1
        if role in ("system", "developer"):
            system.append(content.strip())
        elif role == "user":
            user.append(content.strip())
        else:
            return None  # any assistant/tool turn makes it multi-turn
    if len(user) != 1 or not user[0]:
        return None
    return "\n".join(system), user[0]


def semantic_scope(body: dict, system_prompt: str) -> str:
    fields = {k: body[k] for k in _SCOPE_FIELDS if body.get(k) is not None}
    fields["system"] = system_prompt
    return hashlib.sha256(canonical(fields).encode()).hexdigest()


def cache_directive(headers: Any) -> str:
    """'' (normal), 'no-cache' (don't read, do write) or 'no-store' (neither)."""
    v = (headers.get("x-gateway-cache") or "").lower()
    cc = (headers.get("cache-control") or "").lower()
    if "no-store" in v or "no-store" in cc:
        return "no-store"
    if "no-cache" in v or "no-cache" in cc or "bypass" in v:
        return "no-cache"
    return ""


class CacheLayer:
    def __init__(self, redis: Redis, pool: asyncpg.Pool, metrics: Metrics, embedder: Any | None = None):
        self.redis = redis
        self.pool = pool
        self.metrics = metrics
        self.embedder = embedder

    def semantic_available(self, cfg: CacheConfig) -> bool:
        return cfg.semantic.enabled and self.embedder is not None

    async def lookup(self, tenant: str, alias: str, body: dict, cfg: CacheConfig, directive: str) -> tuple[CachedAnswer | None, Lookup]:
        info = Lookup(store=directive != "no-store")
        if alias in cfg.disabled_aliases or directive == "no-store":
            info.store = False
            return None, info

        if cfg.exact_enabled:
            info.exact_key = exact_key(tenant, body)
            if directive != "no-cache":
                t0 = time.perf_counter()
                raw = await self.redis.get(info.exact_key)
                self.metrics.cache_lookup_duration.labels("exact").observe(time.perf_counter() - t0)
                self.metrics.cache_lookups.labels("exact", "hit" if raw else "miss").inc()
                if raw:
                    d = json.loads(raw)
                    return CachedAnswer(**d, kind="hit-exact"), info

        if self.semantic_available(cfg):
            st = single_turn_prompt(body)
            if st is not None:
                system, user = st
                info.semantic_scope = semantic_scope(body, system)
                info.prompt_text = user
                t0 = time.perf_counter()
                info.embedding = await self.embedder.aembed(user)
                if directive != "no-cache":
                    hit = await self._semantic_search(tenant, info.semantic_scope, info.embedding, cfg.semantic.threshold, user)
                    self.metrics.cache_lookup_duration.labels("semantic").observe(time.perf_counter() - t0)
                    self.metrics.cache_lookups.labels("semantic", "hit" if hit else "miss").inc()
                    if hit:
                        return hit, info
        return None, info

    async def _semantic_search(self, tenant: str, scope: str, emb: np.ndarray, threshold: float, prompt: str) -> CachedAnswer | None:
        rows = await self.pool.fetch(
            """
            SELECT prompt, response, 1 - (embedding <=> $1::vector) AS similarity
            FROM semantic_cache
            WHERE tenant_id = $2 AND scope_hash = $3 AND expires_at > now()
            ORDER BY embedding <=> $1::vector
            LIMIT 3
            """,
            vector_literal(emb), tenant, scope,
        )
        for row in rows:
            if row["similarity"] < threshold:
                break
            ok, reason = compatible(row["prompt"], prompt)
            if ok:
                return CachedAnswer(**row["response"], kind="hit-semantic", similarity=round(float(row["similarity"]), 4))
            self.metrics.cache_lookups.labels("semantic", "guard_rejected").inc()
            logger.debug("semantic candidate rejected by guard: %s", reason)
        return None

    async def store(self, tenant: str, info: Lookup, answer: dict, cfg: CacheConfig, enqueue) -> None:
        """Save a finished answer. Exact: straight to Redis. Semantic: via the worker (off the request path)."""
        if not info.store:
            return
        if info.exact_key:
            await self.redis.set(info.exact_key, json.dumps(answer), ex=cfg.exact_ttl_s)
        if info.embedding is not None and info.semantic_scope:
            enqueue({
                "_kind": "semantic_cache_put", "tenant_id": tenant, "scope_hash": info.semantic_scope,
                "prompt": info.prompt_text, "embedding": [round(float(x), 6) for x in info.embedding],
                "response": answer, "ttl_s": cfg.semantic.ttl_s,
            })


def vector_literal(v: np.ndarray | list[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


async def insert_rows(pool: asyncpg.Pool, events: list[dict]) -> None:
    await pool.executemany(
        """
        INSERT INTO semantic_cache (tenant_id, scope_hash, prompt, embedding, response, expires_at)
        VALUES ($1, $2, $3, $4::vector, $5, now() + make_interval(secs => $6))
        """,
        [(e["tenant_id"], e["scope_hash"], e["prompt"], vector_literal(e["embedding"]), e["response"], float(e["ttl_s"])) for e in events],
    )
