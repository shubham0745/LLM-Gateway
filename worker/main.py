"""Background writer: Redis Stream -> Postgres.

Runs as its own process (``python -m worker.main``). Several workers can run
side by side: they share one consumer group, so each event is written once.
Events are acknowledged only after the database commit, and events a crashed
worker left pending are reclaimed, so nothing is lost between the gateway and
Postgres. Inserts are idempotent (``ON CONFLICT DO NOTHING``) so a redelivered
event is harmless.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from datetime import datetime

import asyncpg
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from gateway import db
from gateway.config import Settings
from gateway.telemetry.logging import setup_logging

logger = logging.getLogger("worker")
GROUP = "writers"

_INSERT_LOG = """
INSERT INTO request_logs (request_id, ts, tenant_id, key_id, alias, provider, model, stream, status, http_status,
    error, cache_status, attempts, ttft_ms, latency_ms, prompt_tokens, completion_tokens, usage_estimated, cost_usd, saved_usd)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
ON CONFLICT (request_id) DO NOTHING
"""


def _log_row(d: dict) -> tuple:
    return (
        d["request_id"], datetime.fromisoformat(d["ts"]), d.get("tenant_id"), d.get("key_id"), d.get("alias"),
        d.get("provider"), d.get("model"), bool(d.get("stream")), d.get("status", "ok"), d.get("http_status"),
        d.get("error"), d.get("cache_status", "miss"), d.get("attempts") or [], d.get("ttft_ms"), d.get("latency_ms"),
        int(d.get("prompt_tokens") or 0), int(d.get("completion_tokens") or 0), bool(d.get("usage_estimated")),
        float(d.get("cost_usd") or 0), float(d.get("saved_usd") or 0),
    )


class Writer:
    def __init__(self, pool: asyncpg.Pool, redis: Redis, stream: str, consumer: str):
        self.pool = pool
        self.redis = redis
        self.stream = stream
        self.consumer = consumer
        self.handlers = {"request": self._write_requests, "semantic_cache_put": self._write_semantic}

    async def ensure_group(self) -> None:
        try:
            await self.redis.xgroup_create(self.stream, GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def _write_requests(self, events: list[dict]) -> None:
        await self.pool.executemany(_INSERT_LOG, [_log_row(e) for e in events])

    async def _write_semantic(self, events: list[dict]) -> None:
        # Imported lazily: the semantic cache module pulls in numpy.
        from gateway.cache.layer import insert_rows

        await insert_rows(self.pool, events)

    async def process(self, entries: list[tuple[bytes, dict]]) -> int:
        by_kind: dict[str, list[dict]] = {}
        ids = []
        for msg_id, fields in entries:
            ids.append(msg_id)
            kind = (fields.get(b"kind") or b"request").decode()
            try:
                by_kind.setdefault(kind, []).append(json.loads(fields[b"data"]))
            except (KeyError, json.JSONDecodeError):
                logger.error("dropping malformed event %s", msg_id)
        for kind, events in by_kind.items():
            handler = self.handlers.get(kind)
            if handler is None:
                logger.error("no handler for event kind %s", kind)
                continue
            await handler(events)
        if ids:
            await self.redis.xack(self.stream, GROUP, *ids)
        return len(ids)

    async def run_once(self, block_ms: int = 1000, count: int = 1000) -> int:
        resp = await self.redis.xreadgroup(GROUP, self.consumer, {self.stream: ">"}, count=count, block=block_ms)
        total = 0
        for _stream, entries in resp or []:
            total += await self.process(entries)
        return total

    async def reclaim(self, min_idle_ms: int = 60_000) -> int:
        """Take over events another worker read but never acknowledged."""
        total, start = 0, "0-0"
        while True:
            start, entries, *_ = await self.redis.xautoclaim(self.stream, GROUP, self.consumer, min_idle_ms, start, count=500)
            if entries:
                total += await self.process(entries)
            if start in (b"0-0", "0-0") or not entries:
                return total


async def main() -> None:
    settings = Settings()
    setup_logging(settings.log_level)
    pool = await db.create_pool(settings.database_url, min_size=1, max_size=4)
    await db.migrate(pool)
    redis = Redis.from_url(settings.redis_url)
    writer = Writer(pool, redis, settings.events_stream, f"worker-{os.environ.get('HOSTNAME', os.getpid())}")
    await writer.ensure_group()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    logger.info("worker started", extra={"fields": {"stream": settings.events_stream}})
    reclaimed = await writer.reclaim()
    if reclaimed:
        logger.info("reclaimed pending events", extra={"fields": {"count": reclaimed}})
    written = 0
    last_cleanup = 0.0
    while not stop.is_set():
        try:
            if loop.time() - last_cleanup > 300:
                deleted = await pool.execute("DELETE FROM semantic_cache WHERE expires_at < now()")
                logger.info("expired semantic cache rows removed", extra={"fields": {"result": deleted}})
                last_cleanup = loop.time()
            written += await writer.run_once()
        except (asyncpg.PostgresError, OSError, ConnectionError):
            logger.exception("write failed; events stay pending and will be retried")
            await asyncio.sleep(2)
            await writer.reclaim(min_idle_ms=0)
    await redis.aclose()
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
