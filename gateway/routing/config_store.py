"""Versioned config in Postgres, hot-reloaded on every instance.

``PUT /admin/config`` validates a new document and inserts it as the next
version, then publishes a message on a Redis channel. Every gateway instance
listens and swaps in the new config (provider adapters, chains, prices,
timeouts) between requests; in-flight requests finish on the config they
started with. A periodic poll of the latest version is the backstop for a
missed message. Old versions are kept, so a rollback is one call.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable

import asyncpg
from redis.asyncio import Redis

from gateway.config import GatewayConfig

logger = logging.getLogger(__name__)
CONTROL_CHANNEL = "gw:control"


class ConfigStore:
    def __init__(self, pool: asyncpg.Pool, redis: Redis):
        self.pool = pool
        self.redis = redis

    async def latest(self) -> tuple[int, GatewayConfig] | None:
        row = await self.pool.fetchrow("SELECT version, document FROM gateway_config ORDER BY version DESC LIMIT 1")
        if row is None:
            return None
        return row["version"], GatewayConfig.model_validate(row["document"])

    async def latest_version(self) -> int:
        return await self.pool.fetchval("SELECT COALESCE(MAX(version), 0) FROM gateway_config")

    async def get(self, version: int) -> GatewayConfig | None:
        doc = await self.pool.fetchval("SELECT document FROM gateway_config WHERE version = $1", version)
        return GatewayConfig.model_validate(doc) if doc is not None else None

    async def versions(self, limit: int = 50) -> list[dict]:
        rows = await self.pool.fetch(
            "SELECT version, comment, created_at FROM gateway_config ORDER BY version DESC LIMIT $1", limit
        )
        return [{"version": r["version"], "comment": r["comment"], "created_at": r["created_at"].isoformat()} for r in rows]

    async def save(self, config: GatewayConfig, comment: str = "") -> int:
        version = await self.pool.fetchval(
            "INSERT INTO gateway_config (document, comment) VALUES ($1, $2) RETURNING version",
            config.model_dump(mode="json"),
            comment,
        )
        await self.publish({"type": "config", "version": version})
        return version

    async def seed_if_empty(self, config: GatewayConfig) -> None:
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute("LOCK TABLE gateway_config IN EXCLUSIVE MODE")
            if await conn.fetchval("SELECT count(*) FROM gateway_config") == 0:
                await conn.execute(
                    "INSERT INTO gateway_config (document, comment) VALUES ($1, 'seeded from file')",
                    config.model_dump(mode="json"),
                )

    async def publish(self, message: dict) -> None:
        await self.redis.publish(CONTROL_CHANNEL, json.dumps(message))


class ControlListener:
    """Reacts to control messages (config reloads, auth cache flushes)."""

    def __init__(
        self,
        store: ConfigStore,
        redis: Redis,
        current_version: Callable[[], int],
        on_config: Callable[[int, GatewayConfig], None],
        on_auth_invalidate: Callable[[], None],
        poll_interval_s: float = 10.0,
    ):
        self.store = store
        self.redis = redis
        self.current_version = current_version
        self.on_config = on_config
        self.on_auth_invalidate = on_auth_invalidate
        self.poll_interval_s = poll_interval_s
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(self._listen()), loop.create_task(self._poll())]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t

    async def reload_if_newer(self) -> None:
        latest = await self.store.latest()
        if latest and latest[0] > self.current_version():
            self.on_config(*latest)
            logger.info("config reloaded", extra={"fields": {"version": latest[0]}})

    async def _listen(self) -> None:
        while True:
            try:
                pubsub = self.redis.pubsub()
                await pubsub.subscribe(CONTROL_CHANNEL)
                async for msg in pubsub.listen():
                    if msg.get("type") != "message":
                        continue
                    try:
                        data = json.loads(msg["data"])
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if data.get("type") == "config":
                        await self.reload_if_newer()
                    elif data.get("type") == "auth":
                        self.on_auth_invalidate()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - reconnect on any Redis error
                logger.warning("control channel error; reconnecting", exc_info=True)
                await asyncio.sleep(1)

    async def _poll(self) -> None:
        while True:
            await asyncio.sleep(self.poll_interval_s)
            try:
                await self.reload_if_newer()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("config poll failed", exc_info=True)
