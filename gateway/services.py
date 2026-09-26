"""The process-wide service container stored on ``app.state.services``."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any

import asyncpg
from redis.asyncio import Redis

from gateway.api.auth import KeyStore
from gateway.config import GatewayConfig, Settings
from gateway.limits.budget import BudgetTracker
from gateway.limits.guard import LimitGuard
from gateway.providers.registry import ProviderRegistry
from gateway.routing.breaker import CircuitBreakers
from gateway.routing.config_store import ConfigStore
from gateway.routing.router import Router
from gateway.telemetry.emitter import Telemetry

logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    config: GatewayConfig
    config_version: int
    pool: asyncpg.Pool
    redis: Redis
    breakers: CircuitBreakers
    keys: KeyStore
    registry: ProviderRegistry
    router: Router
    telemetry: Telemetry
    limits: LimitGuard
    budgets: BudgetTracker
    config_store: ConfigStore
    cache: Any = None  # gateway.cache.CacheLayer, set when caching is wired up
    _background: set[asyncio.Task] = field(default_factory=set)

    def apply_config(self, version: int, config: GatewayConfig) -> None:
        self.registry.apply(config)
        self.router.config = config
        self.breakers.cfg = config.breaker
        self.config = config
        self.config_version = version
        self.telemetry.metrics.config_version.set(version)

    def spawn(self, coro: Coroutine) -> None:
        """Run bookkeeping off the request path, keeping a reference so it isn't GC'd."""
        task = asyncio.get_running_loop().create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._done)

    def _done(self, task: asyncio.Task) -> None:
        self._background.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("background task failed", exc_info=task.exception())

    async def drain(self, timeout: float = 5.0) -> None:
        if self._background:
            await asyncio.wait(list(self._background), timeout=timeout)
