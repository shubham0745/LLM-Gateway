"""The process-wide service container stored on ``app.state.services``."""

from __future__ import annotations

from dataclasses import dataclass

import asyncpg
from redis.asyncio import Redis

from gateway.api.auth import KeyStore
from gateway.config import GatewayConfig, Settings
from gateway.providers.registry import ProviderRegistry
from gateway.routing.breaker import CircuitBreakers
from gateway.routing.router import Router
from gateway.telemetry.emitter import Telemetry


@dataclass
class Services:
    settings: Settings
    config: GatewayConfig
    pool: asyncpg.Pool
    redis: Redis
    breakers: CircuitBreakers
    keys: KeyStore
    registry: ProviderRegistry
    router: Router
    telemetry: Telemetry

    def apply_config(self, config: GatewayConfig) -> None:
        self.registry.apply(config)
        self.router.config = config
        self.breakers.cfg = config.breaker
        self.config = config
