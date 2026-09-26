"""Builds provider adapters from the config document."""

from __future__ import annotations

import itertools
import os

import httpx

from gateway.config import GatewayConfig, ProviderConfig
from gateway.providers.anthropic import AnthropicProvider
from gateway.providers.base import Provider
from gateway.providers.openai_compat import OpenAICompatProvider

_ADAPTERS: dict[str, type[Provider]] = {
    "openai": OpenAICompatProvider,
    "openai_compat": OpenAICompatProvider,
    "anthropic": AnthropicProvider,
}


POOL_SHARDS = 16


class ClientPool:
    """Several pooled httpx clients for one provider, used round-robin.

    Keep-alive connections save a TCP (and TLS) handshake on every call, but
    httpcore's pool checks every idle connection each time it hands one out,
    so one pool holding hundreds of connections costs O(connections) per
    request; under load that scan was the gateway's biggest CPU cost. Splitting
    the connections across shards keeps each scan short.
    """

    def __init__(self, max_connections: int = 1000, shards: int = POOL_SHARDS):
        per = max(1, -(-max_connections // shards))
        limits = httpx.Limits(max_connections=per, max_keepalive_connections=per, keepalive_expiry=60)
        self.clients = [httpx.AsyncClient(limits=limits, http2=False, trust_env=True) for _ in range(shards)]
        self._next = itertools.cycle(self.clients)

    def pick(self) -> httpx.AsyncClient:
        return next(self._next)

    async def aclose(self) -> None:
        for c in self.clients:
            await c.aclose()


def make_client(max_connections: int = 1000) -> ClientPool:
    return ClientPool(max_connections)


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._clients: dict[str, tuple[ProviderConfig, ClientPool]] = {}

    def apply(self, config: GatewayConfig) -> None:
        """(Re)build adapters for a new config version, reusing HTTP pools where unchanged."""
        providers: dict[str, Provider] = {}
        clients: dict[str, tuple[ProviderConfig, ClientPool]] = {}
        for name, pcfg in config.providers.items():
            prev = self._clients.get(name)
            if prev and prev[0] == pcfg:
                client = prev[1]
            else:
                client = make_client(pcfg.max_connections)
            clients[name] = (pcfg, client)
            api_key = os.environ.get(pcfg.api_key_env) if pcfg.api_key_env else None
            providers[name] = _ADAPTERS[pcfg.type](name, pcfg, client, api_key)
        stale = [c for n, (p, c) in self._clients.items() if n not in clients or clients[n][1] is not c]
        self._providers, self._clients = providers, clients
        for c in stale:
            # In-flight requests hold their own reference; closing later is fine
            # for a demo-scale gateway, but we avoid killing live streams.
            _schedule_close(c)

    def get(self, name: str) -> Provider | None:
        return self._providers.get(name)

    def all(self) -> dict[str, Provider]:
        return dict(self._providers)

    async def aclose(self) -> None:
        for _, client in self._clients.values():
            await client.aclose()


def _schedule_close(client: ClientPool, delay: float = 300.0) -> None:
    import asyncio

    async def _close() -> None:
        await asyncio.sleep(delay)
        await client.aclose()

    try:
        asyncio.get_running_loop().create_task(_close())
    except RuntimeError:
        pass
