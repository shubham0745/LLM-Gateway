"""Builds provider adapters from the config document."""

from __future__ import annotations

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


def make_client(max_connections: int = 1000) -> httpx.AsyncClient:
    # One pooled client per provider: keep-alive connections are reused across
    # requests, which saves a TLS handshake (~50-150 ms) on every call.
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections, keepalive_expiry=60)
    return httpx.AsyncClient(limits=limits, http2=False, trust_env=True)


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._clients: dict[str, tuple[ProviderConfig, httpx.AsyncClient]] = {}

    def apply(self, config: GatewayConfig) -> None:
        """(Re)build adapters for a new config version, reusing HTTP pools where unchanged."""
        providers: dict[str, Provider] = {}
        clients: dict[str, tuple[ProviderConfig, httpx.AsyncClient]] = {}
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


def _schedule_close(client: httpx.AsyncClient, delay: float = 300.0) -> None:
    import asyncio

    async def _close() -> None:
        await asyncio.sleep(delay)
        await client.aclose()

    try:
        asyncio.get_running_loop().create_task(_close())
    except RuntimeError:
        pass
