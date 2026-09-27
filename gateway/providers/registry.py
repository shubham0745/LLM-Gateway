"""Maps the ``type`` in a provider's config to its adapter class."""

from __future__ import annotations

from gateway.config import ConfigError, GatewayConfig
from gateway.providers.base import ProviderAdapter
from gateway.providers.openai import OpenAIAdapter

ADAPTER_TYPES: dict[str, type[ProviderAdapter]] = {
    "openai": OpenAIAdapter,
}


def build_adapters(config: GatewayConfig) -> dict[str, ProviderAdapter]:
    adapters: dict[str, ProviderAdapter] = {}
    for name, provider in config.providers.items():
        adapter_cls = ADAPTER_TYPES.get(provider.type)
        if adapter_cls is None:
            known = ", ".join(sorted(ADAPTER_TYPES))
            raise ConfigError(
                f"provider {name!r} has unknown type {provider.type!r} (known: {known})"
            )
        adapters[name] = adapter_cls(name, provider)
    return adapters
