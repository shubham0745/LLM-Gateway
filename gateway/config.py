"""Process settings (from environment) and the routing/pricing config document.

Two kinds of configuration live here:

* ``Settings``: per-process wiring such as database URLs and secrets. Read once
  from the environment at startup.
* ``GatewayConfig``: providers, aliases, pricing, cache and limit defaults. This is
  a versioned document stored in Postgres, editable through the admin API, and
  hot-reloaded by every gateway instance without a restart. The YAML file under
  ``deploy/config`` only seeds the first version.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://gateway:gateway@localhost:5432/gateway"
    redis_url: str = "redis://localhost:6379/0"
    config_path: str = "deploy/config/gateway.yaml"
    admin_token: str = "change-me-admin-token"
    # Server-side secret mixed into API key hashes (HMAC). Rotating it invalidates every key.
    key_pepper: str = "change-me-pepper"
    log_level: str = "INFO"
    # How long an instance trusts its in-memory copy of a key/tenant before re-reading.
    auth_cache_ttl_s: float = 30.0
    # Backup poll interval for config changes, in case a pub/sub reload message was missed.
    config_poll_interval_s: float = 10.0
    embedding_model_dir: str = "models/all-MiniLM-L6-v2"
    # Name of the Redis stream the worker consumes.
    events_stream: str = "gw:events"
    events_stream_maxlen: int = 1_000_000
    instance_id: str = Field(default_factory=lambda: os.environ.get("HOSTNAME", "local"))


# ---------------------------------------------------------------------------
# Config document
# ---------------------------------------------------------------------------


class Timeouts(BaseModel):
    """Seconds. ``ttft`` only applies to streaming; non-streaming uses ``total``."""

    connect: float = 3.0
    ttft: float = 10.0
    # Longest silence tolerated between two chunks once a stream is flowing.
    idle: float = 20.0
    total: float = 120.0


class RetryPolicy(BaseModel):
    max_retries: int = 1  # per target, on top of the first attempt
    base_delay: float = 0.2
    max_delay: float = 2.0
    # A Retry-After longer than this makes us fail over instead of waiting.
    max_retry_after: float = 2.0


class ProviderConfig(BaseModel):
    type: Literal["openai", "anthropic", "openai_compat"]
    base_url: str
    api_key_env: str | None = None
    # Extra static headers (e.g. OpenAI-Organization). Never put secrets here.
    headers: dict[str, str] = Field(default_factory=dict)
    max_connections: int = 1000


class Target(BaseModel):
    provider: str
    model: str
    timeouts: Timeouts | None = None
    retry: RetryPolicy | None = None


class BreakerConfig(BaseModel):
    failure_threshold: int = 5  # consecutive failures that open the circuit
    cooldown_s: float = 15.0  # open -> half-open after this long
    half_open_max_probes: int = 1


class Price(BaseModel):
    """USD per one million tokens."""

    input: float
    output: float


class SemanticCacheConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.95
    ttl_s: int = 86_400


class CacheConfig(BaseModel):
    exact_enabled: bool = True
    exact_ttl_s: int = 3_600
    semantic: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)
    # Aliases the cache should never serve (e.g. creative-writing routes).
    disabled_aliases: list[str] = Field(default_factory=list)


class LimitDefaults(BaseModel):
    rpm: int = 600
    tpm: int = 1_000_000
    monthly_budget_usd: float = 50.0
    # Completion length assumed when the caller does not send max_tokens.
    default_completion_estimate: int = 512


class GatewayConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    aliases: dict[str, list[Target]]
    pricing: dict[str, Price] = Field(default_factory=dict)
    timeouts: Timeouts = Field(default_factory=Timeouts)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    breaker: BreakerConfig = Field(default_factory=BreakerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    limits: LimitDefaults = Field(default_factory=LimitDefaults)

    @field_validator("aliases")
    @classmethod
    def _non_empty_chains(cls, v: dict[str, list[Target]]) -> dict[str, list[Target]]:
        for alias, chain in v.items():
            if not chain:
                raise ValueError(f"alias {alias!r} has an empty fallback chain")
        return v

    @model_validator(mode="after")
    def _targets_reference_known_providers(self) -> GatewayConfig:
        for alias, chain in self.aliases.items():
            for t in chain:
                if t.provider not in self.providers:
                    raise ValueError(f"alias {alias!r} references unknown provider {t.provider!r}")
        return self

    def timeouts_for(self, t: Target) -> Timeouts:
        return t.timeouts or self.timeouts

    def retry_for(self, t: Target) -> RetryPolicy:
        return t.retry or self.retry


_ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def expand_env(text: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` so one file serves Docker and local runs."""
    return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), text)


def load_config_file(path: str | Path) -> GatewayConfig:
    with open(path) as f:
        return GatewayConfig.model_validate(yaml.safe_load(expand_env(f.read())))
