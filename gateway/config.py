"""YAML configuration: providers and the model aliases that route to them.

Example::

    providers:
      openai:
        type: openai
        base_url: https://api.openai.com/v1
        api_key: ${OPENAI_API_KEY}
    models:
      fast: {provider: openai, model: gpt-4o-mini}
      smart: {provider: openai, model: gpt-4o}

``${VAR}`` references are expanded from the environment at load time, so keys
never live in the file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(Exception):
    pass


class ProviderConfig(BaseModel):
    type: str
    base_url: str
    api_key: str | None = None
    timeout_seconds: float = 60.0


class ModelRoute(BaseModel):
    provider: str
    model: str


class GatewayConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelRoute] = Field(min_length=1)

    @model_validator(mode="after")
    def _routes_name_known_providers(self) -> GatewayConfig:
        for alias, route in self.models.items():
            if route.provider not in self.providers:
                raise ValueError(
                    f"model alias {alias!r} points at unknown provider {route.provider!r}"
                )
        return self


def _expand_env(value: object) -> object:
    if isinstance(value, str):

        def sub(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ConfigError(f"environment variable {name} is not set")
            return os.environ[name]

        return _ENV_REF.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def parse_config(text: str) -> GatewayConfig:
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ConfigError("config must be a YAML mapping")
    try:
        return GatewayConfig.model_validate(_expand_env(raw))
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise ConfigError(str(exc)) from exc


def load_config(path: str | Path) -> GatewayConfig:
    return parse_config(Path(path).read_text())
