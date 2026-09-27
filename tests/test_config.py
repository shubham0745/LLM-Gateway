import pytest

from gateway.config import ConfigError, parse_config
from gateway.providers.registry import build_adapters

VALID = """
providers:
  openai: {type: openai, base_url: https://api.openai.com/v1, api_key: "${TEST_KEY}"}
models:
  fast: {provider: openai, model: gpt-4o-mini}
"""


def test_parses_aliases_and_expands_env(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    config = parse_config(VALID)
    assert config.models["fast"].model == "gpt-4o-mini"
    assert config.providers["openai"].api_key == "sk-test"
    assert config.providers["openai"].timeout_seconds == 60.0


def test_missing_env_var_is_an_error(monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    with pytest.raises(ConfigError, match="TEST_KEY"):
        parse_config(VALID)


def test_alias_must_name_a_known_provider():
    text = """
providers:
  openai: {type: openai, base_url: http://x}
models:
  fast: {provider: anthropic, model: claude}
"""
    with pytest.raises(ConfigError, match="unknown provider 'anthropic'"):
        parse_config(text)


def test_needs_at_least_one_model():
    with pytest.raises(ConfigError):
        parse_config("providers: {}\nmodels: {}\n")


def test_unknown_provider_type_is_rejected_when_building_adapters():
    config = parse_config(
        "providers:\n  x: {type: nope, base_url: http://x}\nmodels:\n  a: {provider: x, model: m}\n"
    )
    with pytest.raises(ConfigError, match="unknown type 'nope'"):
        build_adapters(config)


def test_shipped_configs_parse(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MOCK_BASE_URL", "http://mock:9000/v1")
    for path in ("config/gateway.yaml", "config/gateway.mock.yaml"):
        with open(path) as f:
            build_adapters(parse_config(f.read()))
