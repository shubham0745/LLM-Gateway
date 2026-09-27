from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from gateway.config import GatewayConfig, parse_config
from gateway.main import create_app
from mock_provider.app import app as mock_app

MOCK_CONFIG = """
providers:
  mock:
    type: openai
    base_url: http://mock/v1
    api_key: test-key
models:
  fast: {provider: mock, model: mock-small}
  smart: {provider: mock, model: mock-large}
  broken: {provider: mock, model: no-such-model}
"""


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # Tests of hand-written core functions stay red-free in CI while those
    # functions are stubs: they're expected to fail with NotImplementedError,
    # and any *other* failure still fails the run. Once a function is
    # implemented its tests XPASS; then delete the core_todo marker.
    for item in items:
        if item.get_closest_marker("core_todo"):
            item.add_marker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason="core function is still a stub (TODO(core))",
                    strict=False,
                )
            )


@pytest.fixture(autouse=True)
def _fast_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LATENCY_MS", "0")
    monkeypatch.setenv("MOCK_TOKENS_PER_SECOND", "0")


@pytest.fixture
def config() -> GatewayConfig:
    return parse_config(MOCK_CONFIG)


@pytest.fixture
async def client(config: GatewayConfig) -> AsyncIterator[httpx.AsyncClient]:
    """A client for the gateway, whose provider calls go to the in-process mock."""
    app = create_app(config, transport=httpx.ASGITransport(app=mock_app))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as c:
            yield c
