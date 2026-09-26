"""Test harness: real mock provider and real gateway, each in a uvicorn thread.

Needs Postgres and Redis. Defaults match the CI service containers; override
with GATEWAY_TEST_DATABASE_URL / GATEWAY_TEST_REDIS_URL.
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
DB_URL = os.environ.get("GATEWAY_TEST_DATABASE_URL", "postgresql://gateway:gateway@localhost:5432/gateway_test")
REDIS_URL = os.environ.get("GATEWAY_TEST_REDIS_URL", "redis://localhost:6379/15")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    def __init__(self, app, port: int):
        self.port = port
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.time() + 20
        while not self.server.started:
            if time.time() > deadline or not self.thread.is_alive():
                raise RuntimeError("server failed to start")
            time.sleep(0.05)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


async def _reset_state() -> None:
    import asyncpg

    conn = await asyncpg.connect(DB_URL)
    try:
        await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    finally:
        await conn.close()
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(REDIS_URL)
        await r.flushdb()
        await r.aclose()
    except ImportError:
        pass


@pytest.fixture(scope="session")
def mock_server():
    from mock_provider.app import app as mock_app

    srv = ServerThread(mock_app, _free_port())
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture(scope="session")
def gateway_server(mock_server):
    asyncio.run(_reset_state())
    os.environ["MOCK_URL"] = mock_server.url
    from gateway.config import Settings
    from gateway.main import create_app

    settings = Settings(
        database_url=DB_URL,
        redis_url=REDIS_URL,
        config_path=str(ROOT / "tests" / "gateway_test.yaml"),
        admin_token="test-admin",
        key_pepper="test-pepper",
        auth_cache_ttl_s=0.5,
        config_poll_interval_s=0.5,
        log_level="WARNING",
    )
    srv = ServerThread(create_app(settings), _free_port())
    srv.start()
    srv.settings = settings  # type: ignore[attr-defined]
    yield srv
    srv.stop()


@pytest.fixture(scope="session")
def api_key(gateway_server) -> str:
    return _make_key(gateway_server, "t-default", budget=100.0)


def _make_key(gateway_server, tenant: str, budget: float | None = None, rpm: int | None = None, tpm: int | None = None) -> str:
    from gateway import db
    from gateway.api.auth import KeyStore

    async def go() -> str:
        pool = await db.create_pool(DB_URL, min_size=1, max_size=1)
        try:
            store = KeyStore(pool, "test-pepper")
            await store.create_tenant(tenant, tenant, budget, rpm, tpm)
            key, _ = await store.create_key(tenant, "test")
            return key
        finally:
            await pool.close()

    return asyncio.run(go())


@pytest.fixture
def make_key(gateway_server):
    def _mk(tenant: str, **kw) -> str:
        return _make_key(gateway_server, tenant, **kw)

    return _mk


@pytest.fixture(autouse=True)
def reset_mock(request):
    if "mock_server" in request.fixturenames or "gateway_server" in request.fixturenames:
        srv = request.getfixturevalue("mock_server")
        httpx.post(f"{srv.url}/control/reset-all")
    yield


@pytest.fixture
def mock_control(mock_server):
    def _set(instance: str, **patch) -> None:
        r = httpx.put(f"{mock_server.url}/control/{instance}", json=patch)
        r.raise_for_status()

    return _set


@pytest.fixture
def mock_stats(mock_server):
    def _get(instance: str) -> dict:
        return httpx.get(f"{mock_server.url}/stats").json().get(instance, {})

    return _get


@pytest.fixture
def client(gateway_server, api_key):
    from openai import OpenAI

    return OpenAI(base_url=f"{gateway_server.url}/v1", api_key=api_key, max_retries=0, timeout=15)
