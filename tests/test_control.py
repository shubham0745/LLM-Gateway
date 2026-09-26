"""Phase 3: rate limits, budgets, cost accounting, admin API, hot reload, metrics, log worker."""

from __future__ import annotations

import asyncio
import concurrent.futures
import time

import httpx
import openai
import pytest

from tests.conftest import DB_URL, REDIS_URL

ADMIN = {"authorization": "Bearer test-admin"}


def _client(gateway_server, key):
    return openai.OpenAI(base_url=f"{gateway_server.url}/v1", api_key=key, max_retries=0, timeout=15)


def _ask(c, model="mock", **kw):
    return c.chat.completions.with_raw_response.create(model=model, messages=[{"role": "user", "content": kw.pop("prompt", "hi")}], **kw)


def test_cost_and_ratelimit_headers(gateway_server, make_key):
    c = _client(gateway_server, make_key("t-cost"))
    raw = _ask(c)
    resp = raw.parse()
    expected = (resp.usage.prompt_tokens * 0.15 + resp.usage.completion_tokens * 0.60) / 1e6
    assert float(raw.headers["x-gateway-cost-usd"]) == pytest.approx(expected, rel=1e-6)
    assert int(raw.headers["x-ratelimit-limit-requests"]) == 10000
    assert "x-ratelimit-remaining-tokens" in raw.headers


def test_request_rate_limit(gateway_server, make_key):
    c = _client(gateway_server, make_key("t-rpm", rpm=3))
    for _ in range(3):
        _ask(c)
    with pytest.raises(openai.RateLimitError) as ei:
        _ask(c)
    assert ei.value.response.headers["retry-after"]
    assert ei.value.body["code"] == "rate_limit_exceeded"


def test_rate_limit_is_atomic_under_concurrency(gateway_server, make_key):
    c = _client(gateway_server, make_key("t-rpm-conc", rpm=10))

    def one(_):
        try:
            _ask(c)
            return True
        except openai.RateLimitError:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as ex:
        results = list(ex.map(one, range(25)))
    assert results.count(True) == 10


def test_token_limit_rejects_oversized_request(gateway_server, make_key):
    c = _client(gateway_server, make_key("t-tpm", tpm=100))
    with pytest.raises(openai.RateLimitError) as ei:
        _ask(c, max_tokens=500)
    assert ei.value.body["code"] == "request_too_large"


def test_token_reservation_is_settled(gateway_server, make_key):
    """Reserve max_tokens up front, refund what wasn't used: many small answers fit a tight TPM."""
    c = _client(gateway_server, make_key("t-tpm-settle", tpm=1500))
    # Each request reserves ~600 tokens (max_tokens=590) but uses ~50; without settlement only 2 would fit.
    for _ in range(6):
        _ask(c, max_tokens=590)


def test_budget_hard_stop(gateway_server, make_key):
    # mock-small: ~60 prompt+completion tokens ~= $0.00003 per call. Budget allows a handful.
    c = _client(gateway_server, make_key("t-budget", budget=0.0002))
    ok = 0
    with pytest.raises(openai.RateLimitError) as ei:
        for _ in range(50):
            _ask(c, max_tokens=40)
            ok += 1
    assert ei.value.body["code"] == "budget_exceeded"
    assert 2 <= ok < 50
    t = httpx.get(f"{gateway_server.url}/admin/tenants/t-budget", headers=ADMIN).json()
    assert t["spent_this_month_usd"] <= 0.0002 + 1e-9


def test_budget_hard_stop_under_concurrency(gateway_server, make_key):
    c = _client(gateway_server, make_key("t-budget-conc", budget=0.0003))

    def one(_):
        try:
            _ask(c, max_tokens=40)
            return True
        except openai.RateLimitError:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as ex:
        list(ex.map(one, range(60)))
    time.sleep(0.3)  # settlements run in the background
    t = httpx.get(f"{gateway_server.url}/admin/tenants/t-budget-conc", headers=ADMIN).json()
    assert t["spent_this_month_usd"] <= 0.0003 + 1e-9


def test_admin_requires_token(gateway_server):
    assert httpx.get(f"{gateway_server.url}/admin/tenants").status_code == 401
    assert httpx.get(f"{gateway_server.url}/admin/tenants", headers={"authorization": "Bearer nope"}).status_code == 401


def test_admin_tenant_and_key_lifecycle(gateway_server):
    base = gateway_server.url
    r = httpx.post(f"{base}/admin/tenants", headers=ADMIN, json={"id": "acme", "name": "Acme", "monthly_budget_usd": 5})
    assert r.status_code == 201
    k = httpx.post(f"{base}/admin/tenants/acme/keys", headers=ADMIN, json={"name": "ci"}).json()
    c = _client(gateway_server, k["key"])
    _ask(c)
    keys = httpx.get(f"{base}/admin/tenants/acme/keys", headers=ADMIN).json()["data"]
    assert keys[0]["key_prefix"] == k["key"][:10] and "key" not in keys[0]
    assert httpx.delete(f"{base}/admin/keys/{k['id']}", headers=ADMIN).json()["revoked_at"]
    with pytest.raises(openai.AuthenticationError):
        _ask(c)  # revocation is immediate, not after the auth cache TTL


def test_tenant_limit_change_applies_without_restart(gateway_server, make_key):
    key = make_key("t-patch", rpm=1000)
    c = _client(gateway_server, key)
    _ask(c)
    r = httpx.patch(f"{gateway_server.url}/admin/tenants/t-patch", headers=ADMIN, json={"monthly_budget_usd": 0})
    assert r.json()["monthly_budget_usd"] == 0
    time.sleep(0.2)
    with pytest.raises(openai.RateLimitError):
        _ask(c)


def test_config_hot_reload_and_rollback(gateway_server, make_key):
    base = gateway_server.url
    c = _client(gateway_server, make_key("t-reload"))
    cur = httpx.get(f"{base}/admin/config", headers=ADMIN).json()
    cfg = cur["config"]
    cfg["aliases"]["mock"] = [{"provider": "mock-backup", "model": "mock-small"}]
    cfg["aliases"]["brand-new"] = [{"provider": "mock-third", "model": "mock-small"}]
    r = httpx.put(f"{base}/admin/config", headers=ADMIN, json=cfg, params={"comment": "swap"})
    assert r.status_code == 200 and r.json()["version"] > cur["version"]
    assert _ask(c).headers["x-gateway-provider"] == "mock-backup"
    assert _ask(c, model="brand-new").headers["x-gateway-provider"] == "mock-third"
    rb = httpx.post(f"{base}/admin/config/rollback/{cur['version']}", headers=ADMIN)
    assert rb.status_code == 200
    assert _ask(c).headers["x-gateway-provider"] == "mock-primary"


def test_invalid_config_rejected(gateway_server):
    bad = {"providers": {}, "aliases": {"x": [{"provider": "nope", "model": "m"}]}}
    r = httpx.put(f"{gateway_server.url}/admin/config", headers=ADMIN, json=bad)
    assert r.status_code == 400


def test_config_reload_reaches_other_instances(gateway_server):
    """A second gateway process picks up a config change via pub/sub (no restart)."""
    from gateway.config import Settings
    from gateway.main import create_app
    from tests.conftest import ROOT, ServerThread, _free_port

    settings = Settings(
        database_url=DB_URL, redis_url=REDIS_URL, config_path=str(ROOT / "tests" / "gateway_test.yaml"),
        admin_token="test-admin", key_pepper="test-pepper", config_poll_interval_s=60, log_level="WARNING",
    )
    other = ServerThread(create_app(settings), _free_port())
    other.start()
    try:
        cfg = httpx.get(f"{gateway_server.url}/admin/config", headers=ADMIN).json()["config"]
        cfg["aliases"]["only-on-reload"] = [{"provider": "mock-third", "model": "mock-small"}]
        v = httpx.put(f"{gateway_server.url}/admin/config", headers=ADMIN, json=cfg).json()["version"]
        deadline = time.time() + 5
        while time.time() < deadline:
            if httpx.get(f"{other.url}/readyz").json().get("config_version") == v:
                break
            time.sleep(0.05)
        assert httpx.get(f"{other.url}/readyz").json()["config_version"] == v
    finally:
        other.stop()


def test_circuits_endpoint(gateway_server, client, mock_control):
    mock_control("primary", mode="outage")
    for _ in range(3):
        client.chat.completions.create(model="mock", messages=[{"role": "user", "content": "x"}])
    snap = httpx.get(f"{gateway_server.url}/admin/circuits", headers=ADMIN).json()
    assert snap["mock-primary"]["state"] == "open"
    reset = httpx.post(f"{gateway_server.url}/admin/circuits/mock-primary/reset", headers=ADMIN).json()
    assert reset["state"] == "closed"


def test_metrics_exposed(gateway_server, client, mock_control):
    mock_control("primary", mode="outage")
    client.chat.completions.create(model="mock", messages=[{"role": "user", "content": "x"}])
    text = httpx.get(f"{gateway_server.url}/metrics").text
    assert "gateway_requests_total" in text
    assert 'gateway_failovers_total{alias="mock",from_provider="mock-primary",to_provider="mock-backup"}' in text
    assert "gateway_cost_usd_total" in text
    assert 'gateway_circuit_state{provider="mock-primary"}' in text


def test_worker_writes_logs_and_usage_report(gateway_server, make_key):
    """Requests flow gateway -> Redis Stream -> worker -> Postgres, and /admin/usage answers from it."""
    from gateway import db
    from worker.main import Writer

    c = _client(gateway_server, make_key("t-usage"))
    rid = _ask(c).headers["x-request-id"]
    for _ in range(3):
        _ask(c, prompt="another")
    list(c.chat.completions.create(model="mock", messages=[{"role": "user", "content": "s"}], stream=True))
    time.sleep(0.3)  # the gateway batches stream appends

    async def drain():
        from redis.asyncio import Redis

        pool = await db.create_pool(DB_URL, min_size=1, max_size=2)
        redis = Redis.from_url(REDIS_URL)
        w = Writer(pool, redis, "gw:events", "test-worker")
        await w.ensure_group()
        while await w.run_once(block_ms=200):
            pass
        await redis.aclose()
        await pool.close()

    asyncio.run(drain())
    row = httpx.get(f"{gateway_server.url}/admin/requests/{rid}", headers=ADMIN).json()
    assert row["tenant_id"] == "t-usage" and row["provider"] == "mock-primary" and row["cost_usd"] > 0
    assert row["attempts"][0]["outcome"] == "ok"
    today = time.strftime("%Y-%m-%d", time.gmtime())
    rep = httpx.get(f"{gateway_server.url}/admin/usage", headers=ADMIN, params={"start": today, "tenant": "t-usage"}).json()
    assert rep["data"][0]["requests"] == 5
    assert rep["data"][0]["model"] == "mock-small"
    assert rep["total_cost_usd"] > 0
