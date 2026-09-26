"""Phase 2: timeouts, retries, fallback chains and circuit breakers."""

from __future__ import annotations

import concurrent.futures
import time

import httpx
import openai
import pytest

from gateway.config import RetryPolicy
from gateway.routing.router import backoff_delay


def _served_by(raw) -> str:
    return raw.headers["x-gateway-provider"]


def _chat(client, model="mock", stream=False, **kw):
    msgs = [{"role": "user", "content": kw.pop("prompt", "hello")}]
    if stream:
        text = ""
        for c in client.chat.completions.create(model=model, messages=msgs, stream=True, **kw):
            if c.choices:
                text += c.choices[0].delta.content or ""
        return text
    return client.chat.completions.with_raw_response.create(model=model, messages=msgs, **kw)


def test_backoff_has_jitter_and_respects_retry_after():
    p = RetryPolicy(max_retries=3, base_delay=0.1, max_delay=1.0, max_retry_after=2.0)
    delays = {backoff_delay(2, p) for _ in range(50)}
    assert len(delays) > 40 and all(0 <= d <= 0.4 for d in delays)
    assert backoff_delay(0, p, retry_after=1.5) >= 1.5
    assert backoff_delay(0, p, retry_after=30) is None  # too long: fail over instead


@pytest.mark.parametrize("stream", [False, True])
def test_full_outage_fails_over_with_zero_errors(client, mock_control, stream):
    mock_control("primary", mode="outage")
    for _ in range(10):
        r = _chat(client, stream=stream)
        if not stream:
            assert _served_by(r) == "mock-backup"
        else:
            assert r


def test_hanging_provider_hits_ttft_timeout_then_fails_over(client, mock_control):
    mock_control("primary", mode="hang")
    started = time.time()
    text = _chat(client, stream=True)
    assert text
    # 1 s TTFT timeout in the test config, retried once, then the backup answers.
    assert time.time() - started < 4


def test_hanging_non_stream_hits_total_timeout(client, gateway_server, mock_control):
    mock_control("primary", mode="hang")
    r = _chat(client)
    assert _served_by(r) == "mock-backup"


def test_retry_on_429_uses_same_provider(client, mock_control, mock_stats):
    # First call 429s, the retry succeeds: still served by the primary.
    mock_control("primary", fail_next=1, fail_next_status=429, retry_after=0.05)
    r = _chat(client)
    assert _served_by(r) == "mock-primary"
    assert r.headers["x-gateway-attempts"] == "2"
    assert mock_stats("primary")["by_status"] == {"429": 1, "200": 1}


def test_retry_on_500_then_success(client, mock_control, mock_stats):
    mock_control("primary", fail_next=1, fail_next_status=503)
    r = _chat(client, stream=False)
    assert _served_by(r) == "mock-primary"
    assert mock_stats("primary")["by_status"] == {"503": 1, "200": 1}


def test_long_retry_after_fails_over_immediately(client, mock_control, mock_stats):
    mock_control("primary", rate_limit_rate=1.0, retry_after=30)
    r = _chat(client)
    assert _served_by(r) == "mock-backup"
    assert mock_stats("primary")["requests"] == 1  # no retry against a 30 s Retry-After


def test_400_is_not_retried_or_failed_over(gateway_server, api_key, mock_control, mock_stats):
    mock_control("primary", error_rate=1.0, error_status=400)
    r = httpx.post(
        f"{gateway_server.url}/v1/chat/completions",
        headers={"authorization": f"Bearer {api_key}"},
        json={"model": "mock", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 400
    assert mock_stats("primary")["requests"] == 1
    assert mock_stats("backup") == {}


def test_missing_key_provider_is_skipped(client):
    r = _chat(client, model="keyless-first")
    assert _served_by(r) == "mock-backup"


def test_unsupported_feature_fails_over_from_anthropic(client):
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
    r = _chat(client, model="anthropic-then-openai", tools=tools)
    assert _served_by(r) == "mock-backup"


def test_all_providers_down_returns_502(client, mock_control):
    mock_control("primary", mode="outage")
    mock_control("backup", mode="outage")
    with pytest.raises(openai.APIStatusError) as ei:
        _chat(client)
    assert ei.value.status_code == 502


def test_circuit_opens_and_skips_primary(client, gateway_server, mock_control, mock_stats):
    mock_control("primary", mode="outage")
    mock_control("backup", ttft_ms=0, tokens_per_s=1_000_000)  # stay well inside the 0.5 s cooldown
    for _ in range(3):  # threshold is 3 consecutive failures (retries count)
        _chat(client)
    before = mock_stats("primary")["requests"]
    for _ in range(5):
        r = _chat(client)
        assert _served_by(r) == "mock-backup"
    assert mock_stats("primary")["requests"] == before  # open circuit: primary not called at all
    circuits = httpx.get(f"{gateway_server.url}/admin/circuits", headers={"authorization": "Bearer test-admin"})
    if circuits.status_code == 200:  # admin API lands in phase 3
        assert circuits.json()["mock-primary"]["state"] == "open"


def test_circuit_half_open_probe_recovers(client, mock_control, mock_stats):
    mock_control("primary", mode="outage")
    for _ in range(3):
        _chat(client)
    mock_control("primary", mode="ok")
    time.sleep(0.6)  # cooldown is 0.5 s in the test config
    r = _chat(client)
    assert _served_by(r) == "mock-primary"  # the probe succeeded and closed the circuit
    for _ in range(3):
        assert _served_by(_chat(client)) == "mock-primary"


def test_half_open_probe_failure_reopens(client, mock_control, mock_stats):
    mock_control("primary", mode="outage")
    mock_control("backup", ttft_ms=0, tokens_per_s=1_000_000)
    for _ in range(3):
        _chat(client)
    time.sleep(0.6)
    before = mock_stats("primary")["requests"]
    _chat(client)  # probe fails, circuit re-opens
    _chat(client)
    _chat(client)
    assert mock_stats("primary")["requests"] == before + 1  # exactly one probe got through


def test_midstream_disconnect_ends_with_error_event(client, mock_control):
    mock_control("primary", mode="midstream_disconnect", disconnect_after_tokens=3)
    text = ""
    with pytest.raises(openai.APIError) as ei:
        for c in client.chat.completions.create(model="mock", messages=[{"role": "user", "content": "x"}], stream=True):
            if c.choices:
                text += c.choices[0].delta.content or ""
    assert text  # the client already saw partial output, so no silent failover
    assert "after output started" in str(ei.value)


def test_concurrent_outage_zero_failures(gateway_server, api_key, mock_control):
    """The Phase 2 bar: primary fully down under concurrent streaming load, zero failed requests."""
    mock_control("primary", ttft_ms=20, tokens_per_s=500)
    mock_control("backup", ttft_ms=20, tokens_per_s=500)
    c = openai.OpenAI(base_url=f"{gateway_server.url}/v1", api_key=api_key, max_retries=0, timeout=20,
                      default_headers={"x-gateway-cache": "no-store"})

    def one(i: int) -> bool:
        if i == 40:
            mock_control("primary", mode="outage")  # kill the primary mid-run
        text = _chat(c, stream=True, prompt=f"q{i}")
        return bool(text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(one, range(200)))
    assert all(results), f"{results.count(False)} failed requests"
