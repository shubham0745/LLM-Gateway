"""The mock provider itself: tests rely on it behaving like an OpenAI-style API."""

import json
import time

import httpx
import pytest

from mock_provider.app import app

AUTH = {"authorization": "Bearer k"}
BODY = {"model": "mock-small", "messages": [{"role": "user", "content": "one two"}]}


@pytest.fixture
async def mock() -> httpx.AsyncClient:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://mock"
    ) as c:
        yield c


async def test_models(mock):
    ids = [m["id"] for m in (await mock.get("/v1/models")).json()["data"]]
    assert ids == ["mock-small", "mock-large"]


async def test_non_streaming(mock):
    body = (await mock.post("/v1/chat/completions", json=BODY, headers=AUTH)).json()
    assert body["choices"][0]["message"]["content"] == "Mock reply to: one two"
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7}


async def test_streaming_wire_format(mock):
    r = await mock.post(
        "/v1/chat/completions",
        json={**BODY, "stream": True, "stream_options": {"include_usage": True}},
        headers=AUTH,
    )
    blocks = [b for b in r.text.split("\n\n") if b]
    assert all(b.startswith("data: ") for b in blocks)
    assert blocks[-1] == "data: [DONE]"
    chunks = [json.loads(b[6:]) for b in blocks[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]) == (
        "Mock reply to: one two"
    )
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["total_tokens"] == 7


async def test_latency_and_streaming_speed_are_configurable(mock):
    headers = {**AUTH, "x-mock-latency-ms": "100", "x-mock-tokens-per-second": "50"}
    started = time.perf_counter()
    await mock.post("/v1/chat/completions", json={**BODY, "stream": True}, headers=headers)
    # 100 ms latency + 5 tokens at 20 ms each.
    assert time.perf_counter() - started >= 0.2


async def test_failure_injection_and_auth(mock):
    assert (await mock.post("/v1/chat/completions", json=BODY)).status_code == 401
    r = await mock.post("/v1/chat/completions", json=BODY, headers={**AUTH, "x-mock-status": "503"})
    assert r.status_code == 503 and r.json()["error"]["message"]
