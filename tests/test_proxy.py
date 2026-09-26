"""Phase 1: the official OpenAI SDK works through the gateway, streaming and not."""

from __future__ import annotations

import time

import httpx
import openai
import pytest

MODELS = ["mock", "anthropic"]


@pytest.mark.parametrize("model", MODELS)
def test_non_streaming(client, model):
    raw = client.chat.completions.with_raw_response.create(
        model=model, messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    )
    resp = raw.parse()
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.completion_tokens == 40
    assert resp.usage.prompt_tokens > 0
    assert raw.headers["x-request-id"].startswith("req_")
    assert raw.headers["x-gateway-provider"] in ("mock-primary", "mock-anthropic")


@pytest.mark.parametrize("model", MODELS)
def test_streaming_with_usage(client, model):
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "hi"}], stream=True, stream_options={"include_usage": True}
    )
    text, usage, finish, ids = "", None, None, set()
    for chunk in stream:
        ids.add(chunk.id)
        if chunk.usage:
            usage = chunk.usage
        for c in chunk.choices:
            text += c.delta.content or ""
            finish = c.finish_reason or finish
    assert text
    assert finish == "stop"
    assert usage and usage.completion_tokens == 40
    assert len(ids) == 1  # one completion id across all chunks


def test_streaming_without_usage_chunk(client):
    chunks = list(client.chat.completions.create(model="mock", messages=[{"role": "user", "content": "x"}], stream=True))
    assert all(c.usage is None for c in chunks)


def test_stream_and_non_stream_return_same_text(client):
    msgs = [{"role": "user", "content": "determinism"}]
    full = client.chat.completions.create(model="mock", messages=msgs).choices[0].message.content
    streamed = "".join(c.choices[0].delta.content or "" for c in client.chat.completions.create(model="mock", messages=msgs, stream=True) if c.choices)
    assert full == streamed


def test_anthropic_max_tokens_maps_to_length(client):
    resp = client.chat.completions.create(model="anthropic", messages=[{"role": "user", "content": "x"}], max_tokens=5)
    assert resp.choices[0].finish_reason == "length"
    assert resp.usage.completion_tokens == 5


def test_direct_provider_model_syntax(client):
    raw = client.chat.completions.with_raw_response.create(model="mock-backup/mock-small", messages=[{"role": "user", "content": "x"}])
    assert raw.headers["x-gateway-provider"] == "mock-backup"


def test_models_endpoint(client):
    ids = {m.id for m in client.models.list()}
    assert {"mock", "anthropic"} <= ids


def test_unknown_model_is_404(client):
    with pytest.raises(openai.NotFoundError):
        client.chat.completions.create(model="does-not-exist", messages=[{"role": "user", "content": "x"}])


def test_bad_key_is_401(gateway_server):
    c = openai.OpenAI(base_url=f"{gateway_server.url}/v1", api_key="gw-not-a-real-key", max_retries=0)
    with pytest.raises(openai.AuthenticationError):
        c.chat.completions.create(model="mock", messages=[{"role": "user", "content": "x"}])


def test_missing_messages_is_400(gateway_server, api_key):
    r = httpx.post(f"{gateway_server.url}/v1/chat/completions", headers={"authorization": f"Bearer {api_key}"}, json={"model": "mock"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_messages"


def test_client_disconnect_cancels_upstream(gateway_server, api_key, mock_control, mock_stats):
    mock_control("primary", tokens_per_s=10, response_tokens=500)
    with httpx.Client(timeout=10) as h:
        with h.stream(
            "POST",
            f"{gateway_server.url}/v1/chat/completions",
            headers={"authorization": f"Bearer {api_key}"},
            json={"model": "mock", "stream": True, "messages": [{"role": "user", "content": "long"}]},
        ) as r:
            seen = 0
            for line in r.iter_lines():
                if line.startswith("data:"):
                    seen += 1
                if seen == 3:
                    break
    deadline = time.time() + 5
    while time.time() < deadline and mock_stats("primary").get("streams_cancelled", 0) < 1:
        time.sleep(0.05)
    st = mock_stats("primary")
    assert st["streams_cancelled"] == 1
    assert st["tokens_generated"] < 20  # stopped long before 500
