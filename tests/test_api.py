"""End-to-end through the gateway to the in-process mock provider. No real API calls."""

import json

import httpx
import pytest

from mock_provider.app import app as mock_app

HELLO = {"model": "fast", "messages": [{"role": "user", "content": "hello there"}]}


async def test_list_models_returns_aliases(client: httpx.AsyncClient):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert {m["id"]: m["owned_by"] for m in body["data"]} == {
        "fast": "mock",
        "smart": "mock",
        "broken": "mock",
    }


async def test_chat_completion(client: httpx.AsyncClient):
    r = await client.post("/v1/chat/completions", json=HELLO)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "mock-small"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Mock reply to: hello there",
    }
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7}


async def test_alias_routes_to_its_upstream_model_with_provider_key(client: httpx.AsyncClient):
    await client.post("/v1/chat/completions", json={**HELLO, "model": "smart", "seed": 3})
    sent = mock_app.state.last_request
    assert sent["json"]["model"] == "mock-large"
    assert sent["json"]["seed"] == 3
    assert sent["headers"]["authorization"] == "Bearer test-key"


async def test_every_response_has_a_request_id(client: httpx.AsyncClient):
    ok = await client.post("/v1/chat/completions", json=HELLO)
    missing = await client.get("/no-such-route")
    for r in (ok, missing):
        assert r.headers["x-request-id"].startswith("req_")
    assert ok.headers["x-request-id"] != missing.headers["x-request-id"]


async def test_request_id_is_kept_and_forwarded_upstream(client: httpx.AsyncClient):
    r = await client.post("/v1/chat/completions", json=HELLO, headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    assert mock_app.state.last_request["headers"]["x-request-id"] == "abc-123"


async def test_bad_request_id_is_replaced(client: httpx.AsyncClient):
    r = await client.get("/v1/models", headers={"x-request-id": "no spaces allowed"})
    assert r.headers["x-request-id"].startswith("req_")


async def test_unknown_model_is_404_in_openai_error_format(client: httpx.AsyncClient):
    r = await client.post("/v1/chat/completions", json={**HELLO, "model": "gpt-9"})
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "model_not_found"
    assert err["request_id"] == r.headers["x-request-id"]


async def test_invalid_body_is_400(client: httpx.AsyncClient):
    r = await client.post("/v1/chat/completions", json={"model": "fast", "messages": []})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert "messages" in r.json()["error"]["message"]


async def test_upstream_4xx_is_passed_on(client: httpx.AsyncClient):
    r = await client.post("/v1/chat/completions", json={**HELLO, "model": "broken"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "upstream_rejected"
    assert "no-such-model" in r.json()["error"]["message"]


@pytest.mark.parametrize(
    ("mock_status", "gateway_status", "code"),
    [
        (500, 502, "upstream_error"),
        (429, 429, "upstream_rate_limited"),
        (401, 502, "upstream_auth_failed"),
    ],
)
async def test_upstream_errors_are_mapped(mock_status, gateway_status, code, config, monkeypatch):
    from gateway.main import create_app

    # Make the mock fail by injecting its failure header on every upstream call.
    async def inject(request: httpx.Request) -> None:
        request.headers["x-mock-status"] = str(mock_status)

    app = create_app(config, transport=httpx.ASGITransport(app=mock_app))
    async with app.router.lifespan_context(app):
        app.state.http_client.event_hooks["request"].append(inject)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://g"
        ) as c:
            for stream in (False, True):
                r = await c.post("/v1/chat/completions", json={**HELLO, "stream": stream})
                assert r.status_code == gateway_status
                assert r.json()["error"]["code"] == code


async def test_unreachable_provider_is_502(config):
    from gateway.main import create_app

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app = create_app(config, transport=httpx.MockTransport(refuse))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://g"
        ) as c:
            r = await c.post("/v1/chat/completions", json=HELLO)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "upstream_unreachable"


async def test_provider_timeout_is_504(config):
    from gateway.main import create_app

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    app = create_app(config, transport=httpx.MockTransport(slow))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://g"
        ) as c:
            r = await c.post("/v1/chat/completions", json=HELLO)
    assert r.status_code == 504


def sse_data(text: str) -> list[str]:
    return [line[len("data: ") :] for line in text.split("\n") if line.startswith("data: ")]


@pytest.mark.core_todo
async def test_streaming_chat_completion(client: httpx.AsyncClient):
    r = await client.post(
        "/v1/chat/completions",
        json={**HELLO, "stream": True, "stream_options": {"include_usage": True}},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-request-id"].startswith("req_")
    data = sse_data(r.text)
    assert data[-1] == "[DONE]"
    chunks = [json.loads(d) for d in data[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    assert text == "Mock reply to: hello there"
    assert chunks[-1]["usage"]["total_tokens"] == 7


@pytest.mark.core_todo
async def test_mid_stream_error_is_sent_in_band(config):
    from gateway.main import create_app

    def broken_stream(request: httpx.Request) -> httpx.Response:
        body = (
            b'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m",'
            b'"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
            b"data: {oops\n\n"
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    app = create_app(config, transport=httpx.MockTransport(broken_stream))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://g"
        ) as c:
            r = await c.post("/v1/chat/completions", json={**HELLO, "stream": True})
    assert r.status_code == 200
    data = sse_data(r.text)
    assert json.loads(data[0])["choices"][0]["delta"]["content"] == "hi"
    assert json.loads(data[-1])["error"]["request_id"] == r.headers["x-request-id"]
    assert "[DONE]" not in data


async def test_healthz(client: httpx.AsyncClient):
    assert (await client.get("/healthz")).json() == {"status": "ok"}
