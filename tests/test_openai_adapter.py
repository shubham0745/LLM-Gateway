"""Request translation, response translation and usage extraction (no HTTP)."""

from gateway.config import ProviderConfig
from gateway.providers.openai import OpenAIAdapter
from gateway.schemas import ChatCompletionRequest


def adapter(api_key: str | None = "sk-test") -> OpenAIAdapter:
    return OpenAIAdapter(
        "openai", ProviderConfig(type="openai", base_url="https://api.example/v1/", api_key=api_key)
    )


def test_build_request_swaps_model_and_adds_auth():
    req = ChatCompletionRequest(
        model="fast", messages=[{"role": "user", "content": "hi"}], temperature=0.2
    )
    up = adapter().build_request(req, "gpt-4o-mini")
    assert up.method == "POST"
    assert up.url == "https://api.example/v1/chat/completions"
    assert up.headers["authorization"] == "Bearer sk-test"
    assert up.json == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "temperature": 0.2,
    }


def test_build_request_passes_unknown_openai_params_through():
    req = ChatCompletionRequest.model_validate(
        {
            "model": "fast",
            "messages": [{"role": "user", "content": "hi", "name": "sam"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "seed": 7,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )
    up = adapter().build_request(req, "gpt-4o")
    assert up.json["tools"] == [{"type": "function", "function": {"name": "f"}}]
    assert up.json["seed"] == 7
    assert up.json["messages"][0]["name"] == "sam"
    assert up.json["stream_options"] == {"include_usage": True}
    assert up.headers["accept"] == "text/event-stream"


def test_build_request_without_key_sends_no_auth_header():
    req = ChatCompletionRequest(model="fast", messages=[{"role": "user", "content": "hi"}])
    assert "authorization" not in adapter(api_key=None).build_request(req, "m").headers


def test_translate_response_keeps_extra_fields_and_usage():
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini-2024",
        "system_fingerprint": "fp_1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }
    resp = adapter().translate_response(payload)
    dumped = resp.model_dump()
    assert dumped["system_fingerprint"] == "fp_1"
    assert dumped["choices"][0]["message"]["tool_calls"] == [{"id": "c1"}]
    assert resp.usage is not None and resp.usage.total_tokens == 7


def test_extract_usage():
    a = adapter()
    assert a.extract_usage({"usage": None}) is None
    assert a.extract_usage({}) is None
    usage = a.extract_usage(
        {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}
    )
    assert usage is not None and (usage.prompt_tokens, usage.completion_tokens) == (1, 2)


def test_error_message():
    a = adapter()
    assert a.error_message({"error": {"message": "bad key"}}) == "bad key"
    assert a.error_message({"error": "boom"}) == "boom"
    assert a.error_message("nope") is None
