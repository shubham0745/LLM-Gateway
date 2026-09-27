# LLM Gateway

An OpenAI-compatible gateway in front of LLM providers. Clients call it exactly
like the OpenAI API and name a model alias (`fast`, `smart`); the gateway
routes the call to whichever provider and model the alias points at.

v1 supports OpenAI. Anthropic comes next, through the same adapter interface.

## Endpoints

| Method | Path                   | Notes                                |
|--------|------------------------|--------------------------------------|
| POST   | `/v1/chat/completions` | Streaming (`"stream": true`) and not |
| GET    | `/v1/models`           | Lists the configured aliases         |
| GET    | `/healthz`             | Liveness                             |

Every response carries an `x-request-id` header. Send your own to have it
kept and forwarded to the provider.

## Configuration

`config/gateway.yaml` maps aliases to providers. `${VAR}` is read from the
environment, so keys stay out of the file.

```yaml
providers:
  openai:
    type: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
models:
  fast:  {provider: openai, model: gpt-4o-mini}
  smart: {provider: openai, model: gpt-4o}
```

Set `GATEWAY_CONFIG` to use a different file.

## Running

```sh
uv sync
OPENAI_API_KEY=sk-... uv run uvicorn gateway.main:create_app --factory --port 8000
```

Or with no key at all, against the mock provider:

```sh
docker compose up --build
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "fast", "messages": [{"role": "user", "content": "hi"}]}'
```

## Mock provider

`mock_provider/app.py` imitates an OpenAI-style API with deterministic
replies. `MOCK_LATENCY_MS` and `MOCK_TOKENS_PER_SECOND` (or the
`x-mock-latency-ms` and `x-mock-tokens-per-second` headers) set latency and
streaming speed; `x-mock-status` makes it fail with a given status. Tests and
CI only ever talk to it.

## Development

```sh
uv run ruff check . && uv run ruff format --check .
uv run pytest -rxX
```

### Core pieces are written by hand

Some functions are deliberately left as stubs marked `TODO(core)`, with their
tests already written. Those tests carry the `core_todo` marker and show as
`xfail` while the stub raises `NotImplementedError`; any other failure still
fails the run. Implement the function until its tests show `XPASS`, then
delete the marker.

Currently stubbed:

- `parse_sse` in `gateway/streaming/sse.py`: bytes to SSE events
  (`tests/test_sse.py`)
- `OpenAIAdapter.translate_stream` in `gateway/providers/openai.py`: SSE events
  to chunks (`tests/test_openai_stream.py`)

Streaming end to end (`tests/test_api.py`) passes once both are done.
