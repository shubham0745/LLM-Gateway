# LLM Gateway

A self-hosted gateway with an OpenAI-compatible API. Point any OpenAI SDK at it by changing `base_url`.

Work in progress; see `docs/` as phases land.

```bash
docker compose up -d --build
docker compose exec gateway python -m gateway.cli create-tenant demo "Demo"
docker compose exec gateway python -m gateway.cli create-key demo
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="gw-...")
client.chat.completions.create(model="mock", messages=[{"role": "user", "content": "hi"}])
```
