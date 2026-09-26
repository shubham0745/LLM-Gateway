"""Interactive load test with Locust (the scripted numbers come from bench/overhead.py).

    pip install locust
    locust -f bench/locustfile.py --host http://localhost:8080
    # open http://localhost:8089, pick users and spawn rate

Each simulated user streams a chat completion through the ``mock`` alias
(cache bypassed so every request goes upstream), with a short pause between
requests. Time to first token is reported as its own entry ("TTFT").
Set GATEWAY_BENCH_KEY, or run scripts/bootstrap.py first.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from locust import HttpUser, between, task

KEY = os.environ.get("GATEWAY_BENCH_KEY") or json.loads(
    (Path(__file__).resolve().parent.parent / ".gateway-keys.json").read_text()
)["bench"]


class ChatUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def stream_chat(self) -> None:
        body = {"model": "mock", "stream": True, "messages": [{"role": "user", "content": "Explain an LLM gateway briefly."}]}
        headers = {"authorization": f"Bearer {KEY}", "x-gateway-cache": "no-store"}
        t0 = time.perf_counter()
        with self.client.post("/v1/chat/completions", json=body, headers=headers, stream=True,
                              catch_response=True, name="chat (stream)") as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            first = None
            done = False
            for line in r.iter_lines():
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                if b'"error"' in data[:10]:
                    r.failure("stream error event")
                    return
                if first is None and b'"content"' in data:
                    first = time.perf_counter() - t0
                    self.environment.events.request.fire(
                        request_type="SSE", name="TTFT", response_time=first * 1000, response_length=0,
                        exception=None, context={},
                    )
            if not done:
                r.failure("stream ended without [DONE]")
