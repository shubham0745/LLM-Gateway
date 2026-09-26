"""Shared helpers for the benchmark scripts."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "results"
IMG = ROOT / "docs" / "img"
GATEWAY = os.environ.get("GATEWAY_URL", "http://localhost:8080")
MOCK = os.environ.get("MOCK_URL", "http://localhost:9000")
ADMIN_TOKEN = os.environ.get("GATEWAY_ADMIN_TOKEN", "change-me-admin-token")


def bench_key(tenant: str = "bench") -> str:
    env = os.environ.get("GATEWAY_BENCH_KEY")
    if env:
        return env
    path = ROOT / ".gateway-keys.json"
    if not path.exists():
        raise SystemExit("No key found: run `python scripts/bootstrap.py` first (or set GATEWAY_BENCH_KEY).")
    return json.loads(path.read_text())[tenant]


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def summary(values: list[float]) -> dict:
    return {"p50": pct(values, 50), "p95": pct(values, 95), "p99": pct(values, 99), "mean": float(np.mean(values)) if values else float("nan")}


def admin(method: str, path: str, **kw) -> httpx.Response:
    r = httpx.request(method, f"{GATEWAY}{path}", headers={"authorization": f"Bearer {ADMIN_TOKEN}"}, timeout=30, **kw)
    r.raise_for_status()
    return r


def mock_control(instance: str, **patch) -> None:
    httpx.put(f"{MOCK}/control/{instance}", json=patch, timeout=10).raise_for_status()


def mock_reset() -> None:
    httpx.post(f"{MOCK}/control/reset-all", timeout=10).raise_for_status()


@dataclass
class StreamResult:
    started: float
    ok: bool
    ttft: float | None = None
    total: float | None = None
    provider: str | None = None
    cache: str | None = None
    status: int | None = None
    error: str | None = None
    chunks: int = 0
    extra: dict = field(default_factory=dict)


async def stream_once(client: httpx.AsyncClient, url: str, body: dict, headers: dict | None = None) -> StreamResult:
    """One streaming chat completion; measures time to first content token and total time."""
    t0 = time.perf_counter()
    res = StreamResult(started=time.time(), ok=False)
    try:
        async with client.stream("POST", url, json={**body, "stream": True}, headers=headers) as r:
            res.status = r.status_code
            res.provider = r.headers.get("x-gateway-provider")
            res.cache = r.headers.get("x-gateway-cache")
            if r.status_code != 200:
                res.error = f"http {r.status_code}: {(await r.aread())[:200]!r}"
                res.total = time.perf_counter() - t0
                return res
            done = False
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                if res.ttft is None or data.startswith('{"error"'):
                    obj = json.loads(data)
                    if "error" in obj:
                        res.error = "stream error event"
                        break
                    if any((c.get("delta") or {}).get("content") for c in obj.get("choices") or []):
                        res.ttft = time.perf_counter() - t0
                res.chunks += 1
            res.total = time.perf_counter() - t0
            res.ok = done and res.error is None
            if not done and res.error is None:
                res.error = "stream ended without [DONE]"
    except (httpx.HTTPError, OSError) as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        res.total = time.perf_counter() - t0
    return res


def write_result(name: str, data: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    return path
