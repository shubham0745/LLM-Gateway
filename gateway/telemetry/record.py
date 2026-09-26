"""The one structured record every request produces."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def new_request_id() -> str:
    return "req_" + uuid.uuid4().hex


@dataclass
class Attempt:
    provider: str
    model: str
    outcome: str  # ok | error | skipped_circuit_open | skipped_disabled
    latency_ms: float = 0.0
    error_kind: str | None = None
    error: str | None = None
    status: int | None = None
    ttft_ms: float | None = None


@dataclass
class RequestRecord:
    request_id: str = field(default_factory=new_request_id)
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    tenant_id: str | None = None
    key_id: int | None = None
    alias: str | None = None
    provider: str | None = None
    model: str | None = None
    stream: bool = False
    status: str = "ok"
    http_status: int = 200
    error: str | None = None
    cache_status: str = "miss"
    attempts: list[Attempt] = field(default_factory=list)
    ttft_ms: float | None = None
    latency_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_estimated: bool = False
    cost_usd: float = 0.0
    saved_usd: float = 0.0
    _start: float = field(default_factory=time.perf_counter, repr=False)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def mark_first_token(self) -> None:
        if self.ttft_ms is None:
            self.ttft_ms = round(self.elapsed_ms(), 2)

    def finish(self) -> None:
        self.latency_ms = round(self.elapsed_ms(), 2)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_start", None)
        return d
