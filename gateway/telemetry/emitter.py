"""Where finished request records go.

Every record is (1) counted in Prometheus, (2) logged as one JSON line and
(3) queued for Postgres. The Postgres write happens in the worker process:
here we only append to a Redis Stream, batched by a background task, so a
slow database never adds latency to a request. If the in-memory queue fills
(Redis down), records are dropped and counted rather than blocking traffic;
the JSON log line still has them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from redis.asyncio import Redis

from gateway.telemetry.logging import log_fields
from gateway.telemetry.metrics import Metrics
from gateway.telemetry.record import RequestRecord

logger = logging.getLogger("gateway.request")


class Telemetry:
    def __init__(
        self,
        metrics: Metrics | None = None,
        redis: Redis | None = None,
        stream: str = "gw:events",
        maxlen: int = 1_000_000,
        log_requests: bool = True,
        queue_size: int = 100_000,
    ):
        self.metrics = metrics or Metrics()
        self.redis = redis
        self.stream = stream
        self.maxlen = maxlen
        self.log_requests = log_requests
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self.redis is not None and self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._flush_loop())

    async def stop(self) -> None:
        if self._task:
            # Drain what we have before shutting down.
            for _ in range(50):
                if self._queue.empty():
                    break
                await asyncio.sleep(0.05)
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def emit(self, record: RequestRecord) -> None:
        m = self.metrics
        stream = "true" if record.stream else "false"
        m.requests.labels(record.alias or "", record.provider or "", record.status, record.cache_status, stream).inc()
        if record.latency_ms is not None:
            m.request_duration.labels(record.alias or "", stream, record.cache_status).observe(record.latency_ms / 1000)
        if record.ttft_ms is not None and record.stream:
            m.ttft.labels(record.alias or "", record.provider or "").observe(record.ttft_ms / 1000)
        tenant = record.tenant_id or ""
        if record.prompt_tokens or record.completion_tokens:
            model = record.model or ""
            m.tokens.labels(tenant, model, "prompt").inc(record.prompt_tokens)
            m.tokens.labels(tenant, model, "completion").inc(record.completion_tokens)
        if record.cost_usd:
            m.cost.labels(tenant, record.model or "").inc(record.cost_usd)
        if record.saved_usd:
            m.saved.labels(tenant, record.cache_status).inc(record.saved_usd)

        d = record.to_dict()
        if self.log_requests:
            log_fields(logger, "request", **d)
        if self.redis is not None:
            try:
                self._queue.put_nowait(d)
            except asyncio.QueueFull:
                m.events_dropped.inc()

    def enqueue_event(self, event: dict) -> None:
        """Non-request events for the worker (e.g. semantic cache inserts)."""
        if self.redis is None:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.metrics.events_dropped.inc()

    async def _flush_loop(self) -> None:
        assert self.redis is not None
        while True:
            item = await self._queue.get()
            batch = [item]
            while len(batch) < 500 and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            try:
                async with self.redis.pipeline(transaction=False) as pipe:
                    for ev in batch:
                        kind = ev.pop("_kind", "request")
                        pipe.xadd(self.stream, {"kind": kind, "data": json.dumps(ev, default=str)}, maxlen=self.maxlen, approximate=True)
                    await pipe.execute()
            except Exception:  # noqa: BLE001 - Redis hiccup: log and keep going
                logger.warning("failed to publish %d events to %s", len(batch), self.stream, exc_info=True)
                self.metrics.events_dropped.inc(len(batch))
                await asyncio.sleep(0.5)
