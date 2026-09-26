"""Prometheus metrics.

Labels are kept to bounded sets (aliases, providers, models and tenants come
from config and the tenants table, never from free-form request fields) so
series cardinality stays small.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from gateway.routing.breaker import STATE_VALUE
from gateway.routing.router import RouterHooks

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120)
FAST_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)


class Metrics(RouterHooks):
    def __init__(self, registry: CollectorRegistry | None = None):
        r = self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "gateway_requests_total", "Finished requests", ["alias", "provider", "status", "cache", "stream"], registry=r
        )
        self.request_duration = Histogram(
            "gateway_request_duration_seconds", "End-to-end request latency", ["alias", "stream", "cache"],
            buckets=LATENCY_BUCKETS, registry=r,
        )
        self.ttft = Histogram(
            "gateway_ttft_seconds", "Time to first token as seen by the client", ["alias", "provider"],
            buckets=LATENCY_BUCKETS, registry=r,
        )
        self.attempts = Counter(
            "gateway_provider_attempts_total", "Upstream attempts", ["provider", "outcome", "error_kind"], registry=r
        )
        self.attempt_duration = Histogram(
            "gateway_provider_attempt_duration_seconds", "Upstream attempt latency (to first token for streams)",
            ["provider", "outcome"], buckets=LATENCY_BUCKETS, registry=r,
        )
        self.failovers = Counter(
            "gateway_failovers_total", "Moves to the next target in a chain", ["alias", "from_provider", "to_provider"], registry=r
        )
        self.circuit_state = Gauge(
            "gateway_circuit_state", "Circuit breaker state (0 closed, 1 half-open, 2 open)", ["provider"], registry=r
        )
        self.tokens = Counter("gateway_tokens_total", "Tokens billed", ["tenant", "model", "kind"], registry=r)
        self.cost = Counter("gateway_cost_usd_total", "Spend in USD", ["tenant", "model"], registry=r)
        self.saved = Counter("gateway_cache_saved_usd_total", "Spend avoided by cache hits", ["tenant", "cache"], registry=r)
        self.cache_lookups = Counter("gateway_cache_lookups_total", "Cache lookups", ["cache", "result"], registry=r)
        self.cache_lookup_duration = Histogram(
            "gateway_cache_lookup_duration_seconds", "Cache lookup latency", ["cache"], buckets=FAST_BUCKETS, registry=r
        )
        self.rejected = Counter("gateway_rejected_total", "Requests rejected by limits", ["tenant", "reason"], registry=r)
        self.inflight = Gauge("gateway_inflight_requests", "Requests in progress (streams until their last byte)", registry=r)
        self.budget_spent = Gauge("gateway_budget_spent_usd", "Spend this month", ["tenant"], registry=r)
        self.budget_limit = Gauge("gateway_budget_limit_usd", "Monthly budget", ["tenant"], registry=r)
        self.events_dropped = Counter("gateway_events_dropped_total", "Log events dropped (queue full)", registry=r)
        self.config_version = Gauge("gateway_config_version", "Active config version", registry=r)

    # -- RouterHooks --------------------------------------------------------

    def attempt(self, provider: str, outcome: str, error_kind: str | None, latency_s: float) -> None:
        self.attempts.labels(provider, outcome, error_kind or "").inc()
        self.attempt_duration.labels(provider, outcome).observe(latency_s)

    def failover(self, alias: str, from_provider: str, to_provider: str) -> None:
        self.failovers.labels(alias, from_provider, to_provider).inc()

    def circuit(self, provider: str, state: str) -> None:
        self.circuit_state.labels(provider).set(STATE_VALUE.get(state, 0))
