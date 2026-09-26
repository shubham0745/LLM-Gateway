"""Generates the provisioned Grafana dashboards.

    python deploy/grafana/build_dashboards.py

Writing dashboards as code keeps them reviewable; the generated JSON is
committed so ``docker compose up`` needs no build step.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards"
PROM = {"type": "prometheus", "uid": "prometheus"}
PG = {"type": "grafana-postgresql-datasource", "uid": "postgres"}

_next_id = 0


def _id() -> int:
    global _next_id
    _next_id += 1
    return _next_id


def prom(expr: str, legend: str = "", instant: bool = False) -> dict:
    return {"datasource": PROM, "expr": expr, "legendFormat": legend, "instant": instant, "range": not instant}


def sql(query: str, fmt: str = "table") -> dict:
    return {"datasource": PG, "rawSql": query, "format": fmt, "rawQuery": True, "editorMode": "code", "refId": "A"}


def panel(title: str, targets: list[dict], x: int, y: int, w: int = 12, h: int = 8, kind: str = "timeseries",
          unit: str = "short", ds: dict = PROM, **extra) -> dict:
    for i, t in enumerate(targets):
        t["refId"] = chr(ord("A") + i)
    p = {
        "id": _id(), "title": title, "type": kind, "datasource": ds, "targets": targets,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
    }
    p.update(extra)
    return p


def dashboard(uid: str, title: str, panels: list[dict], time_from: str = "now-15m", refresh: str = "5s") -> dict:
    return {
        "uid": uid, "title": title, "tags": ["llm-gateway"], "timezone": "utc", "schemaVersion": 39, "version": 1,
        "editable": True, "refresh": refresh, "time": {"from": time_from, "to": "now"}, "panels": panels,
    }


def live() -> dict:
    rate = "[1m]"
    p = [
        panel("Requests / s by provider", [prom(
            f'label_replace(sum by (provider) (rate(gateway_requests_total{rate})), "provider", "none (failed)", "provider", "^$")',
            "{{provider}}")], 0, 0, unit="reqps"),
        panel("Error rate (all requests)", [prom(
            f'(sum(rate(gateway_requests_total{{status="error"}}{rate})) or vector(0)) / clamp_min(sum(rate(gateway_requests_total{rate})), 1e-9)', "error rate")],
            12, 0, unit="percentunit"),
        panel("End-to-end latency p50 / p95 / p99", [
            prom(f'histogram_quantile({q}, sum by (le) (rate(gateway_request_duration_seconds_bucket{{cache!~"hit.*"}}{rate})))', f"p{int(q * 100)}")
            for q in (0.5, 0.95, 0.99)], 0, 8, unit="s"),
        panel("Time to first token p50 / p95 / p99 (streams)", [
            prom(f'histogram_quantile({q}, sum by (le) (rate(gateway_ttft_seconds_bucket{rate})))', f"p{int(q * 100)}")
            for q in (0.5, 0.95, 0.99)], 12, 8, unit="s"),
        panel("Circuit state per provider (0 closed, 1 half-open, 2 open)", [prom("gateway_circuit_state", "{{provider}}")], 0, 16,
              kind="state-timeline", fieldConfig={"defaults": {"unit": "short", "mappings": [{"type": "value", "options": {
                  "0": {"text": "closed", "color": "green", "index": 0}, "1": {"text": "half-open", "color": "yellow", "index": 1},
                  "2": {"text": "open", "color": "red", "index": 2}}}],
                  # Colours come from the value mappings; a thresholds colour mode would make the
                  # state timeline label bars with threshold ranges ("2+") instead of the mapped text.
                  "color": {"mode": "fixed", "fixedColor": "green"}}, "overrides": []},
              options={"mergeValues": True, "showValue": "auto", "rowHeight": 0.8, "legend": {"showLegend": False}}),
        panel("Failovers / s", [prom(f'sum by (from_provider, to_provider) (rate(gateway_failovers_total{rate}))', "{{from_provider}} → {{to_provider}}")],
              12, 16, unit="ops"),
        panel("Upstream attempts by outcome", [prom(f'sum by (provider, outcome, error_kind) (rate(gateway_provider_attempts_total{rate}))',
                                                     "{{provider}} {{outcome}} {{error_kind}}")], 0, 24, unit="ops"),
        panel("Upstream error rate per provider", [prom(
            f'sum by (provider) (rate(gateway_provider_attempts_total{{outcome="error"}}{rate})) / clamp_min(sum by (provider) (rate(gateway_provider_attempts_total{rate})), 1e-9)',
            "{{provider}}")], 12, 24, unit="percentunit"),
        panel("Cache hit rate", [prom(
            f'sum(rate(gateway_requests_total{{cache=~"hit.*"}}{rate})) / clamp_min(sum(rate(gateway_requests_total{{status="ok"}}{rate})), 1e-9)', "hit rate"),
            prom(f'sum by (cache) (rate(gateway_requests_total{{cache=~"hit.*"}}{rate})) / clamp_min(sum(rate(gateway_requests_total{{status="ok"}}{rate})), 1e-9)', "{{cache}}")],
            0, 32, unit="percentunit"),
        panel("Money saved by the cache (USD, cumulative)", [prom("sum by (cache) (gateway_cache_saved_usd_total)", "{{cache}}")], 12, 32, unit="currencyUSD"),
        panel("Spend rate per tenant (USD / hour)", [prom(f'sum by (tenant) (rate(gateway_cost_usd_total{rate})) * 3600', "{{tenant}}")], 0, 40, unit="currencyUSD"),
        panel("Budget used this month", [prom("gateway_budget_spent_usd / gateway_budget_limit_usd", "{{tenant}}")], 12, 40, kind="bargauge",
              unit="percentunit", options={"orientation": "horizontal", "displayMode": "gradient"}),
        panel("In-flight requests", [prom("sum(gateway_inflight_requests)", "in flight")], 0, 48, w=8),
        panel("Rejected by limits / s", [prom(f'sum by (reason) (rate(gateway_rejected_total{rate}))', "{{reason}}")], 8, 48, w=8, unit="ops"),
        panel("Cache lookup latency p95 (added to every miss)", [
            prom(f'histogram_quantile(0.95, sum by (le, cache) (rate(gateway_cache_lookup_duration_seconds_bucket{rate})))', "{{cache}}")],
            16, 48, w=8, unit="s"),
    ]
    return dashboard("gateway-live", "LLM Gateway — Live", p)


BARS = {"defaults": {"unit": "currencyUSD", "custom": {"drawStyle": "bars", "fillOpacity": 80, "stacking": {"mode": "normal"}}},
        "overrides": []}


def spend() -> dict:
    yesterday = "ts >= date_trunc('day', now() AT TIME ZONE 'utc') - interval '1 day' AND ts < date_trunc('day', now() AT TIME ZONE 'utc')"
    p = [
        panel("Yesterday (UTC): who spent how much, on which model", [sql(f"""
SELECT tenant_id AS tenant, coalesce(model, '(failed)') AS model, count(*) AS requests,
       sum(prompt_tokens) AS prompt_tokens, sum(completion_tokens) AS completion_tokens,
       round(sum(cost_usd)::numeric, 6) AS cost_usd, round(sum(saved_usd)::numeric, 6) AS saved_by_cache_usd
FROM request_logs WHERE {yesterday}
GROUP BY 1, 2 ORDER BY cost_usd DESC""")], 0, 0, w=24, h=9, kind="table", ds=PG),
        panel("Spend per tenant (selected range)", [sql("""
SELECT $__timeGroupAlias(ts, $__interval), tenant_id AS metric, sum(cost_usd)::float AS value
FROM request_logs WHERE $__timeFilter(ts) GROUP BY 1, 2 ORDER BY 1""", "time_series")], 0, 9, ds=PG, unit="currencyUSD",
              fieldConfig=BARS),
        panel("Spend per model (selected range)", [sql("""
SELECT $__timeGroupAlias(ts, $__interval), coalesce(model, '(failed)') AS metric, sum(cost_usd)::float AS value
FROM request_logs WHERE $__timeFilter(ts) GROUP BY 1, 2 ORDER BY 1""", "time_series")], 12, 9, ds=PG, unit="currencyUSD",
              fieldConfig=BARS),
        panel("Spend by tenant and model (selected range)", [sql("""
SELECT tenant_id AS tenant, coalesce(model, '(failed)') AS model, count(*) AS requests, round(sum(cost_usd)::numeric, 6) AS cost_usd,
       round(sum(saved_usd)::numeric, 6) AS saved_usd,
       round(100.0 * count(*) FILTER (WHERE cache_status LIKE 'hit%') / count(*), 1) AS cache_hit_pct
FROM request_logs WHERE $__timeFilter(ts) GROUP BY 1, 2 ORDER BY cost_usd DESC""")], 0, 17, w=24, kind="table", ds=PG),
        panel("Most expensive requests (selected range)", [sql("""
SELECT ts, request_id, tenant_id, alias, provider, model, prompt_tokens, completion_tokens, cost_usd::float, latency_ms
FROM request_logs WHERE $__timeFilter(ts) ORDER BY cost_usd DESC LIMIT 20""")], 0, 25, w=24, kind="table", ds=PG),
        panel("Requests that needed a failover (selected range)", [sql("""
SELECT ts, request_id, alias, coalesce(provider, '(failed)') AS served_by, jsonb_array_length(attempts) AS attempts,
       attempts::text AS attempt_log
FROM request_logs WHERE $__timeFilter(ts) AND jsonb_array_length(attempts) > 1 ORDER BY ts DESC LIMIT 50""")],
              0, 33, w=24, kind="table", ds=PG),
    ]
    return dashboard("gateway-spend", "LLM Gateway — Spend", p, time_from="now-7d", refresh="1m")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for d in (live(), spend()):
        (OUT / f"{d['uid']}.json").write_text(json.dumps(d, indent=2) + "\n")
        print("wrote", OUT / f"{d['uid']}.json")
