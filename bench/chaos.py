"""Chaos suite: inject one failure at a time into the primary provider while
traffic flows, and measure what clients saw.

    python -m bench.chaos                      # every scenario
    python -m bench.chaos --only outage hang

Each scenario runs steady streaming load against the ``mock`` alias
(mock-primary, then mock-backup) in three phases: healthy, fault injected,
fault removed. For each phase it records client-visible errors (split into
"before the first token", which failover is supposed to hide, and "after",
which it cannot), which provider served each request, and latency. A poller
reads ``/admin/circuits`` to time how long the breaker took to open after the
fault began (detection) and how long after the fault ended traffic was back
on the primary (recovery).

Writes docs/results/chaos.json, docs/img/chaos_timelines.png and
docs/img/failover_timeline.png (the outage scenario on its own).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import time
from dataclasses import dataclass

import httpx

from bench.common import ADMIN_TOKEN, GATEWAY, IMG, admin, bench_key, mock_control, mock_reset, stream_once, summary, write_result

PROMPT = [{"role": "user", "content": "Summarise why a gateway needs circuit breakers."}]


@dataclass
class Scenario:
    name: str
    title: str
    primary: dict
    backup: dict | None = None
    config_patch: dict | None = None  # applied through the admin API for the scenario, then reverted
    expect: str = ""


SCENARIOS = [
    Scenario("outage", "Primary returns 503 on every request", {"mode": "outage"},
             expect="no client errors; breaker opens; traffic moves to backup and returns after cooldown"),
    Scenario("hang", "Primary accepts connections but never sends a token", {"mode": "hang"},
             config_patch={"timeouts": {"ttft": 2.0}},
             expect="no client errors; requests wait up to the TTFT timeout until the breaker opens"),
    Scenario("rate_limit", "Primary answers 429 with Retry-After: 30s", {"rate_limit_rate": 1.0, "retry_after": 30},
             expect="no client errors; Retry-After longer than max_retry_after means fail over, not wait"),
    Scenario("flaky", "30% of primary requests fail with 503", {"error_rate": 0.3, "error_status": 503},
             expect="no client errors; retries and failover absorb the failures"),
    Scenario("midstream", "Primary drops the connection after 5 tokens", {"mode": "midstream_disconnect", "disconnect_after_tokens": 5},
             expect="clients that already got tokens see an error event (cannot be hidden); breaker opens, then clean"),
    Scenario("both_down", "Primary and backup both return 503", {"mode": "outage"}, backup={"mode": "outage"},
             expect="every request fails, quickly and with a clear error, once both breakers are open"),
]


@dataclass
class Sample:
    t: float  # seconds since scenario start (request start)
    ok: bool
    provider: str | None
    ttft: float | None
    total: float | None
    error: str | None
    got_tokens: bool


def _merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        out[k] = _merge(out.get(k) or {}, v) if isinstance(v, dict) else v
    return out


async def _load(url: str, headers: dict, concurrency: int, t0: float, end: float, out: list[Sample]) -> None:
    limits = httpx.Limits(max_connections=concurrency + 5, max_keepalive_connections=concurrency + 5)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(60, connect=10)) as client:
        async def worker() -> None:
            while time.perf_counter() < end:
                start = time.perf_counter() - t0
                r = await stream_once(client, url, {"model": "mock", "messages": PROMPT}, headers)
                out.append(Sample(start, r.ok, r.provider, r.ttft, r.total, r.error, r.ttft is not None))
                await asyncio.sleep(0.02)

        await asyncio.gather(*(worker() for _ in range(concurrency)))


async def _poll_circuits(t0: float, end: float, out: list[tuple[float, str, str]]) -> None:
    async with httpx.AsyncClient(timeout=5, headers={"authorization": f"Bearer {ADMIN_TOKEN}"}) as c:
        while time.perf_counter() < end:
            try:
                snap = (await c.get(f"{GATEWAY}/admin/circuits")).json()
                t = time.perf_counter() - t0
                out.append((t, snap.get("mock-primary", {}).get("state", "?"), snap.get("mock-backup", {}).get("state", "?")))
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(0.1)


def _phase_stats(samples: list[Sample]) -> dict:
    ok = [s for s in samples if s.ok]
    before = [s for s in samples if not s.ok and not s.got_tokens]
    after = [s for s in samples if not s.ok and s.got_tokens]
    served: dict[str, int] = {}
    for s in ok:
        served[s.provider or "?"] = served.get(s.provider or "?", 0) + 1
    return {
        "requests": len(samples),
        "errors_before_first_token": len(before),
        "errors_after_first_token": len(after),
        "error_samples": sorted({s.error for s in samples if s.error})[:4],
        "served_by": served,
        "ttft_ms": {k: v * 1000 for k, v in summary([s.ttft for s in ok if s.ttft is not None]).items()},
        "failed_request_ms": {k: v * 1000 for k, v in summary([s.total for s in samples if not s.ok and s.total]).items()},
    }


async def run_scenario(sc: Scenario, key: str, concurrency: int, healthy_s: float, fault_s: float, recover_s: float) -> dict:
    mock_reset()
    for inst in ("primary", "backup"):
        mock_control(inst, ttft_ms=50, tokens_per_s=200, response_tokens=20)
    for p in ("mock-primary", "mock-backup"):
        admin("POST", f"/admin/circuits/{p}/reset")
    original = admin("GET", "/admin/config").json()["config"]
    if sc.config_patch:
        admin("PUT", "/admin/config", json={"config": _merge(original, sc.config_patch)}, params={"comment": f"chaos: {sc.name}"})
        await asyncio.sleep(1.5)  # every gateway process picks up the new version

    headers = {"authorization": f"Bearer {key}", "x-gateway-cache": "no-store"}
    samples: list[Sample] = []
    circuit: list[tuple[float, str, str]] = []
    t0 = time.perf_counter()
    fault_at, heal_at, end = healthy_s, healthy_s + fault_s, healthy_s + fault_s + recover_s

    async def director() -> None:
        await asyncio.sleep(fault_at)
        mock_control("primary", **sc.primary)
        if sc.backup:
            mock_control("backup", **sc.backup)
        await asyncio.sleep(heal_at - fault_at)
        mock_control("primary", mode="ok", error_rate=0.0, rate_limit_rate=0.0)
        mock_control("backup", mode="ok", error_rate=0.0, rate_limit_rate=0.0)

    try:
        await asyncio.gather(
            _load(f"{GATEWAY}/v1/chat/completions", headers, concurrency, t0, t0 + end, samples),
            _poll_circuits(t0, t0 + end + 0.5, circuit),
            director(),
        )
    finally:
        if sc.config_patch:
            admin("PUT", "/admin/config", json={"config": original}, params={"comment": f"chaos: {sc.name} reverted"})

    opened = next((t for t, p, _ in circuit if t >= fault_at and p == "open"), None)
    back_on_primary = next((s.t for s in sorted(samples, key=lambda s: s.t) if s.t >= heal_at and s.ok and s.provider == "mock-primary"), None)
    closed_again = next((t for t, p, _ in circuit if t >= heal_at and p == "closed"), None)
    phases = {
        "healthy": _phase_stats([s for s in samples if s.t < fault_at]),
        "fault": _phase_stats([s for s in samples if fault_at <= s.t < heal_at]),
        "recovered": _phase_stats([s for s in samples if s.t >= heal_at]),
    }
    return {
        "scenario": sc.name, "title": sc.title, "expect": sc.expect, "injected": {"primary": sc.primary, "backup": sc.backup},
        "config_patch": sc.config_patch, "concurrency": concurrency,
        "timeline_s": {"fault_at": fault_at, "healed_at": heal_at, "end": end},
        "breaker_open_after_s": None if opened is None else round(opened - fault_at, 2),
        "back_on_primary_after_s": None if back_on_primary is None else round(back_on_primary - heal_at, 2),
        "breaker_closed_after_s": None if closed_again is None else round(closed_again - heal_at, 2),
        "phases": phases,
        "_samples": [(round(s.t, 3), s.ok, s.provider, s.got_tokens) for s in samples],
        "_circuit": circuit,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--healthy", type=float, default=10)
    ap.add_argument("--fault", type=float, default=30)
    ap.add_argument("--recover", type=float, default=30, help="must exceed the breaker cooldown (15 s by default)")
    args = ap.parse_args()
    key = bench_key()
    results = []
    for sc in SCENARIOS:
        if args.only and sc.name not in args.only:
            continue
        print(f"== {sc.name}: {sc.title}", flush=True)
        r = asyncio.run(run_scenario(sc, key, args.concurrency, args.healthy, args.fault, args.recover))
        f = r["phases"]["fault"]
        print(f"   fault phase: {f['requests']} requests, {f['errors_before_first_token']} errors before first token, "
              f"{f['errors_after_first_token']} after; served by {f['served_by']}; breaker open after {r['breaker_open_after_s']}s; "
              f"back on primary {r['back_on_primary_after_s']}s after the fault ended", flush=True)
        results.append(r)
    mock_reset()
    slim = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    print("wrote", write_result("chaos", {"scenarios": slim}))
    plot(results)


def plot(results: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(ax, r: dict, legend: bool) -> None:
        tl = r["timeline_s"]
        width = 1.0
        nbins = int(tl["end"] / width) + 1
        prim, back, err = [0] * nbins, [0] * nbins, [0] * nbins
        for t, ok, provider, _ in r["_samples"]:
            b = min(int(t / width), nbins - 1)
            if not ok:
                err[b] += 1
            elif provider == "mock-primary":
                prim[b] += 1
            else:
                back[b] += 1
        xs = [i * width for i in range(nbins)]
        ax.stackplot(xs, prim, back, err, labels=["served by primary", "served by backup", "failed"],
                     colors=["#2a6fdb", "#f2a33a", "#d62728"], step="post", alpha=0.9)
        ax.axvspan(tl["fault_at"], tl["healed_at"], color="#d62728", alpha=0.06)
        ymax = max(1, max(p + b + e for p, b, e in zip(prim, back, err, strict=True)))
        opened = [t for t, s, _ in r["_circuit"] if s == "open"]
        if opened:
            ax.hlines([ymax * 1.05] * len(opened), opened, [t + 0.1 for t in opened], color="black", lw=3,
                      label="primary breaker open")
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xlim(0, tl["end"])
        ax.set_title(f"{r['scenario']}: {r['title']}", fontsize=9)
        ax.set_ylabel("requests / s")
        if legend:
            ax.legend(fontsize=7, loc="lower left")

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.3 * n), sharex=True, squeeze=False)
    for i, r in enumerate(results):
        draw(axes[i][0], r, i == 0)
    axes[-1][0].set_xlabel("seconds (shaded: fault injected)")
    fig.tight_layout()
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / "chaos_timelines.png", dpi=120)
    outage = next((r for r in results if r["scenario"] == "outage"), None)
    if outage:
        fig, ax = plt.subplots(figsize=(11, 3.6))
        draw(ax, outage, True)
        ax.set_xlabel("seconds (shaded: primary returning 503)")
        fig.tight_layout()
        fig.savefig(IMG / "failover_timeline.png", dpi=130)


if __name__ == "__main__":
    main()
