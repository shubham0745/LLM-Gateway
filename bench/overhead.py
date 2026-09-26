"""Gateway overhead: the same streaming load against the mock directly and through the gateway.

    python -m bench.overhead --levels 10 100 500 1000

For each concurrency level, N streams run concurrently in a closed loop
(each worker starts its next request as soon as the last one finishes) for a
fixed duration after a short warm-up whose requests are discarded, first straight to the mock provider and then through the
gateway (auth, rate limit, budget, breaker, routing, accounting and logging
all active; response caching bypassed so every request really goes
upstream). Overhead = gateway latency - direct latency, per percentile.

CPU and memory of the gateway container are sampled with ``docker stats``
while each level runs (pass --no-docker to skip).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import threading
import time

import httpx

from bench.common import GATEWAY, IMG, MOCK, bench_key, mock_control, mock_reset, stream_once, summary, write_result

PROMPT = [{"role": "user", "content": "Explain what an LLM gateway does in two sentences."}]


class DockerSampler(threading.Thread):
    def __init__(self, container: str):
        super().__init__(daemon=True)
        self.container = container
        self.cpu: list[float] = []
        self.mem_mb: list[float] = []
        self.stop_flag = threading.Event()

    def run(self) -> None:
        while not self.stop_flag.is_set():
            try:
                out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}} {{.MemUsage}}", self.container],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
                cpu, mem = out.split(" ", 1)
                self.cpu.append(float(cpu.rstrip("%")))
                m = re.match(r"([\d.]+)(KiB|MiB|GiB)", mem)
                if m:
                    self.mem_mb.append(float(m.group(1)) * {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}[m.group(2)])
            except (subprocess.SubprocessError, ValueError):
                pass

    def result(self) -> dict:
        return {"cpu_pct_avg": sum(self.cpu) / len(self.cpu) if self.cpu else None, "cpu_pct_max": max(self.cpu, default=None),
                "mem_mb_max": max(self.mem_mb, default=None)}


async def run_level(url: str, headers: dict, body: dict, concurrency: int, duration: float, warmup: float = 3.0) -> list:
    # Several small client pools instead of one big one: httpx scans every idle
    # connection in a pool per request, which at 1,000 connections would make
    # the load generator itself the bottleneck (the gateway shards for the same reason).
    shards = max(1, min(16, concurrency // 25))
    per = -(-concurrency // shards) + 2
    limits = httpx.Limits(max_connections=per, max_keepalive_connections=per)
    clients = [httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(60, connect=30)) for _ in range(shards)]
    results = []
    start = time.perf_counter()
    measure_from, deadline = start + warmup, start + warmup + duration

    async def worker(client: httpx.AsyncClient) -> None:
        while time.perf_counter() < deadline:
            t = time.perf_counter()
            r = await stream_once(client, url, body, headers)
            if t >= measure_from:  # requests started during warm-up (new connections, cold caches) are discarded
                results.append(r)

    try:
        await asyncio.gather(*(worker(clients[i % shards]) for i in range(concurrency)))
    finally:
        for c in clients:
            await c.aclose()
    return results


def describe(results: list) -> dict:
    ok = [r for r in results if r.ok]
    return {
        "requests": len(results),
        "errors": len(results) - len(ok),
        "error_samples": list({r.error for r in results if not r.ok})[:5],
        "ttft_ms": {k: v * 1000 for k, v in summary([r.ttft for r in ok if r.ttft is not None]).items()},
        "total_ms": {k: v * 1000 for k, v in summary([r.total for r in ok]).items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[10, 100, 500, 1000])
    ap.add_argument("--duration", type=float, default=20.0, help="seconds per level and target")
    ap.add_argument("--container", default="llm-gateway-gateway-1")
    ap.add_argument("--no-docker", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the result file, e.g. 'before-optimisation'")
    ap.add_argument("--ttft-ms", type=int, default=50, help="mock time to first token")
    ap.add_argument("--tokens", type=int, default=40, help="mock tokens per response")
    ap.add_argument("--tps", type=float, default=200, help="mock tokens per second")
    args = ap.parse_args()

    key = bench_key()
    mock_reset()
    # Default: a short, fast stream (50 ms to first token, 40 tokens at 200 tokens/s,
    # ~250 ms), which stresses per-token work. --ttft-ms/--tokens/--tps change it.
    mock_control("primary", ttft_ms=args.ttft_ms, tokens_per_s=args.tps, response_tokens=args.tokens)
    body = {"model": "mock", "messages": PROMPT, "stream_options": {"include_usage": True}}
    direct_body = {"model": "mock-small", "messages": PROMPT, "stream_options": {"include_usage": True}}
    gw_headers = {"authorization": f"Bearer {key}", "x-gateway-cache": "no-store"}

    levels = []
    for c in args.levels:
        print(f"concurrency {c}: direct ...", flush=True)
        direct = describe(asyncio.run(run_level(f"{MOCK}/primary/v1/chat/completions", {}, direct_body, c, args.duration)))
        print(f"concurrency {c}: gateway ...", flush=True)
        sampler = None if args.no_docker else DockerSampler(args.container)
        if sampler:
            sampler.start()
        gw = describe(asyncio.run(run_level(f"{GATEWAY}/v1/chat/completions", gw_headers, body, c, args.duration)))
        if sampler:
            sampler.stop_flag.set()
            sampler.join(timeout=15)
        overhead = {m: {q: gw[m][q] - direct[m][q] for q in ("p50", "p95", "p99")} for m in ("ttft_ms", "total_ms")}
        row = {"concurrency": c, "direct": direct, "gateway": gw, "overhead_ms": overhead,
               "gateway_resources": sampler.result() if sampler else None}
        levels.append(row)
        print(json.dumps({"concurrency": c, "overhead_ms": overhead, "errors": [direct["errors"], gw["errors"]],
                          "rps": round(gw["requests"] / args.duration, 1), "resources": row["gateway_resources"]}), flush=True)

    name = "overhead" + (f"-{args.tag}" if args.tag else "")
    path = write_result(name, {"duration_s": args.duration, "host_cpus": os.cpu_count(), "mock": {"ttft_ms": args.ttft_ms, "tokens": args.tokens, "tokens_per_s": args.tps},
                               "gateway": GATEWAY, "levels": levels})
    print("wrote", path)
    plot(levels, IMG / f"{name}.png")


def plot(levels: list[dict], out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [lv["concurrency"] for lv in levels]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title in ((axes[0], "ttft_ms", "Time to first token"), (axes[1], "total_ms", "Total stream time")):
        for q, style in (("p50", "-"), ("p95", "--"), ("p99", ":")):
            ax.plot(xs, [lv["direct"][metric][q] for lv in levels], style, color="#999999", marker="o", ms=3, label=f"direct {q}")
            ax.plot(xs, [lv["gateway"][metric][q] for lv in levels], style, color="#2a6fdb", marker="o", ms=3, label=f"gateway {q}")
        ax.set_xscale("log")
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_xlabel("concurrent streams")
        ax.set_ylabel("ms")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
