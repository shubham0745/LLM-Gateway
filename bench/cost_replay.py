"""Cost replay: the same request stream with caching off, exact-only, and exact + semantic.

    python -m bench.cost_replay                 # both workloads, 3,000 requests each
    python -m bench.cost_replay --requests 500  # quicker

Workloads are built from Quora Question Pairs (see bench/README.md for the
download), so "the same question asked differently" is real human rephrasing:

  faq     A support-bot-like stream. 70% of requests ask one of 300 popular
          questions (Zipf-distributed popularity), each time phrased as a
          random member of that question's QQP duplicate cluster; 30% are
          one-off questions.
  unique  Every request is a different question. Caching should save ~nothing
          here, and this run shows what the lookups cost when it does not pay.

Each mode runs as a fresh tenant, so caches start empty (entries are
tenant-scoped). Cost is what the gateway billed (x-gateway-cost-usd, zero for
cache hits). A semantic hit counts as wrong when the cached answer was
produced for a question from a different QQP duplicate cluster: the mock's
answers are deterministic per prompt, so each answer identifies the prompt
that produced it.

Routing is not part of this replay: the gateway routes by alias and fallback
chain, not by price, so there is no "routing on/off" to compare yet.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import time
from collections import Counter

import httpx
import numpy as np

from bench.cache_eval import qqp_clusters
from bench.common import GATEWAY, ROOT, admin, mock_control, mock_reset, write_result

QQP = ROOT / "bench" / "data" / "raw" / "qqp.tsv"
# USD per 1M tokens, for re-pricing the billed tokens as if the traffic had gone to these models.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "claude-haiku-4-5": (1.00, 5.00), "gpt-4o": (2.50, 10.00)}


def build_workloads(n: int, seed: int) -> dict[str, list[tuple[str, str]]]:
    """(prompt, cluster id) lists."""
    cluster = qqp_clusters(QQP)
    text: dict[str, str] = {}
    with QQP.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            for q, t in ((r["qid1"], r["question1"]), (r["qid2"], r["question2"])):
                if t and t.strip():
                    text.setdefault(q, t.strip())
    members: dict[str, list[str]] = {}
    for q, root in cluster.items():
        if q in text:
            members.setdefault(root, []).append(q)
    rng = random.Random(seed)
    topics = [m for m in members.values() if len({text[q].lower() for q in m}) >= 3]
    rng.shuffle(topics)
    popular = topics[:300]
    singles = [q for q in text if q not in cluster]
    rng.shuffle(singles)

    ranks = np.arange(1, len(popular) + 1)
    weights = 1 / ranks ** 1.1
    weights /= weights.sum()
    faq: list[tuple[str, str]] = []
    single_iter = iter(singles)
    for _ in range(n):
        if rng.random() < 0.7:
            topic = popular[int(np.searchsorted(np.cumsum(weights), rng.random()))]
            q = rng.choice(topic)
            faq.append((text[q], cluster[q]))
        else:
            q = next(single_iter)
            faq.append((text[q], "single:" + q))
    unique = [(text[q], "single:" + q) for q in singles[len(singles) // 2: len(singles) // 2 + n]]
    return {"faq": faq, "unique": unique}


def fresh_tenant(label: str) -> str:
    tid = f"replay-{label}-{int(time.time())}"
    admin("POST", "/admin/tenants", json={"id": tid, "name": f"cost replay {label}", "monthly_budget_usd": 1_000_000,
                                          "rpm_limit": 100_000_000, "tpm_limit": 2_000_000_000})
    return admin("POST", f"/admin/tenants/{tid}/keys", json={"name": "replay"}).json()["key"]


def set_semantic(enabled: bool) -> None:
    cfg = admin("GET", "/admin/config").json()["config"]
    if cfg["cache"]["semantic"]["enabled"] != enabled:
        cfg["cache"]["semantic"]["enabled"] = enabled
        admin("PUT", "/admin/config", json={"config": cfg}, params={"comment": f"cost replay: semantic={enabled}"})
        time.sleep(1.5)


async def replay(workload: list[tuple[str, str]], key: str, cache_header: str | None, concurrency: int) -> list[dict]:
    headers = {"authorization": f"Bearer {key}"}
    if cache_header:
        headers["x-gateway-cache"] = cache_header
    results: list[dict | None] = [None] * len(workload)
    queue = iter(range(len(workload)))
    async with httpx.AsyncClient(base_url=GATEWAY, headers=headers, timeout=60) as client:
        async def worker() -> None:
            for i in queue:
                prompt, clus = workload[i]
                t0 = time.perf_counter()
                r = await client.post("/v1/chat/completions", json={"model": "mock", "messages": [{"role": "user", "content": prompt}]})
                data = r.json()
                ok = r.status_code == 200
                results[i] = {
                    "cluster": clus, "status": r.status_code, "cache": r.headers.get("x-gateway-cache"),
                    "cost": float(r.headers.get("x-gateway-cost-usd") or 0), "latency": time.perf_counter() - t0,
                    "content": data["choices"][0]["message"]["content"] if ok else None,
                    "usage": data.get("usage") if ok else None,
                }

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    return [r for r in results if r is not None]


def analyse(results: list[dict]) -> dict:
    kinds = Counter(r["cache"] for r in results)
    producer: dict[str, str] = {}  # answer text -> cluster of the prompt that produced it upstream
    for r in results:
        if r["cache"] in ("miss", "bypass") and r["content"]:
            producer.setdefault(r["content"], r["cluster"])
    sem = [r for r in results if r["cache"] == "hit-semantic"]
    wrong = sum(1 for r in sem if producer.get(r["content"]) not in (None, r["cluster"]))
    billed = [r for r in results if r["cache"] in ("miss", "bypass") and r["usage"]]
    p_tok = sum(r["usage"]["prompt_tokens"] for r in billed)
    c_tok = sum(r["usage"]["completion_tokens"] for r in billed)
    lat = {k: float(np.mean([r["latency"] for r in results if r["cache"] == k]) * 1000) for k in kinds}
    return {
        "requests": len(results),
        "errors": sum(1 for r in results if r["status"] != 200),
        "cache": dict(kinds),
        "hit_rate": round((kinds.get("hit-exact", 0) + kinds.get("hit-semantic", 0)) / max(1, len(results)), 4),
        "semantic_hits_wrong": wrong,
        "billed_usd_at_mock_price": round(sum(r["cost"] for r in results), 6),
        "billed_tokens": {"prompt": p_tok, "completion": c_tok},
        "usd_at": {m: round((p_tok * pi + c_tok * po) / 1e6, 6) for m, (pi, po) in PRICES.items()},
        "mean_latency_ms": lat,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if not QQP.exists():
        raise SystemExit(f"{QQP} not found: see bench/README.md for how to download Quora Question Pairs.")

    workloads = build_workloads(args.requests, args.seed)
    mock_reset()
    # Answers of ~150 tokens, returned quickly so the replay is fast; the price table does the rest.
    mock_control("primary", ttft_ms=5, tokens_per_s=100_000, response_tokens=150)
    out: dict = {"requests_per_workload": args.requests, "seed": args.seed, "workloads": {}}
    original = admin("GET", "/admin/config").json()["config"]
    try:
        for name, wl in workloads.items():
            runs = {}
            for mode, header, semantic in (("off", "no-store", False), ("exact", None, False), ("exact+semantic", None, True)):
                set_semantic(semantic)
                key = fresh_tenant(f"{name}-{mode.replace('+', '-')}")
                print(f"{name} / {mode} ...", flush=True)
                runs[mode] = analyse(asyncio.run(replay(wl, key, header, args.concurrency)))
                print("   ", {k: runs[mode][k] for k in ("hit_rate", "semantic_hits_wrong", "billed_usd_at_mock_price", "errors")}, flush=True)
            base = runs["off"]["billed_usd_at_mock_price"] or 1e-12
            for mode in ("exact", "exact+semantic"):
                runs[mode]["saved_vs_off_pct"] = round(100 * (1 - runs[mode]["billed_usd_at_mock_price"] / base), 2)
            out["workloads"][name] = {"distinct_prompts": len({p for p, _ in wl}), "runs": runs}
    finally:
        admin("PUT", "/admin/config", json={"config": original}, params={"comment": "cost replay: restore"})
        mock_reset()
    print("wrote", write_result("cost_replay", out))


if __name__ == "__main__":
    main()
