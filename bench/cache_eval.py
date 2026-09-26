"""Measure the semantic cache: wrong-answer rate against hit rate per threshold.

    python -m bench.cache_eval --qqp bench/data/raw/qqp.tsv

Datasets
  * Quora Question Pairs (public, ~404k pairs, human-labelled duplicates).
    Download: see bench/README.md. A fixed-seed sample is used.
  * bench/data/handwritten_pairs.jsonl: 180 pairs written for this project,
    half of them deliberate near misses ("capital of Austria" vs "capital of
    Australia", "5 km to miles" vs "5 miles to km").

Two views
  * pairwise: q1 is cached, q2 arrives. A hit happens when cos(q1, q2) >= t;
    it is a wrong answer when the pair is labelled "not the same question".
  * retrieval: every q1 is cached at once and each q2 is looked up against all
    of them (top-1 nearest neighbour), which is what the gateway really does.
    A hit is correct only if the neighbour is q2's own duplicate.

Outputs docs/results/cache_eval.json and docs/img/semantic_threshold.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

from gateway.cache.embedder import Embedder
from gateway.cache.guard import compatible

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [round(x, 2) for x in np.arange(0.70, 1.0, 0.01)]


def load_qqp(path: Path, n: int, seed: int) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r.get("question1") and r.get("question2")]
    random.Random(seed).shuffle(rows)
    return [{"q1": r["question1"], "q2": r["question2"], "label": int(r["is_duplicate"]), "qid1": r["qid1"], "qid2": r["qid2"]}
            for r in rows[:n]]


def qqp_clusters(path: Path) -> dict[str, str]:
    """Union-find over every duplicate edge in the full QQP file: qid -> cluster root."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("is_duplicate") == "1":
                a, b = find(r["qid1"]), find(r["qid2"])
                if a != b:
                    parent[a] = b
    return {q: find(q) for q in list(parent)}


def load_pairs(name: str) -> list[dict]:
    with (ROOT / "bench" / "data" / name).open() as f:
        return [json.loads(line) for line in f]


def embed_all(emb: Embedder, texts: list[str], batch: int = 128) -> np.ndarray:
    out = [emb.embed(texts[i:i + batch]) for i in range(0, len(texts), batch)]
    return np.vstack(out).astype(np.float32)


def pairwise(pairs: list[dict], e1: np.ndarray, e2: np.ndarray, use_guard: bool = False) -> list[dict]:
    sims = (e1 * e2).sum(axis=1)
    labels = np.array([p["label"] for p in pairs])
    ok = np.array([compatible(p["q1"], p["q2"])[0] for p in pairs]) if use_guard else np.ones(len(pairs), bool)
    rows = []
    for t in THRESHOLDS:
        hit = (sims >= t) & ok
        hits = int(hit.sum())
        wrong = int((hit & (labels == 0)).sum())
        rows.append({
            "threshold": t,
            "hit_rate": hits / len(pairs),
            "wrong_answer_rate": wrong / hits if hits else 0.0,
            "wrong_per_1000_requests": 1000 * wrong / len(pairs),
            "duplicate_recall": float((hit & (labels == 1)).sum() / max(1, (labels == 1).sum())),
        })
    return rows


def retrieval(pairs: list[dict], e1: np.ndarray, e2: np.ndarray, cluster: dict[str, str], use_guard: bool = False,
              top_k: int = 3) -> list[dict]:
    """Cache every distinct q1, look up every q2, as the gateway does (top-k above threshold, first that passes the guard).

    A hit is correct when the cached question is the same question as the
    query according to QQP's duplicate labels (transitively: duplicates of
    duplicates count), or literally the same text.
    """
    uniq: dict[str, int] = {}
    for i, p in enumerate(pairs):
        uniq.setdefault(p["qid1"], i)
    idx = np.array(list(uniq.values()))
    corpus = e1[idx]
    cq = [pairs[i]["q1"] for i in idx]
    cc = [cluster.get(pairs[i]["qid1"], pairs[i]["qid1"]) for i in idx]
    top_sim = np.empty((len(pairs), top_k), dtype=np.float32)
    top_idx = np.empty((len(pairs), top_k), dtype=np.int64)
    for s in range(0, len(pairs), 1024):
        sims = e2[s:s + 1024] @ corpus.T
        part = np.argpartition(-sims, top_k, axis=1)[:, :top_k]
        order = np.take_along_axis(sims, part, axis=1).argsort(axis=1)[:, ::-1]
        top_idx[s:s + 1024] = np.take_along_axis(part, order, axis=1)
        top_sim[s:s + 1024] = np.take_along_axis(sims, top_idx[s:s + 1024], axis=1)
    rows = []
    for t in THRESHOLDS:
        hits = wrong = 0
        for i, p in enumerate(pairs):
            chosen = None
            for j in range(top_k):
                if top_sim[i, j] < t:
                    break
                cand = top_idx[i, j]
                if not use_guard or compatible(cq[cand], p["q2"])[0]:
                    chosen = cand
                    break
            if chosen is None:
                continue
            hits += 1
            same = cq[chosen].strip().lower() == p["q2"].strip().lower() or cc[chosen] == cluster.get(p["qid2"], p["qid2"])
            wrong += not same
        rows.append({
            "threshold": t,
            "hit_rate": hits / len(pairs),
            "wrong_answer_rate": wrong / hits if hits else 0.0,
            "wrong_per_1000_requests": 1000 * wrong / len(pairs),
        })
    return rows


def pick_threshold(curves: dict[str, list[dict]], max_wrong_per_1000: float) -> tuple[float, str]:
    """Lowest threshold whose wrong answers stay under ``max_wrong_per_1000`` requests.

    Judged on the QQP retrieval view with the guard on, because that is the
    closest to real traffic: a large pool of cached real questions and a
    stream of new ones. The hand-written sets are adversarial by design (half
    near misses) and are reported, not optimised for. Without QQP we fall back
    to the held-out hand-written set.
    """
    for name in ("qqp_retrieval_guard", "heldout_pairwise_guard"):
        rows = curves.get(name)
        if not rows:
            continue
        for row in rows:
            if row["hit_rate"] > 0 and row["wrong_per_1000_requests"] <= max_wrong_per_1000:
                return row["threshold"], name
    return 0.99, "fallback"


def plot(curves: dict[str, list[dict]], chosen: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    styles = {
        "qqp_retrieval": ("#1b9e77", "QQP retrieval, similarity only", "--"),
        "qqp_retrieval_guard": ("#1b9e77", "QQP retrieval, similarity + guard", "-"),
        "heldout_pairwise": ("#d95f02", "Held-out near misses, similarity only", "--"),
        "heldout_pairwise_guard": ("#d95f02", "Held-out near misses, similarity + guard", "-"),
        "qqp_pairwise_guard": ("#7570b3", "QQP pairs, similarity + guard", "-"),
    }
    for name, (color, label, ls) in styles.items():
        rows = curves.get(name)
        if not rows:
            continue
        ax1.plot([r["hit_rate"] * 100 for r in rows], [r["wrong_answer_rate"] * 100 for r in rows], marker="o", ms=3,
                 color=color, ls=ls, label=label)
        if name.endswith("_guard"):
            ax2.plot([r["threshold"] for r in rows], [r["wrong_per_1000_requests"] for r in rows], color=color, label=label)
        for r in rows:
            if abs(r["threshold"] - chosen) < 1e-9:
                ax1.scatter([r["hit_rate"] * 100], [r["wrong_answer_rate"] * 100], s=90, facecolors="none", edgecolors="black", zorder=5)
    ax1.set_xlabel("hit rate (% of lookups answered from cache)")
    ax1.set_ylabel("wrong answers (% of cache hits)")
    ax1.set_title("Wrong-answer rate vs hit rate (circles: chosen threshold)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.axvline(chosen, color="black", lw=1, ls=":")
    ax2.set_xlabel("similarity threshold")
    ax2.set_ylabel("wrong answers per 1,000 requests")
    ax2.set_yscale("symlog", linthresh=1)
    ax2.set_ylim(bottom=0)
    ax2.axhline(2.0, color="grey", lw=1, ls="--", label="budget: 2 wrong per 1,000")
    ax2.set_title(f"Wrong answers per 1,000 requests by threshold (chosen: {chosen})")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qqp", type=Path, default=ROOT / "bench" / "data" / "raw" / "qqp.tsv")
    ap.add_argument("--n", type=int, default=20_000, help="QQP pairs to sample")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-wrong-per-1000", type=float, default=2.0, help="acceptable wrong answers per 1,000 requests")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "all-MiniLM-L6-v2"))
    ap.add_argument("--plot-only", action="store_true", help="redraw the chart from docs/results/cache_eval.json")
    args = ap.parse_args()
    if args.plot_only:
        saved = json.loads((ROOT / "docs" / "results" / "cache_eval.json").read_text())
        plot(saved["curves"], saved["meta"]["chosen_threshold"], ROOT / "docs" / "img" / "semantic_threshold.png")
        return

    emb = Embedder(args.model_dir, threads=4)
    curves: dict[str, list[dict]] = {}
    meta: dict = {"model": "sentence-transformers/all-MiniLM-L6-v2 (ONNX)", "max_wrong_per_1000_requests": args.max_wrong_per_1000}

    for name, file in (("handwritten", "handwritten_pairs.jsonl"), ("heldout", "heldout_pairs.jsonl")):
        pairs = load_pairs(file)
        a, b = embed_all(emb, [p["q1"] for p in pairs]), embed_all(emb, [p["q2"] for p in pairs])
        curves[f"{name}_pairwise"] = pairwise(pairs, a, b)
        curves[f"{name}_pairwise_guard"] = pairwise(pairs, a, b, use_guard=True)
        meta[f"{name}_pairs"] = len(pairs)
        sims = (a * b).sum(axis=1)
        worst = sorted(((float(s), p["q1"], p["q2"]) for s, p in zip(sims, pairs, strict=True) if p["label"] == 0), reverse=True)[:8]
        meta[f"{name}_closest_near_misses"] = [{"similarity": round(s, 4), "q1": x, "q2": y, "guard": compatible(x, y)[1] or "passes"}
                                               for s, x, y in worst]

    if args.qqp.exists():
        qqp = load_qqp(args.qqp, args.n, args.seed)
        cluster = qqp_clusters(args.qqp)
        t0 = time.perf_counter()
        q1, q2 = embed_all(emb, [p["q1"] for p in qqp]), embed_all(emb, [p["q2"] for p in qqp])
        meta["qqp_pairs"] = len(qqp)
        meta["qqp_duplicate_share"] = sum(p["label"] for p in qqp) / len(qqp)
        meta["embed_ms_per_text_batched"] = round((time.perf_counter() - t0) * 1000 / (2 * len(qqp)), 3)
        curves["qqp_pairwise"] = pairwise(qqp, q1, q2)
        curves["qqp_pairwise_guard"] = pairwise(qqp, q1, q2, use_guard=True)
        curves["qqp_retrieval"] = retrieval(qqp, q1, q2, cluster)
        curves["qqp_retrieval_guard"] = retrieval(qqp, q1, q2, cluster, use_guard=True)
    else:
        print(f"QQP not found at {args.qqp}; hand-written pairs only (see bench/README.md to download)")

    # Single-prompt latency, the way the gateway calls it.
    emb1 = Embedder(args.model_dir, threads=1)
    emb1.embed(["warm up"])
    t0 = time.perf_counter()
    for i in range(200):
        emb1.embed([f"How do I configure feature number {i} in my application?"])
    meta["embed_ms_single_prompt_1_thread"] = round((time.perf_counter() - t0) * 1000 / 200, 3)

    chosen, basis = pick_threshold(curves, args.max_wrong_per_1000)
    meta["chosen_threshold"] = chosen
    meta["chosen_on"] = basis
    out = ROOT / "docs" / "results" / "cache_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "curves": curves}, indent=2) + "\n")
    plot(curves, chosen, ROOT / "docs" / "img" / "semantic_threshold.png")

    print(json.dumps(meta, indent=2))
    print("\nthreshold | " + " | ".join(f"{k}: hit% / wrong% of hits / wrong per 1k" for k in curves))
    for i, t in enumerate(THRESHOLDS):
        if round(t * 100) % 2 and t < 0.97:
            continue
        cells = [f"{c[i]['hit_rate'] * 100:5.1f} / {c[i]['wrong_answer_rate'] * 100:5.1f} / {c[i]['wrong_per_1000_requests']:5.1f}"
                 for c in curves.values()]
        print(f"{t:.2f}      | " + " | ".join(cells))


if __name__ == "__main__":
    main()
