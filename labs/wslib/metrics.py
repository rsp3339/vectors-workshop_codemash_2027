"""
Retrieval evaluation metrics.

These are the reference implementations. In the labs you write your own and
assert against these, so they need to be boringly correct rather than clever.

Conventions used everywhere in wslib:

    qrels : {query_id: {doc_id: grade}}   grade 0 = not relevant, >0 = relevant
    run   : {query_id: [doc_id, ...]}     ranked best-first

Grades may be binary (SciFact is: everything relevant is grade 1) or graded
(NFCorpus uses 0/1/2). Every function here handles both.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Mapping, Sequence

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Sequence[str]]


# ── single-query metrics ────────────────────────────────────────────────────


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Fraction of all known-relevant documents that appear in the top k.

    Order inside the top k does not matter — recall only asks whether we found
    them at all. This is the metric to tune an ANN index against.
    """
    total = sum(1 for g in relevant.values() if g > 0)
    if total == 0:
        return 0.0
    found = sum(1 for d in ranked[:k] if relevant.get(d, 0) > 0)
    return found / total


def reciprocal_rank(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """1 / rank of the first relevant document, or 0.0 if none in the top k."""
    for i, d in enumerate(ranked[:k], start=1):
        if relevant.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Discounted cumulative gain.

    gain     = 2**grade - 1     (so a grade-2 doc is worth 3x a grade-1 doc)
    discount = log2(rank + 1)   (rank is 1-based, so rank 1 divides by log2(2)=1)
    """
    total = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        g = relevant.get(d, 0)
        if g > 0:
            total += (2.0 ** g - 1.0) / math.log2(i + 1)
    return total


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """DCG normalised by the best DCG achievable for this query.

    The ideal ranking puts the highest grades first, so IDCG comes from the
    qrels themselves — not from the run.
    """
    ideal_grades = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    idcg = sum(
        (2.0 ** g - 1.0) / math.log2(i + 1)
        for i, g in enumerate(ideal_grades, start=1)
    )
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg


# ── aggregation over a query set ────────────────────────────────────────────


def evaluate(run: Run, qrels: Qrels, ks: Iterable[int] = (1, 5, 10)) -> Dict[str, float]:
    """Mean metrics across every query that has judgements.

    Queries present in `qrels` but missing from `run` score 0 — a retriever that
    returns nothing is not excused from being measured.
    """
    ks = list(ks)
    qids = [q for q in qrels if any(g > 0 for g in qrels[q].values())]
    if not qids:
        return {}

    out: Dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = sum(
            recall_at_k(run.get(q, []), qrels[q], k) for q in qids
        ) / len(qids)
        out[f"ndcg@{k}"] = sum(
            ndcg_at_k(run.get(q, []), qrels[q], k) for q in qids
        ) / len(qids)
    kmax = max(ks)
    out[f"mrr@{kmax}"] = sum(
        reciprocal_rank(run.get(q, []), qrels[q], kmax) for q in qids
    ) / len(qids)
    out["queries"] = float(len(qids))
    return out


def per_query(run: Run, qrels: Qrels, metric: str = "ndcg", k: int = 10) -> Dict[str, float]:
    """Per-query scores — what you need for a paired comparison.

    Aggregates hide which queries improved and which regressed. Two systems with
    the same mean can behave completely differently.
    """
    fn = {"recall": recall_at_k, "ndcg": ndcg_at_k, "mrr": reciprocal_rank}[metric]
    return {
        q: fn(run.get(q, []), qrels[q], k)
        for q in qrels
        if any(g > 0 for g in qrels[q].values())
    }


# ── is the difference real? ────────────────────────────────────────────────


def bootstrap_delta(
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap on the per-query difference (b - a).

    Resamples queries with replacement, so the interval reflects how much the
    result depends on *which* queries happen to be in your golden set. If the
    interval contains 0, you have a hunch with better formatting.
    """
    shared = sorted(set(scores_a) & set(scores_b))
    if not shared:
        raise ValueError("no queries in common between the two runs")

    deltas = [scores_b[q] - scores_a[q] for q in shared]
    mean_delta = sum(deltas) / len(deltas)

    rng = random.Random(seed)
    means: List[float] = []
    m = len(deltas)
    for _ in range(n_resamples):
        s = sum(deltas[rng.randrange(m)] for _ in range(m))
        means.append(s / m)
    means.sort()

    lo_i = int((1 - ci) / 2 * n_resamples)
    hi_i = min(n_resamples - 1, int((1 + ci) / 2 * n_resamples))

    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)

    lo, hi = means[lo_i], means[hi_i]
    return {
        "mean_delta": mean_delta,
        "ci_low": lo,
        "ci_high": hi,
        "significant": float(lo > 0 or hi < 0),
        "n_queries": float(len(deltas)),
        "improved": float(wins),
        "regressed": float(losses),
        "unchanged": float(len(deltas) - wins - losses),
    }


def format_delta(d: Mapping[str, float], label: str = "Δ") -> str:
    """One-line human summary of a bootstrap result."""
    verdict = "SIGNIFICANT" if d["significant"] else "not significant"
    return (
        f"{label} {d['mean_delta']:+.4f}  "
        f"95% CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
        f"{verdict}  "
        f"({int(d['improved'])} better / {int(d['regressed'])} worse / "
        f"{int(d['unchanged'])} same, n={int(d['n_queries'])})"
    )


def leaderboard(results: Mapping[str, Mapping[str, float]], ks=(5, 10)) -> str:
    """Render a comparison table of several named runs."""
    cols = [f"recall@{k}" for k in ks] + [f"ndcg@{k}" for k in ks]
    cols += [c for c in next(iter(results.values())) if c.startswith("mrr@")]

    name_w = max(len(n) for n in results) + 2
    head = "config".ljust(name_w) + "".join(c.rjust(12) for c in cols)
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        row = name.ljust(name_w)
        row += "".join(f"{r.get(c, 0.0):12.4f}" for c in cols)
        lines.append(row)
    return "\n".join(lines)
