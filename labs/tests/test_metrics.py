"""
Hand-checked tests for the reference metrics.

Attendees assert their own implementations against wslib.metrics, so these
numbers are computed by hand in the comments rather than captured from a run.
Run with:  python tests/test_metrics.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wslib.metrics import (  # noqa: E402
    bootstrap_delta, dcg_at_k, evaluate, format_delta, leaderboard,
    ndcg_at_k, per_query, recall_at_k, reciprocal_rank,
)

FAILED = []


def check(label, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {label:<42} got {got:.10f}  want {want:.10f}")
    if not ok:
        FAILED.append(label)


RANKED = ["d1", "d2", "d3", "d4", "d5"]
REL = {"d1": 3, "d3": 2, "d5": 1}          # 3 relevant documents


# ── recall ─────────────────────────────────────────────────────────────────
# top3 = d1,d2,d3 -> found d1,d3 -> 2 of 3
check("recall@3", recall_at_k(RANKED, REL, 3), 2 / 3)
check("recall@5", recall_at_k(RANKED, REL, 5), 1.0)
check("recall@1", recall_at_k(RANKED, REL, 1), 1 / 3)
check("recall, nothing relevant", recall_at_k(RANKED, {"zz": 1}, 5), 0.0)
check("recall, empty qrels", recall_at_k(RANKED, {}, 5), 0.0)

# ── reciprocal rank ────────────────────────────────────────────────────────
check("RR, hit at rank 1", reciprocal_rank(RANKED, REL, 5), 1.0)
check("RR, hit at rank 3", reciprocal_rank(["x", "y", "d1"], REL, 5), 1 / 3)
check("RR, hit outside k", reciprocal_rank(["x", "y", "d1"], REL, 2), 0.0)

# ── DCG ────────────────────────────────────────────────────────────────────
# rank1 d1 g=3 -> (2^3-1)/log2(2) = 7/1     = 7.0
# rank2 d2 g=0 -> 0
# rank3 d3 g=2 -> (2^2-1)/log2(4) = 3/2     = 1.5
DCG3 = 7.0 + 1.5
check("dcg@3", dcg_at_k(RANKED, REL, 3), DCG3)

# ideal top3 grades are [3, 2, 1]
# 7/log2(2) + 3/log2(3) + 1/log2(4)
IDCG3 = 7.0 + 3.0 / math.log2(3) + 1.0 / 2.0
check("idcg@3 (via perfect run)",
      dcg_at_k(["d1", "d3", "d5"], REL, 3), IDCG3)
check("ndcg@3", ndcg_at_k(RANKED, REL, 3), DCG3 / IDCG3)
check("ndcg@3, perfect order", ndcg_at_k(["d1", "d3", "d5"], REL, 3), 1.0)
check("ndcg, nothing relevant", ndcg_at_k(RANKED, {"zz": 1}, 5), 0.0)

# position sensitivity: same documents, worse order -> strictly lower nDCG
good = ndcg_at_k(["d1", "d3", "d5"], REL, 5)
bad = ndcg_at_k(["d5", "d3", "d1"], REL, 5)
print(f"{'PASS' if bad < good else 'FAIL'}  "
      f"{'ndcg penalises worse order':<42} {bad:.6f} < {good:.6f}")
if not bad < good:
    FAILED.append("ndcg order sensitivity")

# recall is blind to that same reordering
check("recall ignores order",
      recall_at_k(["d5", "d3", "d1"], REL, 5), recall_at_k(["d1", "d3", "d5"], REL, 5))


# ── aggregation ────────────────────────────────────────────────────────────
QRELS = {"q1": REL, "q2": {"a1": 1}, "q3": {"z9": 0}}   # q3 has no positives
RUN = {"q1": RANKED, "q2": ["nope", "a1"]}

agg = evaluate(RUN, QRELS, ks=(1, 3))
check("evaluate skips unjudged queries", agg["queries"], 2.0)
# q1 recall@3 = 2/3, q2 recall@3 = 1/1  ->  mean = 5/6
check("evaluate recall@3 mean", agg["recall@3"], (2 / 3 + 1.0) / 2)
# q1 RR = 1/1, q2 RR = 1/2  -> mean = 0.75
check("evaluate mrr@3 mean", agg["mrr@3"], (1.0 + 0.5) / 2)

# a query in qrels but absent from the run must score zero, not vanish
agg_missing = evaluate({"q1": RANKED}, QRELS, ks=(3,))
check("missing run entry scores 0", agg_missing["recall@3"], (2 / 3 + 0.0) / 2)

pq = per_query(RUN, QRELS, metric="recall", k=3)
check("per_query returns judged only", float(len(pq)), 2.0)


# ── bootstrap ──────────────────────────────────────────────────────────────
a = {f"q{i}": 0.5 for i in range(40)}
same = bootstrap_delta(a, dict(a), n_resamples=400, seed=1)
check("bootstrap, no difference", same["mean_delta"], 0.0)
check("bootstrap, not significant", same["significant"], 0.0)

b = {q: v + 0.1 for q, v in a.items()}
better = bootstrap_delta(a, b, n_resamples=400, seed=1)
check("bootstrap, uniform +0.1", better["mean_delta"], 0.1, tol=1e-12)
check("bootstrap, significant", better["significant"], 1.0)
check("bootstrap, all improved", better["improved"], 40.0)

# a real-ish mixed case: tiny mean gain, high variance -> should NOT be called
mixed_a = {f"q{i}": (i % 7) / 10 for i in range(50)}
mixed_b = {f"q{i}": (i % 7) / 10 + (0.4 if i % 5 == 0 else -0.08) for i in range(50)}
mixed = bootstrap_delta(mixed_a, mixed_b, n_resamples=800, seed=3)
print("\n  " + format_delta(mixed, "Δ nDCG@10"))

print("\n" + leaderboard({
    "dense":          {"recall@10": 0.71, "ndcg@10": 0.58, "mrr@10": 0.61},
    "hybrid":         {"recall@10": 0.83, "ndcg@10": 0.64, "mrr@10": 0.67},
    "hybrid+rerank":  {"recall@10": 0.83, "ndcg@10": 0.72, "mrr@10": 0.75},
}, ks=(10,)))

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1)
print("all metric tests passed")
