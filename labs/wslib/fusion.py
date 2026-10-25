"""
Combining two or more ranked lists.

The whole argument of Module 03 lives in the difference between these two
functions:

    linear_blend  fuses by SCORE — needs the scores to be comparable, and they
                  are not. It has a weight you must fit, and refit whenever the
                  corpus, tokenizer, or embedding model changes.

    rrf           fuses by RANK — never looks at a score, so nothing to fit and
                  nothing to break when a score scale moves.

Both take retriever output as (ids, scores) pairs, so a faiss index and a BM25
index drop into either interchangeably.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Result = Tuple[np.ndarray, np.ndarray]      # (ids, scores)


# ── rank-based ─────────────────────────────────────────────────────────────


def rrf(results: Sequence[Result], k_const: int = 60, top_k: Optional[int] = None) -> List[int]:
    """Reciprocal Rank Fusion.

        RRF(d) = sum over retrievers of  1 / (k_const + rank(d))

    rank is 1-based within each list. A document missing from a list simply
    contributes nothing from it — no imputed score, no decision to get wrong.

    k_const = 60 is the constant from Cormack et al. and it works remarkably well
    across corpora. It damps the head of each list: rank 1 is worth only ~1.6%
    more than rank 2, so one over-confident retriever cannot bulldoze the other.
    """
    scores: Dict[int, float] = defaultdict(float)
    for ids, _ in results:
        for rank, doc in enumerate(np.asarray(ids).tolist(), start=1):
            scores[int(doc)] += 1.0 / (k_const + rank)
    ranked = sorted(scores, key=lambda d: (-scores[d], d))
    return ranked[:top_k] if top_k else ranked


def rrf_with_scores(results: Sequence[Result], k_const: int = 60, top_k=None):
    """Same as `rrf` but returns (ids, fused_scores) for inspection."""
    scores: Dict[int, float] = defaultdict(float)
    for ids, _ in results:
        for rank, doc in enumerate(np.asarray(ids).tolist(), start=1):
            scores[int(doc)] += 1.0 / (k_const + rank)
    ranked = sorted(scores, key=lambda d: (-scores[d], d))
    if top_k:
        ranked = ranked[:top_k]
    return np.array(ranked, dtype=np.int64), np.array([scores[d] for d in ranked])


# ── score-based ────────────────────────────────────────────────────────────


def _minmax(x: np.ndarray) -> np.ndarray:
    """Per-query min-max to [0, 1]. The usual 'fix' for incomparable scores.

    It helps, and it is still fragile: the mapping depends on this query's
    particular maximum, so one outlier rescales everything, and a retriever that
    returns a flat list of near-identical scores gets stretched to look decisive.
    """
    if len(x) == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def linear_blend(
    results: Sequence[Result],
    weights: Optional[Sequence[float]] = None,
    normalize: bool = False,
    top_k: Optional[int] = None,
) -> List[int]:
    """Weighted sum of scores.

    Two choices this forces on you, both of which RRF avoids entirely:

    1. **Weights.** With raw scores the weight is not a preference, it is a
       calibration constant fitted to one corpus and one pair of retrievers.
    2. **Missing documents.** A document one retriever never returned gets 0
       here. For BM25 that is defensible (no shared terms). For cosine it is a
       fiction — the true similarity is simply unknown.
    """
    n = len(results)
    w = list(weights) if weights is not None else [1.0 / n] * n

    scores: Dict[int, float] = defaultdict(float)
    for (ids, raw), weight in zip(results, w):
        raw = np.asarray(raw, dtype=np.float64)
        vals = _minmax(raw) if normalize else raw
        for doc, s in zip(np.asarray(ids).tolist(), vals.tolist()):
            scores[int(doc)] += weight * s

    ranked = sorted(scores, key=lambda d: (-scores[d], d))
    return ranked[:top_k] if top_k else ranked


# ── plumbing ───────────────────────────────────────────────────────────────


def scale_scores(result: Result, factor: float) -> Result:
    """Multiply one retriever's scores. Rankings are untouched.

    This is the ×100 experiment: nothing about the retrieval quality changed,
    only the units. Any fusion method whose output moves was depending on
    something it should not have been.
    """
    ids, scores = result
    return ids, np.asarray(scores, dtype=np.float64) * factor


def to_run(fused_per_query: Dict[str, Sequence[int]], corpus, top_k: int = 10) -> dict:
    """Row indices -> document ids, in the {qid: [doc_id, ...]} metrics format."""
    return {
        qid: [corpus.doc_ids[i] for i in list(ids)[:top_k]]
        for qid, ids in fused_per_query.items()
    }
