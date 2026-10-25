"""
Thin wrappers over faiss, so the labs read like retrieval rather than like faiss.

One deliberate choice worth understanding, because Lab 2 depends on it:

    Every vector in these labs is L2-normalised. For unit vectors, ranking by
    inner product and ranking by L2 distance are IDENTICAL, because
        ||u - v||^2 = 2 - 2*(u . v)
    which is monotonically decreasing in the inner product. That is the Module 01
    identity, and it is why we can build ground truth with IndexFlatIP (exact
    cosine) and compare it against an L2-metric HNSW graph without cheating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .quiet import suppress_native_output


def _faiss():
    """Import faiss lazily and fail with something actionable."""
    try:
        import faiss  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "faiss-cpu is required for this lab.\n"
            "  pip install faiss-cpu\n"
            "If the wheel refuses to install on Windows, tell the instructor — "
            "there is an hnswlib fallback."
        ) from exc
    return faiss


# ── results ────────────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """Row indices and scores, best-first, shape (n_queries, k)."""

    ids: np.ndarray
    scores: np.ndarray

    def to_run(self, corpus, query_ids: Sequence[str]) -> dict:
        """Convert to the {query_id: [doc_id, ...]} form the metrics expect."""
        return {
            qid: [corpus.doc_ids[i] for i in row if i >= 0]
            for qid, row in zip(query_ids, self.ids)
        }


# ── base ───────────────────────────────────────────────────────────────────


class BaseIndex:
    name = "base"

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.n, self.d = self.vectors.shape
        self._index = None

    def search(self, queries: np.ndarray, k: int = 10) -> SearchResult:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        scores, ids = self._index.search(q, k)
        return SearchResult(ids=ids, scores=scores)

    def search_one(self, query: np.ndarray, k: int = 10) -> SearchResult:
        return self.search(query[None, :] if query.ndim == 1 else query, k)

    @property
    def bytes_per_vector(self) -> float:
        raise NotImplementedError

    @property
    def total_bytes(self) -> float:
        return self.bytes_per_vector * self.n

    def describe(self) -> str:
        return f"{self.name}: {self.n:,} x {self.d}d"


# ── exact: the ground truth ────────────────────────────────────────────────


class ExactIndex(BaseIndex):
    """Brute-force inner product. On unit vectors this is exact cosine.

    Slow by design — this is what every approximate index gets scored against.
    """

    name = "exact"

    def __init__(self, vectors: np.ndarray):
        super().__init__(vectors)
        faiss = _faiss()
        self._index = faiss.IndexFlatIP(self.d)
        self._index.add(self.vectors)

    @property
    def bytes_per_vector(self) -> float:
        return self.d * 4


# ── HNSW ───────────────────────────────────────────────────────────────────


class HNSWIndex(BaseIndex):
    """Graph index. `ef` is settable after build — that's the whole point."""

    name = "hnsw"

    def __init__(self, vectors: np.ndarray, M: int = 16, ef_construction: int = 200):
        super().__init__(vectors)
        faiss = _faiss()
        self.M = M
        self.ef_construction = ef_construction

        # L2 metric: identical ranking to cosine on unit vectors (see module docstring)
        self._index = faiss.IndexHNSWFlat(self.d, M)
        self._index.hnsw.efConstruction = ef_construction
        self._index.add(self.vectors)
        self.ef = 64

    @property
    def ef(self) -> int:
        return self._index.hnsw.efSearch

    @ef.setter
    def ef(self, value: int) -> None:
        """The one knob you can turn in production without a rebuild."""
        self._index.hnsw.efSearch = int(value)

    @property
    def bytes_per_vector(self) -> float:
        # stored vector + bidirectional links (32-bit ids), doubled at layer 0
        return self.d * 4 + self.M * 2 * 4

    def describe(self) -> str:
        return (
            f"hnsw: M={self.M} efConstruction={self.ef_construction} "
            f"efSearch={self.ef}"
        )


# ── IVF family ─────────────────────────────────────────────────────────────


class IVFFlatIndex(BaseIndex):
    """Partitioned, uncompressed. Isolates the cost of partitioning alone."""

    name = "ivf_flat"

    def __init__(self, vectors: np.ndarray, nlist: Optional[int] = None, seed: int = 0):
        super().__init__(vectors)
        faiss = _faiss()
        self.nlist = int(nlist or max(4, round(np.sqrt(self.n))))

        quantizer = faiss.IndexFlatL2(self.d)
        self._index = faiss.IndexIVFFlat(quantizer, self.d, self.nlist)
        self._index.cp.seed = seed
        with suppress_native_output():
            self._index.train(self.vectors)
        self._index.add(self.vectors)
        self.nprobe = 8

    @property
    def nprobe(self) -> int:
        return self._index.nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        self._index.nprobe = max(1, min(int(value), self.nlist))

    @property
    def bytes_per_vector(self) -> float:
        return self.d * 4

    def describe(self) -> str:
        return f"ivf_flat: nlist={self.nlist} nprobe={self.nprobe}"


class IVFPQIndex(BaseIndex):
    """Partitioned AND compressed — the billion-scale combination.

    `m` sub-vectors, `nbits` bits each, so each vector costs m*nbits/8 bytes
    instead of d*4. m must divide d.
    """

    name = "ivf_pq"

    def __init__(
        self,
        vectors: np.ndarray,
        nlist: Optional[int] = None,
        m: Optional[int] = None,
        nbits: int = 8,
        seed: int = 0,
        verbose: bool = True,
    ):
        super().__init__(vectors)
        faiss = _faiss()

        self.nlist = int(nlist or max(4, round(np.sqrt(self.n))))
        self.m = int(m or _pick_m(self.d))
        self.nbits = nbits

        if self.d % self.m:
            raise ValueError(
                f"m={self.m} must divide the dimension d={self.d}; "
                f"try one of {_divisors(self.d)}"
            )
        # faiss needs at least 2**nbits training points per sub-quantizer
        if self.n < 2 ** nbits:
            raise ValueError(
                f"need at least {2 ** nbits} vectors to train {nbits}-bit codebooks, "
                f"got {self.n}"
            )

        quantizer = faiss.IndexFlatL2(self.d)
        self._index = faiss.IndexIVFPQ(quantizer, self.d, self.nlist, self.m, nbits)
        self._index.cp.seed = seed

        # faiss wants ~39 training points per centroid. With 2**nbits centroids
        # per sub-quantizer and a small corpus that's unreachable, and it says so
        # once per sub-quantizer — 48 identical C++ lines. Say it once instead.
        wanted = 39 * (2 ** nbits)
        self.undertrained = self.n < wanted
        if self.undertrained and verbose:
            print(
                f"  note: {self.n:,} vectors to train {2 ** nbits} centroids per "
                f"sub-quantizer (faiss would prefer {wanted:,}).\n"
                f"        Codebooks will be rougher than at production scale — "
                f"expect slightly worse recall."
            )
        with suppress_native_output():
            self._index.train(self.vectors)
        self._index.add(self.vectors)
        self.nprobe = 8

    @property
    def nprobe(self) -> int:
        return self._index.nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        self._index.nprobe = max(1, min(int(value), self.nlist))

    @property
    def bytes_per_vector(self) -> float:
        return self.m * self.nbits / 8

    @property
    def compression(self) -> float:
        return (self.d * 4) / self.bytes_per_vector

    def describe(self) -> str:
        return (
            f"ivf_pq: nlist={self.nlist} nprobe={self.nprobe} "
            f"m={self.m} nbits={self.nbits} ({self.compression:.0f}x smaller)"
        )


def _divisors(d: int) -> List[int]:
    return [x for x in (8, 12, 16, 24, 32, 48, 64, 96, 128, 192) if d % x == 0]


def _pick_m(d: int) -> int:
    """Default to ~8 dimensions per sub-vector, snapped to a divisor of d."""
    opts = _divisors(d)
    if not opts:
        raise ValueError(f"no convenient PQ sub-vector count divides d={d}")
    target = d / 8
    return min(opts, key=lambda x: abs(x - target))


# ── ground truth ───────────────────────────────────────────────────────────


def ground_truth(
    vectors: np.ndarray, queries: np.ndarray, k: int = 10
) -> np.ndarray:
    """Exact top-k row indices per query. The yardstick for every recall number.

    Computed in blocks so a 5K x 384 corpus doesn't become a memory question.
    """
    index = ExactIndex(vectors)
    out = []
    for start in range(0, len(queries), 256):
        out.append(index.search(queries[start:start + 256], k).ids)
    return np.vstack(out)


def recall_vs_truth(approx_ids: np.ndarray, truth_ids: np.ndarray, k: int) -> float:
    """Mean fraction of the exact top-k that an approximate search also found.

    This is index recall — measured against exact search, NOT against human
    judgements. Both matter and they are different numbers: an index can have
    perfect recall against a retriever that was itself looking in the wrong place.
    """
    hits = 0
    for approx, truth in zip(approx_ids, truth_ids):
        hits += len(set(approx[:k].tolist()) & set(truth[:k].tolist()))
    return hits / (len(truth_ids) * k)
