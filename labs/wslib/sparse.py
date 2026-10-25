"""
BM25 keyword retrieval.

Deliberately the plainest possible implementation: lowercase, split on word
characters, no stemming, no stopword list. Every one of those would help a
little, and every one would be a knob distracting from the point of Lab 3 —
which is that keyword and dense retrieval fail on *different* queries.

Returns the same (ids, scores) shape as the faiss wrappers in `indexes`, so the
two retrievers are interchangeable in a fusion function.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase and keep alphanumeric runs. That's it."""
    return _WORD.findall(text.lower())


class BM25Index:
    """Okapi BM25 over an in-memory corpus.

    k1 controls term saturation (how fast repeated terms stop helping), b
    controls length normalisation. The defaults have survived thirty years of
    retrieval research; changing them is rarely where your wins are.
    """

    name = "bm25"

    def __init__(
        self,
        texts: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
        field_name: str = "bm25",
    ):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install rank-bm25") from exc

        self.k1 = k1
        self.b = b
        self.name = field_name
        self.n = len(texts)
        self._tokenized = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized, k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        """Raw BM25 score for every document. Unbounded and corpus-dependent —
        which is exactly why blending these with cosine needs calibration."""
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)

    def search(self, query: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k (row indices, scores), best first.

        Documents scoring exactly 0 share no query terms at all. We keep them out
        rather than padding the list with noise — a keyword retriever returning
        fewer than k results is normal and honest.
        """
        s = self.scores(query)
        k = min(k, self.n)
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        keep = s[idx] > 0
        return idx[keep], s[idx][keep]

    def search_many(self, queries: Sequence[str], k: int = 10):
        return [self.search(q, k) for q in queries]

    def describe(self) -> str:
        avg = sum(len(t) for t in self._tokenized) / max(self.n, 1)
        vocab = len({t for doc in self._tokenized for t in doc})
        return (
            f"bm25: {self.n:,} docs · k1={self.k1} b={self.b} · "
            f"{vocab:,} term vocabulary · {avg:.0f} tokens/doc"
        )


def dense_search(index, query_vec: np.ndarray, k: int = 10):
    """Adapter so a faiss index has the same call shape as BM25Index.search."""
    res = index.search(query_vec[None, :] if query_vec.ndim == 1 else query_vec, k)
    return res.ids[0], res.scores[0]
