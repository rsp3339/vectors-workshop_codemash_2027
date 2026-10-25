"""
Cross-encoder reranking.

A bi-encoder embeds the query and the document separately, so it can precompute
the whole corpus. A cross-encoder reads the pair *together* with full attention
across both — much better judgement, and nothing cacheable.

That asymmetry is the whole design: retrieve wide and cheap, then re-score a
narrow candidate set expensively. Reranking 50 candidates is affordable;
reranking a million documents is not a thing you do.

Scores are cached to .npy keyed by the exact (model, query, candidate) set, so
re-running a lab cell is instant.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ~90 MB, CPU-friendly, the standard MS MARCO cross-encoder.
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """Relevance score per (query, text) pair. Higher is better.

        These are logits, not probabilities, and not comparable across queries —
        which is fine, because reranking only ever sorts within one query.
        """
        if not texts:
            return np.zeros(0, dtype=np.float32)
        model = self._load()
        pairs = [(query, t) for t in texts]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)

    def rerank(
        self, query: str, candidate_ids: Sequence[int], texts: Sequence[str]
    ) -> List[int]:
        """Reorder candidate ids by cross-encoder score, best first."""
        if len(candidate_ids) == 0:
            return []
        s = self.score(query, [texts[i] for i in candidate_ids])
        order = np.argsort(-s)
        return [int(candidate_ids[i]) for i in order]

    # ── batch over a whole query set, with a cache ─────────────────────────

    def rerank_all(
        self,
        queries: Sequence[str],
        candidates: Sequence[Sequence[int]],
        texts: Sequence[str],
        cache_tag: Optional[str] = None,
        data_dir: Path | str = DATA_DIR,
        verbose: bool = True,
    ) -> List[List[int]]:
        """Rerank every query's candidate list. Cached by content hash.

        The cache key covers the model, the queries, and the exact candidate ids,
        so changing the retrieval config correctly invalidates it.
        """
        data_dir = Path(data_dir)
        path = None
        if cache_tag:
            h = hashlib.sha1()
            h.update(self.model_name.encode())
            for q, cand in zip(queries, candidates):
                h.update(q.encode())
                h.update(np.asarray(cand, dtype=np.int64).tobytes())
            path = data_dir / f"rerank_{cache_tag}_{h.hexdigest()[:12]}.npz"
            if path.exists():
                blob = np.load(path, allow_pickle=True)
                if verbose:
                    print(f"  reranker cache hit ({path.name})")
                return [list(map(int, row)) for row in blob["order"]]

        total = sum(len(c) for c in candidates)
        if verbose:
            print(f"  scoring {total:,} (query, document) pairs on CPU ...")

        out: List[List[int]] = []
        for q, cand in zip(queries, candidates):
            out.append(self.rerank(q, list(cand), texts))

        if path is not None:
            data_dir.mkdir(parents=True, exist_ok=True)
            padded = np.full((len(out), max((len(r) for r in out), default=0)), -1,
                             dtype=np.int64)
            for i, row in enumerate(out):
                padded[i, :len(row)] = row
            np.savez_compressed(path, order=padded)
            if verbose:
                print(f"  cached to {path.name}")
            return [[int(x) for x in row if x >= 0] for row in padded]

        return out
