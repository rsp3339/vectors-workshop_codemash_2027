"""
Chunking, and the bookkeeping that makes chunked retrieval measurable.

The bookkeeping is the part people get wrong. Once you index chunks, your search
returns chunks — but your relevance judgements are about *documents*. So every
chunked pipeline needs a parent map and a pooling rule, and the pooling rule is a
real retrieval decision, not plumbing:

    max pooling  — a document is as good as its best chunk. Standard, and what we
                   use here. Rewards one strong passage.
    sum pooling  — rewards documents with several decent chunks. Biases toward
                   long documents, which is usually not what you want.

`chunk_corpus` returns the chunk texts plus a parent index, and `pool_to_parents`
collapses a chunk ranking back to a document ranking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Deliberately simple: split after . ! ? when followed by whitespace.
# A real system would use a proper sentence splitter; this keeps the lab focused.
_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunking:
    """The result of chunking a corpus."""

    texts: List[str]
    parent: np.ndarray            # parent[i] = row index of chunk i's document
    strategy: str
    size: int
    overlap: int

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def broken_fraction(self) -> float:
        """Share of chunks that do not end on a sentence boundary.

        A chunk cut mid-sentence gets embedded as half an idea, and tends not to
        be retrieved for either half.
        """
        if not self.texts:
            return 0.0
        broken = sum(1 for t in self.texts if not re.search(r"[.!?]\s*$", t.strip()))
        return broken / len(self.texts)

    def summary(self) -> str:
        lens = [len(t) for t in self.texts]
        return (
            f"{self.strategy}(size={self.size}, overlap={self.overlap}): "
            f"{len(self):,} chunks from {len(set(self.parent.tolist())):,} docs · "
            f"{np.mean(lens):.0f} chars avg · "
            f"{self.broken_fraction * 100:.0f}% cut mid-sentence"
        )

    def key(self) -> str:
        return f"{self.strategy}-{self.size}-{self.overlap}"


def chunk_text(text: str, strategy: str, size: int, overlap: int = 0) -> List[str]:
    """Split one document.

    fixed    — hard character windows. Fast, and cuts wherever it lands.
    sentence — pack whole sentences up to the budget. A sentence longer than the
               budget becomes its own oversized chunk rather than being cut.
    """
    text = text.strip()
    if not text:
        return []

    if strategy == "whole":
        return [text]

    if strategy == "fixed":
        step = max(1, size - overlap)
        return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]

    if strategy == "sentence":
        out: List[str] = []
        cur = ""
        for s in _SENT.split(text):
            s = s.strip()
            if not s:
                continue
            if cur and len(cur) + 1 + len(s) > size:
                out.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            out.append(cur)
        return out

    raise ValueError(f"unknown strategy {strategy!r}; use whole, fixed, or sentence")


def chunk_corpus(
    texts: Sequence[str],
    strategy: str = "sentence",
    size: int = 512,
    overlap: int = 0,
) -> Chunking:
    """Chunk every document, keeping track of which document each chunk came from."""
    chunk_texts: List[str] = []
    parent: List[int] = []
    for doc_i, text in enumerate(texts):
        pieces = chunk_text(text, strategy, size, overlap)
        if not pieces:                      # never drop a document entirely
            pieces = [text.strip() or " "]
        chunk_texts.extend(pieces)
        parent.extend([doc_i] * len(pieces))

    return Chunking(
        texts=chunk_texts,
        parent=np.asarray(parent, dtype=np.int64),
        strategy=strategy,
        size=size,
        overlap=overlap,
    )


def pool_to_parents(
    chunk_ids: Sequence[int],
    parent: np.ndarray,
    top_k: int | None = None,
) -> List[int]:
    """Collapse a ranked chunk list into a ranked document list (max pooling).

    Keeps each document at the rank of its best chunk, dropping later duplicates.
    Order in, order out — so this works on any chunk ranking, fused or not.
    """
    seen: Dict[int, None] = {}
    for cid in chunk_ids:
        doc = int(parent[int(cid)])
        if doc not in seen:
            seen[doc] = None
            if top_k and len(seen) >= top_k:
                break
    return list(seen)


def chunk_lookup(chunking: Chunking, chunk_id: int) -> Tuple[int, str]:
    """(parent row index, chunk text) — for showing what the model actually saw."""
    return int(chunking.parent[chunk_id]), chunking.texts[chunk_id]
