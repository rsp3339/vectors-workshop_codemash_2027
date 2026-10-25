"""
Embedding with an on-disk cache.

The point of the cache: nobody should watch a progress bar at the top of a lab.
`prepare_data.py` writes the corpus embeddings once, and every lab loads them
from a .npy in well under a second.

Vectors are L2-normalised on the way out. That is a deliberate teaching choice
as well as a practical one — with unit vectors, inner product IS cosine, so
faiss's IndexFlatIP gives exact cosine ranking and Module 01's identity holds
everywhere downstream.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Small, CPU-friendly, and the standard baseline everyone can reproduce.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Matryoshka-capable, for Lab 1's truncation exercise. Chosen over
# nomic-embed-text-v1.5 because it needs no trust_remote_code and no einops
# (some corporate environments block the former), it is a fraction of the size,
# and being a static model it encodes the whole corpus in ~9s instead of ~72s.
MRL_MODEL = "sentence-transformers/static-retrieval-mrl-en-v1"


def normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Rows that are all zero are left alone."""
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _cache_path(tag: str, model: str, dim: Optional[int], data_dir: Path) -> Path:
    key = f"{tag}|{model}|{dim}"
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    safe = model.split("/")[-1]
    return data_dir / f"emb_{tag}_{safe}_{h}.npy"


def load_model(name: str = DEFAULT_MODEL):
    """Load a SentenceTransformer on CPU.

    Imported lazily so that anything not needing a model — the metrics tests,
    for instance — doesn't pay the multi-second torch import.
    """
    from sentence_transformers import SentenceTransformer

    trust = "nomic" in name          # nomic ships custom modelling code
    return SentenceTransformer(name, device="cpu", trust_remote_code=trust)


def encode(
    texts: Sequence[str],
    model=None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    dim: Optional[int] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed texts and return unit-norm float32 vectors.

    `dim` truncates before normalising, which is the Matryoshka operation from
    Lab 1 — meaningful on an MRL-trained model, destructive on anything else.
    """
    if model is None:
        model = load_model(model_name)

    vecs = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,     # we normalise below, after truncation
        show_progress_bar=show_progress,
    )
    vecs = np.asarray(vecs, dtype=np.float32)
    if dim is not None:
        if dim > vecs.shape[1]:
            raise ValueError(f"dim={dim} exceeds model width {vecs.shape[1]}")
        vecs = vecs[:, :dim]
    return normalize(vecs)


def cached_corpus_embeddings(
    corpus,
    model_name: str = DEFAULT_MODEL,
    dim: Optional[int] = None,
    data_dir: Path | str = DATA_DIR,
    rebuild: bool = False,
) -> np.ndarray:
    """Embeddings for corpus.texts, from disk if we've done this before."""
    data_dir = Path(data_dir)
    path = _cache_path(f"corpus-{corpus.name}", model_name, dim, data_dir)

    if path.exists() and not rebuild:
        vecs = np.load(path)
        if len(vecs) == len(corpus):
            return vecs
        print(f"cache size mismatch at {path.name}, re-encoding", file=sys.stderr)

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"encoding {len(corpus):,} documents with {model_name} ...", file=sys.stderr)
    vecs = encode(corpus.texts, model_name=model_name, dim=dim)
    np.save(path, vecs)
    print(f"  cached to {path.name}", file=sys.stderr)
    return vecs


def cached_query_embeddings(
    corpus,
    model_name: str = DEFAULT_MODEL,
    dim: Optional[int] = None,
    data_dir: Path | str = DATA_DIR,
    rebuild: bool = False,
):
    """Returns (query_ids, vectors) for the judged queries, order aligned."""
    data_dir = Path(data_dir)
    jq = corpus.judged_queries()
    qids: List[str] = sorted(jq)
    path = _cache_path(f"queries-{corpus.name}", model_name, dim, data_dir)

    if path.exists() and not rebuild:
        vecs = np.load(path)
        if len(vecs) == len(qids):
            return qids, vecs

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"encoding {len(qids)} queries ...", file=sys.stderr)
    vecs = encode([jq[q] for q in qids], model_name=model_name, dim=dim,
                  show_progress=False)
    np.save(path, vecs)
    return qids, vecs


def cached_chunk_embeddings(
    corpus,
    chunking,
    model_name: str = DEFAULT_MODEL,
    dim: Optional[int] = None,
    data_dir: Path | str = DATA_DIR,
    rebuild: bool = False,
) -> np.ndarray:
    """Embeddings for a Chunking, cached by its (strategy, size, overlap) key.

    Chunking a corpus multiplies the number of texts to embed, so this is the
    cache that keeps Lab 4 from opening with a two-minute progress bar.
    """
    data_dir = Path(data_dir)
    tag = f"chunks-{corpus.name}-{chunking.key()}"
    path = _cache_path(tag, model_name, dim, data_dir)

    if path.exists() and not rebuild:
        vecs = np.load(path)
        if len(vecs) == len(chunking):
            return vecs
        print(f"cache size mismatch at {path.name}, re-encoding", file=sys.stderr)

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"encoding {len(chunking):,} chunks ({chunking.key()}) ...", file=sys.stderr)
    vecs = encode(chunking.texts, model_name=model_name, dim=dim)
    np.save(path, vecs)
    print(f"  cached to {path.name}", file=sys.stderr)
    return vecs


# ── pooling ────────────────────────────────────────────────────────────────

POOLINGS = ("mean", "cls", "max")


def encode_poolings(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    poolings: Sequence[str] = POOLINGS,
    batch_size: int = 32,
    show_progress: bool = True,
):
    """Return {pooling: unit-norm array} from a SINGLE forward pass.

    Token embeddings are the expensive part, so we compute every pooling from the
    same batch and throw the tokens away — running the model three times to
    compare three poolings would be silly.

    A note that matters for Lab 1: most sentence-transformers (all-MiniLM-L6-v2
    included) end in a Normalize module, so `encode()` returns unit vectors
    whatever you pass for normalize_embeddings. Mean-pooling the token
    embeddings and normalising reproduces `encode()` exactly — verified.
    """
    model = load_model(model_name)
    acc = {p: [] for p in poolings}

    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        toks = model.encode(batch, output_value="token_embeddings",
                            convert_to_numpy=False, show_progress_bar=False)
        for t in toks:
            arr = t.detach().cpu().numpy()
            if "mean" in acc:
                acc["mean"].append(arr.mean(axis=0))
            if "cls" in acc:
                acc["cls"].append(arr[0])            # the [CLS] position
            if "max" in acc:
                acc["max"].append(arr.max(axis=0))
        if show_progress and start % (batch_size * 20) == 0:
            print(f"    {start:,}/{len(texts):,}", file=sys.stderr)

    return {p: normalize(np.stack(v)) for p, v in acc.items()}


def cached_poolings(
    corpus,
    which: str = "corpus",
    model_name: str = DEFAULT_MODEL,
    poolings: Sequence[str] = POOLINGS,
    data_dir: Path | str = DATA_DIR,
    rebuild: bool = False,
):
    """encode_poolings over the corpus or the judged queries, cached to .npz."""
    data_dir = Path(data_dir)
    safe = model_name.split("/")[-1]
    path = data_dir / f"pool_{which}-{corpus.name}_{safe}.npz"

    if which == "corpus":
        texts = corpus.texts
    else:
        jq = corpus.judged_queries()
        texts = [jq[q] for q in sorted(jq)]

    if path.exists() and not rebuild:
        blob = np.load(path)
        if all(p in blob for p in poolings) and len(blob[poolings[0]]) == len(texts):
            return {p: blob[p] for p in poolings}

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"computing {len(poolings)} poolings over {len(texts):,} {which} texts ...",
          file=sys.stderr)
    out = encode_poolings(texts, model_name=model_name, poolings=poolings)
    np.savez_compressed(path, **out)
    print(f"  cached to {path.name}", file=sys.stderr)
    return out
