#!/usr/bin/env python3
"""
Run this ONCE before the workshop (or bake it into the Colab image).

Downloads the corpus, embeds it, and caches the vectors to labs/data/ so that no
lab ever opens with a progress bar. Everything after this runs offline.

    python scripts/prepare_data.py                 # scifact, default model
    python scripts/prepare_data.py --dataset nfcorpus
    python scripts/prepare_data.py --check         # verify the cache, don't build
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from wslib import corpus as corpus_mod
from wslib import embed
from wslib.timing import bytes_human, stopwatch

# Chunk configurations Lab 4 compares. "whole" is the uncut baseline.
CHUNK_CONFIGS = [
    ("sentence", 512, 0),      # respects sentence boundaries
    ("fixed", 256, 0),         # cuts wherever it lands — the cautionary one
]

# Must match Lab 4's constants or the rerank cache key won't hit.
RERANK_CANDIDATES = 100
RERANK_RRF_K = 60
RERANK_DEPTH = 30
RERANK_CHARS = 1200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact", choices=sorted(corpus_mod.DATASETS))
    ap.add_argument("--model", default=embed.DEFAULT_MODEL)
    ap.add_argument("--rebuild", action="store_true", help="ignore existing cache")
    ap.add_argument("--check", action="store_true", help="report status only")
    ap.add_argument("--chunks", action="store_true",
                    help="also build chunk embeddings (needed for Lab 4)")
    ap.add_argument("--rerank", action="store_true",
                    help="also precompute cross-encoder scores (needed for Lab 4)")
    ap.add_argument("--all", action="store_true",
                    help="--chunks, --rerank and --poolings")
    ap.add_argument("--poolings", action="store_true",
                    help="also build mean/cls/max poolings + the MRL model (Lab 1)")
    args = ap.parse_args()
    if args.all:
        args.chunks = args.rerank = args.poolings = True

    print(f"dataset : {args.dataset}")
    print(f"model   : {args.model}")
    print()

    with stopwatch("load corpus"):
        c = corpus_mod.load(args.dataset)
    print(f"  {c.summary()}")

    # a couple of real rows, so anyone reading the output knows what we indexed
    print("\nfirst two documents:")
    for i in range(min(2, len(c))):
        head = c.texts[i][:150].replace("\n", " ")
        print(f"  [{c.doc_ids[i]}] {head}...")

    jq = c.judged_queries()
    print("\nfirst two judged queries:")
    for qid in sorted(jq)[:2]:
        rel = [d for d, g in c.qrels[qid].items() if g > 0]
        print(f"  [{qid}] {jq[qid][:110]}")
        print(f"        relevant: {rel}")

    if args.check:
        print("\n--check: not building embeddings")
        return 0

    print()
    with stopwatch("embed corpus"):
        docs = embed.cached_corpus_embeddings(
            c, model_name=args.model, rebuild=args.rebuild)
    with stopwatch("embed queries"):
        qids, qvecs = embed.cached_query_embeddings(
            c, model_name=args.model, rebuild=args.rebuild)

    # sanity: unit norm is a load-bearing assumption everywhere downstream
    dn = np.linalg.norm(docs, axis=1)
    qn = np.linalg.norm(qvecs, axis=1)
    assert np.allclose(dn, 1.0, atol=1e-4), f"doc norms off: {dn.min()}..{dn.max()}"
    assert np.allclose(qn, 1.0, atol=1e-4), f"query norms off: {qn.min()}..{qn.max()}"

    if args.chunks:
        from wslib import chunking as chunk_mod
        print()
        for strategy, size, overlap in CHUNK_CONFIGS:
            ch = chunk_mod.chunk_corpus(c.texts, strategy, size, overlap)
            print(f"  {ch.summary()}")
            with stopwatch(f"  embed {ch.key()}"):
                embed.cached_chunk_embeddings(c, ch, model_name=args.model,
                                              rebuild=args.rebuild)

    if args.poolings:
        print()
        print("building pooling variants (Lab 1) — one forward pass, three poolings")
        with stopwatch("  corpus poolings"):
            embed.cached_poolings(c, "corpus", model_name=args.model,
                                  rebuild=args.rebuild)
        with stopwatch("  query poolings"):
            embed.cached_poolings(c, "queries", model_name=args.model,
                                  rebuild=args.rebuild)
        print()
        print(f"caching the MRL model ({embed.MRL_MODEL.split('/')[-1]}) for Lab 1")
        with stopwatch("  MRL corpus + queries"):
            embed.cached_corpus_embeddings(c, model_name=embed.MRL_MODEL,
                                           rebuild=args.rebuild)
            embed.cached_query_embeddings(c, model_name=embed.MRL_MODEL,
                                          rebuild=args.rebuild)

    if args.rerank:
        from wslib import fusion, rerank as rerank_mod, sparse
        from wslib.indexes import ExactIndex
        print()
        print("precomputing cross-encoder scores (slow, once — Lab 4 reads the cache)")
        jq_ = c.judged_queries()
        qtexts = [jq_[q] for q in qids]
        dense_i = ExactIndex(docs)
        bm25 = sparse.BM25Index(c.texts)
        d_res = [(dense_i.search(qvecs[i][None, :], RERANK_CANDIDATES).ids[0],
                  dense_i.search(qvecs[i][None, :], RERANK_CANDIDATES).scores[0])
                 for i in range(len(qids))]
        b_res = [bm25.search(t, RERANK_CANDIDATES) for t in qtexts]
        cands = [fusion.rrf([d_res[i], b_res[i]], k_const=RERANK_RRF_K)[:RERANK_DEPTH]
                 for i in range(len(qids))]
        texts = [t[:RERANK_CHARS] for t in c.texts]
        with stopwatch(f"  {len(qids) * RERANK_DEPTH:,} pairs"):
            rerank_mod.CrossEncoderReranker().rerank_all(
                qtexts, cands, texts,
                cache_tag=f"{args.dataset}-h{RERANK_RRF_K}-d{RERANK_DEPTH}-t{RERANK_CHARS}")

    print()
    print(f"corpus vectors : {docs.shape}  {bytes_human(docs.nbytes)}")
    print(f"query vectors  : {qvecs.shape}  {bytes_human(qvecs.nbytes)}")
    print(f"norms          : unit (checked)")
    print(f"cache dir      : {embed.DATA_DIR}")

    # quick end-to-end smoke test: does exact search find judged documents at all?
    from wslib.indexes import ExactIndex
    from wslib.metrics import evaluate

    idx = ExactIndex(docs)
    res = idx.search(qvecs, 10)
    run = res.to_run(c, qids)
    scores = evaluate(run, c.qrels, ks=(10,))
    print()
    print("exact-search sanity check (this is the ceiling for every index):")
    for k, v in scores.items():
        print(f"  {k:12} {v:.4f}")

    # Gate on nDCG, not recall. recall@10 is bounded above by 10/n_relevant, so on
    # a corpus like NFCorpus (~38 relevant documents per query) a *correct*
    # pipeline still reports recall@10 around 0.16. Judging that as a failure is
    # the exact metric-choice mistake Module 04 warns about.
    rel_per_q = [
        sum(1 for g in c.qrels[q].values() if g > 0) for q in c.judged_queries()
    ]
    avg_rel = sum(rel_per_q) / max(len(rel_per_q), 1)
    if avg_rel > 3:
        print(f"\n  note: {avg_rel:.1f} relevant documents per query, so recall@10 is")
        print(f"        capped near {min(1.0, 10 / avg_rel):.2f} by construction. Read nDCG here.")

    ndcg = scores.get("ndcg@10", 0.0)
    if ndcg < 0.15:
        print(f"\nWARNING: nDCG@10 {ndcg:.4f} is too low — something is misaligned.")
        return 1

    print("\nSETUP OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
