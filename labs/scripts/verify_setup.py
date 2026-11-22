#!/usr/bin/env python3
"""
The pre-work check. Send this to attendees before the conference.

Prints SETUP OK, or exactly what is missing and how to fix it. Nothing here
needs the network if prepare_data.py has already been run.

    python scripts/verify_setup.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROBLEMS: list[str] = []


def ok(label: str, detail: str = "") -> None:
    print(f"  OK    {label:<26} {detail}")


def bad(label: str, fix: str) -> None:
    print(f"  MISS  {label:<26} -> {fix}")
    PROBLEMS.append(label)


def main() -> int:
    print(f"\npython   {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")

    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        bad("python >= 3.10", f"you have {major}.{minor}; install 3.12")
    elif (major, minor) != (3, 12):
        print(f"  WARN  python {major}.{minor}: the pinned versions are tested on 3.12 only")
    else:
        ok("python version", f"{major}.{minor}")

    print("\npackages")
    for mod, fix in [
        ("numpy", "pip install numpy"),
        ("faiss", "pip install faiss-cpu"),
        ("sentence_transformers", "pip install sentence-transformers"),
        ("rank_bm25", "pip install rank-bm25"),
        ("matplotlib", "pip install matplotlib"),
    ]:
        try:
            m = __import__(mod)
            ok(mod, getattr(m, "__version__", ""))
        except ImportError:
            bad(mod, fix)

    print("\ndata")
    try:
        from wslib import corpus as corpus_mod
        from wslib import embed

        data = Path(embed.DATA_DIR)
        if not (data / "scifact" / "corpus.jsonl").exists():
            bad("scifact corpus", "python scripts/prepare_data.py")
        else:
            c = corpus_mod.load("scifact")
            ok("scifact corpus", f"{len(c):,} docs, {len(c.judged_queries())} queries")

            npys = list(data.glob("emb_*.npy"))
            if not npys:
                bad("cached embeddings", "python scripts/prepare_data.py")
            else:
                total = sum(p.stat().st_size for p in npys)
                ok("cached embeddings", f"{len(npys)} files, {total / 1e6:.1f} MB")
    except Exception as exc:  # noqa: BLE001
        bad("wslib import", f"{type(exc).__name__}: {exc}")

    print("\nend-to-end")
    if PROBLEMS:
        print("  skipped (fix the above first)")
    else:
        try:
            from wslib import corpus as corpus_mod
            from wslib import embed, indexes, metrics

            c = corpus_mod.load("scifact")
            docs = embed.cached_corpus_embeddings(c)
            qids, qv = embed.cached_query_embeddings(c)
            run = indexes.ExactIndex(docs).search(qv, 10).to_run(c, qids)
            s = metrics.evaluate(run, c.qrels, ks=(10,))
            r = s["recall@10"]
            if r < 0.7:
                bad("exact search recall", f"got {r:.3f}, expected ~0.79")
            else:
                ok("exact search", f"recall@10 {r:.4f}, nDCG@10 {s['ndcg@10']:.4f}")
        except Exception as exc:  # noqa: BLE001
            bad("search smoke test", f"{type(exc).__name__}: {exc}")

    print()
    if PROBLEMS:
        print(f"{len(PROBLEMS)} problem(s): {', '.join(PROBLEMS)}")
        print("\nIf you cannot resolve these, use the Colab notebook instead —")
        print("it needs nothing installed locally.")
        return 1

    print("SETUP OK — you're ready for the workshop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
