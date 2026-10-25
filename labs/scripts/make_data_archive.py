#!/usr/bin/env python3
"""
Package the data cache so Colab users skip the slow prepare_data step.

Upload the result as a GitHub Release asset, then paste its download URL into
DATA_URL in scripts/workshop_config.py and rebuild the notebooks. Attendees then
get the corpora, all embeddings, the pooling variants and the precomputed
cross-encoder scores in one ~30 second download.

    python scripts/make_data_archive.py
    python scripts/make_data_archive.py --out /tmp/workshop-data.tar.gz

The archive is rooted at "data/" so it untars in place from the labs directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# What Lab 1-4 need, and nothing else. Raw BEIR jsonl is included so attendees
# never hit the network for the corpora either.
INCLUDE_SUFFIXES = (".npy", ".npz", ".jsonl", ".tsv")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "workshop-data.tar.gz"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not DATA.exists():
        print(f"no data directory at {DATA}", file=sys.stderr)
        print("run: python scripts/prepare_data.py --poolings", file=sys.stderr)
        print("     python scripts/prepare_data.py --dataset nfcorpus --all",
              file=sys.stderr)
        return 1

    members = sorted(
        p for p in DATA.rglob("*")
        if p.is_file() and p.suffix in INCLUDE_SUFFIXES
    )
    if not members:
        print("nothing to archive — is the cache built?", file=sys.stderr)
        return 1

    total = sum(p.stat().st_size for p in members)
    print(f"{len(members)} files, {human(total)} uncompressed\n")
    for p in members:
        print(f"  {human(p.stat().st_size):>10}  {p.relative_to(ROOT)}")

    # Warn about the pieces the labs actually depend on
    needed = {
        "scifact corpus embeddings": "emb_corpus-scifact",
        "scifact queries": "emb_queries-scifact",
        "scifact poolings": "pool_corpus-scifact",
        "scifact MRL": "static-retrieval-mrl",
        "nfcorpus embeddings": "emb_corpus-nfcorpus",
        "nfcorpus chunks": "emb_chunks-nfcorpus",
        "nfcorpus rerank scores": "rerank_nfcorpus",
    }
    names = " ".join(p.name for p in members)
    missing = [label for label, frag in needed.items() if frag not in names]
    print()
    if missing:
        print("MISSING — some labs will fall back to computing these live:")
        for m in missing:
            print(f"  - {m}")
        print("\nBuild everything with:")
        print("  python scripts/prepare_data.py --poolings")
        print("  python scripts/prepare_data.py --dataset nfcorpus --all")
    else:
        print("all four labs' caches present")

    if args.dry_run:
        print("\n--dry-run: not writing an archive")
        return 0

    out = Path(args.out)
    print(f"\nwriting {out} ...")
    with tarfile.open(out, "w:gz") as tar:
        for p in members:
            tar.add(p, arcname=str(p.relative_to(ROOT)))

    size = out.stat().st_size
    print(f"  {human(size)} compressed ({size / total * 100:.0f}% of original)")
    if size > 1_900_000_000:
        print("  WARNING: over GitHub's 2 GB per-asset limit")

    print("\nnext steps:")
    print(f"  1. gh release create workshop-data '{out}' --notes 'lab data cache'")
    print("  2. copy the asset's download URL into DATA_URL in")
    print("     scripts/workshop_config.py")
    print("  3. python scripts/build_notebooks.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
