"""
wslib — shared plumbing for the four workshop labs.

One corpus, one embedding cache, one set of metrics, shared by every lab, so the
recall you measure in Lab 2 is directly comparable to the recall you improve in
Lab 4.

    from wslib import chunking, corpus, embed, fusion, indexes, metrics
    from wslib import rerank, sparse, timing

Nothing here is clever. It is deliberately readable, because attendees will open
it when they want to know what a number means.
"""

from __future__ import annotations

import os

# ── OpenMP: these MUST be set before faiss or torch loads ──────────────────
#
# pip-installed faiss-cpu and torch each bundle their own copy of libomp. A
# process that uses both — which Lab 4 does, building a faiss index and then
# running a cross-encoder — hits:
#
#     OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
#     already initialized.
#
# KMP_DUPLICATE_LIB_OK silences that message but NOT the consequence: measured
# on macOS + Python 3.12, the process then either segfaults (exit 139) or
# deadlocks inside cross-encoder inference. The warning's "may cause crashes" is
# not theoretical.
#
# What actually fixes it is OMP_NUM_THREADS=1. Measured on this stack:
#
#     OMP_NUM_THREADS unset  -> hangs
#                         1  -> works
#                         2  -> hangs
#                         4  -> hangs
#
# and setting thread counts through the APIs instead (torch.set_num_threads,
# faiss.omp_set_num_threads) is too late — the runtimes are already initialised.
# It has to be the environment, before the first import.
#
# The cost is smaller than it looks. Cross-encoder inference still runs at
# ~1800 pairs/sec single-threaded, and single-threaded faiss is what Lab 2 wants
# anyway: the QPS numbers it reports claim one thread, and now that is true.
#
# The genuinely clean fix is one shared libomp, which means conda-forge instead
# of pip — not worth asking 50 laptops to switch package manager.
#
# Set either variable yourself before importing wslib to override.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from . import (  # noqa: E402,F401
    chunking, corpus, embed, fusion, indexes, metrics, rerank, sparse, timing,
)

__all__ = [
    "chunking", "corpus", "embed", "fusion", "indexes", "metrics", "rerank",
    "sparse", "timing",
]
__version__ = "0.1.0"
