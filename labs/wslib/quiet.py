"""
Suppressing faiss's C++ chatter.

faiss writes some warnings with printf from C++, so `contextlib.redirect_stdout`
does not catch them — the write happens below Python. Training an IVF-PQ index
with 8-bit codebooks on a small corpus emits one "please provide at least N
training points" warning per sub-quantizer, which is 48 identical lines in the
middle of a lab.

The warning is real but expected here: with 5k documents there genuinely aren't
enough points to train 256 centroids per sub-space as well as faiss would like.
We say that once, in English, instead of 48 times in C++.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_native_output(streams=("stdout", "stderr")):
    """Silence writes at the file-descriptor level, C++ included.

    Only use this around calls whose output you have already understood — it
    hides genuine errors too.
    """
    saved = {}
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        for name in streams:
            stream = getattr(sys, name)
            try:
                stream.flush()
            except (ValueError, AttributeError):
                pass
            fd = 1 if name == "stdout" else 2
            saved[name] = os.dup(fd)
            os.dup2(devnull, fd)
        yield
    finally:
        for name, old in saved.items():
            fd = 1 if name == "stdout" else 2
            try:
                getattr(sys, name).flush()
            except (ValueError, AttributeError):
                pass
            os.dup2(old, fd)
            os.close(old)
        os.close(devnull)
