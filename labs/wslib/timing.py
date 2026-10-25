"""
Timing helpers.

Lab 2's whole argument is a recall-versus-speed curve, so the speed half has to
be measured honestly: warm up first, report a median rather than a mean, and
carry p95 alongside because tail latency is what users actually feel.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, List, Sequence


@dataclass
class Timing:
    n: int
    median_ms: float
    p95_ms: float
    mean_ms: float
    qps: float           # single-threaded, derived from the median

    def __str__(self) -> str:
        return (
            f"{self.median_ms:7.2f} ms median · {self.p95_ms:7.2f} ms p95 · "
            f"{self.qps:8.1f} QPS (1 thread, n={self.n})"
        )


def time_queries(
    fn: Callable[[object], object],
    inputs: Sequence[object],
    warmup: int = 5,
    repeat: int = 1,
) -> Timing:
    """Time `fn` once per input and summarise the distribution.

    Warm-up matters more than people expect here: the first few calls pay for
    lazy allocation and cold caches, and including them can inflate a median by
    a factor of two on a small query set.
    """
    if not inputs:
        raise ValueError("no inputs to time")

    for x in inputs[: min(warmup, len(inputs))]:
        fn(x)

    samples: List[float] = []
    for _ in range(repeat):
        for x in inputs:
            t0 = time.perf_counter()
            fn(x)
            samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    median = statistics.median(samples)
    p95 = samples[min(len(samples) - 1, int(0.95 * len(samples)))]
    return Timing(
        n=len(samples),
        median_ms=median,
        p95_ms=p95,
        mean_ms=sum(samples) / len(samples),
        qps=1000.0 / median if median > 0 else float("inf"),
    )


class stopwatch:
    """Context manager for one-off timings: build times, encode passes.

        with stopwatch("build HNSW") as sw:
            index.add(vectors)
        print(sw.seconds)
    """

    def __init__(self, label: str = "", quiet: bool = False):
        self.label = label
        self.quiet = quiet
        self.seconds = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        if not self.quiet:
            print(f"{self.label}: {self.seconds:.2f}s")
        return False


def bytes_human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024
    return f"{n:.1f} GB"
