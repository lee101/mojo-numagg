"""Benchmarks against numbagg on identical inputs."""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time

import numpy as np
import numbagg


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojonumagg


def timed(function, repeats=7):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - start
        if result is None:
            raise RuntimeError("benchmark unexpectedly returned None")
        samples.append(elapsed)
    return statistics.median(samples) * 1000


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    rng = np.random.default_rng(2026)
    n = 2_000_000
    values = rng.normal(size=n)
    other = rng.normal(size=n)
    values[::97] = np.nan
    other[::131] = np.nan
    labels = rng.integers(-1, 4096, size=n, dtype=np.int64)

    cases = [
        (
            "move_sum (w=128)",
            lambda: mojonumagg.move_sum(values, window=128, min_count=64),
            lambda: numbagg.move_sum(values, window=128, min_count=64),
        ),
        (
            "move_mean (w=128)",
            lambda: mojonumagg.move_mean(values, window=128, min_count=64),
            lambda: numbagg.move_mean(values, window=128, min_count=64),
        ),
        (
            "move_var (w=128)",
            lambda: mojonumagg.move_var(values, window=128, min_count=64),
            lambda: numbagg.move_var(values, window=128, min_count=64),
        ),
        (
            "move_corr (w=128)",
            lambda: mojonumagg.move_corr(
                values, other, window=128, min_count=64
            ),
            lambda: numbagg.move_corr(
                values, other, window=128, min_count=64
            ),
        ),
        (
            "group_nansum (4,096 groups)",
            lambda: mojonumagg.group_nansum(
                values, labels, num_labels=4096
            ),
            lambda: numbagg.group_nansum(
                values, labels, num_labels=4096
            ),
        ),
        (
            "group_nanmean (4,096 groups)",
            lambda: mojonumagg.group_nanmean(
                values, labels, num_labels=4096
            ),
            lambda: numbagg.group_nanmean(
                values, labels, num_labels=4096
            ),
        ),
        (
            "group_nanvar (4,096 groups)",
            lambda: mojonumagg.group_nanvar(
                values, labels, num_labels=4096
            ),
            lambda: numbagg.group_nanvar(
                values, labels, num_labels=4096
            ),
        ),
    ]

    rows = []
    for name, mojo_function, reference_function in cases:
        actual = mojo_function()
        expected = reference_function()
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
        mojo_ms = timed(mojo_function)
        numbagg_ms = timed(reference_function)
        rows.append((name, mojo_ms, numbagg_ms, numbagg_ms / mojo_ms))

    print(
        f"Machine: {cpu_name()} ({platform.machine()}), "
        f"Python {platform.python_version()}, numbagg {numbagg.__version__}"
    )
    print()
    print("| Kernel | Elements | numbagg (ms) | Mojo (ms) | Speedup |")
    print("|---|---:|---:|---:|---:|")
    for name, mojo_ms, numbagg_ms, speedup in rows:
        print(
            f"| {name} | {n:,} | {numbagg_ms:.3f} | "
            f"{mojo_ms:.3f} | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
