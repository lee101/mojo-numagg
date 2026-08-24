"""ctypes loader for the Mojo aggregation library."""

from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "src", "kernels.mojo")
LIB = os.environ.get("MOJONUMAGG_LIB") or os.path.join(
    ROOT, "dist", "libmojo-numagg.so"
)
I = ctypes.c_int64
P = ctypes.c_void_p

_SIGNATURES = {
    "mna_move_sum": ([P, P] + [I] * 4, None),
    "mna_move_mean": ([P, P] + [I] * 4, None),
    "mna_move_var": ([P, P] + [I] * 4, None),
    "mna_move_std": ([P, P] + [I] * 4, None),
    "mna_move_cov": ([P, P, P] + [I] * 4, None),
    "mna_move_corr": ([P, P, P] + [I] * 4, None),
    "mna_group_reduce": ([P] * 5 + [I] * 5, None),
    "mna_group_nansum_one_row": ([P] * 3 + [I] * 2, None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    override = os.environ.get("MOJONUMAGG_LIB")
    if override:
        if os.path.isfile(LIB):
            return LIB
        raise BuildError(f"MOJONUMAGG_LIB does not exist or is not a file: {LIB}")
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SOURCE)
    ):
        return LIB
    process = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if process.returncode or not os.path.exists(LIB):
        details = "\n".join(
            part.strip() for part in (process.stdout, process.stderr) if part.strip()
        )
        raise BuildError(details[:4000] or "Mojo build failed without output")
    return LIB


_loaded: ctypes.CDLL | None = None
_executor: ThreadPoolExecutor | None = None
PARALLEL_THRESHOLD = 262_144


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_loaded, name)
            function.argtypes = argtypes
            function.restype = restype
    return _loaded


def addr(array) -> ctypes.c_void_p:
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("cannot pass a null array pointer to Mojo")
    return ctypes.c_void_p(address)


def parallelize_rows(function, nrows: int, total_size: int) -> bool:
    global _executor
    if nrows < 2 or total_size < PARALLEL_THRESHOLD:
        return False
    workers = min(nrows, os.cpu_count() or 1, 8)
    if workers < 2:
        return False
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=min(os.cpu_count() or 1, 8),
            thread_name_prefix="mojonumagg",
        )
    chunk = (nrows + workers - 1) // workers
    futures = [
        _executor.submit(function, start, min(start + chunk, nrows))
        for start in range(0, nrows, chunk)
    ]
    for future in futures:
        future.result()
    return True
