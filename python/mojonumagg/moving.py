"""Fixed-width moving aggregations with numbagg-compatible signatures."""

from __future__ import annotations

import operator

import numpy as np

from ._lib import addr, lib, parallelize_rows


def _axis(axis, ndim: int) -> int | tuple[()]:
    if isinstance(axis, tuple):
        if axis == ():
            return ()
        if len(axis) != 1:
            raise ValueError(f"only one axis can be passed; got {axis}")
        axis = axis[0]
    normalized = operator.index(axis)
    if normalized < 0:
        normalized += ndim
    if not 0 <= normalized < ndim:
        raise np.exceptions.AxisError(normalized, ndim=ndim)
    return normalized


def _parameters(array, window, min_count, axis):
    if array.ndim == 0:
        raise ValueError("moving window functions require ndim > 0")
    normalized_axis = _axis(axis, array.ndim)
    if normalized_axis == ():
        return normalized_axis, 0, 0
    window = operator.index(window)
    if not 0 < window <= array.shape[normalized_axis]:
        raise ValueError(f"window not in valid range: {window}")
    if min_count is None:
        min_count = window
    else:
        min_count = operator.index(min_count)
        if min_count < 0:
            raise ValueError(f"min_count must be positive: {min_count}")
    return normalized_axis, window, min_count


def _unary(arr, *, window, min_count, axis, operation):
    array = np.asarray(arr)
    if array.dtype.kind != "f":
        raise TypeError("moving aggregations require a floating-point array")
    if array.dtype.itemsize > 8:
        raise TypeError("moving aggregations support float16, float32, and float64")
    normalized_axis, window, min_count = _parameters(
        array, window, min_count, axis
    )
    if normalized_axis == ():
        return arr
    data = np.ascontiguousarray(
        np.moveaxis(array, normalized_axis, -1), dtype=np.float64
    )
    result = np.full(data.shape, np.nan, dtype=np.float64)
    if operation in {"var", "std"}:
        effective_min = max(min_count, 2)
    elif operation == "mean":
        effective_min = max(min_count, 1)
    else:
        effective_min = min_count
    function = getattr(lib(), f"mna_move_{operation}")
    nrows = data.size // data.shape[-1]
    data_rows = data.reshape(nrows, data.shape[-1])
    result_rows = result.reshape(nrows, result.shape[-1])

    def process(start, stop):
        function(
            addr(data_rows[start:stop]),
            addr(result_rows[start:stop]),
            stop - start,
            data.shape[-1],
            window,
            effective_min,
        )

    if nrows == 0:
        pass
    elif not parallelize_rows(process, nrows, data.size):
        process(0, nrows)
    restored = np.moveaxis(result, -1, normalized_axis)
    output_dtype = np.dtype(np.float32) if array.dtype.itemsize < 4 else array.dtype
    return restored.astype(output_dtype, copy=False)


def _binary(a, b, *, window, min_count, axis, operation):
    left = np.asarray(a)
    right = np.asarray(b)
    if left.dtype.kind != "f" or right.dtype.kind != "f":
        raise TypeError("moving aggregations require floating-point arrays")
    if left.dtype.itemsize > 8 or right.dtype.itemsize > 8:
        raise TypeError("moving aggregations support float16, float32, and float64")
    left, right = np.broadcast_arrays(left, right)
    normalized_axis, window, min_count = _parameters(
        left, window, min_count, axis
    )
    if normalized_axis == ():
        raise ValueError(
            "`axis` cannot be an empty tuple when passing more than one array"
        )
    left_data = np.ascontiguousarray(
        np.moveaxis(left, normalized_axis, -1), dtype=np.float64
    )
    right_data = np.ascontiguousarray(
        np.moveaxis(right, normalized_axis, -1), dtype=np.float64
    )
    result = np.full(left_data.shape, np.nan, dtype=np.float64)
    effective_min = max(min_count, 2 if operation == "cov" else 1)
    function = getattr(lib(), f"mna_move_{operation}")
    nrows = left_data.size // left_data.shape[-1]
    left_rows = left_data.reshape(nrows, left_data.shape[-1])
    right_rows = right_data.reshape(nrows, right_data.shape[-1])
    result_rows = result.reshape(nrows, result.shape[-1])

    def process(start, stop):
        function(
            addr(left_rows[start:stop]),
            addr(right_rows[start:stop]),
            addr(result_rows[start:stop]),
            stop - start,
            left_data.shape[-1],
            window,
            effective_min,
        )

    if nrows == 0:
        pass
    elif not parallelize_rows(process, nrows, left_data.size):
        process(0, nrows)
    restored = np.moveaxis(result, -1, normalized_axis)
    dtype = np.result_type(np.asarray(a).dtype, np.asarray(b).dtype)
    if dtype.itemsize < 4:
        dtype = np.dtype(np.float32)
    return restored.astype(dtype, copy=False)


def move_sum(arr, /, *, window: int, min_count: int | None = None, axis: int = -1):
    return _unary(
        arr, window=window, min_count=min_count, axis=axis, operation="sum"
    )


def move_mean(arr, /, *, window: int, min_count: int | None = None, axis: int = -1):
    return _unary(
        arr, window=window, min_count=min_count, axis=axis, operation="mean"
    )


def move_var(arr, /, *, window: int, min_count: int | None = None, axis: int = -1):
    return _unary(
        arr, window=window, min_count=min_count, axis=axis, operation="var"
    )


def move_std(arr, /, *, window: int, min_count: int | None = None, axis: int = -1):
    return _unary(
        arr, window=window, min_count=min_count, axis=axis, operation="std"
    )


def move_cov(
    a, b, /, *, window: int, min_count: int | None = None, axis: int = -1
):
    return _binary(
        a, b, window=window, min_count=min_count, axis=axis, operation="cov"
    )


def move_corr(
    a, b, /, *, window: int, min_count: int | None = None, axis: int = -1
):
    return _binary(
        a, b, window=window, min_count=min_count, axis=axis, operation="corr"
    )


__all__ = [
    "move_corr",
    "move_cov",
    "move_mean",
    "move_std",
    "move_sum",
    "move_var",
]
