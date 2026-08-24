"""Grouped reductions with numbagg-compatible signatures."""

from __future__ import annotations

import operator

import numpy as np

from ._lib import addr, lib, parallelize_rows


_OPS = {
    "nansum": 0,
    "nanmean": 1,
    "nanvar": 2,
    "nanstd": 3,
    "nancount": 4,
    "nanprod": 5,
    "nanmin": 6,
    "nanmax": 7,
    "nanfirst": 8,
    "nanlast": 9,
    "nanall": 10,
    "nanany": 11,
    "nansum_of_squares": 12,
    "nanargmax": 13,
    "nanargmin": 14,
}


def _layout(values, labels, axis):
    if axis is None:
        if values.shape != labels.shape:
            raise ValueError(
                "axis required if values and labels have different "
                f"shapes: {values.shape} vs {labels.shape}"
            )
        return values.reshape(1, -1), labels.reshape(-1), ()
    if isinstance(axis, (int, np.integer)):
        normalized = operator.index(axis)
        if normalized < 0:
            normalized += values.ndim
        if not 0 <= normalized < values.ndim:
            raise np.exceptions.AxisError(normalized, ndim=values.ndim)
        expected = (values.shape[normalized],)
        if labels.shape != expected:
            raise ValueError(
                "values must have same shape along axis as labels: "
                f"{expected} vs {labels.shape}"
            )
        moved = np.moveaxis(values, normalized, -1)
        outer_shape = moved.shape[:-1]
        nrows = int(np.prod(outer_shape, dtype=np.intp))
        return (
            moved.reshape(nrows, moved.shape[-1]),
            labels.reshape(-1),
            outer_shape,
        )
    axes_list = []
    for item in axis:
        normalized = operator.index(item)
        if normalized < 0:
            normalized += values.ndim
        if not 0 <= normalized < values.ndim:
            raise np.exceptions.AxisError(normalized, ndim=values.ndim)
        axes_list.append(normalized)
    axes = tuple(axes_list)
    if len(set(axes)) != len(axes):
        raise ValueError("repeated axis")
    expected = tuple(values.shape[item] for item in axes)
    if labels.shape != expected:
        raise ValueError(
            "values must have same shape along axis as labels: "
            f"{expected} vs {labels.shape}"
        )
    destinations = tuple(range(values.ndim - len(axes), values.ndim))
    moved = np.moveaxis(values, axes, destinations)
    n = labels.size
    outer_shape = moved.shape[: values.ndim - len(axes)]
    nrows = int(np.prod(outer_shape, dtype=np.intp))
    return moved.reshape(nrows, n), labels.reshape(-1), outer_shape


def _result_dtype(original, operation):
    if operation in {"nanvar", "nanstd"} and original.dtype.kind in "iu":
        return np.dtype(np.float64)
    if original.dtype.kind == "b":
        if operation in {"nanvar", "nanstd"}:
            raise TypeError(
                f"group_{operation} does not support boolean input. "
                "Convert to a numeric type first."
            )
        return np.dtype(np.int32)
    return original.dtype


def _group_reduce(
    values,
    labels,
    *,
    operation,
    ddof=1,
    num_labels=None,
    axis=None,
):
    original = np.asarray(values)
    label_array = np.asarray(labels)
    if original.dtype.kind not in "biuf":
        raise TypeError("grouped aggregations require a real numeric or boolean array")
    if original.dtype == np.dtype(np.float16):
        raise NotImplementedError("float16 grouped aggregations are not supported")
    if original.dtype.kind == "f" and original.dtype.itemsize > 8:
        raise TypeError("grouped aggregations support float32 and float64")
    if original.dtype.kind in "iu" and original.dtype.itemsize > 4:
        limit = 1 << 53
        if original.size and (
            np.any(original > limit) or np.any(original < -limit)
        ):
            raise ValueError(
                "integer values outside [-2**53, 2**53] cannot be represented "
                "exactly by the float64 Mojo kernel"
            )
    if label_array.dtype.kind != "i":
        raise TypeError(
            "labels must be an integer array; factorize labels before aggregation"
        )
    value_rows, flat_labels, outer_shape = _layout(original, label_array, axis)
    flat_labels = np.ascontiguousarray(flat_labels, dtype=np.int64)
    if num_labels is None:
        if not flat_labels.size:
            raise ValueError("zero-size array to reduction operation maximum")
        num_labels = int(flat_labels.max()) + 1
    else:
        num_labels = operator.index(num_labels)
    if num_labels < 0:
        raise ValueError("num_labels must be non-negative")
    if flat_labels.size and flat_labels.max(initial=-1) >= num_labels:
        raise ValueError("num_labels is smaller than the largest non-negative label")

    rows = np.ascontiguousarray(value_rows, dtype=np.float64)
    result = np.empty((rows.shape[0], num_labels), dtype=np.float64)
    if operation == "nansum" and rows.shape[0] == 1 and result.size:
        lib().mna_group_nansum_one_row(
            addr(rows),
            addr(flat_labels),
            addr(result),
            rows.shape[1],
            num_labels,
        )
    elif result.size:
        work = np.empty_like(result)
        counts = np.empty(result.shape, dtype=np.int64)
        function = lib().mna_group_reduce
        # Empty reduction axes still need the native identity initialization.
        # Non-null one-element sentinels are never dereferenced when n == 0.
        input_buffer = rows if rows.size else np.empty(1, dtype=np.float64)
        label_buffer = (
            flat_labels if flat_labels.size else np.empty(1, dtype=np.int64)
        )

        def process(start, stop):
            function(
                addr(input_buffer if rows.shape[1] == 0 else rows[start:stop]),
                addr(label_buffer),
                addr(result[start:stop]),
                addr(work[start:stop]),
                addr(counts[start:stop]),
                stop - start,
                rows.shape[1],
                num_labels,
                _OPS[operation],
                operator.index(ddof),
            )

        if not parallelize_rows(process, rows.shape[0], rows.size):
            process(0, rows.shape[0])
    result = result.reshape(outer_shape + (num_labels,))
    result_dtype = _result_dtype(original, operation)
    if result_dtype.kind in "iub" and operation in {
        "nanmean",
        "nanfirst",
        "nanlast",
        "nanargmax",
        "nanargmin",
    }:
        result = np.nan_to_num(result, nan=0.0)
    with np.errstate(invalid="ignore"):
        return result.astype(result_dtype, copy=False)


def _make_grouped(name, operation, supports_ddof=False, supports_bool=True, supports_ints=True):
    def function(values, labels, *, ddof=1, num_labels=None, axis=None):
        return _group_reduce(
            values,
            labels,
            operation=operation,
            ddof=ddof,
            num_labels=num_labels,
            axis=axis,
        )

    function.__name__ = name
    function.__qualname__ = name
    function.__doc__ = f"Compute grouped {operation[3:]} while ignoring NaNs."
    function.supports_ddof = supports_ddof
    function.supports_bool = supports_bool
    function.supports_ints = supports_ints
    return function


group_nansum = _make_grouped("group_nansum", "nansum")
group_nanmean = _make_grouped("group_nanmean", "nanmean")
group_nanvar = _make_grouped(
    "group_nanvar", "nanvar", supports_ddof=True, supports_bool=False, supports_ints=False
)
group_nanstd = _make_grouped(
    "group_nanstd", "nanstd", supports_ddof=True, supports_bool=False, supports_ints=False
)
group_nancount = _make_grouped("group_nancount", "nancount")
group_nanprod = _make_grouped("group_nanprod", "nanprod")
group_nanmin = _make_grouped("group_nanmin", "nanmin")
group_nanmax = _make_grouped("group_nanmax", "nanmax")
group_nanfirst = _make_grouped("group_nanfirst", "nanfirst")
group_nanlast = _make_grouped("group_nanlast", "nanlast")
group_nanall = _make_grouped("group_nanall", "nanall")
group_nanany = _make_grouped("group_nanany", "nanany")
group_nansum_of_squares = _make_grouped(
    "group_nansum_of_squares", "nansum_of_squares"
)
group_nanargmax = _make_grouped("group_nanargmax", "nanargmax")
group_nanargmin = _make_grouped("group_nanargmin", "nanargmin")

__all__ = [
    "group_nanall",
    "group_nanany",
    "group_nanargmax",
    "group_nanargmin",
    "group_nancount",
    "group_nanfirst",
    "group_nanlast",
    "group_nanmax",
    "group_nanmean",
    "group_nanmin",
    "group_nanprod",
    "group_nanstd",
    "group_nansum",
    "group_nansum_of_squares",
    "group_nanvar",
]
