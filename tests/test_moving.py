import inspect

import numpy as np
import numbagg
import pytest

import mojonumagg


UNARY = ["move_sum", "move_mean", "move_var", "move_std"]
BINARY = ["move_cov", "move_corr"]


@pytest.fixture(scope="module")
def arrays():
    rng = np.random.default_rng(2026)
    left = rng.normal(size=(4, 73))
    right = rng.normal(size=(4, 73))
    left[:, ::9] = np.nan
    right[:, ::13] = np.nan
    return left, right


@pytest.mark.parametrize("name", UNARY)
@pytest.mark.parametrize("window,min_count", [(1, None), (7, None), (11, 0), (17, 5)])
def test_unary_parity(name, window, min_count, arrays):
    left, _ = arrays
    actual = getattr(mojonumagg, name)(
        left, window=window, min_count=min_count, axis=-1
    )
    expected = getattr(numbagg, name)(
        left, window=window, min_count=min_count, axis=-1
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", BINARY)
@pytest.mark.parametrize("window,min_count", [(3, None), (9, 0), (17, 5)])
def test_binary_parity(name, window, min_count, arrays):
    left, right = arrays
    actual = getattr(mojonumagg, name)(
        left, right, window=window, min_count=min_count
    )
    expected = getattr(numbagg, name)(
        left, right, window=window, min_count=min_count
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", UNARY + BINARY)
def test_non_last_axis(name, arrays):
    left, right = (array.T for array in arrays)
    function = getattr(mojonumagg, name)
    reference = getattr(numbagg, name)
    args = (left, right) if name in BINARY else (left,)
    actual = function(*args, window=13, min_count=4, axis=0)
    expected = reference(*args, window=13, min_count=4, axis=0)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", UNARY + BINARY)
def test_float32_dtype_and_values(name, arrays):
    left, right = (array.astype(np.float32) for array in arrays)
    function = getattr(mojonumagg, name)
    reference = getattr(numbagg, name)
    args = (left, right) if name in BINARY else (left,)
    actual = function(*args, window=15, min_count=4)
    expected = reference(*args, window=15, min_count=4)
    assert actual.dtype == expected.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("name", UNARY + BINARY)
def test_float16_promotes_to_float32(name):
    left = np.arange(20, dtype=np.float16)
    right = np.arange(20, dtype=np.float16)[::-1]
    function = getattr(mojonumagg, name)
    reference = getattr(numbagg, name)
    args = (left, right) if name in BINARY else (left,)
    actual = function(*args, window=5, min_count=2)
    expected = reference(*args, window=5, min_count=2)
    assert actual.dtype == expected.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_numpy_window_reference():
    values = np.array([np.nan, 1.0, 2.0, np.nan, 4.0, 8.0])
    window = 3
    expected_sum = np.full(values.shape, np.nan)
    expected_mean = np.full(values.shape, np.nan)
    for index in range(len(values)):
        part = values[max(0, index - window + 1) : index + 1]
        if np.count_nonzero(~np.isnan(part)) >= 2:
            expected_sum[index] = np.nansum(part)
            expected_mean[index] = np.nanmean(part)
    np.testing.assert_allclose(
        mojonumagg.move_sum(values, window=window, min_count=2), expected_sum
    )
    np.testing.assert_allclose(
        mojonumagg.move_mean(values, window=window, min_count=2), expected_mean
    )


@pytest.mark.parametrize("columns", [131_071, 131_073])
def test_row_parallel_threshold(columns):
    rng = np.random.default_rng(columns)
    values = rng.normal(size=(2, columns))
    values[:, ::101] = np.nan
    actual = mojonumagg.move_var(
        values, window=127, min_count=63
    )
    expected = np.stack(
        [
            mojonumagg.move_var(row, window=127, min_count=63)
            for row in values
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_rolling_variance_nan_replacement_paths():
    values = np.array(
        [np.nan, 2.0, 3.0, 5.0, np.nan, 11.0, 13.0, 17.0, np.nan, 23.0]
    )
    actual = mojonumagg.move_var(values, window=4, min_count=2)
    expected = numbagg.move_var(values, window=4, min_count=2)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_rolling_variance_large_offset_stability():
    rng = np.random.default_rng(407)
    values = 1.0e6 + rng.normal(size=4097)
    values[::97] = np.nan
    actual = mojonumagg.move_var(values, window=128, min_count=64)
    expected = np.full(values.shape, np.nan)
    for index in range(63, values.size):
        part = values[max(0, index - 127) : index + 1]
        if np.count_nonzero(~np.isnan(part)) >= 64:
            expected[index] = np.nanvar(part, ddof=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("name", ["move_var", "move_std"])
def test_rolling_moment_replaces_only_valid_value(name):
    values = np.array([1.0, np.nan, np.nan, 2.0, 3.0])
    actual = getattr(mojonumagg, name)(
        values, window=3, min_count=2
    )
    expected = getattr(numbagg, name)(
        values, window=3, min_count=2
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_broadcast_binary_inputs():
    left = np.arange(24.0).reshape(2, 3, 4)
    right = np.arange(4.0)
    actual = mojonumagg.move_cov(left, right, window=3, min_count=2)
    expected = numbagg.move_cov(
        left, np.broadcast_to(right, left.shape), window=3, min_count=2
    )
    np.testing.assert_allclose(actual, expected)


def test_empty_axis_tuple_returns_original():
    values = np.arange(6.0)
    assert mojonumagg.move_mean(values, window=2, axis=()) is values
    with pytest.raises(ValueError, match="empty tuple"):
        mojonumagg.move_cov(values, values, window=2, axis=())


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"window": 0}, ValueError),
        ({"window": 100}, ValueError),
        ({"window": 2.5}, TypeError),
        ({"window": 2, "min_count": -1}, ValueError),
        ({"window": 2, "axis": (0, 1)}, ValueError),
    ],
)
def test_validation(kwargs, error):
    with pytest.raises(error):
        mojonumagg.move_sum(np.arange(5.0), **kwargs)


def test_rejects_integer_moving_input():
    with pytest.raises(TypeError, match="floating-point"):
        mojonumagg.move_sum(np.arange(5), window=2)


def test_rejects_float128_instead_of_silently_narrowing():
    if not hasattr(np, "float128"):
        pytest.skip("NumPy has no float128 on this platform")
    values = np.arange(5, dtype=np.float128)
    with pytest.raises(TypeError, match="float16, float32, and float64"):
        mojonumagg.move_sum(values, window=2)


@pytest.mark.parametrize("name", UNARY + BINARY)
def test_zero_outer_dimension(name):
    left = np.empty((0, 5), dtype=np.float64)
    right = np.empty((0, 5), dtype=np.float64)
    function = getattr(mojonumagg, name)
    args = (left, right) if name in BINARY else (left,)
    actual = function(*args, window=2)
    assert actual.shape == (0, 5)
    assert actual.dtype == np.dtype(np.float64)


def test_public_signatures_match_numbagg_stubs():
    unary = inspect.signature(mojonumagg.move_mean)
    binary = inspect.signature(mojonumagg.move_cov)
    assert list(unary.parameters) == ["arr", "window", "min_count", "axis"]
    assert list(binary.parameters) == ["a", "b", "window", "min_count", "axis"]
    assert unary.parameters["arr"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert binary.parameters["b"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert unary.parameters["window"].kind is inspect.Parameter.KEYWORD_ONLY
    assert unary.parameters["min_count"].default is None
    assert unary.parameters["axis"].default == -1
