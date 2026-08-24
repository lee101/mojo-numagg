import numpy as np
import numbagg
import pytest
import warnings

import mojonumagg


GROUPED_NAMES = [
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


@pytest.fixture(scope="module")
def grouped_data():
    rng = np.random.default_rng(17)
    labels = rng.integers(-1, 8, size=101, dtype=np.int64)
    values = rng.normal(size=(3, 101))
    values[:, ::11] = np.nan
    values[:, labels == 6] = np.nan
    return values, labels


@pytest.mark.parametrize("name", GROUPED_NAMES)
def test_grouped_float64_parity(name, grouped_data):
    values, labels = grouped_data
    actual = getattr(mojonumagg, name)(values, labels, axis=1, num_labels=10)
    expected = getattr(numbagg, name)(values, labels, axis=1, num_labels=10)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", GROUPED_NAMES)
def test_grouped_nd_labels_parity(name):
    values = np.array([[1.0, np.nan, 3.0], [7.0, 5.0, 2.0]])
    labels = np.array([[0, 0, 1], [1, 2, -1]], dtype=np.int32)
    actual = getattr(mojonumagg, name)(values, labels)
    expected = getattr(numbagg, name)(values, labels)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", GROUPED_NAMES)
def test_grouped_float32_dtype_and_values(name):
    rng = np.random.default_rng(91)
    values = rng.normal(size=211).astype(np.float32)
    values[::17] = np.nan
    labels = rng.integers(-1, 9, size=values.size, dtype=np.int32)
    actual = getattr(mojonumagg, name)(values, labels)
    expected = getattr(numbagg, name)(values, labels)
    assert actual.dtype == expected.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("name", GROUPED_NAMES)
def test_grouped_tuple_axis_parity(name):
    values = np.arange(48.0).reshape(2, 4, 6)
    values[0, 1, 2] = np.nan
    labels = (np.arange(24).reshape(4, 6) % 5).astype(np.int16)
    actual = getattr(mojonumagg, name)(
        values, labels, axis=(1, 2), num_labels=7
    )
    expected = getattr(numbagg, name)(
        values, labels, axis=(1, 2), num_labels=7
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", ["group_nanvar", "group_nanstd"])
@pytest.mark.parametrize("ddof", [0, 1, 3])
def test_ddof_parity(name, ddof, grouped_data):
    values, labels = grouped_data
    actual = getattr(mojonumagg, name)(values, labels, axis=-1, ddof=ddof)
    expected = getattr(numbagg, name)(values, labels, axis=-1, ddof=ddof)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", GROUPED_NAMES)
def test_integer_parity(name):
    labels = np.array([0, 0, 1, -1, 2, 2], dtype=np.int32)
    values = np.array([4, -2, 7, 5, 1, 3], dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = getattr(mojonumagg, name)(values, labels, num_labels=4)
        expected = getattr(numbagg, name)(values, labels, num_labels=4)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name", GROUPED_NAMES)
def test_boolean_parity_or_matching_rejection(name):
    labels = np.array([0, 0, 1, -1, 2, 2], dtype=np.int32)
    values = np.array([True, False, True, False, False, True])
    reference = getattr(numbagg, name)
    function = getattr(mojonumagg, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            expected = reference(values, labels, num_labels=4)
        except TypeError:
            with pytest.raises(TypeError):
                function(values, labels, num_labels=4)
        else:
            actual = function(values, labels, num_labels=4)
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)


def test_grouped_identity_and_missing_semantics():
    values = np.array([np.nan, np.nan, 4.0])
    labels = np.array([0, 0, 1])
    expected = {
        "group_nansum": [0.0, 4.0, 0.0],
        "group_nanprod": [1.0, 4.0, 1.0],
        "group_nancount": [0.0, 1.0, 0.0],
        "group_nanall": [1.0, 1.0, 1.0],
        "group_nanany": [0.0, 1.0, 0.0],
        "group_nanmean": [np.nan, 4.0, np.nan],
        "group_nanmin": [np.nan, 4.0, np.nan],
    }
    for name, target in expected.items():
        np.testing.assert_allclose(
            getattr(mojonumagg, name)(values, labels, num_labels=3),
            target,
            equal_nan=True,
        )


def test_numpy_group_reference():
    values = np.array([3.0, np.nan, -1.0, 8.0, 5.0, 7.0])
    labels = np.array([0, 0, 1, -1, 1, 0])
    expected_sum = np.array([10.0, 4.0])
    expected_mean = np.array([5.0, 2.0])
    np.testing.assert_allclose(
        mojonumagg.group_nansum(values, labels), expected_sum
    )
    np.testing.assert_allclose(
        mojonumagg.group_nanmean(values, labels), expected_mean
    )


@pytest.mark.parametrize("num_labels", range(1, 18))
def test_grouped_simd_initialization_tail(num_labels):
    values = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    labels = np.array([0, -1, 0, 0, 0], dtype=np.int64)
    actual = mojonumagg.group_nanvar(
        values, labels, num_labels=num_labels, ddof=0
    )
    expected = numbagg.group_nanvar(
        values, labels, num_labels=num_labels, ddof=0
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("length", range(1, 18))
def test_grouped_nansum_simd_input_tail(length):
    values = np.arange(length, dtype=np.float64) - 3.0
    values[::5] = np.nan
    labels = np.zeros(length, dtype=np.int64)
    if length > 2:
        labels[-1] = -1
    actual = mojonumagg.group_nansum(values, labels, num_labels=1)
    expected = numbagg.group_nansum(values, labels, num_labels=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_grouped_variance_large_offset_stability():
    rng = np.random.default_rng(404)
    values = 1.0e9 + rng.normal(size=4097)
    labels = np.arange(values.size, dtype=np.int64) % 7
    actual = mojonumagg.group_nanvar(values, labels, num_labels=7)
    expected = np.array(
        [np.var(values[labels == group], ddof=1) for group in range(7)]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("columns", [131_071, 131_073])
def test_grouped_row_parallel_threshold(columns):
    rng = np.random.default_rng(columns)
    values = rng.normal(size=(2, columns))
    values[:, ::103] = np.nan
    labels = rng.integers(-1, 17, size=columns, dtype=np.int64)
    actual = mojonumagg.group_nanvar(
        values, labels, axis=1, num_labels=17
    )
    expected = np.stack(
        [
            mojonumagg.group_nanvar(row, labels, num_labels=17)
            for row in values
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_negative_labels_are_ignored():
    values = np.array([100.0, 1.0, 2.0])
    labels = np.array([-1, 0, 0])
    np.testing.assert_array_equal(
        mojonumagg.group_nansum(values, labels), np.array([3.0])
    )


@pytest.mark.parametrize(
    "values,labels,kwargs,error",
    [
        ([1.0, 2.0], [0.0, 1.0], {}, TypeError),
        (np.ones((2, 3)), [0, 1, 2], {}, ValueError),
        (np.ones((2, 3)), [0, 1], {"axis": 1}, ValueError),
        ([1.0, 2.0], [0, 2], {"num_labels": 2}, ValueError),
        ([1.0, 2.0], [0, 1], {"num_labels": -1}, ValueError),
    ],
)
def test_grouped_validation(values, labels, kwargs, error):
    with pytest.raises(error):
        mojonumagg.group_nansum(values, labels, **kwargs)


def test_bool_variance_rejected():
    with pytest.raises(TypeError, match="does not support boolean"):
        mojonumagg.group_nanvar(
            np.array([True, False]), np.array([0, 0])
        )


def test_float16_grouped_matches_upstream_rejection():
    values = np.array([1.0, 2.0], dtype=np.float16)
    labels = np.array([0, 0])
    with pytest.raises(NotImplementedError):
        mojonumagg.group_nanmean(values, labels)


@pytest.mark.parametrize("dtype", [np.complex128, object])
def test_rejects_non_real_grouped_input(dtype):
    values = np.array([1, 2], dtype=dtype)
    with pytest.raises(TypeError, match="real numeric or boolean"):
        mojonumagg.group_nansum(values, np.array([0, 0]))


def test_rejects_grouped_float128_instead_of_silently_narrowing():
    if not hasattr(np, "float128"):
        pytest.skip("NumPy has no float128 on this platform")
    values = np.array([1, 2], dtype=np.float128)
    with pytest.raises(TypeError, match="float32 and float64"):
        mojonumagg.group_nansum(values, np.array([0, 0]))


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_rejects_integer_values_not_exactly_supported_by_kernel(dtype):
    values = np.array([2**53 + 1], dtype=dtype)
    with pytest.raises(ValueError, match="cannot be represented exactly"):
        mojonumagg.group_nansum(values, np.array([0]))


@pytest.mark.parametrize("name", GROUPED_NAMES)
def test_empty_reduction_axis_with_explicit_groups(name):
    values = np.empty((3, 0), dtype=np.float64)
    labels = np.empty(0, dtype=np.int64)
    actual = getattr(mojonumagg, name)(
        values, labels, axis=1, num_labels=5
    )
    expected = getattr(numbagg, name)(
        values, labels, axis=1, num_labels=5
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_compatibility_metadata():
    assert mojonumagg.group_nanvar.supports_ddof
    assert not mojonumagg.group_nanvar.supports_bool
    assert not mojonumagg.group_nanvar.supports_ints
    assert mojonumagg.group_nansum.supports_bool
    assert mojonumagg.group_nansum.supports_ints
