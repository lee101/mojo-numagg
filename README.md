# mojo-numagg

Grouped and fixed-window numerical aggregations implemented in
[Mojo](https://www.modular.com/mojo) and exposed to Python as NumPy functions.
The public functions use the same names and call signatures as the corresponding
`numbagg` functions, so the covered subset can be adopted with:

```python
import mojonumagg as numbagg
```

There is no Python distribution named `numagg` on PyPI. The target description
matches [numbagg](https://github.com/numbagg/numbagg), an active aggregation
library whose API supplies the names and behavior used here. Tests compare this
port against installed `numbagg` 0.9.4 and against independent NumPy reference
calculations.

## Coverage

| Area | Implemented |
|---|---|
| Fixed windows | `move_sum`, `move_mean`, `move_var`, `move_std`, `move_cov`, `move_corr` |
| Grouped arithmetic | `group_nansum`, `group_nanmean`, `group_nanvar`, `group_nanstd`, `group_nancount`, `group_nanprod`, `group_nansum_of_squares` |
| Grouped selection | `group_nanmin`, `group_nanmax`, `group_nanfirst`, `group_nanlast`, `group_nanargmin`, `group_nanargmax` |
| Grouped logical | `group_nanall`, `group_nanany` |

Moving functions support float16, float32, and float64 arrays, with float16
promoted to float32 as in `numbagg`. Grouped functions support float32,
float64, integer, and boolean inputs wherever `numbagg` supports them, with
matching result dtypes. Integer values outside the exactly representable
float64 range are rejected rather than silently rounded. Both families handle
arbitrary single axes; grouped functions also support `axis=None` and tuples
of reduction axes.

Not covered are exponential windows (`move_exp_*`), moving covariance or
correlation matrices, static whole-array aggregations, fills, and quantiles.
The grouped float16 rejection also follows `numbagg`. These boundaries keep
the port focused on the compute-bound grouped and fixed-window kernels instead
of filling out names with Python fallbacks.

## Install

The repository carries a pinned Mojo nightly and all Python dependencies:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-numagg.so`. Importing the package also
rebuilds the library when the Mojo source is newer. A deployed copy can point
at an existing library with `MOJONUMAGG_LIB=/path/to/libmojo-numagg.so`.
The supplied Pixi environment and shared-library build currently target
Linux; this is not a platform-independent wheel.

## Usage

```python
import numpy as np
import mojonumagg as na

values = np.array([1.0, np.nan, 3.0, 8.0, 5.0, 7.0])
labels = np.array([0, 0, 1, 1, 0, -1])

print(na.group_nanmean(values, labels))
# [3.  5.5]

print(na.move_sum(values, window=3, min_count=2))
# [nan nan  4. 11. 16. 20.]
```

Negative labels are ignored. `num_labels` can be supplied to retain empty
groups, and `ddof` is supported by grouped variance and standard deviation.
NaNs never contribute to counts or arithmetic.

## Performance

Measured with `pixi run bench`, which takes a machine-wide file lock before
timing. Times are medians after warm-up and include Python result allocation
and ctypes calls on both sides.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz (x86_64), Python 3.13.14,
`numbagg` 0.9.4.

| Kernel | Elements | numbagg (ms) | Mojo (ms) | Speedup |
|---|---:|---:|---:|---:|
| `move_sum` (w=128) | 2,000,000 | 21.153 | 14.305 | 1.48x |
| `move_mean` (w=128) | 2,000,000 | 21.091 | 9.614 | 2.19x |
| `move_var` (w=128) | 2,000,000 | 18.545 | 16.471 | 1.13x |
| `move_corr` (w=128) | 2,000,000 | 47.622 | 39.101 | 1.22x |
| `group_nansum` (4,096 groups) | 2,000,000 | 11.990 | 9.902 | 1.21x |
| `group_nanmean` (4,096 groups) | 2,000,000 | 20.489 | 9.706 | 2.11x |
| `group_nanvar` (4,096 groups) | 2,000,000 | 16.538 | 14.790 | 1.12x |

Mojo is faster in all seven cases on this run. No GPU path is provided.

## How it works

All kernels live in one Mojo compilation unit. Python normalizes axes, moves
the aggregation axis to the last dimension, and prepares one C-contiguous
float64 buffer. A single native call then processes every independent row.
Large multi-row inputs use a few concurrent native calls over disjoint row
blocks. Grouped labels cross as contiguous int64 codes and are shared across
those rows.

The C ABI exports only non-parametric functions. NumPy buffers cross the
ctypes boundary as integer addresses, and each Mojo export reconstructs an
`UnsafePointer[..., AnyOrigin[mut=True]]` internally. Python owns inputs,
outputs, and scratch buffers, so the native library performs no allocation
and retains no pointers after a call.

Moving sums and means update a window in constant time. Variance and standard
deviation use removable Welford moments with a direct replacement update.
Covariance and correlation track paired valid observations. Grouped variance
uses centered two-pass moments; other grouped reducers make one pass over each
row. Grouped operations are specialized at compile time, and contiguous state
initialization uses native-width SIMD with a scalar tail.

Independent rows totaling at least 262,144 elements are split across a bounded
host thread pool; smaller calls stay serial to avoid launch overhead. Each
worker passes a disjoint NumPy view directly across the C ABI, so row
parallelism adds no buffer copies. Negative labels and NaN values are skipped
without pre-filtering or materializing index lists.

## License

MIT
