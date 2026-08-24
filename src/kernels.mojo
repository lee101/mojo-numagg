"""Grouped and fixed-window aggregation kernels exposed through a C ABI."""

from std.math import isnan, sqrt
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def nan_value() -> Float64:
    var zero = 0.0
    return zero / zero


def move_sum_mean_row(
    values: FPtr,
    result: FPtr,
    row: Int,
    n: Int,
    window: Int,
    min_count: Int,
    take_mean: Bool,
):
    var base = row * n
    var total = 0.0
    var count = 0
    for i in range(n):
        var value = values[base + i]
        if not isnan(value):
            total += value
            count += 1
        if i >= window:
            var old = values[base + i - window]
            if not isnan(old):
                total -= old
                count -= 1
        if count >= min_count:
            if take_mean:
                if count > 0:
                    result[base + i] = total / Float64(count)
            else:
                result[base + i] = total


def move_sum_mean(
    values: FPtr,
    result: FPtr,
    nrows: Int,
    n: Int,
    window: Int,
    min_count: Int,
    take_mean: Bool,
):
    for row in range(nrows):
        move_sum_mean_row(values, result, row, n, window, min_count, take_mean)


def move_var_std_row[
    take_sqrt: Bool
](
    values: FPtr,
    result: FPtr,
    row: Int,
    n: Int,
    window: Int,
    min_count: Int,
):
    var base = row * n
    var mean = 0.0
    var moment2 = 0.0
    var count = 0
    for i in range(window):
        var value = values[base + i]
        if not isnan(value):
            count += 1
            var delta = value - mean
            mean += delta / Float64(count)
            moment2 += delta * (value - mean)
        if count >= min_count and count > 1:
            var variance = moment2 / Float64(count - 1)
            result[base + i] = sqrt(variance) if take_sqrt else variance

    for i in range(window, n):
        var value = values[base + i]
        var old = values[base + i - window]
        var has_value = not isnan(value)
        var has_old = not isnan(old)
        if has_value and has_old:
            if count > 1:
                var old_mean = mean
                mean += (value - old) / Float64(count)
                moment2 += (value - old) * (
                    value - mean + old - old_mean
                )
                if moment2 < 0.0:
                    moment2 = 0.0
            else:
                mean = value
                moment2 = 0.0
        elif has_old:
            if count == 1:
                mean = 0.0
                moment2 = 0.0
            else:
                var new_count = count - 1
                var new_mean = (Float64(count) * mean - old) / Float64(
                    new_count
                )
                moment2 -= (old - mean) * (old - new_mean)
                mean = new_mean
                if moment2 < 0.0:
                    moment2 = 0.0
            count -= 1
        elif has_value:
            count += 1
            var delta = value - mean
            mean += delta / Float64(count)
            moment2 += delta * (value - mean)
        if count >= min_count and count > 1:
            var variance = moment2 / Float64(count - 1)
            result[base + i] = sqrt(variance) if take_sqrt else variance


def group_nansum_serial(
    values: FPtr,
    labels: IPtr,
    result: FPtr,
    n: Int,
    ngroups: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var zeros = SIMD[DType.float64, W](0.0)
    if ngroups == 1:
        var totals = zeros
        var i = 0
        while i + W <= n:
            var value_vec = values.load[width=W, alignment=1](i)
            var label_vec = labels.load[width=W, alignment=1](i)
            var valid = label_vec.eq(0) & value_vec.eq(value_vec)
            totals += valid.select(value_vec, zeros)
            i += W
        var total = totals.reduce_add()
        while i < n:
            var value = values[i]
            if labels[i] == 0 and not isnan(value):
                total += value
            i += 1
        result[0] = total
        return

    var group = 0
    while group + W <= ngroups:
        result.store[alignment=1](group, zeros)
        group += W
    while group < ngroups:
        result[group] = 0.0
        group += 1
    for i in range(n):
        var label = Int(labels[i])
        if label < 0 or label >= ngroups:
            continue
        var value = values[i]
        if not isnan(value):
            result[label] += value


def move_var_std[
    take_sqrt: Bool
](
    values: FPtr,
    result: FPtr,
    nrows: Int,
    n: Int,
    window: Int,
    min_count: Int,
):
    for row in range(nrows):
        move_var_std_row[take_sqrt](
            values, result, row, n, window, min_count
        )


def move_covariance_row(
    a: FPtr,
    b: FPtr,
    result: FPtr,
    row: Int,
    n: Int,
    window: Int,
    min_count: Int,
):
    var base = row * n
    var sum_a = 0.0
    var sum_b = 0.0
    var sum_ab = 0.0
    var count = 0
    for i in range(n):
        var av = a[base + i]
        var bv = b[base + i]
        if not isnan(av) and not isnan(bv):
            sum_a += av
            sum_b += bv
            sum_ab += av * bv
            count += 1
        if i >= window:
            var old_a = a[base + i - window]
            var old_b = b[base + i - window]
            if not isnan(old_a) and not isnan(old_b):
                sum_a -= old_a
                sum_b -= old_b
                sum_ab -= old_a * old_b
                count -= 1
        if count >= min_count and count > 1:
            result[base + i] = (
                sum_ab - sum_a * sum_b / Float64(count)
            ) / Float64(count - 1)


def move_covariance(
    a: FPtr,
    b: FPtr,
    result: FPtr,
    nrows: Int,
    n: Int,
    window: Int,
    min_count: Int,
):
    for row in range(nrows):
        move_covariance_row(a, b, result, row, n, window, min_count)


def move_correlation_row(
    a: FPtr,
    b: FPtr,
    result: FPtr,
    row: Int,
    n: Int,
    window: Int,
    min_count: Int,
):
    var base = row * n
    var sum_a = 0.0
    var sum_b = 0.0
    var sum_ab = 0.0
    var sum_a2 = 0.0
    var sum_b2 = 0.0
    var count = 0
    for i in range(n):
        var av = a[base + i]
        var bv = b[base + i]
        if not isnan(av) and not isnan(bv):
            sum_a += av
            sum_b += bv
            sum_ab += av * bv
            sum_a2 += av * av
            sum_b2 += bv * bv
            count += 1
        if i >= window:
            var old_a = a[base + i - window]
            var old_b = b[base + i - window]
            if not isnan(old_a) and not isnan(old_b):
                sum_a -= old_a
                sum_b -= old_b
                sum_ab -= old_a * old_b
                sum_a2 -= old_a * old_a
                sum_b2 -= old_b * old_b
                count -= 1
        if count >= min_count and count > 1:
            var count_f = Float64(count)
            var var_a = count_f * sum_a2 - sum_a * sum_a
            var var_b = count_f * sum_b2 - sum_b * sum_b
            var denom2 = var_a * var_b
            if denom2 > 0.0:
                var covariance = count_f * sum_ab - sum_a * sum_b
                result[base + i] = covariance / sqrt(denom2)


def move_correlation(
    a: FPtr,
    b: FPtr,
    result: FPtr,
    nrows: Int,
    n: Int,
    window: Int,
    min_count: Int,
):
    for row in range(nrows):
        move_correlation_row(a, b, result, row, n, window, min_count)


def group_reduce_row[
    op: Int
](
    values: FPtr,
    labels: IPtr,
    result: FPtr,
    work: FPtr,
    counts: IPtr,
    row: Int,
    n: Int,
    ngroups: Int,
    ddof: Int,
):
    var value_base = row * n
    var result_base = row * ngroups
    if op == 2 or op == 3:
        for i in range(n):
            var group = Int(labels[i])
            if group < 0 or group >= ngroups:
                continue
            var value = values[value_base + i]
            if isnan(value):
                continue
            var k = result_base + group
            counts[k] += 1
            result[k] += value

        for group in range(ngroups):
            var k = result_base + group
            var count = Int(counts[k])
            if count > 0:
                work[k] = result[k] / Float64(count)
            result[k] = 0.0

        for i in range(n):
            var group = Int(labels[i])
            if group < 0 or group >= ngroups:
                continue
            var value = values[value_base + i]
            if isnan(value):
                continue
            var k = result_base + group
            var delta = value - work[k]
            result[k] += delta * delta

        for group in range(ngroups):
            var k = result_base + group
            var count = Int(counts[k])
            if count > ddof:
                result[k] /= Float64(count - ddof)
                if op == 3:
                    result[k] = sqrt(result[k])
            else:
                result[k] = nan_value()
        return

    for i in range(n):
        var group = Int(labels[i])
        if group < 0 or group >= ngroups:
            continue
        var value = values[value_base + i]
        if isnan(value):
            continue
        var k = result_base + group
        counts[k] += 1
        var count = Int(counts[k])
        if op == 0 or op == 1:
            result[k] += value
        elif op == 2 or op == 3:
            var delta = value - work[k]
            work[k] += delta / Float64(count)
            result[k] += delta * (value - work[k])
        elif op == 5:
            result[k] *= value
        elif op == 6:
            if count == 1 or value < result[k]:
                result[k] = value
        elif op == 7:
            if count == 1 or value > result[k]:
                result[k] = value
        elif op == 8:
            if count == 1:
                result[k] = value
        elif op == 9:
            result[k] = value
        elif op == 10:
            if value == 0.0:
                result[k] = 0.0
        elif op == 11:
            if value != 0.0:
                result[k] = 1.0
        elif op == 12:
            result[k] += value * value
        elif op == 13:
            if count == 1 or value > work[k]:
                work[k] = value
                result[k] = Float64(i)
        elif op == 14:
            if count == 1 or value < work[k]:
                work[k] = value
                result[k] = Float64(i)

    for group in range(ngroups):
        var k = result_base + group
        var count = Int(counts[k])
        if op == 1:
            result[k] = result[k] / Float64(count) if count > 0 else nan_value()
        elif op == 2 or op == 3:
            if count > ddof:
                result[k] /= Float64(count - ddof)
                if op == 3:
                    result[k] = sqrt(result[k])
            else:
                result[k] = nan_value()
        elif op == 4:
            result[k] = Float64(count)


def group_reduce[
    op: Int
](
    values: FPtr,
    labels: IPtr,
    result: FPtr,
    work: FPtr,
    counts: IPtr,
    nrows: Int,
    n: Int,
    ngroups: Int,
    ddof: Int,
):
    var size = nrows * ngroups
    comptime W = simd_width_of[DType.float64]()
    var k = 0
    var initial = 0.0
    if op == 5 or op == 10:
        initial = 1.0
    elif op == 6 or op == 7 or op == 8 or op == 9 or op == 13 or op == 14:
        initial = nan_value()
    var initial_vec = SIMD[DType.float64, W](initial)
    var zero_vec = SIMD[DType.float64, W](0.0)
    var zero_counts = SIMD[DType.int64, W](0)
    while k + W <= size:
        counts.store(k, zero_counts)
        work.store(k, zero_vec)
        result.store(k, initial_vec)
        k += W
    while k < size:
        counts[k] = 0
        work[k] = 0.0
        result[k] = initial
        k += 1

    for row in range(nrows):
        group_reduce_row[op](
            values, labels, result, work, counts, row, n, ngroups, ddof
        )


@export("mna_move_sum")
def mna_move_sum(
    values: Int,
    result: Int,
    nrows: Int,
    n: Int,
    window: Int,
    min_count: Int,
) abi("C"):
    move_sum_mean(fp(values), fp(result), nrows, n, window, min_count, False)


@export("mna_move_mean")
def mna_move_mean(
    values: Int,
    result: Int,
    nrows: Int,
    n: Int,
    window: Int,
    min_count: Int,
) abi("C"):
    move_sum_mean(fp(values), fp(result), nrows, n, window, min_count, True)


@export("mna_move_var")
def mna_move_var(
    values: Int,
    result: Int,
    nrows: Int,
    n: Int,
    window: Int,
    min_count: Int,
) abi("C"):
    move_var_std[False](fp(values), fp(result), nrows, n, window, min_count)


@export("mna_move_std")
def mna_move_std(
    values: Int,
    result: Int,
    nrows: Int,
    n: Int,
    window: Int,
    min_count: Int,
) abi("C"):
    move_var_std[True](fp(values), fp(result), nrows, n, window, min_count)


@export("mna_move_cov")
def mna_move_cov(
    a: Int,
    b: Int,
    result: Int,
    nrows: Int,
    n: Int,
    window: Int,
    min_count: Int,
) abi("C"):
    move_covariance(fp(a), fp(b), fp(result), nrows, n, window, min_count)


@export("mna_move_corr")
def mna_move_corr(
    a: Int,
    b: Int,
    result: Int,
    nrows: Int,
    n: Int,
    window: Int,
    min_count: Int,
) abi("C"):
    move_correlation(fp(a), fp(b), fp(result), nrows, n, window, min_count)


@export("mna_group_reduce")
def mna_group_reduce(
    values: Int,
    labels: Int,
    result: Int,
    work: Int,
    counts: Int,
    nrows: Int,
    n: Int,
    ngroups: Int,
    op: Int,
    ddof: Int,
) abi("C"):
    if op == 0:
        group_reduce[0](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )
    elif op == 1:
        group_reduce[1](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )
    elif op == 2:
        group_reduce[2](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )
    elif op == 3:
        group_reduce[3](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )
    elif op == 4:
        group_reduce[4](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )
    elif op == 5:
        group_reduce[5](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )
    elif op == 6:
        group_reduce[6](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )
    elif op == 7:
        group_reduce[7](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )
    elif op == 8:
        group_reduce[8](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )
    elif op == 9:
        group_reduce[9](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )
    elif op == 10:
        group_reduce[10](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )
    elif op == 11:
        group_reduce[11](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )
    elif op == 12:
        group_reduce[12](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )
    elif op == 13:
        group_reduce[13](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )
    elif op == 14:
        group_reduce[14](
            fp(values),
            ip(labels),
            fp(result),
            fp(work),
            ip(counts),
            nrows,
            n,
            ngroups,
            ddof,
        )


@export("mna_group_nansum_one_row")
def mna_group_nansum_one_row(
    values: Int,
    labels: Int,
    result: Int,
    n: Int,
    ngroups: Int,
) abi("C"):
    group_nansum_serial(fp(values), ip(labels), fp(result), n, ngroups)
