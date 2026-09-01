"""Small statistics helpers, so the test suite needs no SciPy.

Only what the speculative-decoding equivalence check requires: a chi-square
survival function and a two-sample test of homogeneity.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

_MAX_ITER = 300
_EPS = 3e-14


def _gamma_series(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x) by series expansion."""
    term = 1.0 / a
    total = term
    ap = a
    for _ in range(_MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) by continued fraction."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(statistic: float, df: int) -> float:
    """Upper tail probability of a chi-square distribution: P(X > statistic)."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if statistic <= 0:
        return 1.0
    a, x = df / 2.0, statistic / 2.0
    # The series converges fast below the mean, the continued fraction above.
    return 1.0 - _gamma_series(a, x) if x < a + 1.0 else _gamma_cf(a, x)


@dataclass
class ChiSquareResult:
    statistic: float
    dof: int
    p_value: float
    num_categories: int
    pooled_low_count: int

    def __str__(self) -> str:  # pragma: no cover - reporting aid
        return (
            f"chi2={self.statistic:.2f} dof={self.dof} p={self.p_value:.4f} "
            f"(k={self.num_categories}, {self.pooled_low_count} rare merged)"
        )


def chi_square_two_sample(
    sample_a: Iterable[int],
    sample_b: Iterable[int],
    min_expected: float = 5.0,
) -> ChiSquareResult:
    """Test whether two categorical samples come from the same distribution.

    Categories whose pooled count is too small for the chi-square
    approximation are merged into a single bucket.
    """
    count_a, count_b = Counter(sample_a), Counter(sample_b)
    n_a, n_b = sum(count_a.values()), sum(count_b.values())
    if not n_a or not n_b:
        raise ValueError("both samples must be non-empty")
    total = n_a + n_b

    keep: list[int] = []
    merged = 0
    for key in set(count_a) | set(count_b):
        pooled = count_a[key] + count_b[key]
        if pooled * n_a / total >= min_expected and pooled * n_b / total >= min_expected:
            keep.append(key)
        else:
            merged += 1

    rows = [(count_a[k], count_b[k]) for k in keep]
    if merged:
        rest_a = n_a - sum(r[0] for r in rows)
        rest_b = n_b - sum(r[1] for r in rows)
        if rest_a or rest_b:
            rows.append((rest_a, rest_b))

    statistic = 0.0
    for observed_a, observed_b in rows:
        pooled = observed_a + observed_b
        expected_a = pooled * n_a / total
        expected_b = pooled * n_b / total
        if expected_a > 0:
            statistic += (observed_a - expected_a) ** 2 / expected_a
        if expected_b > 0:
            statistic += (observed_b - expected_b) ** 2 / expected_b

    dof = max(1, len(rows) - 1)
    return ChiSquareResult(
        statistic=statistic,
        dof=dof,
        p_value=chi2_sf(statistic, dof),
        num_categories=len(rows),
        pooled_low_count=merged,
    )


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile; ``q`` in [0, 100]."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
