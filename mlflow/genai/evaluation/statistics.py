"""Confidence-interval estimation for evaluation sweeps.

A sweep runs each configuration ``n_repeats`` times, producing a sample of
per-run scores per scorer. This module reduces that sample to a point estimate
plus a confidence interval, choosing the interval method automatically:

* boolean / pass-fail scorers -> Wilson score interval (correct near 0 and 1,
  where the normal approximation degrades),
* numeric scorers with >= 2 repeats -> Student's t interval over the run means,
* numeric scorers with a single repeat -> bootstrap over the single run's rows,
  so a confidence interval is still available when repetition is not possible.

The confidence level and bootstrap resample count are fixed constants rather
than user-facing knobs to keep the sweep API small.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

# Fixed defaults. Intentionally not exposed as API parameters — a two-sided 95%
# interval and 1000 bootstrap resamples are standard choices that cover the vast
# majority of use cases. Revisit via environment variables if a need arises.
CONFIDENCE_LEVEL = 0.95
N_BOOTSTRAP = 1000
# Deterministic seed so a sweep's reported interval is reproducible across calls.
_BOOTSTRAP_SEED = 0

CIMethod = Literal["t", "wilson", "bootstrap", "none"]


@dataclass(frozen=True)
class Interval:
    """A point estimate and confidence interval for a single scorer."""

    mean: float
    ci_low: float
    ci_high: float
    std: float
    n_samples: int
    method: CIMethod


def _z_for(confidence_level: float) -> float:
    """Two-sided z critical value for the given confidence level."""
    # Inverse CDF of the standard normal via the error function's inverse.
    return math.sqrt(2.0) * _erfinv(confidence_level)


def _erfinv(x: float) -> float:
    """Inverse error function (Winitzki approximation, refined by one Newton step).

    Accurate to ~1e-6 over the range we need, and avoids a scipy dependency in
    the skinny client. ``x`` is the confidence level (e.g. 0.95); the standard
    normal quantile is ``sqrt(2) * erfinv(x)``.
    """
    a = 0.147
    ln = math.log(1.0 - x * x)
    term = 2.0 / (math.pi * a) + ln / 2.0
    approx = math.copysign(math.sqrt(math.sqrt(term * term - ln / a) - term), x)
    # One Newton-Raphson refinement against erf.
    err = math.erf(approx) - x
    approx -= err / (2.0 / math.sqrt(math.pi) * math.exp(-approx * approx))
    return approx


# Small table of two-sided Student's t critical values at 95% confidence, keyed
# by degrees of freedom (n - 1). Avoids a scipy dependency for the common case
# of a handful of repeats. Falls back to the normal quantile for large dof.
_T_TABLE_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    30: 2.042,
    60: 2.000,
}


def _t_critical(dof: int, confidence_level: float) -> float:
    if confidence_level == CONFIDENCE_LEVEL and dof in _T_TABLE_95:
        return _T_TABLE_95[dof]
    # For dof not in the table (or a non-default confidence level) approximate
    # with the largest tabulated dof <= dof, or the normal quantile for large dof.
    if confidence_level == CONFIDENCE_LEVEL:
        if candidates := [d for d in _T_TABLE_95 if d <= dof]:
            return _T_TABLE_95[max(candidates)]
    return _z_for(confidence_level)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float], ddof: int = 1) -> float:
    n = len(values)
    if n - ddof <= 0:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - ddof))


def t_interval(values: list[float], confidence_level: float = CONFIDENCE_LEVEL) -> Interval:
    """Student's t confidence interval over per-run scores.

    Args:
        values: One score per repeat (typically each run's per-row mean).
        confidence_level: Two-sided confidence level.
    """
    n = len(values)
    mean = _mean(values)
    if n < 2:
        return Interval(mean, mean, mean, 0.0, n, "none")
    std = _std(values, ddof=1)
    stderr = std / math.sqrt(n)
    margin = _t_critical(n - 1, confidence_level) * stderr
    return Interval(mean, mean - margin, mean + margin, std, n, "t")


def wilson_interval(successes: int, n: int, confidence_level: float = CONFIDENCE_LEVEL) -> Interval:
    """Wilson score interval for a binomial proportion.

    Correct near the 0 and 1 boundaries where the normal approximation produces
    intervals that leave [0, 1]. Used for boolean / pass-fail scorers.
    """
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0.0, 0, "none")
    z = _z_for(confidence_level)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    std = math.sqrt(p * (1 - p))
    return Interval(p, max(0.0, center - margin), min(1.0, center + margin), std, n, "wilson")


def bootstrap_interval(
    row_values: list[float],
    confidence_level: float = CONFIDENCE_LEVEL,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Interval:
    """Percentile bootstrap over row-level scores from a single run.

    The sweep falls back to this when ``n_repeats == 1``: with no repeats to
    estimate run-to-run variance, it resamples the run's rows to estimate how
    much the mean depends on which rows were sampled. This answers a different
    question than the across-repeat t interval and is only used as a fallback.
    """
    n = len(row_values)
    mean = _mean(row_values) if n else 0.0
    if n < 2:
        return Interval(mean, mean, mean, 0.0, n, "none")

    rng = random.Random(_BOOTSTRAP_SEED)
    boot_means: list[float] = []
    for _ in range(n_bootstrap):
        resample = [row_values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(_mean(resample))
    boot_means.sort()
    lo_idx = int((1 - confidence_level) / 2 * n_bootstrap)
    hi_idx = int((1 + confidence_level) / 2 * n_bootstrap) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))
    return Interval(
        mean, boot_means[lo_idx], boot_means[hi_idx], _std(row_values, ddof=1), n, "bootstrap"
    )


def _is_binary(values: list[float]) -> bool:
    """Whether every value is 0 or 1 (a pass/fail scorer after float-casting)."""
    return bool(values) and all(v in (0.0, 1.0) for v in values)


def compute_scorer_interval(
    *,
    per_run_means: list[float],
    row_values: list[float],
) -> Interval:
    """Pick and compute the confidence interval for one scorer in a sweep.

    The method is chosen automatically — there is no user-facing knob:

    * pass/fail scorer (all row values are 0/1) -> Wilson over the pooled
      (row, repeat) Bernoulli trials,
    * numeric scorer with >= 2 repeats -> Student's t over the per-run means,
    * numeric scorer with a single repeat -> bootstrap over that run's rows.

    Args:
        per_run_means: One aggregated score per repeat (e.g. each run's row mean).
        row_values: Every row-level score across all repeats, float-cast. Used to
            detect pass/fail scorers and for the single-repeat bootstrap fallback.
    """
    if _is_binary(row_values):
        successes = sum(1 for v in row_values if v == 1.0)
        return wilson_interval(successes, len(row_values))
    if len(per_run_means) >= 2:
        return t_interval(per_run_means)
    return bootstrap_interval(row_values)
