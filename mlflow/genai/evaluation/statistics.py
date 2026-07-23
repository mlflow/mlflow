"""Confidence-interval estimation for evaluation sweeps.

Reduces a scorer's repeated-run scores to a point estimate and a confidence
interval, picking the method automatically: Wilson for pass/fail scorers,
Student's t for numeric scorers with >= 2 repeats, bootstrap over rows when
there is only one repeat. Approximations are scipy-free to keep the skinny
client dependency-light.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

CONFIDENCE_LEVEL = 0.95
N_BOOTSTRAP = 1000
_BOOTSTRAP_SEED = 0

CIMethod = Literal["t", "wilson", "bootstrap", "none"]


@dataclass(frozen=True)
class Interval:
    """A point estimate and confidence interval for a single scorer.

    ``method`` records which estimator produced the bounds (``"t"``,
    ``"wilson"``, ``"bootstrap"``, or ``"none"`` when there were too few
    samples), and ``n_samples`` is the count backing it — the repeat count for
    ``t``, the pooled trial/row count for Wilson/bootstrap.
    """

    mean: float
    ci_low: float
    ci_high: float
    std: float
    n_samples: int
    method: CIMethod


def _z_for(confidence_level: float) -> float:
    return math.sqrt(2.0) * _erfinv(confidence_level)


def _erfinv(x: float) -> float:
    """Inverse error function (Winitzki approximation + one Newton step)."""
    a = 0.147
    ln = math.log(1.0 - x * x)
    term = 2.0 / (math.pi * a) + ln / 2.0
    approx = math.copysign(math.sqrt(math.sqrt(term * term - ln / a) - term), x)
    err = math.erf(approx) - x
    approx -= err / (2.0 / math.sqrt(math.pi) * math.exp(-approx * approx))
    return approx


# Two-sided t critical values at 95%, keyed by degrees of freedom (n - 1).
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

    ``values`` is one score per repeat (typically each run's per-row mean).
    Returns a ``"none"`` interval when there are fewer than 2 repeats.
    """
    n = len(values)
    mean = _mean(values)
    if n < 2:
        return Interval(mean, mean, mean, 0.0, n, "none")
    std = _std(values, ddof=1)
    margin = _t_critical(n - 1, confidence_level) * std / math.sqrt(n)
    return Interval(mean, mean - margin, mean + margin, std, n, "t")


def wilson_interval(successes: int, n: int, confidence_level: float = CONFIDENCE_LEVEL) -> Interval:
    """Wilson score interval for a binomial proportion.

    Correct near the 0 and 1 boundaries where the normal approximation produces
    bounds outside [0, 1]. Used for boolean / pass-fail scorers.
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

    The fallback when ``n_repeats == 1``: with no repeats to measure run-to-run
    variance, resample the run's rows to estimate how much the mean depends on
    which rows were sampled. A different question than the across-repeat t
    interval, so it is only used when repetition is unavailable.
    """
    n = len(row_values)
    mean = _mean(row_values) if n else 0.0
    if n < 2:
        return Interval(mean, mean, mean, 0.0, n, "none")

    rng = random.Random(_BOOTSTRAP_SEED)
    boot_means = sorted(
        _mean([row_values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_bootstrap)
    )
    lo = max(0, min(int((1 - confidence_level) / 2 * n_bootstrap), n_bootstrap - 1))
    hi = max(0, min(int((1 + confidence_level) / 2 * n_bootstrap) - 1, n_bootstrap - 1))
    return Interval(mean, boot_means[lo], boot_means[hi], _std(row_values, ddof=1), n, "bootstrap")


def _is_binary(values: list[float]) -> bool:
    return bool(values) and all(v in (0.0, 1.0) for v in values)


def compute_scorer_interval(*, per_run_means: list[float], row_values: list[float]) -> Interval:
    """Pick and compute the confidence interval for one scorer in a sweep.

    The method is chosen automatically — there is no user-facing knob:

    * pass/fail scorer (all row values 0/1) -> Wilson over the pooled
      (row, repeat) Bernoulli trials,
    * numeric scorer with >= 2 repeats -> Student's t over the per-run means,
    * numeric scorer with a single repeat -> bootstrap over that run's rows.

    Args:
        per_run_means: One aggregated score per repeat (each run's row mean).
        row_values: Every row-level score across all repeats, float-cast; used
            to detect pass/fail scorers and for the single-repeat bootstrap.
    """
    if _is_binary(row_values):
        return wilson_interval(sum(1 for v in row_values if v == 1.0), len(row_values))
    if len(per_run_means) >= 2:
        return t_interval(per_run_means)
    return bootstrap_interval(row_values)
