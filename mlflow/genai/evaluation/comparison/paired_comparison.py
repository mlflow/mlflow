"""Paired comparison of one scorer's values between two evaluation runs."""

import logging
from typing import TYPE_CHECKING, Literal

from mlflow.genai.evaluation.comparison.entities import ScorerComparison
from mlflow.genai.evaluation.comparison.significance_tests import (
    _bootstrap_mean_ci,
    _mcnemar,
    _wilcoxon,
)

# This module is imported by `import mlflow`, which must work on `mlflow-skinny`, so the data
# science libraries are only imported inside the functions that use them.
if TYPE_CHECKING:
    import numpy as np

_logger = logging.getLogger(__name__)

# The smallest n at which the exact two-sided signed-rank p-value can fall below 0.05
# (2 / 2**6 = 0.03125). Below it no result could ever be significant.
_MIN_PAIRS = 6
# Below this many pairs the result is reported but a warning is logged.
_SMALL_SAMPLE_PAIRS = 10


def _compare_paired_values(
    scorer: str,
    baseline: "np.ndarray",
    candidate: "np.ndarray",
    *,
    value_type: Literal["binary", "numeric"],
    greater_is_better: bool = True,
    confidence_level: float = 0.95,
) -> ScorerComparison:
    import numpy as np
    from scipy import stats

    n_paired = len(baseline)
    if n_paired < _MIN_PAIRS:
        return ScorerComparison(
            scorer=scorer,
            status="insufficient_pairs",
            n_paired=n_paired,
            greater_is_better=greater_is_better,
        )
    if n_paired < _SMALL_SAMPLE_PAIRS:
        _logger.warning(
            f"Scorer {scorer!r} has only {n_paired} paired rows. With fewer than "
            f"{_SMALL_SAMPLE_PAIRS} pairs the tests have little power and the confidence "
            "interval is unreliable; evaluate more rows before trusting the result."
        )

    # The test is chosen from the paired values, not from the Python type of the feedback:
    # McNemar's test applies iff every value of both runs is 0 or 1. An ``int(x > 0.5)`` or
    # ``1.0``/``0.0`` scorer is pass/fail in substance, whereas a boolean scorer whose
    # per-trace averages are fractional is not.
    binary = bool(np.isin(np.concatenate([baseline, candidate]), (0.0, 1.0)).all())
    deltas = candidate - baseline
    diff = float(deltas.mean())
    # Differences that are equal up to floating point noise have no variance to test against.
    constant = np.allclose(deltas, deltas[0], rtol=1e-9, atol=1e-12)
    std = 0.0 if constant else float(deltas.std(ddof=1))
    # Resampling a constant can only return it, and the BCa acceleration is undefined there.
    ci_low, ci_high = (diff, diff) if constant else _bootstrap_mean_ci(deltas, confidence_level)

    n_nonzero = int(np.sum(deltas != 0))
    t_test_p_value = None
    if binary:
        method, p_value = _mcnemar(deltas)
    else:
        if constant:
            # Without variance the t statistic is undefined, unless the differences are all
            # zero, which is no evidence of a change.
            t_test_p_value = float("nan") if n_nonzero else 1.0
        else:
            t_test_p_value = float(stats.ttest_rel(candidate, baseline).pvalue)
        method, p_value = _wilcoxon(deltas)

    return ScorerComparison(
        scorer=scorer,
        status="ok",
        n_paired=n_paired,
        greater_is_better=greater_is_better,
        value_type=value_type,
        method=method,
        baseline_mean=float(baseline.mean()),
        candidate_mean=float(candidate.mean()),
        diff=diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        t_test_p_value=t_test_p_value,
        effect_size=diff / std if std > 0 else float("nan"),
        ties=n_paired - n_nonzero,
        n_nonzero=n_nonzero,
    )
