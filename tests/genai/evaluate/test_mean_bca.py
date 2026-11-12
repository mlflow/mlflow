import inspect

import numpy as np
import pytest
from scipy import stats

from mlflow.genai.evaluation.comparison.paired_comparison import _compare_paired_values
from mlflow.genai.evaluation.comparison.significance_tests import _bootstrap_mean_ci


def scipy_mean_bca(deltas, confidence_level):
    seed_kwarg = "rng" if "rng" in inspect.signature(stats.bootstrap).parameters else "random_state"
    interval = stats.bootstrap(
        (deltas,),
        np.mean,
        method="BCa",
        confidence_level=confidence_level,
        n_resamples=10_000,
        batch=max(1, 1_000_000 // len(deltas)),
        **{seed_kwarg: np.random.default_rng(0)},
    ).confidence_interval
    return interval.low, interval.high


@pytest.mark.parametrize("confidence_level", [0.8, 0.95, 0.99])
@pytest.mark.parametrize(
    "deltas",
    [
        pytest.param(np.array([-4.0, -1.0, 0.0, 0.0, 2.0, 9.0]), id="small-sample"),
        pytest.param(np.random.default_rng(41).normal(size=127), id="normal"),
        pytest.param(np.random.default_rng(42).lognormal(size=43), id="skewed"),
        pytest.param(np.tile([-2.0, 0.0, 1.0], 20), id="tied"),
        pytest.param(np.array([0.0] * 39 + [1.0]), id="binary"),
        pytest.param(np.linspace(-3.0, 2.0, 70)[::-2], id="negative-stride"),
    ],
)
def test_mean_bca_matches_scipy_jackknife(deltas, confidence_level):
    original = deltas.copy()
    expected = scipy_mean_bca(deltas, confidence_level)

    actual = _bootstrap_mean_ci(deltas, confidence_level)

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(deltas, original)
    assert all(isinstance(bound, float) for bound in actual)


@pytest.mark.parametrize("confidence_level", [0.8, 0.99])
def test_paired_comparison_retains_scipy_bca_bounds(confidence_level):
    baseline = np.random.default_rng(31).normal(size=150)
    candidate = baseline - np.random.default_rng(32).lognormal(size=150)
    expected = scipy_mean_bca(candidate - baseline, confidence_level)

    result = _compare_paired_values(
        "score", baseline, candidate, value_type="numeric", confidence_level=confidence_level
    )

    assert result.status == "ok"
    assert result.n_paired == 150
    assert result.diff == pytest.approx(np.mean(candidate - baseline))
    np.testing.assert_allclose((result.ci_low, result.ci_high), expected, rtol=2e-12, atol=2e-12)
