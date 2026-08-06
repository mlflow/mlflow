import math

import pytest

from mlflow.genai.evaluation.statistics import (
    Interval,
    _z_for,
    bootstrap_interval,
    compute_scorer_interval,
    t_interval,
    wilson_interval,
)


def test_z_for_95_matches_normal_quantile():
    assert _z_for(0.95) == pytest.approx(1.96, abs=1e-3)


def test_t_interval_matches_manual_computation():
    # mean=0.85, std(ddof=1)=0.05, n=3, t_crit(dof=2)=4.303, stderr=0.05/sqrt(3)
    interval = t_interval([0.8, 0.9, 0.85])
    stderr = 0.05 / math.sqrt(3)
    margin = 4.303 * stderr
    assert interval.mean == pytest.approx(0.85)
    assert interval.std == pytest.approx(0.05)
    assert interval.ci_low == pytest.approx(0.85 - margin, abs=1e-3)
    assert interval.ci_high == pytest.approx(0.85 + margin, abs=1e-3)
    assert interval.method == "t"
    assert interval.n_samples == 3


def test_t_interval_single_value_has_no_interval():
    interval = t_interval([0.7])
    assert interval.mean == 0.7
    assert interval.ci_low == 0.7
    assert interval.ci_high == 0.7
    assert interval.method == "none"


@pytest.mark.parametrize(
    ("successes", "n", "expected_low", "expected_high"),
    [
        # Wilson 95% CI reference values (match R prop.test / statsmodels).
        (8, 10, 0.4902, 0.9433),
        (5, 10, 0.2366, 0.7634),
    ],
)
def test_wilson_interval_reference_values(successes, n, expected_low, expected_high):
    interval = wilson_interval(successes, n)
    assert interval.mean == pytest.approx(successes / n)
    assert interval.ci_low == pytest.approx(expected_low, abs=1e-3)
    assert interval.ci_high == pytest.approx(expected_high, abs=1e-3)
    assert interval.method == "wilson"


@pytest.mark.parametrize(("successes", "n"), [(10, 10), (0, 10)])
def test_wilson_interval_stays_within_unit_range_at_boundaries(successes, n):
    interval = wilson_interval(successes, n)
    assert 0.0 <= interval.ci_low <= interval.ci_high <= 1.0


def test_bootstrap_interval_is_deterministic_and_brackets_mean():
    values = [0.7, 0.9, 0.85, 0.95, 0.8]
    a = bootstrap_interval(values)
    b = bootstrap_interval(values)
    assert a == b  # seeded, reproducible
    assert a.method == "bootstrap"
    assert a.ci_low <= a.mean <= a.ci_high


def test_bootstrap_interval_single_value_has_no_interval():
    interval = bootstrap_interval([0.5])
    assert interval.method == "none"


def test_compute_scorer_interval_binary_uses_wilson():
    interval = compute_scorer_interval(
        per_run_means=[0.8, 1.0],
        row_values=[1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    )
    assert interval.method == "wilson"
    assert 0.0 <= interval.ci_low <= interval.ci_high <= 1.0


def test_compute_scorer_interval_numeric_multi_repeat_uses_t():
    interval = compute_scorer_interval(
        per_run_means=[0.8, 0.9, 0.85],
        row_values=[0.7, 0.9, 0.85, 0.95],
    )
    assert interval.method == "t"


def test_compute_scorer_interval_single_repeat_falls_back_to_bootstrap():
    interval = compute_scorer_interval(
        per_run_means=[0.83],
        row_values=[0.7, 0.9, 0.85, 0.95, 0.8],
    )
    assert interval.method == "bootstrap"


def test_interval_is_frozen():
    interval = Interval(0.5, 0.4, 0.6, 0.1, 3, "t")
    with pytest.raises(AttributeError, match="cannot assign to field"):
        interval.mean = 0.9
