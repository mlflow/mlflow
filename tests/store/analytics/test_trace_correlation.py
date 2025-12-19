import math

import pytest

from mlflow.store.analytics.trace_correlation import (
    calculate_npmi_from_counts,
    calculate_smoothed_npmi,
)


@pytest.mark.parametrize(
    (
        "joint_count",
        "filter1_count",
        "filter2_count",
        "total_count",
        "expected_npmi",
        "expected_smoothed_range",
    ),
    [
        (10, 10, 10, 100, 1.0, (0.95, 1.0)),
        (0, 20, 30, 100, -1.0, None),
        (10, 20, 50, 100, 0.0, None),
        (100, 100, 100, 100, 1.0, None),
    ],
    ids=["perfect_positive", "perfect_negative", "independence", "all_match_both"],
)
def test_npmi_correlations(
    joint_count, filter1_count, filter2_count, total_count, expected_npmi, expected_smoothed_range
):
    result = calculate_npmi_from_counts(joint_count, filter1_count, filter2_count, total_count)

    if expected_npmi == 0.0:
        assert abs(result.npmi) < 0.01
    else:
        assert result.npmi == expected_npmi

    if expected_smoothed_range:
        assert expected_smoothed_range[0] < result.npmi_smoothed <= expected_smoothed_range[1]


@pytest.mark.parametrize(
    ("joint_count", "filter1_count", "filter2_count", "total_count"),
    [
        (0, 0, 10, 100),
        (0, 10, 0, 100),
        (0, 0, 0, 100),
        (0, 0, 0, 0),
        (50, 30, 40, 100),
    ],
    ids=["zero_filter1", "zero_filter2", "both_zero", "empty_dataset", "inconsistent"],
)
def test_npmi_undefined_cases(joint_count, filter1_count, filter2_count, total_count):
    result = calculate_npmi_from_counts(joint_count, filter1_count, filter2_count, total_count)
    assert math.isnan(result.npmi)


def test_npmi_partial_overlap():
    result = calculate_npmi_from_counts(
        joint_count=15, filter1_count=40, filter2_count=30, total_count=100
    )
    assert 0 < result.npmi < 1
    assert 0.1 < result.npmi < 0.2


def test_npmi_with_smoothing():
    result = calculate_npmi_from_counts(
        joint_count=0, filter1_count=2, filter2_count=3, total_count=10
    )
    assert result.npmi == -1.0
    assert result.npmi_smoothed is not None
    assert -1.0 < result.npmi_smoothed < 0

    npmi_smooth = calculate_smoothed_npmi(
        joint_count=0, filter1_count=2, filter2_count=3, total_count=10
    )
    assert -1.0 < npmi_smooth < 0


def test_npmi_all_traces_match_both():
    result = calculate_npmi_from_counts(
        joint_count=100, filter1_count=100, filter2_count=100, total_count=100
    )
    assert result.npmi == 1.0


@pytest.mark.parametrize(
    ("joint_count", "filter1_count", "filter2_count", "total_count"),
    [
        (50, 50, 50, 100),
        (1, 2, 3, 100),
        (99, 99, 99, 100),
        (25, 50, 75, 100),
    ],
    ids=["half_match", "small_counts", "near_all", "quarter_match"],
)
def test_npmi_clamping(joint_count, filter1_count, filter2_count, total_count):
    result = calculate_npmi_from_counts(joint_count, filter1_count, filter2_count, total_count)
    if not math.isnan(result.npmi):
        assert -1.0 <= result.npmi <= 1.0


def test_both_npmi_values_returned():
    result = calculate_npmi_from_counts(
        joint_count=0, filter1_count=10, filter2_count=15, total_count=100
    )

    assert result.npmi == -1.0
    assert result.npmi_smoothed is not None
    assert -1.0 < result.npmi_smoothed < 0

    result2 = calculate_npmi_from_counts(
        joint_count=5, filter1_count=10, filter2_count=15, total_count=100
    )

    assert result2.npmi > 0
    assert result2.npmi_smoothed > 0
    assert abs(result2.npmi - result2.npmi_smoothed) > 0.001


def test_symmetry():
    result_ab = calculate_npmi_from_counts(15, 30, 40, 100)
    result_reversed = calculate_npmi_from_counts(15, 40, 30, 100)
    assert abs(result_ab.npmi - result_reversed.npmi) < 1e-10


def test_monotonicity_joint_count():
    npmis = []
    for joint in range(0, 21):
        result = calculate_npmi_from_counts(joint, 30, 40, 100)
        npmis.append(result.npmi)

    for i in range(1, len(npmis)):
        if not math.isnan(npmis[i]) and not math.isnan(npmis[i - 1]):
            assert npmis[i] >= npmis[i - 1]


@pytest.mark.parametrize(
    ("joint_count", "filter1_count", "filter2_count", "total_count", "expected_range"),
    [
        (30, 30, 50, 100, (0.5, 1.0)),
        (1, 30, 50, 100, (-0.7, -0.5)),
    ],
    ids=["high_overlap", "low_overlap"],
)
def test_boundary_values(joint_count, filter1_count, filter2_count, total_count, expected_range):
    result = calculate_npmi_from_counts(joint_count, filter1_count, filter2_count, total_count)
    assert expected_range[0] < result.npmi < expected_range[1]
