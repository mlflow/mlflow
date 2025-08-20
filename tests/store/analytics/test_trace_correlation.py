import math

from mlflow.store.analytics.trace_correlation import (
    calculate_expected_and_lift,
    calculate_npmi_from_counts,
)


def test_npmi_perfect_positive_correlation():
    result = calculate_npmi_from_counts(
        joint_count=10, filter1_count=10, filter2_count=10, total_count=100
    )
    assert result.npmi == 1.0
    assert 0.95 < result.npmi_smoothed <= 1.0


def test_npmi_perfect_negative_correlation():
    result = calculate_npmi_from_counts(
        joint_count=0, filter1_count=20, filter2_count=30, total_count=100
    )
    assert result.npmi == -1.0


def test_npmi_independence():
    result = calculate_npmi_from_counts(
        joint_count=10, filter1_count=20, filter2_count=50, total_count=100
    )
    assert abs(result.npmi) < 0.01


def test_npmi_undefined_zero_filter1():
    result = calculate_npmi_from_counts(
        joint_count=0, filter1_count=0, filter2_count=10, total_count=100
    )
    assert math.isnan(result.npmi)


def test_npmi_undefined_zero_filter2():
    result = calculate_npmi_from_counts(
        joint_count=0, filter1_count=10, filter2_count=0, total_count=100
    )
    assert math.isnan(result.npmi)


def test_npmi_undefined_both_zero():
    result = calculate_npmi_from_counts(
        joint_count=0, filter1_count=0, filter2_count=0, total_count=100
    )
    assert math.isnan(result.npmi)


def test_npmi_empty_dataset():
    result = calculate_npmi_from_counts(
        joint_count=0, filter1_count=0, filter2_count=0, total_count=0
    )
    assert math.isnan(result.npmi)


def test_npmi_partial_overlap():
    result = calculate_npmi_from_counts(
        joint_count=15, filter1_count=40, filter2_count=30, total_count=100
    )
    assert 0 < result.npmi < 1
    assert 0.1 < result.npmi < 0.2


def test_npmi_inconsistent_counts():
    result = calculate_npmi_from_counts(
        joint_count=50, filter1_count=30, filter2_count=40, total_count=100
    )
    assert math.isnan(result.npmi)


def test_npmi_with_smoothing():
    result = calculate_npmi_from_counts(
        joint_count=0, filter1_count=2, filter2_count=3, total_count=10
    )
    assert result.npmi == -1.0
    assert result.npmi_smoothed is not None
    assert -1.0 < result.npmi_smoothed < 0

    from mlflow.store.analytics.trace_correlation import calculate_smoothed_npmi

    npmi_smooth = calculate_smoothed_npmi(
        joint_count=0, filter1_count=2, filter2_count=3, total_count=10
    )
    assert -1.0 < npmi_smooth < 0


def test_npmi_all_traces_match_both():
    result = calculate_npmi_from_counts(
        joint_count=100, filter1_count=100, filter2_count=100, total_count=100
    )
    assert result.npmi == 1.0


def test_npmi_clamping():
    test_cases = [
        (50, 50, 50, 100),
        (1, 2, 3, 100),
        (99, 99, 99, 100),
        (25, 50, 75, 100),
    ]

    for joint, f1, f2, total in test_cases:
        result = calculate_npmi_from_counts(joint, f1, f2, total)
        if not math.isnan(result.npmi):
            assert -1.0 <= result.npmi <= 1.0


def test_expected_joint_independence():
    lift_result = calculate_expected_and_lift(
        joint_count=12, filter1_count=20, filter2_count=60, total_count=100
    )
    assert lift_result.expected_joint == 12.0
    assert lift_result.lift == 1.0


def test_lift_positive_association():
    lift_result = calculate_expected_and_lift(
        joint_count=20, filter1_count=30, filter2_count=40, total_count=100
    )
    assert lift_result.expected_joint == 12.0
    assert lift_result.lift > 1.0
    assert abs(lift_result.lift - 20 / 12) < 0.01


def test_lift_negative_association():
    lift_result = calculate_expected_and_lift(
        joint_count=5, filter1_count=30, filter2_count=40, total_count=100
    )
    assert lift_result.expected_joint == 12.0
    assert lift_result.lift < 1.0
    assert abs(lift_result.lift - 5 / 12) < 0.01


def test_lift_zero_joint():
    lift_result = calculate_expected_and_lift(
        joint_count=0, filter1_count=30, filter2_count=40, total_count=100
    )
    assert lift_result.expected_joint == 12.0
    assert lift_result.lift == 0.0


def test_lift_zero_expected():
    lift_result = calculate_expected_and_lift(
        joint_count=0, filter1_count=0, filter2_count=40, total_count=100
    )
    assert lift_result.expected_joint == 0.0
    assert lift_result.lift is None


def test_empty_dataset():
    lift_result = calculate_expected_and_lift(
        joint_count=0, filter1_count=0, filter2_count=0, total_count=0
    )
    assert lift_result.expected_joint is None
    assert lift_result.lift is None


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


def test_boundary_values():
    result = calculate_npmi_from_counts(
        joint_count=30, filter1_count=30, filter2_count=50, total_count=100
    )
    assert 0.5 < result.npmi < 1.0

    result = calculate_npmi_from_counts(
        joint_count=1, filter1_count=30, filter2_count=50, total_count=100
    )
    assert -0.7 < result.npmi < -0.5
