import pytest

from mlflow.insights.constants import (
    INSIGHTS_ANALYSIS_FILE_NAME,
    INSIGHTS_RUN_TAG_NAME_KEY,
    AnalysisStatus,
    HypothesisStatus,
    IssueSeverity,
    IssueStatus,
)


def test_file_constants():
    assert INSIGHTS_ANALYSIS_FILE_NAME == "analysis.yaml"
    assert INSIGHTS_RUN_TAG_NAME_KEY == "mlflow.insights.name"


@pytest.mark.parametrize(
    ("constant_class", "expected_values"),
    [
        (
            AnalysisStatus,
            {
                AnalysisStatus.ACTIVE.value,
                AnalysisStatus.COMPLETED.value,
                AnalysisStatus.ARCHIVED.value,
                AnalysisStatus.ERROR.value,
            },
        ),
        (
            HypothesisStatus,
            {
                HypothesisStatus.TESTING.value,
                HypothesisStatus.VALIDATED.value,
                HypothesisStatus.REJECTED.value,
                HypothesisStatus.ERROR.value,
            },
        ),
        (
            IssueSeverity,
            {
                IssueSeverity.LOW.value,
                IssueSeverity.MEDIUM.value,
                IssueSeverity.HIGH.value,
                IssueSeverity.CRITICAL.value,
            },
        ),
        (
            IssueStatus,
            {
                IssueStatus.OPEN.value,
                IssueStatus.IN_PROGRESS.value,
                IssueStatus.RESOLVED.value,
                IssueStatus.REJECTED.value,
                IssueStatus.ERROR.value,
            },
        ),
    ],
)
def test_constant_values(constant_class, expected_values):
    assert {member.value for member in constant_class} == expected_values


@pytest.mark.parametrize(
    ("constant_class", "valid", "invalid"),
    [
        (AnalysisStatus, AnalysisStatus.ACTIVE.value, "INVALID"),
        (HypothesisStatus, HypothesisStatus.TESTING.value, "UNKNOWN"),
        (IssueSeverity, IssueSeverity.HIGH.value, "EXTREME"),
        (IssueStatus, IssueStatus.OPEN.value, "PENDING"),
    ],
)
def test_is_valid(constant_class, valid, invalid):
    assert constant_class(valid).value == valid

    with pytest.raises(ValueError, match=f"'{invalid}' is not a valid"):
        constant_class(invalid)


@pytest.mark.parametrize(
    ("constant_class", "invalid_value"),
    [
        (AnalysisStatus, "PENDING"),
        (HypothesisStatus, "MAYBE"),
        (IssueSeverity, "VERY_HIGH"),
        (IssueStatus, "CLOSED"),
    ],
)
def test_validate_raises_on_invalid(constant_class, invalid_value):
    match_pattern = f"'{invalid_value}' is not a valid {constant_class.__name__}"
    with pytest.raises(ValueError, match=match_pattern):
        constant_class(invalid_value)
