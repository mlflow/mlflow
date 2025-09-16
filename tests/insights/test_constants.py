import pytest

from mlflow.exceptions import MlflowException
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
        (AnalysisStatus, {"ACTIVE", "COMPLETED", "ARCHIVED", "ERROR"}),
        (HypothesisStatus, {"TESTING", "VALIDATED", "REJECTED", "ERROR"}),
        (IssueSeverity, {"LOW", "MEDIUM", "HIGH", "CRITICAL"}),
        (IssueStatus, {"OPEN", "IN_PROGRESS", "RESOLVED", "REJECTED", "ERROR"}),
    ],
)
def test_constant_values(constant_class, expected_values):
    assert constant_class.values() == expected_values


@pytest.mark.parametrize(
    ("constant_class", "valid", "invalid"),
    [
        (AnalysisStatus, "ACTIVE", "INVALID"),
        (HypothesisStatus, "TESTING", "UNKNOWN"),
        (IssueSeverity, "HIGH", "EXTREME"),
        (IssueStatus, "OPEN", "PENDING"),
    ],
)
def test_is_valid(constant_class, valid, invalid):
    assert constant_class.is_valid(valid) is True
    assert constant_class.is_valid(invalid) is False


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
    with pytest.raises(
        MlflowException, match=f"Invalid configuration supplied for {constant_class.__name__}"
    ):
        constant_class.validate(invalid_value)
