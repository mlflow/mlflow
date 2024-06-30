import pytest

from mlflow.entities import AssessmentSource
from mlflow.exceptions import MlflowException


def test_assessment_source_type_validation():
    # Valid source types
    try:
        AssessmentSource(source_type="HUMAN", source_id="user_1")
        AssessmentSource(source_type="AI_JUDGE", source_id="judge_1")
    except MlflowException:
        pytest.fail("Valid source type raised exception")

    # Invalid source type
    with pytest.raises(MlflowException, match="Invalid assessment source type"):
        AssessmentSource(source_type="ROBOT", source_id="robot_1")


def test_assessment_source_case_insensitivity():
    # Valid source types with different cases
    try:
        source_1 = AssessmentSource(source_type="human", source_id="user_1")
        source_2 = AssessmentSource(source_type="Human", source_id="user_2")
        source_3 = AssessmentSource(source_type="HUMAN", source_id="user_3")
        source_4 = AssessmentSource(source_type="ai_judge", source_id="judge_1")
        source_5 = AssessmentSource(source_type="Ai_Judge", source_id="judge_2")
        source_6 = AssessmentSource(source_type="AI_JUDGE", source_id="judge_3")
    except MlflowException:
        pytest.fail("Valid source type raised exception")

    # Verify that the source type is normalized to uppercase
    assert source_1.source_type == "HUMAN"
    assert source_2.source_type == "HUMAN"
    assert source_3.source_type == "HUMAN"
    assert source_4.source_type == "AI_JUDGE"
    assert source_5.source_type == "AI_JUDGE"
    assert source_6.source_type == "AI_JUDGE"
