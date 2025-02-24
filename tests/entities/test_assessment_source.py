import pytest

from mlflow.entities.assessment_source import AssessmentSourceV3
from mlflow.exceptions import MlflowException


def test_assessment_source_equality():
    source1 = AssessmentSourceV3(source_type="HUMAN", source_id="user_1")
    source2 = AssessmentSourceV3(source_type="HUMAN", source_id="user_1")
    source3 = AssessmentSourceV3(source_type="LLM_JUDGE", source_id="ai_1")
    source5 = AssessmentSourceV3(source_type="HUMAN", source_id="user_2")

    assert source1 == source2  # Same type and ID
    assert source1 != source3  # Different type
    assert source1 != source5  # Different ID


def test_assessment_source_properties():
    source = AssessmentSourceV3(source_type="HUMAN", source_id="user_1")

    assert source.source_type == "HUMAN"
    assert source.source_id == "user_1"


def test_assessment_source_to_from_dictionary():
    source = AssessmentSourceV3(source_type="HUMAN", source_id="user_1")
    source_dict = source.to_dictionary()

    expected_dict = {
        "source_type": "HUMAN",
        "source_id": "user_1",
    }
    assert source_dict == expected_dict

    recreated_source = AssessmentSourceV3.from_dictionary(source_dict)
    assert recreated_source == source


def test_assessment_source_type_validation():
    # Valid source types
    AssessmentSourceV3(source_type="HUMAN", source_id="user_1")
    AssessmentSourceV3(source_type="LLM_JUDGE", source_id="judge_1")

    # Invalid source type
    with pytest.raises(MlflowException, match="Invalid assessment source type"):
        AssessmentSourceV3(source_type="ROBOT", source_id="robot_1")


def test_assessment_source_case_insensitivity():
    # Valid source types with different cases
    source_1 = AssessmentSourceV3(source_type="human", source_id="user_1")
    source_2 = AssessmentSourceV3(source_type="Human", source_id="user_2")
    source_3 = AssessmentSourceV3(source_type="HUMAN", source_id="user_3")
    source_4 = AssessmentSourceV3(source_type="llm_judge", source_id="judge_1")
    source_5 = AssessmentSourceV3(source_type="Llm_Judge", source_id="judge_2")
    source_6 = AssessmentSourceV3(source_type="LLM_JUDGE", source_id="judge_3")

    # Verify that the source type is normalized to uppercase
    assert source_1.source_type == "HUMAN"
    assert source_2.source_type == "HUMAN"
    assert source_3.source_type == "HUMAN"
    assert source_4.source_type == "LLM_JUDGE"
    assert source_5.source_type == "LLM_JUDGE"
    assert source_6.source_type == "LLM_JUDGE"
