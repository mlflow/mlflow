from unittest.mock import patch

import pytest

from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.evaluation import Assessment
from mlflow.evaluation.assessment import AssessmentSource
from mlflow.exceptions import MlflowException


def test_assessment_equality():
    source_1 = AssessmentSource(source_type="HUMAN", source_id="user_1")
    source_2 = AssessmentSource(source_type="HUMAN", source_id="user_1")
    source_3 = AssessmentSource(source_type="AI_JUDGE", source_id="ai_1")

    # Valid assessments
    assessment_1 = Assessment(
        name="relevance",
        source=source_1,
        value=0.9,
    )
    assessment_2 = Assessment(
        name="relevance",
        source=source_2,
        value=0.9,
    )
    assessment_3 = Assessment(
        name="relevance",
        source=source_1,
        value=0.8,
    )
    assessment_4 = Assessment(
        name="relevance",
        source=source_3,
        value=0.9,
    )
    assessment_5 = Assessment(
        name="relevance",
        source=source_1,
        error_code="E002",
        error_message="A different error occurred.",
    )
    assessment_6 = Assessment(
        name="relevance",
        source=source_1,
        error_code="E001",
        error_message="Another error message.",
    )

    # Same name, source, and value
    assert assessment_1 == assessment_2
    assert assessment_1 != assessment_3  # Different value
    assert assessment_1 != assessment_4  # Different source
    assert assessment_1 != assessment_5  # One has value, other has error_code
    assert assessment_5 != assessment_6  # Different error_code


def test_assessment_properties():
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")
    assessment = Assessment(
        name="relevance",
        source=source,
        value=0.9,
        rationale="Rationale text",
        metadata={"key1": "value1"},
    )

    assert assessment.name == "relevance"
    assert assessment.source == source
    assert assessment.value == 0.9
    assert assessment.rationale == "Rationale text"
    assert assessment.metadata == {"key1": "value1"}
    assert assessment.error_code is None
    assert assessment.error_message is None


def test_assessment_to_from_dictionary():
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")
    assessment = Assessment(
        name="relevance",
        source=source,
        value=0.9,
        rationale="Rationale text",
        metadata={"key1": "value1"},
    )
    assessment_dict = assessment.to_dictionary()

    expected_dict = {
        "name": "relevance",
        "source": source.to_dictionary(),
        "value": 0.9,
        "rationale": "Rationale text",
        "metadata": {"key1": "value1"},
        "error_code": None,
        "error_message": None,
    }
    assert assessment_dict == expected_dict

    recreated_assessment = Assessment.from_dictionary(assessment_dict)
    assert recreated_assessment == assessment


def test_assessment_value_validation():
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")

    # Valid cases
    try:
        Assessment(
            name="relevance",
            source=source,
            value=True,
        )
        Assessment(
            name="relevance",
            source=source,
            value=0.9,
        )
        Assessment(
            name="relevance",
            source=source,
            value="value",
        )
        Assessment(
            name="relevance",
            source=source,
            error_code="E001",
            error_message="Error",
        )
    except MlflowException:
        pytest.fail("Valid value raised exception")

    # Invalid case: more than one value type specified
    with pytest.raises(
        MlflowException,
        match="Exactly one of value or error_code must be specified for an assessment.",
    ):
        Assessment(
            name="relevance",
            source=source,
            value=True,
            error_code="E002",
        )

    # Invalid case: no value type specified
    with pytest.raises(
        MlflowException,
        match="Exactly one of value or error_code must be specified for an assessment.",
    ):
        Assessment(
            name="relevance",
            source=source,
        )

    # Invalid case: error_message specified with another value type
    with pytest.raises(
        MlflowException,
        match="error_message cannot be specified when value is specified.",
    ):
        Assessment(
            name="relevance",
            source=source,
            value=0.9,
            error_message="An error occurred",
        )

    with pytest.raises(
        MlflowException,
        match="error_message cannot be specified when value is specified.",
    ):
        Assessment(
            name="relevance",
            source=source,
            value="value",
            error_message="An error occurred",
        )

    with pytest.raises(
        MlflowException,
        match="error_message cannot be specified when value is specified.",
    ):
        Assessment(
            name="relevance",
            source=source,
            value=True,
            error_message="An error occurred",
        )


def test_assessment_to_entity():
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")
    assessment = Assessment(
        name="relevance",
        source=source,
        value=0.9,
        rationale="Rationale text",
        metadata={"key1": "value1"},
    )

    evaluation_id = "evaluation_1"
    with patch("time.time", return_value=1234567890):
        assessment_entity = assessment._to_entity(evaluation_id)

    assert assessment_entity.evaluation_id == evaluation_id
    assert assessment_entity.name == assessment.name
    assert assessment_entity.source == assessment.source
    assert assessment_entity.boolean_value is None
    assert assessment_entity.numeric_value == assessment.value
    assert assessment_entity.string_value is None
    assert assessment_entity.rationale == assessment.rationale
    assert assessment_entity.metadata == assessment.metadata
    assert assessment_entity.error_code is None
    assert assessment_entity.error_message is None
    assert assessment_entity.timestamp == 1234567890000  # Mocked timestamp


def test_assessment_to_entity_with_error():
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")
    assessment = Assessment(
        name="relevance",
        source=source,
        error_code="E001",
        error_message="An error occurred",
    )

    evaluation_id = "evaluation_1"
    with patch("time.time", return_value=1234567890):
        assessment_entity = assessment._to_entity(evaluation_id)

    assert assessment_entity.evaluation_id == evaluation_id
    assert assessment_entity.name == assessment.name
    assert assessment_entity.source == assessment.source
    assert assessment_entity.boolean_value is None
    assert assessment_entity.numeric_value is None
    assert assessment_entity.string_value is None
    assert assessment_entity.rationale == assessment.rationale
    assert assessment_entity.metadata == assessment.metadata
    assert assessment_entity.error_code == assessment.error_code
    assert assessment_entity.error_message == assessment.error_message
    assert assessment_entity.timestamp == 1234567890000  # Mocked timestamp


def test_assessment_to_entity_with_boolean_value():
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")
    assessment = Assessment(
        name="relevance",
        source=source,
        value=True,
    )

    evaluation_id = "evaluation_1"
    with patch("time.time", return_value=1234567890):
        assessment_entity = assessment._to_entity(evaluation_id)

    assert assessment_entity.evaluation_id == evaluation_id
    assert assessment_entity.name == assessment.name
    assert assessment_entity.source == assessment.source
    assert assessment_entity.boolean_value == assessment.value
    assert assessment_entity.numeric_value is None
    assert assessment_entity.string_value is None
    assert assessment_entity.rationale == assessment.rationale
    assert assessment_entity.metadata == assessment.metadata
    assert assessment_entity.error_code is None
    assert assessment_entity.error_message is None
    assert assessment_entity.timestamp == 1234567890000  # Mocked timestamp


def test_assessment_to_entity_with_string_value():
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")
    assessment = Assessment(
        name="relevance",
        source=source,
        value="string value",
    )

    evaluation_id = "evaluation_1"
    with patch("time.time", return_value=1234567890):
        assessment_entity = assessment._to_entity(evaluation_id)

    assert assessment_entity.evaluation_id == evaluation_id
    assert assessment_entity.name == assessment.name
    assert assessment_entity.source == assessment.source
    assert assessment_entity.boolean_value is None
    assert assessment_entity.numeric_value is None
    assert assessment_entity.string_value == assessment.value
    assert assessment_entity.rationale == assessment.rationale
    assert assessment_entity.metadata == assessment.metadata
    assert assessment_entity.error_code is None
    assert assessment_entity.error_message is None
    assert assessment_entity.timestamp == 1234567890000  # Mocked timestamp


def test_assessment_with_full_metadata():
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")
    assessment = Assessment(
        name="relevance",
        source=source,
        value=0.9,
        rationale="Detailed rationale",
        metadata={"key1": "value1", "key2": "value2"},
    )

    assert assessment.metadata == {"key1": "value1", "key2": "value2"}
    assert assessment.rationale == "Detailed rationale"


def test_assessment_without_metadata():
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")
    assessment = Assessment(
        name="relevance",
        source=source,
        value=0.9,
    )

    assert assessment.metadata == {}
    assert assessment.rationale is None


def test_assessment_without_source():
    assessment = Assessment(
        name="relevance",
        value=0.9,
    )

    assert assessment.source.source_type == AssessmentSourceType.SOURCE_TYPE_UNSPECIFIED

    entity = assessment._to_entity("evaluation_1")
    assert entity.source.source_type == AssessmentSourceType.SOURCE_TYPE_UNSPECIFIED
