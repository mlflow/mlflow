import pytest

from mlflow.entities.assessment import Assessment, AssessmentSource
from mlflow.exceptions import MlflowException


def test_assessment_equality():
    source_1 = AssessmentSource(source_type="HUMAN", source_id="user_1")
    source_2 = AssessmentSource(source_type="HUMAN", source_id="user_1")
    source_3 = AssessmentSource(source_type="AI_JUDGE", source_id="ai_1")

    # Valid assessments
    assessment_1 = Assessment(
        evaluation_id="eval1",
        name="relevance",
        source=source_1,
        timestamp=123456789,
        numeric_value=0.9,
    )
    assessment_2 = Assessment(
        evaluation_id="eval1",
        name="relevance",
        source=source_2,
        timestamp=123456789,
        numeric_value=0.9,
    )
    assessment_3 = Assessment(
        evaluation_id="eval1",
        name="relevance",
        source=source_1,
        timestamp=123456789,
        numeric_value=0.8,
    )
    assessment_4 = Assessment(
        evaluation_id="eval1",
        name="relevance",
        source=source_3,
        timestamp=123456789,
        numeric_value=0.9,
    )
    assessment_5 = Assessment(
        evaluation_id="eval1",
        name="relevance",
        source=source_1,
        timestamp=123456789,
        error_code="E002",
        error_message="A different error occurred.",
    )
    assessment_6 = Assessment(
        evaluation_id="eval1",
        name="relevance",
        source=source_1,
        timestamp=123456789,
        error_code="E001",
        error_message="Another error message.",
    )

    # Same evaluation_id, name, source, timestamp, and numeric_value
    assert assessment_1 == assessment_2
    assert assessment_1 != assessment_3  # Different numeric_value
    assert assessment_1 != assessment_4  # Different source
    assert assessment_1 != assessment_5  # One has numeric_value, other has error_code
    assert assessment_5 != assessment_6  # Different error_code


def test_assessment_properties():
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")
    assessment = Assessment(
        evaluation_id="eval1",
        name="relevance",
        source=source,
        timestamp=123456789,
        numeric_value=0.9,
        rationale="Rationale text",
        metadata={"key1": "value1"},
    )

    assert assessment.evaluation_id == "eval1"
    assert assessment.name == "relevance"
    assert assessment.source == source
    assert assessment.timestamp == 123456789
    assert assessment.numeric_value == 0.9
    assert assessment.rationale == "Rationale text"
    assert assessment.metadata == {"key1": "value1"}
    assert assessment.error_code is None
    assert assessment.error_message is None


def test_assessment_to_from_dictionary():
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")
    assessment = Assessment(
        evaluation_id="eval1",
        name="relevance",
        source=source,
        timestamp=123456789,
        numeric_value=0.9,
        rationale="Rationale text",
        metadata={"key1": "value1"},
    )
    assessment_dict = assessment.to_dictionary()

    expected_dict = {
        "evaluation_id": "eval1",
        "name": "relevance",
        "source": source.to_dictionary(),
        "timestamp": 123456789,
        "boolean_value": None,
        "numeric_value": 0.9,
        "string_value": None,
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
    Assessment(
        evaluation_id="eval1",
        name="relevance",
        source=source,
        timestamp=123456789,
        boolean_value=True,
    )
    Assessment(
        evaluation_id="eval1",
        name="relevance",
        source=source,
        timestamp=123456789,
        numeric_value=0.9,
    )
    Assessment(
        evaluation_id="eval1",
        name="relevance",
        source=source,
        timestamp=123456789,
        string_value="value",
    )
    Assessment(
        evaluation_id="eval1",
        name="relevance",
        source=source,
        timestamp=123456789,
        error_code="E001",
        error_message="Error",
    )

    # Invalid case: more than one value type specified
    with pytest.raises(
        MlflowException,
        match="Exactly one of boolean_value, numeric_value, string_value, or error_code must be "
        "specified for an assessment.",
    ):
        Assessment(
            evaluation_id="eval1",
            name="relevance",
            source=source,
            timestamp=123456789,
            boolean_value=True,
            numeric_value=0.9,
        )

    # Invalid case: no value type specified
    with pytest.raises(
        MlflowException,
        match="Exactly one of boolean_value, numeric_value, string_value, or error_code must be "
        "specified for an assessment.",
    ):
        Assessment(
            evaluation_id="eval1",
            name="relevance",
            source=source,
            timestamp=123456789,
        )

    # Invalid case: error_message specified with another value type
    with pytest.raises(
        MlflowException,
        match="error_message cannot be specified when boolean_value, numeric_value, or "
        "string_value is specified.",
    ):
        Assessment(
            evaluation_id="eval1",
            name="relevance",
            source=source,
            timestamp=123456789,
            numeric_value=0,
            error_message="An error occurred",
        )

    with pytest.raises(
        MlflowException,
        match="error_message cannot be specified when boolean_value, numeric_value, or "
        "string_value is specified.",
    ):
        Assessment(
            evaluation_id="eval1",
            name="relevance",
            source=source,
            timestamp=123456789,
            string_value="value",
            error_message="An error occurred",
        )

    with pytest.raises(
        MlflowException,
        match="error_message cannot be specified when boolean_value, numeric_value, or "
        "string_value is specified.",
    ):
        Assessment(
            evaluation_id="eval1",
            name="relevance",
            source=source,
            timestamp=123456789,
            boolean_value=False,
            error_message="An error occurred",
        )
