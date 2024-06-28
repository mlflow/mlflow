import pytest

from mlflow.entities import AssessmentSource, EvaluationTag, Metric
from mlflow.evaluation import (
    Assessment,
    Evaluation,
)
from mlflow.exceptions import MlflowException


def test_assessment_equality():
    source_1 = AssessmentSource(source_type="HUMAN", source_id="user_1")
    source_2 = AssessmentSource(source_type="HUMAN", source_id="user_1")
    source_3 = AssessmentSource(source_type="AI_JUDGE", source_id="ai_1")

    # Valid assessments
    assessment_1 = Assessment(
        name="relevance",
        value=0.9,
        source=source_1,
    )
    assessment_2 = Assessment(
        name="relevance",
        value=0.9,
        source=source_2,
    )
    assessment_3 = Assessment(
        name="relevance",
        value=0.8,
        source=source_1,
    )
    assessment_4 = Assessment(
        name="relevance",
        value=0.9,
        source=source_3,
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

    assert assessment_1 == assessment_2  # Same name, value, source
    assert assessment_1 != assessment_3  # Different value
    assert assessment_1 != assessment_4  # Different source
    assert assessment_1 != assessment_5  # One has value, other has error_code
    assert assessment_1 != assessment_6  # One has value, other has error_code


def test_assessment_must_specify_value_or_error_code():
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")

    # Valid assessments
    valid_assessment_with_value = Assessment(name="relevance", value=0.9, source=source)
    valid_assessment_with_error = Assessment(
        name="relevance",
        value=None,
        source=source,
        error_code="E001",
        error_message="An error occurred.",
    )

    assert valid_assessment_with_value.value == 0.9
    assert valid_assessment_with_value.error_code is None
    assert valid_assessment_with_error.value is None
    assert valid_assessment_with_error.error_code == "E001"

    # Invalid assessments
    with pytest.raises(
        MlflowException,
        match="Exactly one of value or error_code must be specified for an assessment",
    ):
        Assessment(name="relevance", value=None, source=source)  # Neither value nor error_code

    with pytest.raises(
        MlflowException,
        match="Exactly one of value or error_code must be specified for an assessment",
    ):
        Assessment(
            name="relevance",
            value=0.9,
            source=source,
            error_code="E001",
            error_message="An error occurred.",
        )  # Both value and error_code


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


def test_evaluation_equality():
    inputs_1 = {"feature1": 1.0, "feature2": 2.0}
    outputs_1 = {"prediction": 0.5}
    targets_1 = {"actual": 0.6}
    metrics_1 = [Metric(key="metric1", value=1.1, timestamp=0, step=0)]
    request_id_1 = "req1"
    inputs_id_1 = "id1"
    error_code_1 = "E001"
    error_message_1 = "An error occurred during evaluation."
    tags_1 = {"tag1": "value1", "tag2": "value2"}

    inputs_2 = {"feature1": 1.0, "feature2": 2.0}
    outputs_2 = {"prediction": 0.5}
    targets_2 = {"actual": 0.6}
    metrics_2 = [Metric(key="metric1", value=1.1, timestamp=0, step=0)]
    request_id_2 = "req1"
    inputs_id_2 = "id1"
    error_code_2 = "E001"
    error_message_2 = "An error occurred during evaluation."
    tags_2 = {"tag1": "value1", "tag2": "value2"}

    inputs_3 = {"feature1": 3.0, "feature2": 4.0}
    outputs_3 = {"prediction": 0.7}
    tags_3 = {"tag3": "value3", "tag4": "value4"}

    inputs_4 = {"feature1": 1.0, "feature2": 2.0}
    outputs_4 = {"prediction": 0.5}
    targets_4 = {"actual": 0.6}
    metrics_4 = [Metric(key="metric1", value=1.1, timestamp=0, step=0)]
    request_id_4 = "req1"
    inputs_id_4 = "id1"
    error_code_4 = "E002"
    error_message_4 = "A different error occurred during evaluation."
    tags_4 = {"tag1": "new_value1", "tag2": "new_value2"}

    source_1 = AssessmentSource(source_type="HUMAN", source_id="user_1")
    assessment_1 = Assessment(
        name="relevance",
        source=source_1,
        value=0.9,
    )
    assessment_2 = Assessment(
        name="relevance",
        source=source_1,
        value=0.9,
    )
    assessment_3 = Assessment(
        name="relevance",
        source=source_1,
        value=1.0,
    )

    evaluation_1 = Evaluation(
        inputs=inputs_1,
        outputs=outputs_1,
        targets=targets_1,
        metrics=metrics_1,
        inputs_id=inputs_id_1,
        request_id=request_id_1,
        assessments=[assessment_1],
        error_code=error_code_1,
        error_message=error_message_1,
        tags=tags_1,
    )
    evaluation_2 = Evaluation(
        inputs=inputs_2,
        outputs=outputs_2,
        targets=targets_2,
        metrics=metrics_2,
        inputs_id=inputs_id_2,
        request_id=request_id_2,
        assessments=[assessment_2],
        error_code=error_code_2,
        error_message=error_message_2,
        tags=tags_2,
    )
    evaluation_3 = Evaluation(inputs=inputs_3, outputs=outputs_3, tags=tags_3)
    evaluation_4 = Evaluation(
        inputs=inputs_4,
        outputs=outputs_4,
        targets=targets_4,
        metrics=metrics_4,
        inputs_id=inputs_id_4,
        request_id=request_id_4,
        assessments=[assessment_3],
        error_code=error_code_4,
        error_message=error_message_4,
        tags=tags_4,
    )

    # Same inputs, outputs, targets, metrics, assessments, inputs_id, request_id,
    # error_code, error_message, tags
    assert evaluation_1 == evaluation_2
    assert evaluation_1 != evaluation_3  # Different inputs, outputs, and tags
    assert evaluation_1 != evaluation_4  # Different error_code, error_message, and tags
    assert (
        evaluation_2 != evaluation_4
    )  # Different assessments, error_code, error_message, and tags


def test_evaluation_tag_equality():
    tag_1 = EvaluationTag(key="tag1", value="value1")
    tag_2 = EvaluationTag(key="tag1", value="value1")
    tag_3 = EvaluationTag(key="tag1", value="value2")
    tag_4 = EvaluationTag(key="tag2", value="value1")

    assert tag_1 == tag_2  # Same key and value
    assert tag_1 != tag_3  # Different value
    assert tag_1 != tag_4  # Different key
