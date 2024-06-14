from mlflow.entities import AssessmentSource, Metric
from mlflow.evaluation import (
    Assessment,
    Evaluation,
)


def test_assessment_equality():
    source_1 = AssessmentSource(source_type="HUMAN", source_id="user_1")
    source_2 = AssessmentSource(source_type="HUMAN", source_id="user_1")
    source_3 = AssessmentSource(source_type="AI", source_id="ai_1")

    assessment_1 = Assessment(name="relevance", value=0.9, source=source_1)
    assessment_2 = Assessment(name="relevance", value=0.9, source=source_2)
    assessment_3 = Assessment(name="relevance", value=0.8, source=source_1)
    assessment_4 = Assessment(name="relevance", value=0.9, source=source_3)

    assert assessment_1 == assessment_2  # Same name, value, and source
    assert assessment_1 != assessment_3  # Different value
    assert assessment_1 != assessment_4  # Different source


def test_evaluation_equality():
    inputs_1 = {"feature1": 1.0, "feature2": 2.0}
    outputs_1 = {"prediction": 0.5}
    targets_1 = {"actual": 0.6}
    metrics_1 = [Metric(key="metric1", value=1.1, timestamp=0, step=0)]
    request_id_1 = "req1"
    inputs_id_1 = "id1"
    error_code_1 = "E001"
    error_message_1 = "An error occurred during evaluation."

    inputs_2 = {"feature1": 1.0, "feature2": 2.0}
    outputs_2 = {"prediction": 0.5}
    targets_2 = {"actual": 0.6}
    metrics_2 = [Metric(key="metric1", value=1.1, timestamp=0, step=0)]
    request_id_2 = "req1"
    inputs_id_2 = "id1"
    error_code_2 = "E001"
    error_message_2 = "An error occurred during evaluation."

    inputs_3 = {"feature1": 3.0, "feature2": 4.0}
    outputs_3 = {"prediction": 0.7}

    inputs_4 = {"feature1": 1.0, "feature2": 2.0}
    outputs_4 = {"prediction": 0.5}
    targets_4 = {"actual": 0.6}
    metrics_4 = [Metric(key="metric1", value=1.1, timestamp=0, step=0)]
    request_id_4 = "req1"
    inputs_id_4 = "id1"
    error_code_4 = "E002"
    error_message_4 = "A different error occurred during evaluation."

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
    )
    evaluation_3 = Evaluation(inputs=inputs_3, outputs=outputs_3)
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
    )

    # Same inputs, outputs, targets, metrics, assessments, inputs_id, request_id,
    # error_code, error_message
    assert evaluation_1 == evaluation_2
    assert evaluation_1 != evaluation_3  # Different inputs and outputs
    assert evaluation_1 != evaluation_4  # Different error_code and error_message
    assert evaluation_2 != evaluation_4  # Different assessments, error_code, and error_message
