from mlflow.entities import Metric
from mlflow.entities.assessment import Assessment
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.entities.evaluation import Evaluation
from mlflow.entities.evaluation_tag import EvaluationTag


def test_evaluation_equality():
    source_1 = AssessmentSource(source_type="HUMAN", source_id="user_1")
    metric_1 = Metric(key="metric1", value=1.1, timestamp=123, step=0)
    tag_1 = EvaluationTag(key="tag1", value="value1")

    # Valid evaluations
    evaluation_1 = Evaluation(
        evaluation_id="eval1",
        run_id="run1",
        inputs_id="inputs1",
        inputs={"feature1": 1.0, "feature2": 2.0},
        outputs={"prediction": 0.5},
        request_id="req1",
        targets={"actual": 0.6},
        assessments=[
            Assessment(
                evaluation_id="eval1",
                name="relevance",
                source=source_1,
                timestamp=123456789,
                numeric_value=0.9,
            )
        ],
        metrics=[metric_1],
        tags=[tag_1],
        error_code="E001",
        error_message="An error occurred",
    )
    evaluation_2 = Evaluation(
        evaluation_id="eval1",
        run_id="run1",
        inputs_id="inputs1",
        inputs={"feature1": 1.0, "feature2": 2.0},
        outputs={"prediction": 0.5},
        request_id="req1",
        targets={"actual": 0.6},
        assessments=[
            Assessment(
                evaluation_id="eval1",
                name="relevance",
                source=source_1,
                timestamp=123456789,
                numeric_value=0.9,
            )
        ],
        metrics=[metric_1],
        tags=[tag_1],
        error_code="E001",
        error_message="An error occurred",
    )
    evaluation_3 = Evaluation(
        evaluation_id="eval2",
        run_id="run2",
        inputs_id="inputs2",
        inputs={"feature1": 1.0, "feature2": 2.0},
        outputs={"prediction": 0.5},
        request_id="req2",
        targets={"actual": 0.7},
        assessments=[
            Assessment(
                evaluation_id="eval2",
                name="relevance",
                source=source_1,
                timestamp=123456789,
                numeric_value=0.8,
            )
        ],
        metrics=[Metric(key="metric1", value=1.2, timestamp=123, step=0)],
        tags=[EvaluationTag(key="tag2", value="value2")],
        error_code="E002",
        error_message="Another error occurred",
    )

    assert evaluation_1 == evaluation_2  # Same evaluation data
    assert evaluation_1 != evaluation_3  # Different evaluation data


def test_evaluation_properties():
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")
    metric = Metric(key="metric1", value=1.1, timestamp=123, step=0)
    tag = EvaluationTag(key="tag1", value="value1")
    assessment = Assessment(
        evaluation_id="eval1",
        name="relevance",
        source=source,
        timestamp=123456789,
        numeric_value=0.9,
        rationale="Rationale text",
        metadata={"key1": "value1"},
    )
    evaluation = Evaluation(
        evaluation_id="eval1",
        run_id="run1",
        inputs_id="inputs1",
        inputs={"feature1": 1.0, "feature2": 2.0},
        outputs={"prediction": 0.5},
        request_id="req1",
        targets={"actual": 0.6},
        assessments=[assessment],
        metrics=[metric],
        tags=[tag],
        error_code="E001",
        error_message="An error occurred",
    )

    assert evaluation.evaluation_id == "eval1"
    assert evaluation.run_id == "run1"
    assert evaluation.inputs_id == "inputs1"
    assert evaluation.inputs == {"feature1": 1.0, "feature2": 2.0}
    assert evaluation.outputs == {"prediction": 0.5}
    assert evaluation.request_id == "req1"
    assert evaluation.targets == {"actual": 0.6}
    assert evaluation.error_code == "E001"
    assert evaluation.error_message == "An error occurred"
    assert evaluation.assessments == [assessment]
    assert evaluation.metrics == [metric]
    assert evaluation.tags == [tag]


def test_evaluation_to_from_dictionary():
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")
    metric = Metric(key="metric1", value=1.1, timestamp=123, step=0)
    tag = EvaluationTag(key="tag1", value="value1")
    assessment = Assessment(
        evaluation_id="eval1",
        name="relevance",
        source=source,
        timestamp=123456789,
        numeric_value=0.9,
        rationale="Rationale text",
        metadata={"key1": "value1"},
    )
    evaluation = Evaluation(
        evaluation_id="eval1",
        run_id="run1",
        inputs_id="inputs1",
        inputs={"feature1": 1.0, "feature2": 2.0},
        outputs={"prediction": 0.5},
        request_id="req1",
        targets={"actual": 0.6},
        assessments=[assessment],
        metrics=[metric],
        tags=[tag],
        error_code="E001",
        error_message="An error occurred",
    )
    evaluation_dict = evaluation.to_dictionary()

    expected_dict = {
        "evaluation_id": "eval1",
        "run_id": "run1",
        "inputs_id": "inputs1",
        "inputs": {"feature1": 1.0, "feature2": 2.0},
        "outputs": {"prediction": 0.5},
        "request_id": "req1",
        "targets": {"actual": 0.6},
        "assessments": [assessment.to_dictionary()],
        "metrics": [metric.to_dictionary()],
        "tags": [tag.to_dictionary()],
        "error_code": "E001",
        "error_message": "An error occurred",
    }
    assert evaluation_dict == expected_dict

    recreated_evaluation = Evaluation.from_dictionary(evaluation_dict)
    assert recreated_evaluation == evaluation


def test_evaluation_construction_with_minimal_required_fields():
    evaluation = Evaluation(
        evaluation_id="eval1",
        run_id="run1",
        inputs_id="inputs1",
        inputs={"feature1": 1.0, "feature2": 2.0},
    )
    evaluation_dict = evaluation.to_dictionary()
    recreated_evaluation = Evaluation.from_dictionary(evaluation_dict)
    assert recreated_evaluation == evaluation
