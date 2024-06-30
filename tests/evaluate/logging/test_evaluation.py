from unittest.mock import patch

from mlflow.entities import Metric
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.entities.evaluation_tag import EvaluationTag
from mlflow.evaluation import Assessment, Evaluation


def test_evaluation_equality():
    inputs = {"feature1": 1.0, "feature2": 2.0}
    outputs = {"prediction": 0.5}
    assessments = [
        Assessment(
            name="assessment1",
            source=AssessmentSource(source_type="HUMAN", source_id="user_1"),
            value=0.9,
        )
    ]
    metrics = [Metric(key="metric1", value=1.1, timestamp=0, step=0)]
    tags = {"tag1": "value1", "tag2": "value2"}

    evaluation_1 = Evaluation(
        inputs=inputs,
        outputs=outputs,
        assessments=assessments,
        metrics=metrics,
        tags=tags,
    )
    evaluation_2 = Evaluation(
        inputs=inputs,
        outputs=outputs,
        assessments=assessments,
        metrics=metrics,
        tags=tags,
    )
    assert evaluation_1 == evaluation_2

    evaluation_3 = Evaluation(
        inputs={"feature1": 3.0, "feature2": 4.0},
        outputs=outputs,
    )
    assert evaluation_1 != evaluation_3


def test_evaluation_properties():
    inputs = {"feature1": 1.0, "feature2": 2.0}
    outputs = {"prediction": 0.5}
    assessments = [
        Assessment(
            name="assessment1",
            source=AssessmentSource(source_type="HUMAN", source_id="user_1"),
            value=0.9,
        )
    ]
    metrics = [Metric(key="metric1", value=1.1, timestamp=0, step=0)]
    tags = {"tag1": "value1", "tag2": "value2"}

    evaluation = Evaluation(
        inputs=inputs,
        outputs=outputs,
        assessments=assessments,
        metrics=metrics,
        tags=tags,
        request_id="req1",
        targets={"target1": 1.0},
        error_code="E001",
        error_message="An error occurred",
    )

    assert evaluation.inputs == inputs
    assert evaluation.outputs == outputs
    assert evaluation.assessments == assessments
    assert evaluation.metrics == metrics
    assert evaluation.tags == [
        EvaluationTag(key="tag1", value="value1"),
        EvaluationTag(key="tag2", value="value2"),
    ]
    assert evaluation.request_id == "req1"
    assert evaluation.targets == {"target1": 1.0}
    assert evaluation.error_code == "E001"
    assert evaluation.error_message == "An error occurred"


def test_evaluation_to_from_dictionary():
    inputs = {"feature1": 1.0, "feature2": 2.0}
    outputs = {"prediction": 0.5}
    assessments = [
        Assessment(
            name="assessment1",
            source=AssessmentSource(source_type="HUMAN", source_id="user_1"),
            value=0.9,
        )
    ]
    metrics = [Metric(key="metric1", value=1.1, timestamp=0, step=0)]
    tags = {"tag1": "value1", "tag2": "value2"}

    evaluation = Evaluation(
        inputs=inputs,
        outputs=outputs,
        assessments=assessments,
        metrics=metrics,
        tags=tags,
        request_id="req1",
        targets={"target1": 1.0},
        error_code="E001",
        error_message="An error occurred",
    )
    evaluation_dict = evaluation.to_dictionary()

    expected_dict = {
        "inputs_id": evaluation.inputs_id,
        "inputs": inputs,
        "outputs": outputs,
        "request_id": "req1",
        "targets": {"target1": 1.0},
        "error_code": "E001",
        "error_message": "An error occurred",
        "assessments": [assessment.to_dictionary() for assessment in assessments],
        "metrics": [metric.to_dictionary() for metric in metrics],
        "tags": [tag.to_dictionary() for tag in evaluation.tags],
    }
    assert evaluation_dict == expected_dict

    recreated_evaluation = Evaluation.from_dictionary(evaluation_dict)
    assert recreated_evaluation == evaluation


# def test_evaluation_inputs_id_hashing():
#     inputs = {"feature1": 1.0, "feature2": 2.0}
#     expected_inputs_id = _generate_inputs_id(inputs)
#
#     evaluation = Evaluation(inputs=inputs)
#     assert evaluation.inputs_id == expected_inputs_id
#
#     evaluation_with_id = Evaluation(inputs=inputs, inputs_id="custom_id")
#     assert evaluation_with_id.inputs_id == "custom_id"


@patch("time.time", return_value=1234567890)
def test_evaluation_to_entity(mock_time):
    inputs = {"feature1": 1.0, "feature2": 2.0}
    outputs = {"prediction": 0.5}
    assessments = [
        Assessment(
            name="assessment1",
            source=AssessmentSource(source_type="HUMAN", source_id="user_1"),
            value=0.9,
        )
    ]
    metrics = [Metric(key="metric1", value=1.1, timestamp=0, step=0)]
    tags = {"tag1": "value1", "tag2": "value2"}

    evaluation = Evaluation(
        inputs=inputs,
        outputs=outputs,
        assessments=assessments,
        metrics=metrics,
        tags=tags,
        request_id="req1",
        targets={"target1": 1.0},
        error_code="E001",
        error_message="An error occurred",
    )

    entity = evaluation._to_entity(run_id="run1", evaluation_id="eval1")
    assert entity.evaluation_id == "eval1"
    assert entity.run_id == "run1"
    assert entity.inputs_id == evaluation.inputs_id
    assert entity.inputs == inputs
    assert entity.outputs == outputs
    assert entity.request_id == "req1"
    assert entity.targets == {"target1": 1.0}
    assert entity.error_code == "E001"
    assert entity.error_message == "An error occurred"
    assert entity.assessments == [a._to_entity("eval1") for a in assessments]
    assert entity.metrics == metrics
    assert entity.tags == [
        EvaluationTag(key="tag1", value="value1"),
        EvaluationTag(key="tag2", value="value2"),
    ]
