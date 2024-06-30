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


def test_evaluation_inputs_id_uniqueness():
    # Define a few different input objects
    inputs_1 = {"feature1": 1.0, "feature2": 2.0}
    inputs_2 = {"feature1": 1.0, "feature2": 2.0}  # Same as inputs_1
    inputs_3 = {"feature1": 3.0, "feature2": 4.0}  # Different from inputs_1 and inputs_2
    inputs_4 = {"feature1": "value1", "feature2": "value2"}
    inputs_5 = {"feature1": "value1", "feature2": "value2"}  # Same as inputs_4
    inputs_6 = {"feature1": "value3", "feature2": "value4"}  # Different from inputs_4 and inputs_5

    # Create Evaluation objects
    evaluation_1 = Evaluation(inputs=inputs_1)
    evaluation_2 = Evaluation(inputs=inputs_2)
    evaluation_3 = Evaluation(inputs=inputs_3)
    evaluation_4 = Evaluation(inputs=inputs_4)
    evaluation_5 = Evaluation(inputs=inputs_5)
    evaluation_6 = Evaluation(inputs=inputs_6)

    # Verify that inputs_id is the same for equivalent inputs
    assert evaluation_1.inputs_id == evaluation_2.inputs_id
    assert evaluation_4.inputs_id == evaluation_5.inputs_id

    # Verify that inputs_id is different for different inputs
    assert evaluation_1.inputs_id != evaluation_3.inputs_id
    assert evaluation_1.inputs_id != evaluation_4.inputs_id
    assert evaluation_1.inputs_id != evaluation_6.inputs_id
    assert evaluation_4.inputs_id != evaluation_6.inputs_id

    # Additional verification for unique inputs_id generation
    inputs_7 = {"feature1": 5.0, "feature2": 6.0}
    inputs_8 = {"feature1": 7.0, "feature2": 8.0}
    evaluation_7 = Evaluation(inputs=inputs_7)
    evaluation_8 = Evaluation(inputs=inputs_8)

    assert evaluation_7.inputs_id != evaluation_8.inputs_id

    # Ensure different orders of the same inputs result in the same inputs_id
    inputs_9 = {"feature2": 2.0, "feature1": 1.0}  # Same values as inputs_1, but different order
    evaluation_9 = Evaluation(inputs=inputs_9)

    assert evaluation_1.inputs_id == evaluation_9.inputs_id


def test_evaluation_with_non_json_serializable_inputs():
    class NonSerializable:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return f"NonSerializable(value={self.value})"

    # Define non-JSON-serializable inputs
    inputs_1 = {"feature1": NonSerializable(1), "feature2": NonSerializable(2)}
    inputs_2 = {"feature1": NonSerializable(1), "feature2": NonSerializable(2)}  # Same as inputs_1
    inputs_3 = {
        "feature1": NonSerializable(3),
        "feature2": NonSerializable(4),
    }  # Different from inputs_1

    # Create Evaluation objects
    evaluation_1 = Evaluation(inputs=inputs_1)
    evaluation_2 = Evaluation(inputs=inputs_2)
    evaluation_3 = Evaluation(inputs=inputs_3)

    # Verify that inputs_id is the same for equivalent inputs
    assert evaluation_1.inputs_id == evaluation_2.inputs_id

    # Verify that inputs_id is different for different inputs
    assert evaluation_1.inputs_id != evaluation_3.inputs_id

    # Verify that inputs_id is generated
    assert evaluation_1.inputs_id is not None
    assert evaluation_2.inputs_id is not None
    assert evaluation_3.inputs_id is not None
