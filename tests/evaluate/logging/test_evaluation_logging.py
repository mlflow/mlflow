import pytest

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType, Metric
from mlflow.evaluation import Assessment, get_evaluation, log_assessments, log_evaluation
from mlflow.exceptions import MlflowException


def test_log_evaluation_with_minimal_params_succeeds():
    inputs = {"feature1": 1.0, "feature2": 2.0}
    outputs = {"prediction": 0.5}

    with mlflow.start_run():
        logged_evaluation = log_evaluation(inputs=inputs, outputs=outputs)
        assert logged_evaluation is not None
        assert logged_evaluation.inputs_id is not None
        assert logged_evaluation.inputs == inputs
        assert logged_evaluation.outputs == outputs

        retrieved_evaluation = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=mlflow.active_run().info.run_id
        )
        assert retrieved_evaluation == logged_evaluation


class CustomClassA:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, CustomClassA) and self.value == other.value


class CustomClassB:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, CustomClassB) and self.value == other.value


@pytest.mark.parametrize(
    ("inputs", "outputs"),
    [
        ({"feature1": 1.0, "feature2": 2.0}, {"prediction": 0.5}),
        (
            {"feature1": CustomClassA(1), "feature2": CustomClassB(2)},
            {"prediction": CustomClassA(0.5)},
        ),
        (
            {"feature1": [1.0, 2.0, 3.0], "feature2": {"subfeature": CustomClassB(2)}},
            {"prediction": [0.1, 0.2, 0.3]},
        ),
        (
            {"feature1": {"nested": {"subnested": CustomClassA(5)}}, "feature2": CustomClassB(2)},
            {"prediction": {"complex": CustomClassB(0.5)}},
        ),
    ],
)
def test_log_evaluation_with_complex_inputs_outputs(inputs, outputs):
    def compare_dict_keys(dict1, dict2):
        if dict1.keys() != dict2.keys():
            return False
        for key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                if not compare_dict_keys(dict1[key], dict2[key]):
                    return False
        return True

    with mlflow.start_run():
        logged_evaluation = log_evaluation(inputs=inputs, outputs=outputs)
        assert logged_evaluation is not None
        assert logged_evaluation.inputs_id is not None
        assert logged_evaluation.inputs == inputs
        assert logged_evaluation.outputs == outputs

        retrieved_evaluation = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=mlflow.active_run().info.run_id
        )
        assert retrieved_evaluation.inputs_id == logged_evaluation.inputs_id
        assert compare_dict_keys(
            logged_evaluation.inputs, retrieved_evaluation.inputs
        ), "The keys of the nested inputs dictionaries do not match."
        assert compare_dict_keys(
            logged_evaluation.outputs, retrieved_evaluation.outputs
        ), "The keys of the nested outputs dictionaries do not match."


@pytest.mark.parametrize(
    ("inputs", "outputs"),
    [
        ({"feature1": 1.0, "feature2": 2.0}, {"prediction": 0.5}),
        (
            {"feature1": CustomClassA(1), "feature2": CustomClassB(2)},
            {"prediction": CustomClassA(0.5)},
        ),
        (
            {"feature1": [1.0, 2.0, 3.0], "feature2": {"subfeature": CustomClassB(2)}},
            {"prediction": [0.1, 0.2, 0.3]},
        ),
        (
            {"feature1": {"nested": {"subnested": CustomClassA(5)}}, "feature2": CustomClassB(2)},
            {"prediction": {"complex": CustomClassB(0.5)}},
        ),
    ],
)
def test_log_evaluation_with_same_inputs_has_same_inputs_id(inputs, outputs):
    with mlflow.start_run():
        # Log the first evaluation
        first_logged_evaluation = log_evaluation(inputs=inputs, outputs=outputs)
        assert first_logged_evaluation is not None
        assert first_logged_evaluation.inputs_id is not None

        # Log the second evaluation with the same inputs
        second_logged_evaluation = log_evaluation(inputs=inputs, outputs=outputs)
        assert second_logged_evaluation is not None
        assert second_logged_evaluation.inputs_id is not None

        # Assert that the inputs_id is the same for both logged evaluations
        assert (
            first_logged_evaluation.inputs_id == second_logged_evaluation.inputs_id
        ), "The inputs_id should be the same for evaluations logged with the same inputs."

        # Retrieve and verify the evaluations
        run_id = mlflow.active_run().info.run_id
        retrieved_first_evaluation = get_evaluation(
            evaluation_id=first_logged_evaluation.evaluation_id, run_id=run_id
        )
        retrieved_second_evaluation = get_evaluation(
            evaluation_id=second_logged_evaluation.evaluation_id, run_id=run_id
        )

        assert (
            retrieved_first_evaluation.inputs_id == first_logged_evaluation.inputs_id
        ), "inputs_id of the retrieved first evaluation must match the logged first evaluation."
        assert (
            retrieved_second_evaluation.inputs_id == second_logged_evaluation.inputs_id
        ), "inputs_id of the retrieved second evaluation must match the logged second evaluation."


@pytest.mark.parametrize(
    ("inputs", "outputs", "targets", "assessments", "metrics"),
    [
        (
            {"feature1": 1.0, "feature2": 2.0},
            {"prediction": 0.5},
            {"actual": 1.0},
            [
                {
                    "name": "assessment1",
                    "value": 1.0,
                    "source": {"source_type": "HUMAN", "source_id": "user_1"},
                },
                {
                    "name": "assessment2",
                    "value": 0.84,
                    "source": {"source_type": "HUMAN", "source_id": "user_1"},
                },
            ],
            [
                Metric(key="metric1", value=1.4, timestamp=1717047609503, step=0),
                Metric(key="metric2", value=1.2, timestamp=1717047609504, step=0),
            ],
        ),
        (
            {"feature1": "text1", "feature2": "text2"},
            {"prediction": "output_text"},
            {"actual": "expected_text"},
            [
                Assessment(
                    name="accuracy",
                    value=0.8,
                    source=AssessmentSource(
                        source_type=AssessmentSourceType.HUMAN, source_id="user-1"
                    ),
                )
            ],
            {"metric1": 0.8, "metric2": 0.84},
        ),
    ],
)
def test_log_evaluation_with_all_params(inputs, outputs, targets, assessments, metrics):
    inputs_id = "unique-inputs-id"
    request_id = "unique-request-id"

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        # Log the evaluation
        logged_evaluation = log_evaluation(
            inputs=inputs,
            outputs=outputs,
            inputs_id=inputs_id,
            request_id=request_id,
            targets=targets,
            assessments=assessments,
            metrics=metrics,
            run_id=run_id,
        )

        # Assert the fields of the logged evaluation
        assert logged_evaluation.inputs == inputs
        assert logged_evaluation.outputs == outputs
        assert logged_evaluation.inputs_id == inputs_id
        assert logged_evaluation.request_id == request_id
        assert logged_evaluation.targets == targets

        metrics = (
            {metric.key: metric.value for metric in logged_evaluation.metrics}
            if isinstance(metrics, list) and isinstance(metrics[0], Metric)
            else metrics
        )
        assert {metric.key: metric.value for metric in logged_evaluation.metrics} == metrics

        assessments = [
            Assessment.from_dictionary(assessment)
            for assessment in assessments
            if isinstance(assessment, dict)
        ]
        assessment_entities = [
            assessment._to_entity(evaluation_id=logged_evaluation.evaluation_id)
            for assessment in assessments
        ]
        for logged_assessment, assessment_entity in zip(
            logged_evaluation.assessments, assessment_entities
        ):
            assert logged_assessment.name == assessment_entity.name
            assert logged_assessment.boolean_value == assessment_entity.boolean_value
            assert logged_assessment.numeric_value == assessment_entity.numeric_value
            assert logged_assessment.string_value == assessment_entity.string_value
            assert logged_assessment.metadata == assessment_entity.metadata
            assert logged_assessment.source == assessment_entity.source

        retrieved_evaluation = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=run_id
        )
        assert logged_evaluation == retrieved_evaluation


def test_log_evaluation_starts_run_if_not_started():
    inputs = {"feature1": 1.0, "feature2": {"nested_feature": 2.0}}
    outputs = {"prediction": 0.5}

    # Ensure there is no active run
    if mlflow.active_run() is not None:
        mlflow.end_run()

    # Log evaluation without explicitly starting a run
    logged_evaluation = log_evaluation(inputs=inputs, outputs=outputs)

    # Verify that a run has been started
    active_run = mlflow.active_run()
    assert active_run is not None, "Expected a run to be started automatically."

    # Retrieve the evaluation using the run ID
    retrieved_evaluation = get_evaluation(
        evaluation_id=logged_evaluation.evaluation_id, run_id=active_run.info.run_id
    )
    assert retrieved_evaluation == logged_evaluation

    # End the run to clean up
    mlflow.end_run()


def test_log_assessments_with_minimal_params_succeeds():
    inputs = {"feature1": 1.0, "feature2": 2.0}
    outputs = {"prediction": 0.5}

    assessments = [
        Assessment(
            name="relevance",
            value=0.9,
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1"),
        )
    ]

    with mlflow.start_run():
        logged_evaluation = log_evaluation(inputs=inputs, outputs=outputs)

        log_assessments(evaluation_id=logged_evaluation.evaluation_id, assessments=assessments)

        retrieved_evaluation = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=mlflow.active_run().info.run_id
        )

        assert len(retrieved_evaluation.assessments) == 1
        assessment_entities = [
            assessment._to_entity(evaluation_id=logged_evaluation.evaluation_id)
            for assessment in assessments
        ]
        for retrieved_assessment, assessment_entity in zip(
            retrieved_evaluation.assessments, assessment_entities
        ):
            assert retrieved_assessment.name == assessment_entity.name
            assert retrieved_assessment.boolean_value == assessment_entity.boolean_value
            assert retrieved_assessment.numeric_value == assessment_entity.numeric_value
            assert retrieved_assessment.string_value == assessment_entity.string_value
            assert retrieved_assessment.metadata == assessment_entity.metadata
            assert retrieved_assessment.source == assessment_entity.source


def test_log_assessments_without_nonexistent_evaluation_fails():
    with mlflow.start_run():
        with pytest.raises(
            MlflowException, match="The specified run does not contain any evaluations"
        ):
            log_assessments(
                evaluation_id="nonexistent",
                assessments=Assessment(
                    name="assessment_name",
                    value=0.5,
                    source=AssessmentSource(source_type="AI_JUDGE", source_id="judge_id"),
                ),
            )

        log_evaluation(inputs={"feature1": 1.0, "feature2": 2.0}, outputs={"prediction": 0.5})
        with pytest.raises(
            MlflowException,
            match="The specified evaluation ID 'nonexistent' does not exist in the run",
        ):
            log_assessments(
                evaluation_id="nonexistent",
                assessments=Assessment(
                    name="assessment_name",
                    value=0.5,
                    source=AssessmentSource(source_type="AI_JUDGE", source_id="judge_id"),
                ),
            )
