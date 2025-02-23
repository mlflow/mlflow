import pytest

import mlflow
from mlflow.entities import Metric
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.evaluation_tag import EvaluationTag
from mlflow.evaluation import Assessment, Evaluation, log_evaluations

from tests.evaluate.logging.utils import get_evaluation


@pytest.fixture
def end_run_at_test_end():
    yield
    mlflow.end_run()


def test_log_evaluations_with_minimal_params_succeeds():
    inputs1 = {"feature1": 1.0, "feature2": 2.0}
    outputs1 = {"prediction": 0.5}

    inputs2 = {"feature3": 3.0, "feature4": 4.0}
    outputs2 = {"prediction": 0.8}

    with mlflow.start_run():
        # Create evaluation objects
        evaluation1 = Evaluation(inputs=inputs1, outputs=outputs1)
        evaluation2 = Evaluation(inputs=inputs2, outputs=outputs2)

        # Log the evaluations
        logged_evaluations = log_evaluations(evaluations=[evaluation1, evaluation2])
        assert len(logged_evaluations) == 2

        for logged_evaluation, expected_evaluation in zip(
            logged_evaluations, [evaluation1, evaluation2]
        ):
            assert logged_evaluation.inputs == expected_evaluation.inputs
            assert logged_evaluation.outputs == expected_evaluation.outputs
            retrieved_evaluation = get_evaluation(
                evaluation_id=logged_evaluation.evaluation_id,
                run_id=mlflow.active_run().info.run_id,
            )
            assert retrieved_evaluation is not None
            assert retrieved_evaluation.inputs == logged_evaluation.inputs
            assert retrieved_evaluation.outputs == logged_evaluation.outputs


def test_log_evaluations_with_all_params():
    evaluations_data = [
        (
            {"feature1": 1.0, "feature2": 2.0},
            {"prediction": 0.5},
            {"actual": 1.0},
            [
                {
                    "name": "assessment1",
                    "value": 1.0,
                    "source": {
                        "source_type": "HUMAN",
                        "source_id": "user_1",
                        "metadata": {"sourcekey1": "sourcevalue1"},
                    },
                },
                {
                    "name": "assessment2",
                    "value": 0.84,
                    "source": {
                        "source_type": "HUMAN",
                        "source_id": "user_1",
                        "metadata": {"sourcekey2": "sourcevalue2"},
                    },
                },
            ],
            [
                Metric(key="metric1", value=1.4, timestamp=1717047609503, step=0),
                Metric(key="metric2", value=1.2, timestamp=1717047609504, step=0),
            ],
            {"tag1": "value1", "tag2": "value2"},
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
                        source_type=AssessmentSourceType.HUMAN,
                        source_id="user-1",
                        metadata={"sourcekey3": "sourcevalue3"},
                    ),
                )
            ],
            {"metric1": 0.8, "metric2": 0.84},
            {"tag3": "value3", "tag4": "value4"},
        ),
    ]

    inputs_id = "unique-inputs-id"
    request_id = "unique-request-id"

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        evaluations = []
        for inputs, outputs, targets, assessments, metrics, tags in evaluations_data:
            if isinstance(assessments[0], dict):
                assessments = [Assessment.from_dictionary(assessment) for assessment in assessments]

            if isinstance(metrics, dict):
                metrics = [
                    Metric(key=key, value=value, timestamp=0, step=0)
                    for key, value in metrics.items()
                ]

            evaluation = Evaluation(
                inputs=inputs,
                outputs=outputs,
                inputs_id=inputs_id,
                request_id=request_id,
                targets=targets,
                assessments=assessments,
                metrics=metrics,
                tags=tags,
            )
            evaluations.append(evaluation)

        # Log the evaluations
        logged_evaluations = log_evaluations(evaluations=evaluations, run_id=run_id)

        for logged_evaluation, (inputs, outputs, targets, assessments, metrics, tags) in zip(
            logged_evaluations, evaluations_data
        ):
            # Assert the fields of the logged evaluation
            assert logged_evaluation.inputs == inputs
            assert logged_evaluation.outputs == outputs
            assert logged_evaluation.inputs_id == inputs_id
            assert logged_evaluation.request_id == request_id
            assert logged_evaluation.targets == targets

            logged_metrics = (
                {metric.key: metric.value for metric in logged_evaluation.metrics}
                if isinstance(metrics, list) and isinstance(metrics[0], Metric)
                else metrics
            )
            assert {
                metric.key: metric.value for metric in logged_evaluation.metrics
            } == logged_metrics

            logged_tags = (
                {tag.key: tag.value for tag in logged_evaluation.tags}
                if isinstance(tags, list) and isinstance(tags[0], EvaluationTag)
                else tags
            )
            assert {tag.key: tag.value for tag in logged_evaluation.tags} == logged_tags

            assessment_entities = [
                Assessment.from_dictionary(assessment)._to_entity(
                    evaluation_id=logged_evaluation.evaluation_id
                )
                if isinstance(assessment, dict)
                else assessment._to_entity(evaluation_id=logged_evaluation.evaluation_id)
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


def test_log_evaluations_starts_run_if_not_started(end_run_at_test_end):
    inputs = {"feature1": 1.0, "feature2": {"nested_feature": 2.0}}
    outputs = {"prediction": 0.5}

    # Ensure there is no active run
    if mlflow.active_run() is not None:
        mlflow.end_run()

    # Log evaluation without explicitly starting a run
    logged_evaluation = log_evaluations(evaluations=[Evaluation(inputs=inputs, outputs=outputs)])[0]

    # Verify that a run has been started
    active_run = mlflow.active_run()
    assert active_run is not None, "Expected a run to be started automatically."

    # Retrieve the evaluation using the run ID
    retrieved_evaluation = get_evaluation(
        evaluation_id=logged_evaluation.evaluation_id, run_id=active_run.info.run_id
    )
    assert retrieved_evaluation == logged_evaluation


def test_evaluation_module_exposes_relevant_apis_for_logging():
    import mlflow.evaluation

    assert hasattr(mlflow.evaluation, "log_evaluations")
    assert hasattr(mlflow.evaluation, "Evaluation")
    assert hasattr(mlflow.evaluation, "Assessment")
    assert hasattr(mlflow.evaluation, "AssessmentSource")
    assert hasattr(mlflow.evaluation, "AssessmentSourceType")


def test_log_evaluations_works_properly_with_empty_evaluations_list():
    with mlflow.start_run():
        log_evaluations(evaluations=[])

        artifacts = mlflow.MlflowClient().list_artifacts(mlflow.active_run().info.run_id)
        assert len(artifacts) == 0
