import pandas as pd
import pytest

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType, EvaluationTag, Metric
from mlflow.evaluation import (
    Assessment,
    Evaluation,
    get_evaluation,
    log_assessments,
    log_evaluation,
    log_evaluations,
    log_evaluations_df,
    search_evaluations,
    set_evaluation_tags,
)
from mlflow.exceptions import MlflowException


def test_log_evaluation_with_minimal_params_succeeds():
    inputs = {"feature1": 1.0, "feature2": 2.0}

    with mlflow.start_run():
        logged_evaluation = log_evaluation(inputs=inputs)
        assert logged_evaluation is not None
        assert logged_evaluation.inputs_id is not None
        assert logged_evaluation.inputs == inputs

        assert logged_evaluation.outputs is None
        assert logged_evaluation.request_id is None
        assert logged_evaluation.targets is None
        assert logged_evaluation.assessments is None
        assert logged_evaluation.metrics is None
        assert logged_evaluation.error_code is None
        assert logged_evaluation.error_message is None

        retrieved_evaluation = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=mlflow.active_run().info.run_id
        )
        assert retrieved_evaluation == logged_evaluation


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


def test_log_evaluations_df_with_minimal_params_succeeds():
    with mlflow.start_run():
        # Define the input DataFrame
        data = {
            "feature1": [1.0, 3.0],
            "feature2": [2.0, 4.0],
            "prediction": [0.5, 0.72],
        }
        evaluations_df = pd.DataFrame(data)

        # Define the columns
        input_cols = ["feature1", "feature2"]
        output_cols = ["prediction"]

        # Log the evaluations
        result_df = log_evaluations_df(
            evaluations_df=evaluations_df,
            input_cols=input_cols,
            output_cols=output_cols,
        )

        # Verify that the evaluation IDs have been added to the DataFrame
        assert "evaluation_id" in result_df.columns
        assert len(result_df["evaluation_id"]) == len(evaluations_df)

        # Verify that the evaluations have been logged correctly
        for evaluation_id in result_df["evaluation_id"]:
            retrieved_evaluation = get_evaluation(
                evaluation_id=evaluation_id, run_id=mlflow.active_run().info.run_id
            )
            assert retrieved_evaluation is not None

            # Check that the inputs and outputs match the original DataFrame
            original_row = evaluations_df[result_df["evaluation_id"] == evaluation_id].iloc[0]
            assert retrieved_evaluation.inputs["feature1"] == original_row["feature1"]
            assert retrieved_evaluation.inputs["feature2"] == original_row["feature2"]
            assert retrieved_evaluation.outputs["prediction"] == original_row["prediction"]


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
    (
        "inputs",
        "outputs",
        "targets",
        "assessments",
        "metrics",
        "tags",
        "error_code",
        "error_message",
    ),
    [
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
                {
                    "name": "assessment_error",
                    "source": {
                        "source_type": "HUMAN",
                        "source_id": "user_2",
                        "metadata": {"sourcekey3": "sourcevalue3"},
                    },
                    "error_code": "E002",
                    "error_message": "Another error occurred during assessment.",
                },
            ],
            [
                Metric(key="metric1", value=1.4, timestamp=1717047609503, step=0),
                Metric(key="metric2", value=1.2, timestamp=1717047609504, step=0),
            ],
            {"tag1": "value1", "tag2": "value2"},
            "E001",
            "An error occurred during evaluation.",
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
                        metadata={"sourcekey4": "sourcevalue4"},
                    ),
                )
            ],
            {"metric1": 0.8, "metric2": 0.84},
            {"tag3": "value3", "tag4": "value4"},
            "E002",
            "Another error occurred.",
        ),
    ],
)
def test_log_evaluation_with_all_params(
    inputs, outputs, targets, assessments, metrics, tags, error_code, error_message
):
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
            tags=tags,
            error_code=error_code,
            error_message=error_message,
            run_id=run_id,
        )

        # Assert the fields of the logged evaluation
        assert logged_evaluation.inputs == inputs
        assert logged_evaluation.outputs == outputs
        assert logged_evaluation.inputs_id == inputs_id
        assert logged_evaluation.request_id == request_id
        assert logged_evaluation.targets == targets
        assert logged_evaluation.error_code == error_code
        assert logged_evaluation.error_message == error_message

        metrics = (
            {metric.key: metric.value for metric in logged_evaluation.metrics}
            if isinstance(metrics, list) and isinstance(metrics[0], Metric)
            else metrics
        )
        assert {metric.key: metric.value for metric in logged_evaluation.metrics} == metrics

        tags = (
            {tag.key: tag.value for tag in logged_evaluation.tags}
            if isinstance(tags, list) and isinstance(tags[0], EvaluationTag)
            else tags
        )
        assert {tag.key: tag.value for tag in logged_evaluation.tags} == tags

        # Process assessments
        processed_assessments = []
        for assessment in assessments:
            if isinstance(assessment, dict):
                if "value" in assessment:
                    assessment_obj = Assessment.from_dictionary(assessment)
                else:
                    assessment_obj = Assessment(
                        name=assessment["name"],
                        source=AssessmentSource.from_dictionary(assessment["source"]),
                        value=None,
                        error_code=assessment.get("error_code"),
                        error_message=assessment.get("error_message"),
                    )
                processed_assessments.append(assessment_obj)

        assessment_entities = [
            assessment._to_entity(evaluation_id=logged_evaluation.evaluation_id)
            for assessment in processed_assessments
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
            assert logged_assessment.error_code == assessment_entity.error_code
            assert logged_assessment.error_message == assessment_entity.error_message

        retrieved_evaluation = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=run_id
        )
        assert logged_evaluation == retrieved_evaluation


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


def test_log_evaluation_df_with_all_params():
    with mlflow.start_run():
        # Define the input DataFrame
        data = {
            "feature1": [1.0, 3.0],
            "feature2": [2.0, 4.0],
            "prediction": [0.5, 0.72],
            "actual": [0.62, 0.74],
            "inputs_id": ["id1", "id2"],
            "request_id": ["req1", "req2"],
            "metrics": [{"metric1": 1.1}, [Metric(key="metric2", value=1.2, timestamp=0, step=0)]],
            "error_code": ["E001", "E002"],
            "error_message": ["An error occurred during evaluation.", "Another error occurred."],
        }
        evaluations_df = pd.DataFrame(data)

        # Define the columns
        input_cols = ["feature1", "feature2"]
        output_cols = ["prediction"]
        target_cols = ["actual"]
        inputs_id_col = "inputs_id"

        # Log the evaluations
        result_df = log_evaluations_df(
            run_id=mlflow.active_run().info.run_id,
            evaluations_df=evaluations_df,
            input_cols=input_cols,
            output_cols=output_cols,
            target_cols=target_cols,
            inputs_id_col=inputs_id_col,
        )

        # Verify that the evaluation IDs have been added to the DataFrame
        assert "evaluation_id" in result_df.columns
        assert len(result_df["evaluation_id"]) == len(evaluations_df)

        # Verify that the evaluations have been logged correctly
        for evaluation_id in result_df["evaluation_id"]:
            retrieved_evaluation = get_evaluation(
                evaluation_id=evaluation_id, run_id=mlflow.active_run().info.run_id
            )
            assert retrieved_evaluation is not None

            # Check that the inputs, outputs, targets, metrics, request_id, error_code, and
            # error_message match the original DataFrame
            original_row = evaluations_df[result_df["evaluation_id"] == evaluation_id].iloc[0]
            assert retrieved_evaluation.inputs["feature1"] == original_row["feature1"]
            assert retrieved_evaluation.inputs["feature2"] == original_row["feature2"]
            assert retrieved_evaluation.outputs["prediction"] == original_row["prediction"]
            assert retrieved_evaluation.targets["actual"] == original_row["actual"]
            assert retrieved_evaluation.inputs_id == original_row["inputs_id"]
            assert retrieved_evaluation.request_id == original_row["request_id"]
            assert retrieved_evaluation.error_code == original_row["error_code"]
            assert retrieved_evaluation.error_message == original_row["error_message"]

            if isinstance(original_row["metrics"], dict):
                assert {met.key: met.value for met in retrieved_evaluation.metrics} == original_row[
                    "metrics"
                ]
            else:
                assert retrieved_evaluation.metrics == original_row["metrics"]


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


@pytest.mark.parametrize(
    "assessments",
    [
        Assessment(
            name="relevance",
            value=0.9,
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1"),
        ),
        [
            Assessment(
                name="relevance",
                value=0.9,
                source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1"),
            ),
            Assessment(
                name="accuracy",
                value=0.8,
                source=AssessmentSource(
                    source_type=AssessmentSourceType.AI_JUDGE, source_id="judge_1"
                ),
            ),
        ],
        {
            "name": "relevance",
            "value": 0.9,
            "source": {"source_type": "HUMAN", "source_id": "user_1"},
        },
        [
            {
                "name": "relevance",
                "value": 0.9,
                "source": {"source_type": "HUMAN", "source_id": "user_1"},
            },
            {
                "name": "accuracy",
                "value": 0.8,
                "source": {"source_type": "AI_JUDGE", "source_id": "judge_1"},
            },
        ],
    ],
)
def test_log_assessments_with_varying_formats(assessments):
    inputs = {"feature1": 1.0, "feature2": 2.0}
    outputs = {"prediction": 0.5}

    with mlflow.start_run() as run:
        logged_evaluation = log_evaluation(inputs=inputs, outputs=outputs)

        log_assessments(evaluation_id=logged_evaluation.evaluation_id, assessments=assessments)

        # Verify that the evaluation and assessments have been logged correctly
        run_id = run.info.run_id
        retrieved_evaluation = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=run_id
        )

        # Convert the expected assessments to Assessment objects for comparison
        if isinstance(assessments, dict):
            expected_assessments = [Assessment.from_dictionary(assessments)]
        elif isinstance(assessments, list) and all(isinstance(a, dict) for a in assessments):
            expected_assessments = [Assessment.from_dictionary(a) for a in assessments]
        else:
            expected_assessments = assessments if isinstance(assessments, list) else [assessments]

        assert len(retrieved_evaluation.assessments) == len(expected_assessments)
        expected_assessment_entities = [
            assessment._to_entity(evaluation_id=logged_evaluation.evaluation_id)
            for assessment in expected_assessments
        ]
        for retrieved_assessment, assessment_entity in zip(
            retrieved_evaluation.assessments, expected_assessment_entities
        ):
            assert retrieved_assessment.name == assessment_entity.name
            assert retrieved_assessment.boolean_value == assessment_entity.boolean_value
            assert retrieved_assessment.numeric_value == assessment_entity.numeric_value
            assert retrieved_assessment.string_value == assessment_entity.string_value
            assert retrieved_assessment.metadata == assessment_entity.metadata
            assert retrieved_assessment.source == assessment_entity.source


def test_incremental_logging_of_assessments():
    inputs = {"feature1": 1.0, "feature2": 2.0}
    outputs = {"prediction": 0.5}

    assessment1 = Assessment(
        name="relevance",
        value=0.9,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1"),
    )

    assessment2 = Assessment(
        name="accuracy",
        value=0.8,
        source=AssessmentSource(source_type=AssessmentSourceType.AI_JUDGE, source_id="judge_1"),
    )

    assessment3 = Assessment(
        name="error_assessment",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_2"),
        error_code="E001",
        error_message="An error occurred during the assessment.",
    )

    def assert_assessments_equal(assessment, expected_assessment):
        assert assessment.name == expected_assessment.name
        assert assessment.boolean_value == expected_assessment.boolean_value
        assert assessment.numeric_value == expected_assessment.numeric_value
        assert assessment.string_value == expected_assessment.string_value
        assert assessment.metadata == expected_assessment.metadata
        assert assessment.source == expected_assessment.source
        assert assessment.error_code == expected_assessment.error_code
        assert assessment.error_message == expected_assessment.error_message

    with mlflow.start_run() as run:
        logged_evaluation = log_evaluation(inputs=inputs, outputs=outputs)

        log_assessments(evaluation_id=logged_evaluation.evaluation_id, assessments=assessment1)

        run_id = run.info.run_id
        retrieved_evaluation1 = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=run_id
        )
        assert len(retrieved_evaluation1.assessments) == 1
        retrieved_assessment1 = retrieved_evaluation1.assessments[0]
        assert_assessments_equal(
            retrieved_assessment1,
            assessment1._to_entity(evaluation_id=logged_evaluation.evaluation_id),
        )

        log_assessments(evaluation_id=logged_evaluation.evaluation_id, assessments=assessment2)

        retrieved_evaluation2 = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=run_id
        )
        assert len(retrieved_evaluation2.assessments) == 2
        for retrieved_assessment, expected_assessment in zip(
            retrieved_evaluation2.assessments, [assessment1, assessment2]
        ):
            assert_assessments_equal(
                retrieved_assessment,
                expected_assessment._to_entity(evaluation_id=logged_evaluation.evaluation_id),
            )

        log_assessments(evaluation_id=logged_evaluation.evaluation_id, assessments=assessment3)

        retrieved_evaluation3 = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=run_id
        )
        assert len(retrieved_evaluation3.assessments) == 3
        for retrieved_assessment, expected_assessment in zip(
            retrieved_evaluation3.assessments, [assessment1, assessment2, assessment3]
        ):
            assert_assessments_equal(
                retrieved_assessment,
                expected_assessment._to_entity(evaluation_id=logged_evaluation.evaluation_id),
            )


@pytest.mark.parametrize(
    ("assessment", "log_with_evaluation"),
    [
        (
            Assessment(
                name="boolean_assessment",
                value=True,
                source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1"),
            ),
            True,
        ),
        (
            Assessment(
                name="string_assessment",
                value="good",
                source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_2"),
            ),
            False,
        ),
        (
            Assessment(
                name="float_assessment",
                value=0.9,
                source=AssessmentSource(
                    source_type=AssessmentSourceType.AI_JUDGE, source_id="judge_1"
                ),
            ),
            True,
        ),
        (
            Assessment(
                name="integer_assessment",
                value=10,
                source=AssessmentSource(
                    source_type=AssessmentSourceType.AI_JUDGE, source_id="judge_2"
                ),
            ),
            False,
        ),
    ],
)
def test_log_assessments_with_varying_value_types(assessment, log_with_evaluation):
    inputs = {"feature1": 1.0, "feature2": 2.0}
    outputs = {"prediction": 0.5}

    with mlflow.start_run() as run:
        if log_with_evaluation:
            logged_evaluation = log_evaluation(
                inputs=inputs, outputs=outputs, assessments=[assessment]
            )
        else:
            logged_evaluation = log_evaluation(inputs=inputs, outputs=outputs)
            log_assessments(evaluation_id=logged_evaluation.evaluation_id, assessments=assessment)

        run_id = run.info.run_id
        retrieved_evaluation = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=run_id
        )
        assert len(retrieved_evaluation.assessments) == 1

        retrieved_assessment = retrieved_evaluation.assessments[0]
        expected_assessment_entity = assessment._to_entity(
            evaluation_id=logged_evaluation.evaluation_id
        )
        assert retrieved_assessment.name == expected_assessment_entity.name
        assert retrieved_assessment.boolean_value == expected_assessment_entity.boolean_value
        assert retrieved_assessment.numeric_value == expected_assessment_entity.numeric_value
        assert retrieved_assessment.string_value == expected_assessment_entity.string_value
        assert retrieved_assessment.metadata == expected_assessment_entity.metadata
        assert retrieved_assessment.source == expected_assessment_entity.source
        if isinstance(assessment.value, bool):
            assert retrieved_assessment.boolean_value == assessment.value
            assert retrieved_assessment.string_value is None
            assert retrieved_assessment.numeric_value is None
        elif isinstance(assessment.value, str):
            assert retrieved_assessment.string_value == assessment.value
            assert retrieved_assessment.boolean_value is None
            assert retrieved_assessment.numeric_value is None
        elif isinstance(assessment.value, (int, float)):
            assert retrieved_assessment.numeric_value == assessment.value
            assert retrieved_assessment.boolean_value is None
            assert retrieved_assessment.string_value is None
        else:
            raise ValueError(f"Unexpected assessment value type: {type(assessment.value)}.")


def test_logging_assessments_to_multiple_evaluations():
    inputs_1 = {"feature1": 1.0, "feature2": 2.0}
    outputs_1 = {"prediction": 0.5}
    inputs_2 = {"feature1": 3.0, "feature2": 4.0}
    outputs_2 = {"prediction": 0.7}

    assessment1 = Assessment(
        name="relevance",
        value=0.9,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1"),
    )

    assessment2 = Assessment(
        name="relevance",
        value=0.8,
        source=AssessmentSource(source_type=AssessmentSourceType.AI_JUDGE, source_id="judge_1"),
    )

    with mlflow.start_run() as run:
        # Log the first evaluation and assessments
        logged_evaluation1 = log_evaluation(inputs=inputs_1, outputs=outputs_1)
        log_assessments(evaluation_id=logged_evaluation1.evaluation_id, assessments=assessment1)

        # Log the second evaluation and assessments
        logged_evaluation2 = log_evaluation(inputs=inputs_2, outputs=outputs_2)
        log_assessments(evaluation_id=logged_evaluation2.evaluation_id, assessments=assessment2)

        def assert_assessments_equal(assessment, expected_assessment):
            assert assessment.name == expected_assessment.name
            assert assessment.boolean_value == expected_assessment.boolean_value
            assert assessment.numeric_value == expected_assessment.numeric_value
            assert assessment.string_value == expected_assessment.string_value
            assert assessment.metadata == expected_assessment.metadata
            assert assessment.source == expected_assessment.source

    run_id = run.info.run_id

    retrieved_evaluation1 = get_evaluation(
        evaluation_id=logged_evaluation1.evaluation_id, run_id=run_id
    )
    assert len(retrieved_evaluation1.assessments) == 1
    retrieved_assessment1 = retrieved_evaluation1.assessments[0]

    assert_assessments_equal(
        retrieved_assessment1,
        assessment1._to_entity(evaluation_id=logged_evaluation1.evaluation_id),
    )

    retrieved_evaluation2 = get_evaluation(
        evaluation_id=logged_evaluation2.evaluation_id, run_id=run_id
    )
    assert len(retrieved_evaluation2.assessments) == 1
    retrieved_assessment2 = retrieved_evaluation2.assessments[0]

    assert_assessments_equal(
        retrieved_assessment2,
        assessment2._to_entity(evaluation_id=logged_evaluation2.evaluation_id),
    )


def test_log_multiple_assessments_with_same_name_different_sources():
    inputs = {"feature1": 1.0, "feature2": 2.0}
    outputs = {"prediction": 0.5}

    assessments = [
        Assessment(
            name="relevance",
            value=0.9,
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1"),
        ),
        Assessment(
            name="relevance",
            value=0.8,
            source=AssessmentSource(source_type=AssessmentSourceType.AI_JUDGE, source_id="judge_1"),
        ),
        Assessment(
            name="relevance",
            value=0.85,
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_2"),
        ),
        Assessment(
            name="relevance",
            value=0.8,
            source=AssessmentSource(source_type=AssessmentSourceType.AI_JUDGE, source_id="judge_2"),
        ),
    ]

    with mlflow.start_run() as run:
        logged_evaluation = log_evaluation(inputs=inputs, outputs=outputs)
        log_assessments(evaluation_id=logged_evaluation.evaluation_id, assessments=assessments)

        run_id = run.info.run_id
        retrieved_evaluation = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=run_id
        )
        assert len(retrieved_evaluation.assessments) == len(assessments) == 4

        expected_assessment_entities = [
            assessment._to_entity(evaluation_id=logged_evaluation.evaluation_id)
            for assessment in assessments
        ]
        for retrieved_assessment, assessment_entity in zip(
            retrieved_evaluation.assessments, expected_assessment_entities
        ):
            assert retrieved_assessment.name == assessment_entity.name
            assert retrieved_assessment.boolean_value == assessment_entity.boolean_value
            assert retrieved_assessment.numeric_value == assessment_entity.numeric_value
            assert retrieved_assessment.string_value == assessment_entity.string_value
            assert retrieved_assessment.metadata == assessment_entity.metadata
            assert retrieved_assessment.source == assessment_entity.source


def test_log_assessments_with_same_name_and_source():
    inputs_1 = {"feature1": 1.0, "feature2": 2.0}
    outputs_1 = {"prediction": 0.5}
    inputs_2 = {"feature1": 3.0, "feature2": 4.0}
    outputs_2 = {"prediction": 0.7}

    assessment_1 = Assessment(
        name="relevance",
        value=0.9,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1"),
    )

    assessment_2 = Assessment(
        name="relevance",
        value=0.8,
        source=AssessmentSource(source_type=AssessmentSourceType.AI_JUDGE, source_id="judge_1"),
    )

    updated_assessment_1 = Assessment(
        name="relevance",
        value=0.96,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1"),
    )

    with mlflow.start_run() as run:
        # Log the first evaluation and the first assessment
        logged_evaluation_1 = log_evaluation(inputs=inputs_1, outputs=outputs_1)
        log_assessments(evaluation_id=logged_evaluation_1.evaluation_id, assessments=[assessment_1])

        # Log the second evaluation and the second assessment
        logged_evaluation_2 = log_evaluation(inputs=inputs_2, outputs=outputs_2)
        log_assessments(evaluation_id=logged_evaluation_2.evaluation_id, assessments=[assessment_2])

        # Log the updated first assessment to the first evaluation
        log_assessments(
            evaluation_id=logged_evaluation_1.evaluation_id, assessments=[updated_assessment_1]
        )

    def assert_assessments_equal(assessment, expected_assessment):
        assert assessment.name == expected_assessment.name
        assert assessment.boolean_value == expected_assessment.boolean_value
        assert assessment.numeric_value == expected_assessment.numeric_value
        assert assessment.string_value == expected_assessment.string_value
        assert assessment.metadata == expected_assessment.metadata
        assert assessment.source == expected_assessment.source

    # Verify that the first evaluation contains the second assessment logged to the first evaluation
    run_id = run.info.run_id
    retrieved_evaluation_1 = get_evaluation(
        evaluation_id=logged_evaluation_1.evaluation_id, run_id=run_id
    )
    assert len(retrieved_evaluation_1.assessments) == 1
    retrieved_assessment_1 = retrieved_evaluation_1.assessments[0]
    assert_assessments_equal(
        retrieved_assessment_1,
        updated_assessment_1._to_entity(evaluation_id=logged_evaluation_1.evaluation_id),
    )

    # Verify that the second evaluation contains the first (and only) assessment logged to the
    # second evaluation
    run_id = run.info.run_id
    retrieved_evaluation_2 = get_evaluation(
        evaluation_id=logged_evaluation_2.evaluation_id, run_id=run_id
    )
    assert len(retrieved_evaluation_2.assessments) == 1
    retrieved_assessment_2 = retrieved_evaluation_2.assessments[0]
    assert_assessments_equal(
        retrieved_assessment_2,
        assessment_2._to_entity(evaluation_id=logged_evaluation_2.evaluation_id),
    )


def test_log_assessments_with_same_name_and_source_and_metadata():
    inputs_1 = {"feature1": 1.0, "feature2": 2.0}
    outputs_1 = {"prediction": 0.5}
    inputs_2 = {"feature1": 3.0, "feature2": 4.0}
    outputs_2 = {"prediction": 0.7}

    metadata_1 = {"key1": "value1"}
    metadata_2 = {"key2": "value2"}

    assessment_1 = Assessment(
        name="relevance",
        value=0.9,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1"),
        metadata=metadata_1,
    )

    assessment_2 = Assessment(
        name="relevance",
        value=0.8,
        source=AssessmentSource(source_type=AssessmentSourceType.AI_JUDGE, source_id="judge_1"),
        metadata=metadata_2,
    )

    updated_assessment_1 = Assessment(
        name="relevance",
        value=0.96,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1"),
        metadata=metadata_1,
    )

    assessment_3 = Assessment(
        name="relevance",
        value=0.96,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1"),
        metadata=metadata_2,
    )

    with mlflow.start_run() as run:
        # Log the first evaluation and the first assessment
        logged_evaluation_1 = log_evaluation(inputs=inputs_1, outputs=outputs_1)
        log_assessments(evaluation_id=logged_evaluation_1.evaluation_id, assessments=[assessment_1])

        # Log the second evaluation and the second assessment
        logged_evaluation_2 = log_evaluation(inputs=inputs_2, outputs=outputs_2)
        log_assessments(evaluation_id=logged_evaluation_2.evaluation_id, assessments=[assessment_2])

        # Log the updated first assessment to the first evaluation
        log_assessments(
            evaluation_id=logged_evaluation_1.evaluation_id, assessments=[updated_assessment_1]
        )

        # Log the third assessment to the first evaluation
        log_assessments(evaluation_id=logged_evaluation_1.evaluation_id, assessments=[assessment_3])

    def assert_assessments_equal(assessment, expected_assessment):
        assert assessment.name == expected_assessment.name
        assert assessment.boolean_value == expected_assessment.boolean_value
        assert assessment.numeric_value == expected_assessment.numeric_value
        assert assessment.string_value == expected_assessment.string_value
        assert assessment.metadata == expected_assessment.metadata
        assert assessment.source == expected_assessment.source

    # Verify that the first evaluation contains the updated assessment logged to
    # the first evaluation
    run_id = run.info.run_id
    retrieved_evaluation_1 = get_evaluation(
        evaluation_id=logged_evaluation_1.evaluation_id, run_id=run_id
    )
    assert len(retrieved_evaluation_1.assessments) == 2
    for retrieved_assessment, expected_assessment in zip(
        retrieved_evaluation_1.assessments, [updated_assessment_1, assessment_3]
    ):
        assert_assessments_equal(
            retrieved_assessment,
            expected_assessment._to_entity(evaluation_id=logged_evaluation_1.evaluation_id),
        )

    # Verify that the second evaluation contains the first (and only) assessment logged to the
    # second evaluation
    run_id = run.info.run_id
    retrieved_evaluation_2 = get_evaluation(
        evaluation_id=logged_evaluation_2.evaluation_id, run_id=run_id
    )
    assert len(retrieved_evaluation_2.assessments) == 1
    retrieved_assessment_2 = retrieved_evaluation_2.assessments[0]
    assert_assessments_equal(
        retrieved_assessment_2,
        assessment_2._to_entity(evaluation_id=logged_evaluation_2.evaluation_id),
    )


def test_log_evaluation_with_assessments_supporting_none_value():
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id

        inputs = {"feature1": 1.0, "feature2": 2.0}
        outputs = {"prediction": 0.5}
        targets = {"actual": 0.5}
        metrics = [Metric(key="metric1", value=1.1, timestamp=0, step=0)]
        request_id = "req1"
        inputs_id = "id1"

        source = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1")
        assessment_with_none_value = Assessment(
            name="relevance", value=None, source=source, error_code="E001"
        )

        # Log the evaluation
        logged_evaluation = log_evaluation(
            inputs=inputs,
            outputs=outputs,
            targets=targets,
            metrics=metrics,
            inputs_id=inputs_id,
            request_id=request_id,
            assessments=[assessment_with_none_value],
            run_id=run_id,
        )

        # Retrieve the evaluation
        retrieved_evaluation = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=run_id
        )

        # Verify the evaluation details
        assert retrieved_evaluation == logged_evaluation

        # Verify the assessments

        def assert_assessments_equal(assessment, expected_assessment):
            assert assessment.name == expected_assessment.name
            assert assessment.boolean_value == expected_assessment.boolean_value
            assert assessment.numeric_value == expected_assessment.numeric_value
            assert assessment.string_value == expected_assessment.string_value
            assert assessment.metadata == expected_assessment.metadata
            assert assessment.source == expected_assessment.source
            assert assessment.error_code == expected_assessment.error_code
            assert assessment.error_message == expected_assessment.error_message

        assert len(retrieved_evaluation.assessments) == 1
        assert_assessments_equal(
            retrieved_evaluation.assessments[0],
            assessment_with_none_value._to_entity(evaluation_id=logged_evaluation.evaluation_id),
        )


def test_log_assessments_with_nonexistent_evaluation_fails():
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


@pytest.mark.parametrize(
    ("first_value", "second_value"),
    [(0.95, "high"), ("low", 0.75), (True, "true_string"), (False, 0.85), ("string_value", 0.8)],
)
def test_assessment_name_with_different_value_types_fails(first_value, second_value):
    inputs = {"feature1": 1.0, "feature2": 2.0}
    outputs = {"prediction": 0.5}

    with mlflow.start_run():
        logged_evaluation = log_evaluation(inputs=inputs, outputs=outputs)

        assessment1 = Assessment(
            name="accuracy",
            value=first_value,
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1"),
        )

        assessment2 = Assessment(
            name="accuracy",
            value=second_value,
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1"),
        )

        log_assessments(evaluation_id=logged_evaluation.evaluation_id, assessments=assessment1)

        with pytest.raises(MlflowException, match="does not match the value type"):
            log_assessments(evaluation_id=logged_evaluation.evaluation_id, assessments=assessment2)

        with pytest.raises(MlflowException, match="different value types"):
            log_evaluation(inputs=inputs, outputs=outputs, assessments=[assessment1, assessment2])


def test_set_evaluation_tags_with_minimal_params_succeeds():
    inputs = {"feature1": 1.0, "feature2": 2.0}
    outputs = {"prediction": 0.5}
    tags = {"tag1": "value1", "tag2": "value2"}

    with mlflow.start_run():
        logged_evaluation = log_evaluation(inputs=inputs, outputs=outputs)

        set_evaluation_tags(evaluation_id=logged_evaluation.evaluation_id, tags=tags)

        retrieved_evaluation = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=mlflow.active_run().info.run_id
        )

        assert len(retrieved_evaluation.tags) == len(tags)
        for tag_key, tag_value in tags.items():
            assert any(
                tag.key == tag_key and tag.value == tag_value for tag in retrieved_evaluation.tags
            )


def test_set_evaluation_tags_updates_values():
    inputs = {"feature1": 1.0, "feature2": 2.0}
    outputs = {"prediction": 0.5}
    initial_tags = {"tag1": "value1", "tag2": "value2", "tag3": "value3"}
    updated_tags = {"tag1": "new_value1", "tag2": "new_value2"}
    final_tags = {"tag1": "new_value1", "tag2": "new_value2", "tag3": "value3"}

    with mlflow.start_run():
        logged_evaluation = log_evaluation(inputs=inputs, outputs=outputs)

        # Set initial tags
        set_evaluation_tags(evaluation_id=logged_evaluation.evaluation_id, tags=initial_tags)

        # Update the tags
        set_evaluation_tags(evaluation_id=logged_evaluation.evaluation_id, tags=updated_tags)

        retrieved_evaluation = get_evaluation(
            evaluation_id=logged_evaluation.evaluation_id, run_id=mlflow.active_run().info.run_id
        )

        assert len(retrieved_evaluation.tags) == len(final_tags)
        for tag_key, tag_value in final_tags.items():
            assert any(
                tag.key == tag_key and tag.value == tag_value for tag in retrieved_evaluation.tags
            )


def test_set_evaluation_tags_with_empty_tags_succeeds():
    inputs = {"feature1": 1.0, "feature2": 2.0}
    outputs = {"prediction": 0.5}

    with mlflow.start_run():
        logged_evaluation = log_evaluation(inputs=inputs, outputs=outputs)
        set_evaluation_tags(evaluation_id=logged_evaluation.evaluation_id, tags={})


def test_set_evaluation_tags_with_nonexistent_evaluation_fails():
    with mlflow.start_run():
        with pytest.raises(
            MlflowException, match="The specified run does not contain any evaluations"
        ):
            set_evaluation_tags(
                evaluation_id="nonexistent",
                tags={"tag1": "value1", "tag2": "value2"},
            )

        log_evaluation(inputs={"feature1": 1.0, "feature2": 2.0}, outputs={"prediction": 0.5})
        with pytest.raises(
            MlflowException,
            match="The specified evaluation ID 'nonexistent' does not exist in the run",
        ):
            set_evaluation_tags(
                evaluation_id="nonexistent",
                tags={"tag1": "value1", "tag2": "value2"},
            )


def test_search_evaluations():
    inputs_1 = {"feature1": 1.0, "feature2": 2.0}
    outputs_1 = {"prediction": 0.5}
    targets_1 = {"actual": 0.5}
    metrics_1 = [Metric(key="metric1", value=1.1, timestamp=0, step=0)]
    request_id_1 = "req1"
    inputs_id_1 = "id1"

    inputs_2 = {"feature1": 3.0, "feature2": 4.0}
    outputs_2 = {"prediction": 0.75}
    targets_2 = {"actual": 0.8}
    metrics_2 = [Metric(key="metric2", value=1.2, timestamp=0, step=0)]
    request_id_2 = "req2"
    inputs_id_2 = "id2"

    source_1 = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user_1")
    assessment_1 = Assessment(name="assessment1", source=source_1, value=0.8)
    source_2 = AssessmentSource(source_type=AssessmentSourceType.AI_JUDGE, source_id="judge_1")
    assessment_2 = Assessment(name="assessment2", source=source_2, value=0.8)

    with mlflow.start_run() as run1:
        run_id1 = run1.info.run_id
        log_evaluation(
            inputs=inputs_1,
            outputs=outputs_1,
            targets=targets_1,
            metrics=metrics_1,
            assessments=[assessment_1],
            inputs_id=inputs_id_1,
            request_id=request_id_1,
        )
        log_evaluation(
            inputs=inputs_2,
            outputs=outputs_2,
            targets=targets_2,
            metrics=metrics_2,
            assessments=[assessment_2],
            inputs_id=inputs_id_2,
            request_id=request_id_2,
        )

    with mlflow.start_run() as run2:
        run_id2 = run2.info.run_id
        log_evaluation(
            inputs=inputs_2,
            outputs=outputs_2,
            targets=targets_2,
            metrics=metrics_2,
            assessments=[assessment_2],
            inputs_id=inputs_id_2,
            request_id=request_id_2,
        )

    # Search for the evaluations
    evaluations = search_evaluations(run_ids=[run_id1, run_id2])

    assert len(evaluations) == 3  # 3 evaluations should be retrieved

    # Verify the details of the retrieved evaluations
    eval1 = next(e for e in evaluations if e.inputs_id == inputs_id_1)
    assert eval1.inputs == inputs_1
    assert eval1.outputs == outputs_1
    assert eval1.targets == targets_1
    assert eval1.request_id == request_id_1
    assert eval1.assessments[0].name == assessment_1.name
    assert eval1.assessments[0].numeric_value == assessment_1.value
    assert eval1.metrics[0].key == metrics_1[0].key
    assert eval1.metrics[0].value == metrics_1[0].value

    eval2_list = [e for e in evaluations if e.inputs_id == inputs_id_2]
    assert len(eval2_list) == 2  # There should be two evaluations with inputs_id_2

    for eval2 in eval2_list:
        assert eval2.inputs == inputs_2
        assert eval2.outputs == outputs_2
        assert eval2.targets == targets_2
        assert eval2.request_id == request_id_2
        assert eval2.assessments[0].name == assessment_2.name
        assert eval2.assessments[0].numeric_value == assessment_2.value
        assert eval2.metrics[0].key == metrics_2[0].key
        assert eval2.metrics[0].value == metrics_2[0].value


def test_search_evaluations_no_runids():
    search_results = search_evaluations(run_ids=[])
    assert search_results == []


def test_search_evaluations_with_run_lacking_evaluations():
    with mlflow.start_run() as run:
        run_id = run.info.run_id

    search_results = search_evaluations(run_ids=[run_id])
    assert search_results == []
