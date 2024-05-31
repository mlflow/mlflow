import pytest

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType, Metric
from mlflow.evaluation import (
    Assessment,
    Evaluation,
    get_evaluation,
    log_assessments,
    log_evaluation,
    log_evaluations,
)
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
        ),
    ]

    inputs_id = "unique-inputs-id"
    request_id = "unique-request-id"

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        evaluations = []
        for inputs, outputs, targets, assessments, metrics in evaluations_data:
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
            )
            evaluations.append(evaluation)

        # Log the evaluations
        logged_evaluations = log_evaluations(evaluations=evaluations, run_id=run_id)

        for logged_evaluation, (inputs, outputs, targets, assessments, metrics) in zip(
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

    def assert_assessments_equal(assessment, expected_assessment):
        assert assessment.name == expected_assessment.name
        assert assessment.boolean_value == expected_assessment.boolean_value
        assert assessment.numeric_value == expected_assessment.numeric_value
        assert assessment.string_value == expected_assessment.string_value
        assert assessment.metadata == expected_assessment.metadata
        assert assessment.source == expected_assessment.source

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
