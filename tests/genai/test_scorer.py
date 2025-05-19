import importlib
from collections import defaultdict
from unittest.mock import patch

import pandas as pd
import pytest
from packaging.version import Version

import mlflow
from mlflow.entities import Assessment, AssessmentSource, AssessmentSourceType, Feedback
from mlflow.entities.assessment import FeedbackValue
from mlflow.entities.assessment_error import AssessmentError
from mlflow.evaluation import Assessment as LegacyAssessment
from mlflow.genai import Scorer, scorer

if importlib.util.find_spec("databricks.agents") is None:
    pytest.skip(reason="databricks-agents is not installed", allow_module_level=True)

agent_sdk_version = Version(importlib.import_module("databricks.agents").__version__)


def always_yes(inputs, outputs, expectations, trace):
    return "yes"


class AlwaysYesScorer(Scorer):
    def __call__(self, inputs, outputs, expectations, trace):
        return "yes"


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "inputs": [
                {"message": [{"role": "user", "content": "What is Spark??"}]},
                {
                    "messages": [
                        {"role": "user", "content": "How can you minimize data shuffling in Spark?"}
                    ]
                },
            ],
            "outputs": [
                {"choices": [{"message": {"content": "actual response for first question"}}]},
                {"choices": [{"message": {"content": "actual response for second question"}}]},
            ],
            "expectations": [
                {"expected_response": "expected response for first question"},
                {"expected_response": "expected response for second question"},
            ],
        }
    )


@pytest.mark.parametrize("dummy_scorer", [AlwaysYesScorer(name="always_yes"), scorer(always_yes)])
def test_scorer_existence_in_metrics(sample_data, dummy_scorer):
    result = mlflow.genai.evaluate(data=sample_data, scorers=[dummy_scorer])
    assert any("always_yes" in metric for metric in result.metrics.keys())


@pytest.mark.parametrize(
    "dummy_scorer", [AlwaysYesScorer(name="always_no"), scorer(name="always_no")(always_yes)]
)
def test_scorer_name_works(sample_data, dummy_scorer):
    _SCORER_NAME = "always_no"
    result = mlflow.genai.evaluate(data=sample_data, scorers=[dummy_scorer])
    assert any(_SCORER_NAME in metric for metric in result.metrics.keys())


def test_scorer_is_called_with_correct_arguments(sample_data):
    actual_call_args_list = []

    @scorer
    def dummy_scorer(inputs, outputs, expectations, trace) -> float:
        actual_call_args_list.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "expectations": expectations,
            }
        )
        return 0.0

    mlflow.genai.evaluate(data=sample_data, scorers=[dummy_scorer])

    assert len(actual_call_args_list) == len(sample_data)

    # Prepare expected arguments, keyed by expected_response for matching
    sample_data_set = defaultdict(set)
    for i in range(len(sample_data)):
        sample_data_set["inputs"].add(str(sample_data["inputs"][i]))
        sample_data_set["outputs"].add(str(sample_data["outputs"][i]))
        sample_data_set["expectations"].add(
            str(sample_data["expectations"][i]["expected_response"])
        )

    for actual_args in actual_call_args_list:
        # do any check since actual passed input could be reformatted and larger than sample input
        assert any(
            sample_data_input in str(actual_args["inputs"])
            for sample_data_input in sample_data_set["inputs"]
        )
        assert str(actual_args["outputs"]) in sample_data_set["outputs"]
        assert (
            str(actual_args["expectations"]["expected_response"]) in sample_data_set["expectations"]
        )


def test_scorer_receives_extra_arguments():
    received_args = []

    @scorer
    def dummy_scorer(inputs, outputs, retrieved_context) -> float:
        received_args.append((inputs, outputs, retrieved_context))
        return 0

    mlflow.genai.evaluate(
        data=[
            {
                "inputs": {"question": "What is Spark?"},
                "outputs": "actual response for first question",
                "retrieved_context": [{"doc_uri": "document_1", "content": "test"}],
            },
        ],
        scorers=[dummy_scorer],
    )

    inputs, outputs, retrieved_context = received_args[0]
    assert inputs == {"question": "What is Spark?"}
    assert outputs == "actual response for first question"
    assert retrieved_context == [{"doc_uri": "document_1", "content": "test"}]


def test_trace_passed_correctly():
    @mlflow.trace
    def predict_fn(question):
        return "output: " + str(question)

    actual_call_args_list = []

    @scorer
    def dummy_scorer(inputs, outputs, trace):
        actual_call_args_list.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "trace": trace,
            }
        )
        return 0.0

    data = [
        {"inputs": {"question": "input1"}},
        {"inputs": {"question": "input2"}},
    ]
    mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=data,
        scorers=[dummy_scorer],
    )

    assert len(actual_call_args_list) == len(data)
    for actual_args in actual_call_args_list:
        assert actual_args["trace"] is not None
        trace = actual_args["trace"]
        # check if the input is present in the trace
        assert any(
            str(data[i]["inputs"]["question"]) in str(trace.data.request) for i in range(len(data))
        )
        # check if predict_fn was run by making output it starts with "output:"
        assert "output:" in str(trace.data.response)[:10]


@pytest.mark.parametrize(
    "scorer_return",
    [
        "yes",
        42,
        42.0,
        # Feedback object.
        Feedback(name="big_question", value=42, rationale="It's the answer to everything"),
        # List of Feedback objects.
        [
            Feedback(name="big_question", value=42, rationale="It's the answer to everything"),
            Feedback(name="small_question", value=1, rationale="Not sure, just a guess"),
        ],
        # Raw Assessment object. This construction should only be done internally.
        Assessment(
            name="big_question",
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="123"),
            feedback=FeedbackValue(value=42),
            rationale="It's the answer to everything",
        ),
        # Legacy mlflow.evaluation.Assessment object. Still used by managed judges.
        LegacyAssessment(name="big_question", value=True),
    ],
)
def test_scorer_on_genai_evaluate(sample_data, scorer_return):
    @scorer
    def dummy_scorer(inputs, outputs):
        return scorer_return

    results = mlflow.genai.evaluate(
        data=sample_data,
        scorers=[dummy_scorer],
    )

    assert any("metric/dummy_scorer" in metric for metric in results.metrics.keys())

    dummy_scorer_cols = [
        col for col in results.result_df.keys() if "dummy_scorer" in col and "value" in col
    ]
    dummy_scorer_values = set()
    for col in dummy_scorer_cols:
        for _val in results.result_df[col]:
            dummy_scorer_values.add(_val)

    scorer_return_values = set()
    if isinstance(scorer_return, list):
        for _assessment in scorer_return:
            scorer_return_values.add(_assessment.feedback.value)
    elif isinstance(scorer_return, Assessment):
        scorer_return_values.add(scorer_return.feedback.value)
    elif isinstance(scorer_return, mlflow.evaluation.Assessment):
        scorer_return_values.add(scorer_return.value)
    else:
        scorer_return_values.add(scorer_return)

    assert dummy_scorer_values == scorer_return_values


def test_scorer_returns_feedback_with_error(sample_data):
    @scorer
    def dummy_scorer(inputs):
        return Feedback(
            name="feedback_with_error",
            error=AssessmentError(error_code="500", error_message="This is an error"),
            source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt"),
            metadata={"index": 0},
        )

    with patch("mlflow.get_tracking_uri", return_value="databricks"):
        results = mlflow.genai.evaluate(
            data=sample_data,
            scorers=[dummy_scorer],
        )

    # Scorer should not be in result when it returns an error
    assert all("metric/dummy_scorer" not in metric for metric in results.metrics.keys())


def test_builtin_scorers_are_callable():
    from mlflow.genai.scorers import safety

    # test with new scorer signature format
    with patch("databricks.agents.evals.judges.safety") as mock_safety:
        safety()(
            inputs={"question": "What is the capital of France?"},
            outputs="The capital of France is Paris.",
        )

        mock_safety.assert_called_once_with(
            request={"question": "What is the capital of France?"},
            response="The capital of France is Paris.",
        )


@pytest.mark.parametrize(
    ("scorer_return", "expected_feedback_name"),
    [
        # Single feedback object with default name -> should be renamed to "my_scorer"
        (
            Feedback(value=42, rationale="It's the answer to everything"),
            "my_scorer",
        ),
        # Single feedback object -> should be renamed to "my_scorer"
        (
            Feedback(name="big_question", value=42, rationale="It's the answer to everything"),
            "my_scorer",
        ),
    ],
)
def test_custom_scorer_overwrites_feedback_name(scorer_return, expected_feedback_name):
    @scorer
    def my_scorer(inputs, outputs):
        return scorer_return

    feedback = my_scorer.run(
        inputs={"question": "What is the capital of France?"},
        outputs="The capital of France is Paris.",
    )
    assert feedback.name == expected_feedback_name
    assert feedback.value == 42


def test_custom_scorer_does_not_overwrite_feedback_name_when_returning_list():
    @scorer
    def my_scorer(inputs, outputs):
        return [
            Feedback(name="big_question", value=42, rationale="It's the answer to everything"),
            Feedback(name="small_question", value=1, rationale="Not sure, just a guess"),
        ]

    feedbacks = my_scorer.run(
        inputs={"question": "What is the capital of France?"},
        outputs="The capital of France is Paris.",
    )
    assert feedbacks[0].name == "big_question"
    assert feedbacks[1].name == "small_question"
