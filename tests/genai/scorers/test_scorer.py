import importlib
from collections import defaultdict
from unittest.mock import call, patch

import pandas as pd
import pytest
from databricks.rag_eval.evaluation.entities import CategoricalRating
from packaging.version import Version

import mlflow
from mlflow.entities import Assessment, AssessmentSource, AssessmentSourceType, Feedback
from mlflow.entities.assessment import FeedbackValue
from mlflow.entities.assessment_error import AssessmentError
from mlflow.genai import Scorer, scorer
from mlflow.genai.scorers import Correctness, Guidelines, RetrievalGroundedness

from tests.tracing.helper import get_traces, purge_traces

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


def test_trace_passed_to_builtin_scorers_correctly(sample_rag_trace):
    with (
        patch(
            "databricks.agents.evals.judges.correctness",
            return_value=Feedback(name="correctness", value=CategoricalRating.YES),
        ) as mock_correctness,
        patch(
            "databricks.agents.evals.judges.guideline_adherence",
            return_value=Feedback(name="guideline_adherence", value=CategoricalRating.YES),
        ) as mock_guideline,
        patch(
            "databricks.agents.evals.judges.groundedness",
            return_value=Feedback(name="groundedness", value=CategoricalRating.YES),
        ) as mock_groundedness,
    ):
        mlflow.genai.evaluate(
            data=pd.DataFrame({"trace": [sample_rag_trace]}),
            scorers=[
                RetrievalGroundedness(name="retrieval_groundedness"),
                Correctness(name="correctness"),
                Guidelines(name="english", guidelines=["write in english"]),
            ],
        )

    assert mock_correctness.call_count == 1
    assert mock_guideline.call_count == 1
    assert mock_groundedness.call_count == 2  # Called per retriever span

    mock_correctness.assert_called_once_with(
        request="{'question': 'query'}",
        response="answer",
        expected_facts=["fact1", "fact2"],
        expected_response="expected answer",
        assessment_name="correctness",
    )
    mock_guideline.assert_called_once_with(
        guidelines=["write in english"],
        guidelines_context={"request": "{'question': 'query'}", "response": "answer"},
        assessment_name="english",
    )
    mock_groundedness.assert_has_calls(
        [
            call(
                request="{'question': 'query'}",
                response="answer",
                retrieved_context=[
                    {"content": "content_1", "doc_uri": "url_1"},
                    {"content": "content_2", "doc_uri": "url_2"},
                ],
                assessment_name="retrieval_groundedness",
            ),
            call(
                request="{'question': 'query'}",
                response="answer",
                retrieved_context=[
                    {"content": "content_3"},
                ],
                assessment_name="retrieval_groundedness",
            ),
        ]
    )


def test_trace_passed_to_custom_scorer_correctly(sample_data):
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


def test_custom_scorer_allow_none_return():
    @scorer
    def dummy_scorer(inputs, outputs):
        return None

    assert dummy_scorer.run(inputs={"question": "query"}, outputs="answer") is None


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


@pytest.mark.parametrize(
    ("scorer_return", "expected_feedback_name"),
    [
        # Single feedback object with default name -> should be renamed to "my_scorer"
        (Feedback(value=42, rationale="rationale"), "my_scorer"),
        # Single feedback object with custom name -> should NOT be renamed to "my_scorer"
        (Feedback(name="custom_name", value=42, rationale="rationale"), "custom_name"),
    ],
)
def test_custom_scorer_overwrites_default_feedback_name(scorer_return, expected_feedback_name):
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


def test_extra_traces_from_customer_scorer_should_be_cleaned_up():
    @scorer
    def my_scorer_1(inputs, outputs):
        with mlflow.start_span(name="scorer_trace_1") as span:
            # Tracing is disabled during evaluation but this should not NPE
            span.set_inputs(inputs)
            span.set_outputs(outputs)

        with mlflow.start_span(name="scorer_trace_2"):
            pass
        return 1

    @scorer
    @mlflow.trace
    def my_scorer_2():
        return 0.5

    def predict(question: str) -> str:
        return "output: " + str(question)

    result = mlflow.genai.evaluate(
        data=[{"inputs": {"question": "Hello"}} for _ in range(100)],
        scorers=[my_scorer_1, my_scorer_2],
        predict_fn=predict,
    )
    # Scorers should be computed correctly
    assert result.metrics["metric/my_scorer_1/average"] == 1
    assert result.metrics["metric/my_scorer_2/average"] == 0.5

    # Traces should only be generated for predict_fn
    traces = get_traces()
    assert len(traces) == 100
    assert all(trace.data.spans[0].name == "predict" for trace in traces)
    purge_traces()

    # When invoked directly, the scorer should generate traces
    score = my_scorer_2()
    assert score == 0.5
    assert len(get_traces()) == 1


def test_extra_traces_before_evaluation_execution_should_not_be_cleaned_up():
    def predict(question: str) -> str:
        return "output: " + str(question)

    @scorer
    @mlflow.trace
    def my_scorer(inputs, outputs):
        return 0.5

    with mlflow.start_run():
        # Generate another trace in the run before running the evaluation
        with mlflow.start_span(name="should_be_kept"):
            pass

        result = mlflow.genai.evaluate(
            data=[{"inputs": {"question": "Hello"}}],
            scorers=[my_scorer],
            predict_fn=predict,
        )
    # Scorers should be computed correctly
    assert result.metrics["metric/my_scorer/average"] == 0.5

    # Traces should only be generated for predict_fn
    traces = get_traces()
    assert len(traces) == 2  # 1 for predict_fn, 1 for a trace generated before evaluation
    assert traces[0].data.spans[0].name == "predict"
    assert traces[1].data.spans[0].name == "should_be_kept"
