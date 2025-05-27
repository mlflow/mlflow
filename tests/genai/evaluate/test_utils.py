import importlib
import os
from typing import Any, Literal
from unittest.mock import patch

import pandas as pd
import pytest

import mlflow
from mlflow.entities.assessment import Expectation, Feedback
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.entities.span import SpanType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai import scorer
from mlflow.genai.evaluation.utils import _convert_to_legacy_eval_set
from mlflow.genai.scorers.builtin_scorers import safety

if importlib.util.find_spec("databricks.agents") is None:
    pytest.skip(reason="databricks-agents is not installed", allow_module_level=True)


@pytest.fixture(scope="module")
def spark():
    try:
        from pyspark.sql import SparkSession

        with SparkSession.builder.getOrCreate() as spark:
            yield spark
    except Exception as e:
        pytest.skip(f"Failed to create a spark session: {e}")


@pytest.fixture
def sample_dict_data_single():
    return [
        {
            "inputs": {"question": "What is Spark?"},
            "outputs": "actual response for first question",
            "expectations": {"expected_response": "expected response for first question"},
        },
    ]


@pytest.fixture
def sample_dict_data_multiple():
    return [
        {
            "inputs": {"question": "What is Spark?"},
            "outputs": "actual response for first question",
            "expectations": {"expected_response": "expected response for first question"},
            # Additional columns required by the judges
            "retrieved_context": [
                {
                    "content": "doc content 1",
                    "doc_uri": "doc_uri_2_1",
                },
                {
                    "content": "doc content 2.",
                    "doc_uri": "doc_uri_6_extra",
                },
            ],
        },
        {
            "inputs": {"question": "How can you minimize data shuffling in Spark?"},
            "outputs": "actual response for second question",
            "expectations": {"expected_response": "expected response for second question"},
            "retrieved_context": [],
        },
        # Some records might not have expectations
        {
            "inputs": {"question": "What is MLflow?"},
            "outputs": "actual response for third question",
        },
    ]


@pytest.fixture
def sample_dict_data_multiple_with_custom_expectations():
    return [
        {
            "inputs": {"question": "What is Spark?"},
            "outputs": "actual response for first question",
            "expectations": {
                "expected_response": "expected response for first question",
                "my_custom_expectation": "custom expectation for the first question",
            },
        },
        {
            "inputs": {"question": "How can you minimize data shuffling in Spark?"},
            "outputs": "actual response for second question",
            "expectations": {
                "expected_response": "expected response for second question",
                "my_custom_expectation": "custom expectation for the second question",
            },
        },
        # Some records might not have all expectations
        {
            "inputs": {"question": "What is MLflow?"},
            "outputs": "actual response for third question",
            "expectations": {
                "my_custom_expectation": "custom expectation for the third question",
            },
        },
    ]


@pytest.fixture
def sample_pd_data(sample_dict_data_multiple):
    """Returns a pandas DataFrame with sample data"""
    return pd.DataFrame(sample_dict_data_multiple)


@pytest.fixture
def sample_spark_data(sample_pd_data, spark):
    """Convert pandas DataFrame to PySpark DataFrame"""
    return spark.createDataFrame(sample_pd_data)


_ALL_DATA_FIXTURES = [
    "sample_dict_data_single",
    "sample_dict_data_multiple",
    "sample_dict_data_multiple_with_custom_expectations",
    "sample_pd_data",
    "sample_spark_data",
]


class TestModel:
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, question: str) -> str:
        response = self.call_llm(messages=[{"role": "user", "content": question}])
        return response["choices"][0]["message"]["content"]

    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"choices": [{"message": {"role": "assistant", "content": "I don't know"}}]}


def get_test_traces(type=Literal["pandas", "list"]):
    model = TestModel()

    model.predict("What is MLflow?")
    traces = mlflow.search_traces(return_type=type, order_by=["timestamp_ms ASC"])

    # Add assessments. Since log_assessment API is not supported in OSS MLflow yet, we
    # need to add it to the trace info manually.
    source = AssessmentSource(source_id="test", source_type="HUMAN")
    trace = traces[0] if type == "list" else traces.iloc[0]["trace"]
    trace.info.assessments.extend(
        [
            # 1. Expectation with reserved name "expected_response"
            Expectation(
                name="expected_response",
                source=source,
                trace_id=trace.info.trace_id,
                value="expected response for first question",
            ),
            # 2. Expectation with reserved name "expected_facts"
            Expectation(
                name="expected_facts",
                source=source,
                trace_id=trace.info.trace_id,
                value=["fact1", "fact2"],
            ),
            # 3. Expectation with reserved name "guidelines"
            Expectation(
                name="guidelines",
                source=source,
                trace_id=trace.info.trace_id,
                value=["Be polite", "Be kind"],
            ),
            # 4. Expectation with custom name "ground_truth"
            Expectation(
                name="my_custom_expectation",
                source=source,
                trace_id=trace.info.trace_id,
                value="custom expectation for the first question",
            ),
            # 5. Non-expectation assessment
            Feedback(
                name="feedback",
                source=source,
                trace_id=trace.info.trace_id,
                value="some feedback",
            ),
        ]
    )
    return [{"trace": trace} for trace in traces] if type == "list" else traces


@pytest.mark.parametrize("input_type", ["list", "pandas"])
def test_convert_to_legacy_eval_traces(input_type):
    sample_data = get_test_traces(type=input_type)
    data = _convert_to_legacy_eval_set(sample_data)

    assert "trace" in data.columns

    # "request" column should be derived from the trace
    assert "request" in data.columns
    assert list(data["request"]) == [{"question": "What is MLflow?"}]
    assert data["expectations"][0] == {
        "expected_response": "expected response for first question",
        "expected_facts": ["fact1", "fact2"],
        "guidelines": ["Be polite", "Be kind"],
        "my_custom_expectation": "custom expectation for the first question",
    }
    # Assessment with type "Feedback" should not be present in the transformed data
    assert "feedback" not in data.columns


@pytest.mark.parametrize("data_fixture", _ALL_DATA_FIXTURES)
def test_convert_to_legacy_eval_set_has_no_errors(data_fixture, request):
    sample_data = request.getfixturevalue(data_fixture)

    transformed_data = _convert_to_legacy_eval_set(sample_data)

    assert "request" in transformed_data.columns
    assert "response" in transformed_data.columns
    assert "expectations" in transformed_data.columns


@pytest.mark.parametrize("data_fixture", _ALL_DATA_FIXTURES)
def test_scorer_receives_correct_data(data_fixture, request):
    sample_data = request.getfixturevalue(data_fixture)

    received_args = []

    @scorer
    def dummy_scorer(inputs, outputs, expectations):
        received_args.append(
            (
                inputs["question"],
                outputs,
                expectations.get("expected_response"),
                expectations.get("my_custom_expectation"),
            )
        )
        return 0

    mlflow.genai.evaluate(
        data=sample_data,
        scorers=[dummy_scorer],
    )

    all_inputs, all_outputs, all_expectations, all_custom_expectations = zip(*received_args)
    expected_inputs = [
        "What is Spark?",
        "How can you minimize data shuffling in Spark?",
        "What is MLflow?",
    ][: len(sample_data)]
    expected_outputs = [
        "actual response for first question",
        "actual response for second question",
        "actual response for third question",
    ][: len(sample_data)]
    expected_expectations = [
        "expected response for first question",
        "expected response for second question",
        None,
    ][: len(sample_data)]

    assert set(all_inputs) == set(expected_inputs)
    assert set(all_outputs) == set(expected_outputs)
    assert set(all_expectations) == set(expected_expectations)

    if data_fixture == "sample_dict_data_multiple_with_custom_expectations":
        expected_custom_expectations = [
            "custom expectation for the first question",
            "custom expectation for the second question",
            "custom expectation for the third question",
        ]
        assert set(all_custom_expectations) == set(expected_custom_expectations)


def test_input_is_required_if_trace_is_not_provided():
    with patch("mlflow.models.evaluate") as mock_evaluate:
        with pytest.raises(MlflowException, match="inputs.*required"):
            mlflow.genai.evaluate(
                data=pd.DataFrame({"outputs": ["Paris"]}),
                scorers=[safety],
            )

        mock_evaluate.assert_not_called()

        mlflow.genai.evaluate(
            data=pd.DataFrame(
                {"inputs": [{"question": "What is the capital of France?"}], "outputs": ["Paris"]}
            ),
            scorers=[safety],
        )
        mock_evaluate.assert_called_once()


def test_input_is_optional_if_trace_is_provided():
    with mlflow.start_span() as span:
        span.set_inputs({"question": "What is the capital of France?"})
        span.set_outputs("Paris")

    trace = mlflow.get_trace(span.trace_id)

    with patch("mlflow.models.evaluate") as mock_evaluate:
        mlflow.genai.evaluate(
            data=pd.DataFrame({"trace": [trace]}),
            scorers=[safety],
        )

        mock_evaluate.assert_called_once()


# TODO: Remove this skip once databricks-agents 1.0 is released
@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping test in CI because this test requires Agent SDK pre-release wheel",
)
@pytest.mark.parametrize("input_type", ["list", "pandas"])
def test_scorer_receives_correct_data_with_trace_data(input_type):
    sample_data = get_test_traces(type=input_type)
    received_args = []

    @scorer
    def dummy_scorer(inputs, outputs, expectations, trace):
        received_args.append((inputs, outputs, expectations, trace))
        return 0

    mlflow.genai.evaluate(
        data=sample_data,
        scorers=[dummy_scorer],
    )

    inputs, outputs, expectations, trace = received_args[0]
    assert inputs == {"question": "What is MLflow?"}
    assert outputs == "I don't know"
    assert expectations == {
        "expected_response": "expected response for first question",
        "expected_facts": ["fact1", "fact2"],
        "guidelines": ["Be polite", "Be kind"],
        "my_custom_expectation": "custom expectation for the first question",
    }
    assert isinstance(trace, Trace)


@pytest.mark.parametrize("data_fixture", _ALL_DATA_FIXTURES)
def test_predict_fn_receives_correct_data(data_fixture, request):
    sample_data = request.getfixturevalue(data_fixture)

    received_args = []

    def predict_fn(question: str):
        received_args.append(question)
        return question

    mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=sample_data,
        scorers=[safety],
    )

    received_args.pop(0)  # Remove the one-time prediction to check if a model is traced
    assert len(received_args) == len(sample_data)
    expected_contents = [
        "What is Spark?",
        "How can you minimize data shuffling in Spark?",
        "What is MLflow?",
    ][: len(sample_data)]
    # Using set because eval harness runs predict_fn in parallel
    assert set(received_args) == set(expected_contents)
