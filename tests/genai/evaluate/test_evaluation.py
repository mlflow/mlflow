import warnings
from importlib import import_module
from unittest import mock
from unittest.mock import patch

import pandas as pd
import pytest
from packaging.version import Version

import mlflow
from mlflow.entities.assessment import Expectation, Feedback
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.entities.span import SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import scorer
from mlflow.genai.scorers.builtin_scorers import GENAI_CONFIG_NAME, safety

from tests.evaluate.test_evaluation import _DUMMY_CHAT_RESPONSE
from tests.tracing.helper import get_traces

_IS_AGENT_SDK_V1 = Version(import_module("databricks.agents").__version__).major >= 1


class TestModel:
    def predict(self, question: str) -> str:
        return "I don't know"


@scorer
def exact_match(outputs, expectations):
    return outputs == expectations["expected_response"]


@scorer
def max_length(outputs, expectations):
    return len(outputs) <= expectations["max_length"]


@scorer
def relevance(inputs, outputs):
    return Feedback(
        name="relevance",
        value="yes",
        rationale="The response is relevant to the question",
        source=AssessmentSource(source_id="gpt", source_type="LLM_JUDGE"),
    )


@scorer
def has_trace(trace):
    return trace is not None


@pytest.mark.skipif(not _IS_AGENT_SDK_V1, reason="Databricks Agent SDK v1 is required")
def test_evaluate_with_static_dataset():
    data = [
        {
            "inputs": {"question": "What is MLflow?"},
            "outputs": "MLflow is a tool for ML",
            "expectations": {
                "expected_response": "MLflow is a tool for ML",
                "max_length": 100,
            },
        },
        {
            "inputs": {"question": "What is Spark?"},
            "outputs": "Spark is a fast data processing engine",
            "expectations": {
                "expected_response": "Spark is a fast data processing engine",
                "max_length": 1,
            },
        },
    ]

    result = mlflow.genai.evaluate(
        data=data,
        scorers=[exact_match, max_length, relevance, has_trace],
    )

    metrics = result.metrics
    assert metrics["metric/exact_match/average"] == 1.0
    assert metrics["metric/max_length/average"] == 0.5
    assert metrics["metric/relevance/relevance/average"] == 1.0
    assert metrics["metric/has_trace/average"] == 1.0

    # Exact number of traces should be generated
    traces = get_traces()
    assert len(traces) == len(data)


@pytest.mark.skipif(not _IS_AGENT_SDK_V1, reason="Databricks Agent SDK v1 is required")
@pytest.mark.parametrize("is_predict_fn_traced", [True, False])
def test_evaluate_with_predict_fn(is_predict_fn_traced):
    data = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {
                "expected_response": "MLflow is a tool for ML",
                "max_length": 100,
            },
        },
        {
            "inputs": {"question": "What is Spark?"},
            "expectations": {
                "expected_response": "Spark is a fast data processing engine",
                "max_length": 1,
            },
        },
    ]
    model = TestModel()
    predict_fn = mlflow.trace(model.predict) if is_predict_fn_traced else model.predict

    result = mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=data,
        scorers=[exact_match, max_length, relevance, has_trace],
    )

    metrics = result.metrics
    assert metrics["metric/exact_match/average"] == 0.0
    assert metrics["metric/max_length/average"] == 0.5
    assert metrics["metric/relevance/relevance/average"] == 1.0
    assert metrics["metric/has_trace/average"] == 1.0

    # Exact number of traces should be generated
    traces = get_traces()
    assert len(traces) == len(data)


@pytest.mark.skipif(not _IS_AGENT_SDK_V1, reason="Databricks Agent SDK v1 is required")
@pytest.mark.parametrize("pass_full_dataframe", [True, False])
def test_evaluate_with_traces(pass_full_dataframe):
    questions = ["What is MLflow?", "What is Spark?"]

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(question: str) -> str:
        return TestModel().predict(question)

    for question in questions:
        predict(question)

    data = mlflow.search_traces()
    assert len(data) == len(questions)

    # OSS MLflow backend doesn't support assessment APIs now, so we need to manually add them
    data.iloc[0]["trace"].info.assessments = [
        Expectation(
            name="expected_response",
            trace_id="tr-123",
            value="MLflow is a tool for ML",
            source=AssessmentSource(source_id="me", source_type="HUMAN"),
        ),
        Expectation(
            name="max_length",
            trace_id="tr-123",
            value=100,
            source=AssessmentSource(source_id="me", source_type="HUMAN"),
        ),
    ]
    data.iloc[1]["trace"].info.assessments = [
        Expectation(
            name="expected_response",
            trace_id="tr-123",
            value="Spark is a fast data processing engine",
            source=AssessmentSource(source_id="me", source_type="HUMAN"),
        ),
        Expectation(
            name="max_length",
            trace_id="tr-123",
            value=1,
            source=AssessmentSource(source_id="me", source_type="HUMAN"),
        ),
    ]

    if not pass_full_dataframe:
        data = data[["trace"]]

    result = mlflow.genai.evaluate(
        data=data,
        scorers=[exact_match, max_length, relevance, has_trace],
    )

    metrics = result.metrics
    assert metrics["metric/exact_match/average"] == 0.0
    assert metrics["metric/max_length/average"] == 0.5
    assert metrics["metric/relevance/relevance/average"] == 1.0
    assert metrics["metric/has_trace/average"] == 1.0

    # Assessments should be added to the traces in-place and no new trace should be created
    assert len(get_traces()) == len(questions)


@mock.patch("mlflow.deployments.get_deploy_client")
def test_model_from_deployment_endpoint(mock_get_deploy_client):
    mock_client = mock_get_deploy_client.return_value
    mock_client.predict.return_value = _DUMMY_CHAT_RESPONSE
    mock_client.get_endpoint.return_value = {"task": "llm/v1/chat"}

    data = [
        {
            "inputs": {
                "messages": [
                    {"content": "You are a helpful assistant.", "role": "system"},
                    {"content": "What is Spark?", "role": "user"},
                ],
                "max_tokens": 10,
            }
        },
        {
            "inputs": {
                "messages": [
                    {"content": "What is MLflow?", "role": "user"},
                ]
            }
        },
    ]

    predict_fn = mlflow.genai.to_predict_fn("endpoints:/chat")

    # predict_fn should be callable with a single input
    response = predict_fn(**data[0]["inputs"])

    mock_client.predict.assert_called_once_with(
        endpoint="chat",
        inputs=data[0]["inputs"],
    )
    assert response == _DUMMY_CHAT_RESPONSE  # Chat response should not be parsed
    mock_client.reset_mock()

    # Running evaluation
    result = mlflow.genai.evaluate(
        data=data,
        predict_fn=predict_fn,
        scorers=[has_trace],
    )

    mock_client.predict.assert_has_calls(
        [
            # Test call to check if the function is traced or not
            mock.call(endpoint="chat", inputs=data[0]["inputs"]),
            # First evaluation call
            mock.call(endpoint="chat", inputs=data[0]["inputs"]),
            # Second evaluation call
            mock.call(endpoint="chat", inputs=data[1]["inputs"]),
        ],
        any_order=True,
    )

    # Validate traces
    traces = mlflow.search_traces(run_id=result.run_id, return_type="list")

    assert len(traces) == 2
    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "predict"
    assert spans[0].attributes["endpoint"] == "endpoints:/chat"
    # Eval harness runs prediction in parallel, so the order is not deterministic
    assert spans[0].inputs in (data[0]["inputs"], data[1]["inputs"])
    assert spans[0].outputs == _DUMMY_CHAT_RESPONSE


def test_evaluate_passes_model_id_to_mlflow_evaluate():
    # Tracking URI = databricks is required to use mlflow.genai.evaluate()
    mlflow.set_tracking_uri("databricks")
    data = [
        {"inputs": {"x": "bar"}, "outputs": "response from model"},
        {"inputs": {"x": "qux"}, "outputs": "response from model"},
    ]

    with mock.patch("mlflow.models.evaluate") as mock_evaluate:

        @mlflow.trace
        def model(x):
            return x

        mlflow.genai.evaluate(
            data=data,
            predict_fn=model,
            model_id="test_model_id",
            scorers=[safety()],
        )

        # Verify the call was made with the right parameters
        mock_evaluate.assert_called_once_with(
            model=mock.ANY,
            data=mock.ANY,
            evaluator_config={GENAI_CONFIG_NAME: {"metrics": ["safety"]}},
            model_type="databricks-agent",
            extra_metrics=[],
            model_id="test_model_id",
            _called_from_genai_evaluate=True,
        )


@patch("mlflow.get_tracking_uri", return_value="databricks")
def test_no_scorers(mock_get_tracking_uri):
    with pytest.raises(TypeError, match=r"evaluate\(\) missing 1 required positional"):
        mlflow.genai.evaluate(data=[{"inputs": "Hello", "outputs": "Hi"}])

    with pytest.raises(MlflowException, match=r"The `scorers` argument must be a list of"):
        mlflow.genai.evaluate(data=[{"inputs": "Hello", "outputs": "Hi"}], scorers=[])


def test_genai_evaluate_does_not_warn_about_deprecated_model_type():
    """
    MLflow shows a warning when model_type="databricks-agent" is used for mlflow.evaluate()
    API. This test verifies that the warning is not shown when mlflow.genai.evaluate() is used.
    """
    with (
        patch("mlflow.genai.evaluation.base.is_databricks_uri", return_value=True),
        patch("mlflow.models.evaluation.base._evaluate") as mock_evaluate_impl,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error", FutureWarning)
        mlflow.genai.evaluate(
            data=[{"inputs": {"question": "Hello"}, "outputs": "Hi"}],
            scorers=[safety()],
        )

    mock_evaluate_impl.assert_called_once()

    # Warning should be shown when "databricks-agent" model type is used with direct call
    data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    with (
        patch("mlflow.models.evaluation.base.warnings") as mock_warnings,
        patch("mlflow.models.evaluation.base._evaluate") as mock_evaluate_impl,
    ):
        mlflow.models.evaluate(
            data=data,
            model=lambda x: x["x"] * 2,
            model_type="databricks-agent",
            extra_metrics=[mlflow.metrics.latency()],
        )
    mock_warnings.warn.assert_called_once()
    assert mock_warnings.warn.call_args[0][0].startswith(
        "The 'databricks-agent' model type is deprecated"
    )
    mock_evaluate_impl.assert_called_once()


@pytest.mark.parametrize("pass_full_dataframe", [True, False])
def test_trace_input_can_contain_string_input(pass_full_dataframe):
    """
    The `inputs` column must be a dictionary when a static dataset is provided.
    However, when a trace is provided, it doesn't need to be validated and the
    harness can handle it nicely.
    """
    with mlflow.start_span() as span:
        span.set_inputs("What is MLflow?")
        span.set_outputs("MLflow is a tool for ML")

    traces = mlflow.search_traces()
    if not pass_full_dataframe:
        traces = traces[["trace"]]

    # Harness should run without an error
    mlflow.genai.evaluate(data=traces, scorers=[safety()])
