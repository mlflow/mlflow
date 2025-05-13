from importlib import import_module
from unittest import mock
from unittest.mock import patch

import pandas as pd
import pytest
from packaging.version import Version

import mlflow
from mlflow.entities.assessment import Assessment, Expectation, Feedback
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.entities.span import SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import scorer
from mlflow.genai.scorers.builtin_scorers import GENAI_CONFIG_NAME, safety

from tests.evaluate.test_evaluation import _DUMMY_CHAT_RESPONSE
from tests.genai.conftest import mock_init_auth

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
    return Assessment(
        name="relevance",
        feedback=Feedback(value="yes"),
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

    with mock.patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
        result = mlflow.genai.evaluate(
            data=data,
            scorers=[exact_match, max_length, relevance, has_trace],
        )

    metrics = result.metrics
    assert metrics["metric/exact_match/average"] == 1.0
    assert metrics["metric/max_length/average"] == 0.5
    assert metrics["metric/relevance/relevance/average"] == 1.0
    assert metrics["metric/has_trace/average"] == 1.0


@pytest.mark.skipif(not _IS_AGENT_SDK_V1, reason="Databricks Agent SDK v1 is required")
def test_evaluate_with_predict_fn():
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

    with mock.patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
        result = mlflow.genai.evaluate(
            predict_fn=model.predict,
            data=data,
            scorers=[exact_match, max_length, relevance, has_trace],
        )

    metrics = result.metrics
    assert metrics["metric/exact_match/average"] == 0.0
    assert metrics["metric/max_length/average"] == 0.5
    assert metrics["metric/relevance/relevance/average"] == 1.0
    assert metrics["metric/has_trace/average"] == 1.0


@pytest.mark.skipif(not _IS_AGENT_SDK_V1, reason="Databricks Agent SDK v1 is required")
def test_evaluate_with_traces():
    questions = ["What is MLflow?", "What is Spark?"]

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(question: str) -> str:
        return TestModel().predict(question)

    for question in questions:
        predict(question)

    data = mlflow.search_traces()

    # OSS MLflow backend doesn't support assessment APIs now, so we need to manually add them
    data.iloc[0]["trace"].info.assessments = [
        Assessment(
            name="expected_response",
            trace_id="tr-123",
            expectation=Expectation(value="MLflow is a tool for ML"),
            source=AssessmentSource(source_id="me", source_type="HUMAN"),
        ),
        Assessment(
            name="max_length",
            trace_id="tr-123",
            expectation=Expectation(value=100),
            source=AssessmentSource(source_id="me", source_type="HUMAN"),
        ),
    ]
    data.iloc[1]["trace"].info.assessments = [
        Assessment(
            name="expected_response",
            trace_id="tr-123",
            expectation=Expectation(value="Spark is a fast data processing engine"),
            source=AssessmentSource(source_id="me", source_type="HUMAN"),
        ),
        Assessment(
            name="max_length",
            trace_id="tr-123",
            expectation=Expectation(value=1),
            source=AssessmentSource(source_id="me", source_type="HUMAN"),
        ),
    ]

    with mock.patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
        result = mlflow.genai.evaluate(
            data=data,
            scorers=[exact_match, max_length, relevance, has_trace],
        )

    metrics = result.metrics
    assert metrics["metric/exact_match/average"] == 0.0
    assert metrics["metric/max_length/average"] == 0.5
    assert metrics["metric/relevance/relevance/average"] == 1.0
    assert metrics["metric/has_trace/average"] == 1.0


def mock_init_auth(config_instance):
    config_instance.host = "https://databricks.com/"
    config_instance._header_factory = lambda: {}


@pytest.mark.parametrize(
    "model_input",
    [
        # Case 1: Single chat dictionary.
        # This is an expected input format from the Databricks RAG Evaluator.
        {
            "messages": [{"content": "What is MLflow?", "role": "user"}],
            "max_tokens": 10,
        },
        # Case 2: List of chat dictionaries.
        # This is not a typical input format from either default or Databricks RAG evaluators,
        # but we support it for compatibility with the normal Pyfunc models.
        [
            {"messages": [{"content": "What is MLflow?", "role": "user"}]},
            {"messages": [{"content": "What is Spark?", "role": "user"}]},
        ],
        # Case 3: DataFrame with a column of dictionaries
        pd.DataFrame(
            {
                "inputs": [
                    {
                        "messages": [{"content": "What is MLflow?", "role": "user"}],
                        "max_tokens": 10,
                    },
                    {
                        "messages": [{"content": "What is Spark?", "role": "user"}],
                    },
                ]
            }
        ),
        # Case 4: DataFrame with a column of strings
        pd.DataFrame(
            {
                "inputs": ["What is MLflow?", "What is Spark?"],
            }
        ),
    ],
)
@mock.patch("mlflow.deployments.get_deploy_client")
def test_model_from_deployment_endpoint(mock_deploy_client, model_input):
    mock_deploy_client.return_value.predict.return_value = _DUMMY_CHAT_RESPONSE
    mock_deploy_client.return_value.get_endpoint.return_value = {"task": "llm/v1/chat"}

    predict_fn = mlflow.genai.to_predict_fn("endpoints:/chat")

    response = predict_fn(model_input)

    if isinstance(model_input, dict):
        assert mock_deploy_client.return_value.predict.call_count == 1
        # Chat response should be unwrapped
        assert response == "This is a response"
    else:
        assert mock_deploy_client.return_value.predict.call_count == 2
        assert pd.Series(response).equals(pd.Series(["This is a response"] * 2))


def test_evaluate_passes_model_id_to_mlflow_evaluate():
    # Tracking URI = databricks is required to use mlflow.genai.evaluate()
    mlflow.set_tracking_uri("databricks")
    data = [
        {"inputs": {"foo": "bar"}, "outputs": "response from model"},
        {"inputs": {"baz": "qux"}, "outputs": "response from model"},
    ]

    with mock.patch("mlflow.evaluate") as mock_evaluate:

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
        )


@patch("mlflow.get_tracking_uri", return_value="databricks")
def test_no_scorers(mock_get_tracking_uri):
    with pytest.raises(TypeError, match=r"evaluate\(\) missing 1 required positional"):
        mlflow.genai.evaluate(data=[{"inputs": "Hello", "outputs": "Hi"}])

    with pytest.raises(MlflowException, match=r"At least one scorer is required"):
        mlflow.genai.evaluate(data=[{"inputs": "Hello", "outputs": "Hi"}], scorers=[])
