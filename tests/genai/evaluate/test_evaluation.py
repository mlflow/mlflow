from unittest import mock
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.builtin_scorers import GENAI_CONFIG_NAME, groundedness

from tests.evaluate.test_evaluation import _DUMMY_CHAT_RESPONSE


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
            scorers=[groundedness()],
        )

        # Verify the call was made with the right parameters
        mock_evaluate.assert_called_once_with(
            model=model,
            data=mock.ANY,
            evaluator_config={GENAI_CONFIG_NAME: {"metrics": ["groundedness"]}},
            model_type="databricks-agent",
            extra_metrics=[],
            model_id="test_model_id",
        )


def test_evaluate_accepts_managed_dataset():
    from databricks.rag_eval.datasets.rest_entities import DatasetRecord, Expectation, Input

    from mlflow.genai.datasets import EvaluationDataset

    record = DatasetRecord(
        inputs=[Input(key="question", value="foo")],
        expectations={"expected_response": Expectation(value="bar")},
    )

    # build a fake client whose list_dataset_records() returns your record
    fake_client = MagicMock()
    fake_client.list_dataset_records.return_value = [record]

    with (
        patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth),
        patch("databricks.rag_eval.datasets.entities._get_client", return_value=fake_client),
        patch("mlflow.get_tracking_uri", return_value="databricks"),
    ):
        results = mlflow.genai.evaluate(data=EvaluationDataset("test-id"), scorers=[groundedness()])

        assert results.result_df["request"].tolist() == [{"question": "foo"}]
        assert results.result_df["expected_response"].tolist() == ["bar"]


@patch("mlflow.get_tracking_uri", return_value="databricks")
def test_no_scorers(mock_get_tracking_uri):
    with pytest.raises(TypeError, match=r"evaluate\(\) missing 1 required positional"):
        mlflow.genai.evaluate(data=[{"inputs": "Hello", "outputs": "Hi"}])

    with pytest.raises(MlflowException, match=r"At least one scorer is required"):
        mlflow.genai.evaluate(data=[{"inputs": "Hello", "outputs": "Hi"}], scorers=[])
