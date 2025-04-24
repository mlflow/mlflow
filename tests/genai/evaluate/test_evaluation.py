from unittest import mock

import pandas as pd
import pytest

import mlflow.genai.evaluation

from tests.evaluate.test_evaluation import _DUMMY_CHAT_RESPONSE


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
