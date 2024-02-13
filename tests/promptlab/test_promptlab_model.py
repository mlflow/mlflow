from unittest import mock

import pandas as pd
import pytest

from mlflow.deployments import set_deployments_target
from mlflow.entities.param import Param
from mlflow.promptlab import _PromptlabModel


@pytest.fixture(autouse=True, scope="module")
def set_target():
    set_deployments_target("http://localhost:5000")
    yield
    set_deployments_target(None)


def construct_model(route):
    return _PromptlabModel(
        "Write me a story about {{ thing }}.",
        [Param(key="thing", value="books")],
        [Param(key="temperature", value=0.5), Param(key="max_tokens", value=10)],
        route,
    )


def test_promptlab_prompt_replacement():
    data = pd.DataFrame(
        data=[
            {"thing": "books"},
            {"thing": "coffee"},
            {"thing": "nothing"},
        ]
    )

    model = construct_model("completions")
    with mock.patch(
        "mlflow.deployments.mlflow.MlflowDeploymentClient.get_endpoint",
        return_value=mock.Mock(endpoint_type="llm/v1/completions"),
    ) as mock_get_endpoint, mock.patch(
        "mlflow.deployments.mlflow.MlflowDeploymentClient.predict"
    ) as mock_query:
        model.predict(data)
        mock_get_endpoint.assert_called()
        calls = [
            mock.call(
                endpoint="completions",
                inputs={
                    "prompt": f"Write me a story about {thing}.",
                    "temperature": 0.5,
                    "max_tokens": 10,
                },
            )
            for thing in data["thing"]
        ]

        mock_query.assert_has_calls(calls, any_order=True)


def test_promptlab_works_with_chat_route():
    mock_response = {
        "choices": [
            {"message": {"role": "user", "content": "test"}, "metadata": {"finish_reason": "stop"}}
        ]
    }
    model = construct_model("chat")

    with mock.patch(
        "mlflow.deployments.mlflow.MlflowDeploymentClient.get_endpoint",
        return_value=mock.Mock(endpoint_type="llm/v1/chat"),
    ) as mock_get_endpoint, mock.patch(
        "mlflow.deployments.mlflow.MlflowDeploymentClient.predict", return_value=mock_response
    ) as mock_predict:
        response = model.predict(pd.DataFrame(data=[{"thing": "books"}]))
        mock_get_endpoint.assert_called()
        mock_predict.assert_called()
        assert response == ["test"]


def test_promptlab_works_with_completions_route():
    mock_response = {
        "choices": [
            {
                "text": "test",
                "metadata": {"finish_reason": "stop"},
            }
        ]
    }
    model = construct_model("completions")
    with mock.patch(
        "mlflow.deployments.mlflow.MlflowDeploymentClient.get_endpoint",
        return_value=mock.Mock(endpoint_type="llm/v1/completions"),
    ) as mock_get_endpoint, mock.patch(
        "mlflow.deployments.mlflow.MlflowDeploymentClient.predict", return_value=mock_response
    ) as mock_predict:
        response = model.predict(pd.DataFrame(data=[{"thing": "books"}]))
        mock_get_endpoint.assert_called()
        mock_predict.assert_called()
        assert response == ["test"]
