import os
from unittest import mock
from numpy import isin

import pytest

from mlflow.deployments.batch_inference import (
    BatchInferenceHandlerFactory,
    EmbeddingsV1Handler,
    ChatV1Handler,
)
from mlflow.exceptions import MlflowException


@pytest.fixture(autouse=True)
def mock_databricks_credentials(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "secret")
    

def test_batch_inference_handler_factory():
    with mock.patch(
        "mlflow.deployments.databricks.DatabricksDeploymentClient.get_endpoint", return_value={"task": "llm/v1/embeddings"}
    ):
        handler = BatchInferenceHandlerFactory.create(target_uri="databricks", endpoint="databricks-bge")
        assert isinstance(handler, EmbeddingsV1Handler)

    with mock.patch(
        "mlflow.deployments.databricks.DatabricksDeploymentClient.get_endpoint", return_value={"task": "llm/v1/chat"}
    ):
        handler = BatchInferenceHandlerFactory.create(target_uri="databricks", endpoint="databricks-mixtral")
        assert isinstance(handler, ChatV1Handler)

    with mock.patch(
        "mlflow.deployments.databricks.DatabricksDeploymentClient.get_endpoint", return_value={"task": "unknown"}
    ):
        with pytest.raises(MlflowException):
            BatchInferenceHandlerFactory.create(target_uri="databricks", endpoint="databricks-unkown")


def test_batch_inference_embedding_v1():
    def _predict(inputs, *args, **kwargs):
        data = inputs["input"]
        if isinstance(data, str):
            return {"data": [{"embedding": [1.0]}]}
        else:
            return {"data": [{"embedding": [float(i)]} for i in range(len(data))]}

    with mock.patch(
        "mlflow.deployments.databricks.DatabricksDeploymentClient.get_endpoint", return_value={"task": "llm/v1/embeddings"}
    ):
        handler = BatchInferenceHandlerFactory.create("databricks", "databricks-bge", batch_size=2)

    with mock.patch("mlflow.deployments.databricks.DatabricksDeploymentClient.predict", side_effect=_predict) as mock_predict:
        response = handler.predict(["first", "second", "third"])
        assert response == {'embedding': [[0.0], [1.0], [0.0]], 'error': [None, None, None]}
        assert mock_predict.call_count == 2


def test_batch_inference_chat_v1():
    def _predict(inputs, *args, **kwargs):
        return {"choices": [{"message": {"content": "fake"}}]}

    with mock.patch(
        "mlflow.deployments.databricks.DatabricksDeploymentClient.get_endpoint", return_value={"task": "llm/v1/chat"}
    ):
        handler = BatchInferenceHandlerFactory.create("databricks", "databricks-mixtral")

    with mock.patch("mlflow.deployments.databricks.DatabricksDeploymentClient.predict", side_effect=_predict) as mock_predict:
        # support pass in list of string
        response = handler.predict(["first message", "second message", "third message"])
        assert response == {'chat': ["fake", "fake", "fake"], 'error': [None, None, None]}
        assert mock_predict.call_count == 3

    with mock.patch("mlflow.deployments.databricks.DatabricksDeploymentClient.predict", side_effect=_predict) as mock_predict:
        # support pass in list of messages (message type is List[Dict[str, str]])
        response = handler.predict([
            [{"role": "user", "content": "hello"}, {"role": "user", "content": "world"}],
            [{"role": "user", "content": "foo"}]
        ])
        assert response == {'chat': ["fake", "fake"], 'error': [None, None]}
        assert mock_predict.call_count == 2
