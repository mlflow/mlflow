from types import SimpleNamespace
from unittest import mock

import pytest

from mlflow.assistant.gateway_connection import ensure_gateway_connection
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST


def _not_found() -> MlflowException:
    return MlflowException("missing", error_code=RESOURCE_DOES_NOT_EXIST)


def _missing_gateway_store() -> mock.MagicMock:
    store = mock.MagicMock()
    store.get_secret_info.side_effect = _not_found()
    store.get_gateway_model_definition.side_effect = _not_found()
    store.get_gateway_endpoint.side_effect = _not_found()
    store.create_gateway_secret.return_value = SimpleNamespace(secret_id="sec-1")
    store.create_gateway_model_definition.return_value = SimpleNamespace(
        model_definition_id="md-1",
    )
    store.create_gateway_endpoint.return_value = SimpleNamespace(
        name="mlflow-assistant-anthropic",
    )
    return store


@pytest.fixture
def store():
    store = _missing_gateway_store()
    with mock.patch("mlflow.assistant.gateway_connection._get_store", return_value=store):
        yield store


def test_ensure_gateway_connection_creates_secret_model_and_endpoint(store):
    endpoint_name = ensure_gateway_connection("anthropic", "sk-secret")

    assert endpoint_name == "mlflow-assistant-anthropic"
    store.create_gateway_secret.assert_called_once_with(
        secret_name="mlflow-assistant-anthropic",
        secret_value={"api_key": "sk-secret"},
        provider="anthropic",
    )
    store.create_gateway_model_definition.assert_called_once_with(
        name="mlflow-assistant-anthropic",
        secret_id="sec-1",
        provider="anthropic",
        model_name="claude-sonnet-5",
    )
    store.create_gateway_endpoint.assert_called_once()


def test_ensure_gateway_connection_rotates_existing_secret(store):
    store.get_secret_info.side_effect = None
    store.get_secret_info.return_value = SimpleNamespace(secret_id="sec-existing")
    store.get_gateway_model_definition.side_effect = None
    store.get_gateway_model_definition.return_value = SimpleNamespace(
        model_definition_id="md-existing",
    )
    store.get_gateway_endpoint.side_effect = None
    store.get_gateway_endpoint.return_value = SimpleNamespace(
        name="mlflow-assistant-openai",
    )

    endpoint_name = ensure_gateway_connection("openai", "sk-new")

    assert endpoint_name == "mlflow-assistant-openai"
    store.update_gateway_secret.assert_called_once_with(
        secret_id="sec-existing",
        secret_value={"api_key": "sk-new"},
    )
    store.create_gateway_secret.assert_not_called()
    store.create_gateway_model_definition.assert_not_called()
    store.create_gateway_endpoint.assert_not_called()
