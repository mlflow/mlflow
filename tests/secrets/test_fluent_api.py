from unittest import mock

import mlflow
from mlflow.secrets.scope import SecretScope


def test_fluent_set_secret():
    with mock.patch("mlflow.tracking.MlflowClient") as MockClient:
        mock_client = MockClient.return_value
        mlflow.set_secret("api_key", "secret_value", SecretScope.GLOBAL)
        mock_client.set_secret.assert_called_once_with(
            "api_key", "secret_value", SecretScope.GLOBAL, None
        )


def test_fluent_list_secret_names():
    with mock.patch("mlflow.tracking.MlflowClient") as MockClient:
        mock_client = MockClient.return_value
        mock_client.list_secret_names.return_value = ["key1", "key2"]
        result = mlflow.list_secret_names(SecretScope.GLOBAL)
        assert result == ["key1", "key2"]
        mock_client.list_secret_names.assert_called_once_with(SecretScope.GLOBAL, None)


def test_fluent_delete_secret():
    with mock.patch("mlflow.tracking.MlflowClient") as MockClient:
        mock_client = MockClient.return_value
        mlflow.delete_secret("api_key", SecretScope.GLOBAL)
        mock_client.delete_secret.assert_called_once_with("api_key", SecretScope.GLOBAL, None)


def test_fluent_set_secret_with_scope_id():
    with mock.patch("mlflow.tracking.MlflowClient") as MockClient:
        mock_client = MockClient.return_value
        mlflow.set_secret("scorer_key", "secret", SecretScope.SCORER, 123)
        mock_client.set_secret.assert_called_once_with(
            "scorer_key", "secret", SecretScope.SCORER, 123
        )


def test_fluent_api_default_parameters():
    with mock.patch("mlflow.tracking.MlflowClient") as MockClient:
        mock_client = MockClient.return_value
        mlflow.set_secret("api_key", "secret_value")
        mock_client.set_secret.assert_called_once_with(
            "api_key", "secret_value", SecretScope.GLOBAL, None
        )
