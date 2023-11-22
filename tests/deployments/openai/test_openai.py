from unittest import mock

import pytest

from mlflow.deployments import get_deploy_client


@pytest.fixture
def mock_openai_creds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "my-secret-key")


@pytest.fixture
def mock_azure_openai_creds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "my-secret-key")
    monkeypatch.setenv("OPENAI_API_TYPE", "azure")
    monkeypatch.setenv("OPENAI_API_BASE", "my-base")
    monkeypatch.setenv("OPENAI_DEPLOYMENT_NAME", "my-deployment")


def test_get_deploy_client(mock_openai_creds):
    get_deploy_client("openai")


def test_predict(mock_openai_creds):
    client = get_deploy_client("openai")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"foo": "bar"}
    mock_resp.status_code = 200
    with mock.patch(
        "mlflow.openai.api_request_parallel_processor.process_api_requests", return_value=mock_resp
    ) as mock_request:
        resp = client.predict(endpoint="test", inputs={})
        mock_request.assert_called_once()
        assert resp == {"foo": "bar"}
