from unittest import mock

import pytest

from mlflow.deployments import get_deploy_client


@pytest.fixture(autouse=True)
def mock_databricks_credentials(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "secret")


def test_get_deploy_client():
    get_deploy_client("openai")


def test_predict():
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
