import os
from unittest import mock

import pytest

from mlflow.deployments import get_deploy_client


@pytest.fixture(autouse=True)
def mock_databricks_credentials(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "secret")


def test_get_deploy_client():
    get_deploy_client("databricks")
    get_deploy_client("databricks://scope:prefix")


def test_create_endpoint():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"name": "test"}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.create_endpoint(
            name="test",
            config={
                "served_entities": [
                    {
                        "name": "test",
                        "external_model": {
                            "name": "gpt-4",
                            "provider": "openai",
                            "openai_config": {
                                "openai_api_key": "secret",
                            },
                        },
                    }
                ],
                "task": "llm/v1/chat",
            },
        )
        mock_request.assert_called_once()
        assert resp == {"name": "test"}


def test_create_endpoint_config_only():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"name": "test"}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.create_endpoint(
            config={
                "name": "test_new",
                "config": {
                    "served_entities": [
                        {
                            "name": "test_entity",
                            "external_model": {
                                "name": "gpt-4",
                                "provider": "openai",
                                "task": "llm/v1/chat",
                                "openai_config": {
                                    "openai_api_key": "secret",
                                },
                            },
                        }
                    ],
                    "route_optimized": True,
                },
            },
        )
        mock_request.assert_called_once()
        assert resp == {"name": "test"}


def test_get_endpoint():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"name": "test"}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.get_endpoint(endpoint="test")
        mock_request.assert_called_once()
        assert resp == {"name": "test"}


def test_list_endpoints():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"endpoints": [{"name": "test"}]}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.list_endpoints()
        mock_request.assert_called_once()
        assert resp == [{"name": "test"}]


def test_update_endpoint():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.update_endpoint(
            endpoint="test",
            config={
                "served_entities": [
                    {
                        "name": "test",
                        "external_model": {
                            "name": "gpt-4",
                            "provider": "openai",
                            "openai_config": {
                                "openai_api_key": "secret",
                            },
                        },
                    }
                ],
                "task": "llm/v1/chat",
            },
        )
        mock_request.assert_called_once()
        assert resp == {}


def test_update_endpoint_config():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.update_endpoint_config(
            endpoint="test",
            config={
                "served_entities": [
                    {
                        "name": "gpt-4-mini",
                        "external_model": {
                            "name": "gpt-4-mini",
                            "provider": "openai",
                            "task": "llm/v1/chat",
                            "openai_config": {
                                "openai_api_key": "{{secrets/scope/key}}",
                            },
                        },
                    }
                ],
            },
        )
        mock_request.assert_called_once()
        assert resp == {}


def test_update_endpoint_tags():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.update_endpoint_tags(
            endpoint="test",
            config={"add_tags": [{"key": "project", "value": "test"}]},
        )
        mock_request.assert_called_once()
        assert resp == {}


def test_update_endpoint_rate_limits():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.update_endpoint_rate_limits(
            endpoint="test",
            config={"rate_limits": [{"calls": 10, "key": "endpoint", "renewal_period": "minute"}]},
        )
        mock_request.assert_called_once()
        assert resp == {}


def test_update_endpoint_ai_gateway():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.update_endpoint_ai_gateway(
            endpoint="test",
            config={
                "usage_tracking_config": {"enabled": True},
                "inference_table_config": {
                    "enabled": True,
                    "catalog_name": "my_catalog",
                    "schema_name": "my_schema",
                },
            },
        )
        mock_request.assert_called_once()
        assert resp == {}


def test_delete_endpoint():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.delete_endpoint(endpoint="test")
        mock_request.assert_called_once()
        assert resp == {}


def test_predict():
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"foo": "bar"}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.predict(endpoint="test", inputs={})
        mock_request.assert_called_once()
        assert resp == {"foo": "bar"}
