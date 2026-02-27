import os
import warnings
from unittest import mock

import pytest

from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException


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


def test_create_endpoint_name_match():
    """Test when name is provided both in config and as named arg with matching values.
    Should emit a deprecation warning about using name parameter.
    """
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"name": "test"}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        with pytest.warns(
            UserWarning,
            match="Passing 'name' as a parameter is deprecated. "
            "Please specify 'name' only within the config dictionary.",
        ):
            resp = client.create_endpoint(
                name="test",
                config={
                    "name": "test",
                    "config": {
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
                },
            )
        mock_request.assert_called_once()
        assert resp == {"name": "test"}


def test_create_endpoint_name_mismatch():
    """Test when name is provided both in config and as named arg with different values.
    Should raise an MlflowException.
    """
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"name": "test"}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200

    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        with pytest.raises(
            MlflowException,
            match="Name mismatch. Found 'test1' as parameter and 'test2' "
            "in config. Please specify 'name' only within the config "
            "dictionary as this parameter is deprecated.",
        ):
            client.create_endpoint(
                name="test1",
                config={
                    "name": "test2",
                    "config": {
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
                },
            )
        mock_request.assert_not_called()


def test_create_endpoint_route_optimized_match():
    """Test when route_optimized is provided both in config and as named arg with matching values.
    Should emit a deprecation warning.
    """
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"name": "test"}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200

    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        with pytest.warns(
            UserWarning,
            match="Passing 'route_optimized' as a parameter is deprecated. "
            "Please specify 'route_optimized' only within the config dictionary.",
        ):
            resp = client.create_endpoint(
                name="test",
                route_optimized=True,
                config={
                    "name": "test",
                    "route_optimized": True,
                    "config": {
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
                },
            )
        mock_request.assert_called_once()
        assert resp == {"name": "test"}


def test_create_endpoint_route_optimized_mismatch():
    """Test when route_optimized is provided both in config and as named arg with different values.
    Should raise an MlflowException.
    """
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"name": "test"}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200

    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        with pytest.raises(
            MlflowException,
            match="Conflicting 'route_optimized' values found. "
            "Please specify 'route_optimized' only within the config dictionary "
            "as this parameter is deprecated.",
        ):
            client.create_endpoint(
                name="test",
                route_optimized=True,
                config={
                    "name": "test",
                    "route_optimized": False,
                    "config": {
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
                },
            )
        mock_request.assert_not_called()


def test_create_endpoint_named_name():
    """Test using the legacy format with separate parameters instead of full API payload.
    Should emit a deprecation warning about the old format.
    """
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"name": "test"}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200

    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        with pytest.warns(
            UserWarning,
            match="Passing 'name', 'config', and 'route_optimized' as separate parameters is "
            "deprecated. Please pass the full API request payload as a single dictionary "
            "in the 'config' parameter.",
        ):
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


def test_create_endpoint_named_route_optimized():
    """Test using the legacy format with route_optimized parameter.
    Should emit a deprecation warning about the old format.
    """
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"name": "test"}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200

    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        with pytest.warns(
            UserWarning,
            match="Passing 'name', 'config', and 'route_optimized' as separate parameters is "
            "deprecated. Please pass the full API request payload as a single dictionary "
            "in the 'config' parameter.",
        ):
            resp = client.create_endpoint(
                name="test",
                route_optimized=True,
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
        with pytest.warns(
            UserWarning,
            match="The `update_endpoint` method is deprecated. Use the specific update methods—"
            "`update_endpoint_config`, `update_endpoint_tags`, `update_endpoint_rate_limits`, "
            "`update_endpoint_ai_gateway`—instead.",
        ):
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


def test_predict_with_total_timeout_env_var(monkeypatch):
    monkeypatch.setenv("MLFLOW_DEPLOYMENT_PREDICT_TOTAL_TIMEOUT", "900")
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"foo": "bar"}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200

    with mock.patch(
        "mlflow.deployments.databricks.http_request", return_value=mock_resp
    ) as mock_http:
        resp = client.predict(endpoint="test", inputs={})
        mock_http.assert_called_once()
        call_kwargs = mock_http.call_args[1]
        assert call_kwargs["retry_timeout_seconds"] == 900
        assert resp == {"foo": "bar"}


def test_predict_stream_with_total_timeout_env_var(monkeypatch):
    monkeypatch.setenv("MLFLOW_DEPLOYMENT_PREDICT_TOTAL_TIMEOUT", "900")
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.iter_lines.return_value = [
        "data: " + '{"id": "1", "choices": [{"delta": {"content": "Hello"}}]}',
        "data: [DONE]",
    ]
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200
    mock_resp.encoding = "utf-8"

    with mock.patch(
        "mlflow.deployments.databricks.http_request", return_value=mock_resp
    ) as mock_http:
        chunks = list(client.predict_stream(endpoint="test", inputs={}))
        mock_http.assert_called_once()
        call_kwargs = mock_http.call_args[1]
        assert call_kwargs["retry_timeout_seconds"] == 900
        assert len(chunks) == 1


def test_predict_warns_on_misconfigured_timeouts(monkeypatch):
    monkeypatch.setenv("MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT", "300")
    monkeypatch.setenv("MLFLOW_DEPLOYMENT_PREDICT_TOTAL_TIMEOUT", "120")
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"foo": "bar"}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200

    with mock.patch(
        "mlflow.deployments.databricks.http_request", return_value=mock_resp
    ) as mock_http:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resp = client.predict(endpoint="test", inputs={})

        mock_http.assert_called_once()
        assert resp == {"foo": "bar"}
        assert len(w) == 1
        warning_msg = str(w[0].message)
        assert "MLFLOW_DEPLOYMENT_PREDICT_TOTAL_TIMEOUT" in warning_msg
        assert "(120s)" in warning_msg
        assert "(300s)" in warning_msg


def test_predict_stream_warns_on_misconfigured_timeouts(monkeypatch):
    monkeypatch.setenv("MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT", "300")
    monkeypatch.setenv("MLFLOW_DEPLOYMENT_PREDICT_TOTAL_TIMEOUT", "120")
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.iter_lines.return_value = [
        "data: " + '{"id": "1", "choices": [{"delta": {"content": "Hello"}}]}',
        "data: [DONE]",
    ]
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200
    mock_resp.encoding = "utf-8"

    with mock.patch(
        "mlflow.deployments.databricks.http_request", return_value=mock_resp
    ) as mock_http:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chunks = list(client.predict_stream(endpoint="test", inputs={}))

        mock_http.assert_called_once()
        assert len(chunks) == 1
        assert len(w) == 1
        warning_msg = str(w[0].message)
        assert "MLFLOW_DEPLOYMENT_PREDICT_TOTAL_TIMEOUT" in warning_msg
        assert "(120s)" in warning_msg
        assert "(300s)" in warning_msg


def test_predict_no_warning_when_timeouts_properly_configured(monkeypatch):
    monkeypatch.setenv("MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT", "120")
    monkeypatch.setenv("MLFLOW_DEPLOYMENT_PREDICT_TOTAL_TIMEOUT", "600")
    client = get_deploy_client("databricks")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"foo": "bar"}
    mock_resp.url = os.environ["DATABRICKS_HOST"]
    mock_resp.status_code = 200

    with (
        mock.patch(
            "mlflow.deployments.databricks.http_request", return_value=mock_resp
        ) as mock_http,
        mock.patch("mlflow.utils.rest_utils._logger.warning") as mock_warning,
    ):
        resp = client.predict(endpoint="test", inputs={})
        mock_http.assert_called_once()
        assert resp == {"foo": "bar"}
        mock_warning.assert_not_called()
