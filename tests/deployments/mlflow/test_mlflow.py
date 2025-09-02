from unittest import mock

import pytest

from mlflow.deployments import get_deploy_client
from mlflow.deployments.mlflow import MlflowDeploymentClient
from mlflow.environment_variables import MLFLOW_DEPLOYMENT_CLIENT_HTTP_REQUEST_TIMEOUT


def test_get_deploy_client():
    client = get_deploy_client("http://localhost:5000")
    assert isinstance(client, MlflowDeploymentClient)


def test_create_endpoint():
    client = get_deploy_client("http://localhost:5000")
    with pytest.raises(NotImplementedError, match=r".*"):
        client.create_endpoint(name="test")


def test_update_endpoint():
    client = get_deploy_client("http://localhost:5000")
    with pytest.raises(NotImplementedError, match=r".*"):
        client.update_endpoint(endpoint="test")


def test_delete_endpoint():
    client = get_deploy_client("http://localhost:5000")
    with pytest.raises(NotImplementedError, match=r".*"):
        client.delete_endpoint(endpoint="test")


def test_get_endpoint():
    client = get_deploy_client("http://localhost:5000")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {
        "model": {"name": "gpt-4", "provider": "openai"},
        "name": "completions",
        "endpoint_type": "llm/v1/completions",
        "endpoint_url": "http://localhost:5000/endpoints/chat/invocations",
        "limit": None,
    }
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.get_endpoint(endpoint="test")
        mock_request.assert_called_once()
        assert resp.dict() == {
            "name": "completions",
            "endpoint_type": "llm/v1/completions",
            "model": {"name": "gpt-4", "provider": "openai"},
            "endpoint_url": "http://localhost:5000/endpoints/chat/invocations",
            "limit": None,
        }
        ((_, url), _) = mock_request.call_args
        assert url == "http://localhost:5000/api/2.0/endpoints/test"


def test_list_endpoints():
    client = get_deploy_client("http://localhost:5000")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {
        "endpoints": [
            {
                "model": {"name": "gpt-4", "provider": "openai"},
                "name": "completions",
                "endpoint_type": "llm/v1/completions",
                "endpoint_url": "http://localhost:5000/endpoints/chat/invocations",
                "limit": None,
            }
        ]
    }
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.list_endpoints()
        mock_request.assert_called_once()
        assert [r.dict() for r in resp] == [
            {
                "model": {"name": "gpt-4", "provider": "openai"},
                "name": "completions",
                "endpoint_type": "llm/v1/completions",
                "endpoint_url": "http://localhost:5000/endpoints/chat/invocations",
                "limit": None,
            }
        ]
        ((_, url), _) = mock_request.call_args
        assert url == "http://localhost:5000/api/2.0/endpoints/"


def test_list_endpoints_paginated():
    client = get_deploy_client("http://localhost:5000")
    mock_resp = mock.Mock()
    mock_resp.json.side_effect = [
        {
            "endpoints": [
                {
                    "model": {"name": "gpt-4", "provider": "openai"},
                    "name": "chat",
                    "endpoint_type": "llm/v1/chat",
                    "endpoint_url": "http://localhost:5000/endpoints/chat/invocations",
                    "limit": None,
                }
            ],
            "next_page_token": "token",
        },
        {
            "endpoints": [
                {
                    "model": {"name": "gpt-4", "provider": "openai"},
                    "name": "completions",
                    "endpoint_type": "llm/v1/completions",
                    "endpoint_url": "http://localhost:5000/endpoints/chat/invocations",
                    "limit": None,
                }
            ]
        },
    ]
    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.list_endpoints()
        assert mock_request.call_count == 2
        assert [r.dict() for r in resp] == [
            {
                "model": {"name": "gpt-4", "provider": "openai"},
                "name": "chat",
                "endpoint_type": "llm/v1/chat",
                "endpoint_url": "http://localhost:5000/endpoints/chat/invocations",
                "limit": None,
            },
            {
                "model": {"name": "gpt-4", "provider": "openai"},
                "name": "completions",
                "endpoint_type": "llm/v1/completions",
                "endpoint_url": "http://localhost:5000/endpoints/chat/invocations",
                "limit": None,
            },
        ]


def test_predict():
    client = get_deploy_client("http://localhost:5000")
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
        },
    }

    mock_resp.status_code = 200
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        resp = client.predict(endpoint="test", inputs={})
        mock_request.assert_called_once()
        assert resp == {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21,
            },
        }
        ((_, url), _) = mock_request.call_args
        assert url == "http://localhost:5000/endpoints/test/invocations"


def test_call_endpoint_uses_default_timeout():
    client = get_deploy_client("http://localhost:5000")

    with mock.patch("mlflow.deployments.mlflow.http_request") as mock_http_request:
        mock_http_request.return_value.json.return_value = {"test": "response"}
        mock_http_request.return_value.status_code = 200

        client._call_endpoint("GET", "/test")

        mock_http_request.assert_called_once()
        call_args = mock_http_request.call_args
        assert call_args.kwargs["timeout"] == MLFLOW_DEPLOYMENT_CLIENT_HTTP_REQUEST_TIMEOUT.get()


def test_call_endpoint_respects_custom_timeout():
    client = get_deploy_client("http://localhost:5000")
    custom_timeout = 600

    with mock.patch("mlflow.deployments.mlflow.http_request") as mock_http_request:
        mock_http_request.return_value.json.return_value = {"test": "response"}
        mock_http_request.return_value.status_code = 200

        client._call_endpoint("GET", "/test", timeout=custom_timeout)

        mock_http_request.assert_called_once()
        call_args = mock_http_request.call_args
        assert call_args.kwargs["timeout"] == custom_timeout


def test_call_endpoint_timeout_with_environment_variable(monkeypatch):
    custom_timeout = 420
    monkeypatch.setenv("MLFLOW_DEPLOYMENT_CLIENT_HTTP_REQUEST_TIMEOUT", str(custom_timeout))

    client = get_deploy_client("http://localhost:5000")

    with mock.patch("mlflow.deployments.mlflow.http_request") as mock_http_request:
        mock_http_request.return_value.json.return_value = {"test": "response"}
        mock_http_request.return_value.status_code = 200

        client._call_endpoint("GET", "/test")

        mock_http_request.assert_called_once()
        call_args = mock_http_request.call_args
        assert call_args.kwargs["timeout"] == custom_timeout


def test_get_endpoint_uses_deployment_client_timeout():
    client = get_deploy_client("http://localhost:5000")

    with mock.patch("mlflow.deployments.mlflow.http_request") as mock_http_request:
        mock_http_request.return_value.json.return_value = {
            "model": {"name": "gpt-4", "provider": "openai"},
            "name": "test",
            "endpoint_type": "llm/v1/chat",
            "endpoint_url": "http://localhost:5000/endpoints/test/invocations",
            "limit": None,
        }
        mock_http_request.return_value.status_code = 200

        client.get_endpoint("test")

        mock_http_request.assert_called_once()
        call_args = mock_http_request.call_args
        assert call_args.kwargs["timeout"] == MLFLOW_DEPLOYMENT_CLIENT_HTTP_REQUEST_TIMEOUT.get()


def test_list_endpoints_uses_deployment_client_timeout():
    client = get_deploy_client("http://localhost:5000")

    with mock.patch("mlflow.deployments.mlflow.http_request") as mock_http_request:
        mock_http_request.return_value.json.return_value = {"endpoints": []}
        mock_http_request.return_value.status_code = 200

        client.list_endpoints()

        mock_http_request.assert_called_once()
        call_args = mock_http_request.call_args
        assert call_args.kwargs["timeout"] == MLFLOW_DEPLOYMENT_CLIENT_HTTP_REQUEST_TIMEOUT.get()
