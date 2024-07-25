from unittest import mock

import pytest

from mlflow.deployments import get_deploy_client
from mlflow.deployments.mlflow import MlflowDeploymentClient


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
