from unittest import mock

import pytest
from fastapi.testclient import TestClient

from mlflow.deployments.server.app import create_app_from_config, create_app_from_env
from mlflow.deployments.server.constants import (
    MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE,
    MLFLOW_DEPLOYMENTS_ENDPOINTS_BASE,
)
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import GatewayConfig

from tests.gateway.tools import MockAsyncResponse


@pytest.fixture
def client() -> TestClient:
    config = GatewayConfig(
        **{
            "endpoints": [
                {
                    "name": "completions-gpt4",
                    "endpoint_type": "llm/v1/completions",
                    "model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "mykey",
                            "openai_api_base": "https://api.openai.com/v1",
                            "openai_api_version": "2023-05-10",
                            "openai_api_type": "openai",
                        },
                    },
                },
                {
                    "name": "chat-gpt4",
                    "endpoint_type": "llm/v1/chat",
                    "model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "MY_API_KEY",
                        },
                    },
                    "limit": {"calls": 10, "key": None, "renewal_period": "minute"},
                },
            ]
        }
    )
    app = create_app_from_config(config)
    return TestClient(app)


model_response = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "gpt-4o-mini",
    "usage": {
        "prompt_tokens": 13,
        "completion_tokens": 7,
        "total_tokens": 20,
    },
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "\n\nThis is a test!",
            },
            "finish_reason": "stop",
            "index": 0,
        }
    ],
    "headers": {"Content-Type": "application/json"},
}

test_response = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "gpt-4o-mini",
    "usage": {
        "prompt_tokens": 13,
        "completion_tokens": 7,
        "total_tokens": 20,
    },
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "\n\nThis is a test!",
                "tool_calls": None,
            },
            "finish_reason": "stop",
            "index": 0,
        }
    ],
}


def test_index(client: TestClient):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["Location"] == "/docs"


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_favicon(client: TestClient):
    response = client.get("/favicon.ico")
    assert response.status_code == 200


def test_docs(client: TestClient):
    response = client.get("/docs")
    assert response.status_code == 200


def test_list_endpoints(client: TestClient):
    response = client.get(MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE)
    assert response.status_code == 200
    assert response.json()["endpoints"] == [
        {
            "name": "completions-gpt4",
            "endpoint_type": "llm/v1/completions",
            "endpoint_url": "/gateway/completions-gpt4/invocations",
            "model": {
                "name": "gpt-4",
                "provider": "openai",
            },
            "limit": None,
        },
        {
            "name": "chat-gpt4",
            "endpoint_type": "llm/v1/chat",
            "endpoint_url": "/gateway/chat-gpt4/invocations",
            "model": {
                "name": "gpt-4",
                "provider": "openai",
            },
            "limit": {"calls": 10, "key": None, "renewal_period": "minute"},
        },
    ]


def test_get_endpoint(client: TestClient):
    response = client.get(f"{MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE}chat-gpt4")
    assert response.status_code == 200
    assert response.json() == {
        "name": "chat-gpt4",
        "endpoint_type": "llm/v1/chat",
        "endpoint_url": "/gateway/chat-gpt4/invocations",
        "model": {
            "name": "gpt-4",
            "provider": "openai",
        },
        "limit": {"calls": 10, "key": None, "renewal_period": "minute"},
    }


def test_dynamic_endpoint():
    config = GatewayConfig(
        **{
            "endpoints": [
                {
                    "name": "chat",
                    "endpoint_type": "llm/v1/chat",
                    "model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "mykey",
                            "openai_api_base": "https://api.openai.com/v1",
                        },
                    },
                    "limit": {"calls": 10, "key": None, "renewal_period": "minute"},
                }
            ]
        }
    )
    app = create_app_from_config(config)
    client = TestClient(app)

    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(model_response)
    ) as mock_post:
        resp = client.post(
            f"{MLFLOW_DEPLOYMENTS_ENDPOINTS_BASE}chat/invocations",
            json={"messages": [{"role": "user", "content": "Tell me a joke"}]},
        )
        mock_post.assert_called_once()
        assert resp.status_code == 200
        assert resp.json() == test_response


def test_rate_limit():
    config = GatewayConfig(
        **{
            "endpoints": [
                {
                    "name": "chat",
                    "endpoint_type": "llm/v1/chat",
                    "model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "mykey",
                            "openai_api_base": "https://api.openai.com/v1",
                        },
                    },
                    "limit": {"calls": 1, "key": None, "renewal_period": "minute"},
                }
            ]
        }
    )
    app = create_app_from_config(config)
    client = TestClient(app)

    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(model_response)
    ) as mock_post:
        resp = client.post(
            f"{MLFLOW_DEPLOYMENTS_ENDPOINTS_BASE}chat/invocations",
            json={"messages": [{"role": "user", "content": "Tell me a joke"}]},
        )
        mock_post.assert_called_once()
        assert resp.status_code == 200
        assert resp.json() == test_response
        # second call
        resp = client.post(
            f"{MLFLOW_DEPLOYMENTS_ENDPOINTS_BASE}chat/invocations",
            json={"messages": [{"role": "user", "content": "Tell me a joke again"}]},
        )
        assert resp.status_code == 429


def test_create_app_from_env_fails_if_MLFLOW_DEPLOYMENTS_CONFIG_is_not_set(monkeypatch):
    monkeypatch.delenv("MLFLOW_DEPLOYMENTS_CONFIG", raising=False)
    with pytest.raises(MlflowException, match="'MLFLOW_DEPLOYMENTS_CONFIG' is not set"):
        create_app_from_env()
