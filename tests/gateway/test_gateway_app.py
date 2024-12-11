from unittest import mock

import pytest
from fastapi.testclient import TestClient

from mlflow.exceptions import MlflowException
from mlflow.gateway.app import create_app_from_config, create_app_from_env
from mlflow.gateway.config import GatewayConfig
from mlflow.gateway.constants import MLFLOW_GATEWAY_CRUD_ROUTE_BASE, MLFLOW_GATEWAY_ROUTE_BASE

from tests.gateway.tools import MockAsyncResponse


@pytest.fixture
def client() -> TestClient:
    config = GatewayConfig(
        **{
            "routes": [
                {
                    "name": "completions-gpt4",
                    "route_type": "llm/v1/completions",
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
                    "route_type": "llm/v1/chat",
                    "model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "MY_API_KEY",
                        },
                    },
                },
            ]
        }
    )
    app = create_app_from_config(config)
    return TestClient(app)


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


def test_search_routes(client: TestClient):
    response = client.get(MLFLOW_GATEWAY_CRUD_ROUTE_BASE)
    assert response.status_code == 200
    assert response.json()["routes"] == [
        {
            "name": "completions-gpt4",
            "route_type": "llm/v1/completions",
            "route_url": "/gateway/completions-gpt4/invocations",
            "model": {
                "name": "gpt-4",
                "provider": "openai",
            },
            "limit": None,
        },
        {
            "name": "chat-gpt4",
            "route_type": "llm/v1/chat",
            "route_url": "/gateway/chat-gpt4/invocations",
            "model": {
                "name": "gpt-4",
                "provider": "openai",
            },
            "limit": None,
        },
    ]


def test_get_route(client: TestClient):
    response = client.get(f"{MLFLOW_GATEWAY_CRUD_ROUTE_BASE}chat-gpt4")
    assert response.status_code == 200
    assert response.json() == {
        "name": "chat-gpt4",
        "route_type": "llm/v1/chat",
        "route_url": "/gateway/chat-gpt4/invocations",
        "model": {
            "name": "gpt-4",
            "provider": "openai",
        },
        "limit": None,
    }


def test_dynamic_route():
    config = GatewayConfig(
        **{
            "routes": [
                {
                    "name": "chat",
                    "route_type": "llm/v1/chat",
                    "model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "mykey",
                            "openai_api_base": "https://api.openai.com/v1",
                        },
                    },
                    "limit": None,
                }
            ]
        }
    )
    app = create_app_from_config(config)
    client = TestClient(app)

    resp = {
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
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        resp = client.post(
            f"{MLFLOW_GATEWAY_ROUTE_BASE}chat/invocations",
            json={"messages": [{"role": "user", "content": "Tell me a joke"}]},
        )
        mock_post.assert_called_once()
        assert resp.status_code == 200
        assert resp.json() == {
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


def test_create_app_from_env_fails_if_MLFLOW_GATEWAY_CONFIG_is_not_set(monkeypatch):
    monkeypatch.delenv("MLFLOW_GATEWAY_CONFIG", raising=False)
    with pytest.raises(MlflowException, match="'MLFLOW_GATEWAY_CONFIG' is not set"):
        create_app_from_env()
