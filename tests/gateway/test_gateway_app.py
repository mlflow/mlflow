from fastapi.testclient import TestClient

from mlflow.gateway.app import create_app_from_config
from mlflow.gateway.config import GatewayConfig


def test_create_app():
    config = GatewayConfig(
        **{
            "routes": [
                {
                    "name": "completions-gpt4",
                    "type": "llm/v1/completions",
                    "model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "mykey",
                            "openai_api_base": "https://api.openai.com/v1",
                            "openai_api_version": "2023-05-10",
                            "openai_api_type": "openai/v1/chat/completions",
                        },
                    },
                },
                {
                    "name": "chat-gpt4",
                    "type": "llm/v1/chat",
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

    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

    response = client.get("/gateway/routes")
    assert response.status_code == 200
    assert response.json() == {
        "routes": [
            {
                "name": "completions-gpt4",
                "type": "llm/v1/completions",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                },
            },
            {
                "name": "chat-gpt4",
                "type": "llm/v1/chat",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                },
            },
        ]
    }

    response = client.get("/gateway/routes/chat-gpt4")
    assert response.status_code == 200
    assert response.json() == {
        "route": {
            "name": "chat-gpt4",
            "type": "llm/v1/chat",
            "model": {
                "name": "gpt-4",
                "provider": "openai",
            },
        }
    }
