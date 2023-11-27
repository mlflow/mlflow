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
    monkeypatch.setenv("OPENAI_API_VERSION", "2023-05-15")


def test_get_deploy_client(mock_openai_creds):
    get_deploy_client("openai")


def test_predict_openai(mock_openai_creds):
    client = get_deploy_client("openai")
    mock_resp = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-0301",
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
        "mlflow.openai.api_request_parallel_processor.process_api_requests",
        return_value=[mock_resp],
    ) as mock_request:
        resp = client.predict(endpoint="test", inputs={"prompt": "my prompt", "temperature": 0.1})
        mock_request.assert_called_once_with(
            [
                {
                    "model": "test",
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": "my prompt"}],
                    "api_base": "https://api.openai.com/v1",
                    "api_type": "open_ai",
                }
            ],
            mock.ANY,
            api_token=mock.ANY,
            throw_original_error=True,
            max_workers=1,
        )
        assert resp == {
            "candidates": [{"text": "\n\nThis is a test!", "metadata": {"finish_reason": "stop"}}]
        }


def test_predict_azure_openai(mock_azure_openai_creds):
    client = get_deploy_client("openai")
    mock_resp = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-0301",
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
        "mlflow.openai.api_request_parallel_processor.process_api_requests",
        return_value=[mock_resp],
    ) as mock_request:
        resp = client.predict(endpoint="test", inputs={"prompt": "my prompt", "temperature": 0.1})
        mock_request.assert_called_once_with(
            [
                {
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": "my prompt"}],
                    "api_base": "my-base",
                    "api_version": "2023-05-15",
                    "api_type": "azure",
                    "deployment_id": "my-deployment",
                }
            ],
            mock.ANY,
            api_token=mock.ANY,
            throw_original_error=True,
            max_workers=1,
        )
        assert resp == {
            "candidates": [{"text": "\n\nThis is a test!", "metadata": {"finish_reason": "stop"}}]
        }
