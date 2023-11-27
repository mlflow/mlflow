from unittest import mock

import openai
import pytest

from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException


@pytest.fixture
def mock_openai_creds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "my-secret-key")


@pytest.fixture
def mock_azure_openai_creds(monkeypatch):
    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "my-secret-key",
            "OPENAI_API_TYPE": "azure",
            "OPENAI_API_BASE": "my-base",
            "OPENAI_DEPLOYMENT_NAME": "my-deployment",
            "OPENAI_API_VERSION": "2023-05-15",
        }
    )


@pytest.fixture
def mock_bad_azure_openai_creds(monkeypatch):
    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_API_TYPE": "azure",
            "OPENAI_API_VERSION": "2023-05-15",
            "OPENAI_API_BASE": "https://openai-for.openai.azure.com/",
        }
    )


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
        assert resp == mock_resp


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
        assert resp == mock_resp


def test_no_openai_api_key():
    client = get_deploy_client("openai")
    with pytest.raises(MlflowException, match="OPENAI_API_KEY environment variable not set"):
        client.predict(endpoint="test", inputs={"prompt": "my prompt", "temperature": 0.1})


def test_score_model_azure_openai_bad_envs(mock_bad_azure_openai_creds):
    client = get_deploy_client("openai")
    with pytest.raises(
        MlflowException, match="Either engine or deployment_id must be set for Azure OpenAI API"
    ):
        client.predict(endpoint="test", inputs={"prompt": "my prompt", "temperature": 0.1})


def test_openai_authentication_error(mock_openai_creds):
    client = get_deploy_client("openai")
    with mock.patch(
        "mlflow.openai.api_request_parallel_processor.process_api_requests",
        side_effect=openai.error.AuthenticationError("foo"),
    ) as mock_post:
        with pytest.raises(
            MlflowException, match="Authentication Error for OpenAI. Error response"
        ):
            client.predict(endpoint="test", inputs={"prompt": "my prompt", "temperature": 0.1})
        mock_post.assert_called_once()


def test_openai_invalid_request_error(mock_openai_creds):
    client = get_deploy_client("openai")
    with mock.patch(
        "mlflow.openai.api_request_parallel_processor.process_api_requests",
        side_effect=openai.error.InvalidRequestError("foo", "bar"),
    ) as mock_post:
        with pytest.raises(MlflowException, match="Invalid Request to OpenAI. Error response"):
            client.predict(endpoint="test", inputs={"prompt": "my prompt", "temperature": 0.1})
        mock_post.assert_called_once()


def test_openai_mlflow_exception(mock_openai_creds):
    client = get_deploy_client("openai")
    with mock.patch(
        "mlflow.openai.api_request_parallel_processor.process_api_requests",
        side_effect=MlflowException("foo"),
    ) as mock_post:
        with pytest.raises(MlflowException, match="foo"):
            client.predict(endpoint="test", inputs={"prompt": "my prompt", "temperature": 0.1})
        mock_post.assert_called_once()


def test_openai_mlflow_exception(mock_openai_creds):
    client = get_deploy_client("openai")
    with mock.patch(
        "mlflow.openai.api_request_parallel_processor.process_api_requests",
        side_effect=Exception("foo"),
    ) as mock_post:
        with pytest.raises(MlflowException, match="Error response from OpenAI:"):
            client.predict(endpoint="test", inputs={"prompt": "my prompt", "temperature": 0.1})
        mock_post.assert_called_once()
