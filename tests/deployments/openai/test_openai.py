from unittest import mock

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
        "mlflow.openai.api_request_parallel_processor.process_api_requests",
        return_value=[mock_resp],
    ) as mock_request:
        resp = client.predict(
            endpoint="test",
            inputs={
                "messages": [
                    {"role": "user", "content": "Hello!"},
                ],
            },
        )
        mock_request.assert_called_once_with(
            [{"model": "test", "messages": [{"role": "user", "content": "Hello!"}]}],
            "https://api.openai.com/v1/chat/completions",
            api_token=mock.ANY,
            throw_original_error=mock.ANY,
            max_workers=1,
        )
        assert resp == mock_resp


def test_list_endpoints_openai(mock_openai_creds):
    client = get_deploy_client("openai")

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "object": "list",
        "data": [
            {
                "id": "model-id-0",
                "object": "model",
                "created": 1686935002,
                "owned_by": "organization-owner",
            },
            {
                "id": "model-id-1",
                "object": "model",
                "created": 1686935002,
                "owned_by": "organization-owner",
            },
            {"id": "model-id-2", "object": "model", "created": 1686935002, "owned_by": "openai"},
        ],
    }

    with mock.patch(
        "requests.get",
        return_value=mock_response,
    ) as mock_request:
        resp = client.list_endpoints()
        mock_request.assert_called_once()
        assert resp == mock_response.json.return_value


def test_list_endpoints_azure_openai(mock_azure_openai_creds):
    client = get_deploy_client("openai")

    with pytest.raises(
        NotImplementedError, match="List endpoints is not implemented for Azure OpenAI API"
    ):
        client.list_endpoints()


def test_get_endpoint_openai(mock_openai_creds):
    client = get_deploy_client("openai")

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "gpt-4o-mini",
        "object": "model",
        "created": 1686935002,
        "owned_by": "openai",
    }

    with mock.patch(
        "requests.get",
        return_value=mock_response,
    ) as mock_request:
        resp = client.get_endpoint("gpt-4o-mini")
        mock_request.assert_called_once()
        assert resp == mock_response.json.return_value


def test_get_endpoint_azure_openai(mock_azure_openai_creds):
    client = get_deploy_client("openai")

    with pytest.raises(
        NotImplementedError, match="Get endpoint is not implemented for Azure OpenAI API"
    ):
        client.get_endpoint("gpt-4o-mini")


def test_predict_azure_openai(mock_azure_openai_creds):
    client = get_deploy_client("openai")
    mock_resp = {
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
        "mlflow.openai.api_request_parallel_processor.process_api_requests",
        return_value=[mock_resp],
    ) as mock_request:
        resp = client.predict(
            endpoint="test",
            inputs={
                "messages": [
                    {"role": "user", "content": "Hello!"},
                ],
            },
        )
        mock_request.assert_called_once_with(
            [{"messages": [{"role": "user", "content": "Hello!"}]}],
            "my-base/openai/deployments/my-deployment/chat/completions?api-version=2023-05-15",
            api_token=mock.ANY,
            throw_original_error=mock.ANY,
            max_workers=1,
        )
        assert resp == mock_resp


def test_no_openai_api_key():
    client = get_deploy_client("openai")
    with pytest.raises(MlflowException, match="OPENAI_API_KEY environment variable not set"):
        client.predict(
            endpoint="test",
            inputs={
                "messages": [
                    {"role": "user", "content": "Hello!"},
                ],
            },
        )


def test_score_model_azure_openai_bad_envs(mock_bad_azure_openai_creds):
    client = get_deploy_client("openai")
    with pytest.raises(
        MlflowException, match="Either engine or deployment_id must be set for Azure OpenAI API"
    ):
        client.predict(
            endpoint="test",
            inputs={
                "messages": [
                    {"role": "user", "content": "Hello!"},
                ],
            },
        )


def test_openai_authentication_error(mock_openai_creds):
    client = get_deploy_client("openai")
    mock_response = mock.Mock()
    mock_response.status_code = 401
    mock_response.json.return_value = {
        "error": {
            "message": "Incorrect API key provided: redacted. You can find your API key at https://platform.openai.com/account/api-keys.",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_api_key",
        }
    }

    with mock.patch("requests.post", return_value=mock_response):
        with pytest.raises(MlflowException, match="Authentication Error for OpenAI"):
            client.predict(
                endpoint="test",
                inputs={
                    "messages": [
                        {"role": "user", "content": "Hello!"},
                    ],
                },
            )


def test_openai_mlflow_exception(mock_openai_creds):
    client = get_deploy_client("openai")
    with mock.patch(
        "mlflow.openai.api_request_parallel_processor.process_api_requests",
        side_effect=MlflowException("foo"),
    ) as mock_post:
        with pytest.raises(MlflowException, match="foo"):
            client.predict(
                endpoint="test",
                inputs={
                    "messages": [
                        {"role": "user", "content": "Hello!"},
                    ],
                },
            )
        mock_post.assert_called_once()


def test_openai_exception(mock_openai_creds):
    client = get_deploy_client("openai")
    with mock.patch(
        "mlflow.openai.api_request_parallel_processor.process_api_requests",
        side_effect=Exception("foo"),
    ) as mock_post:
        with pytest.raises(MlflowException, match="Error response from OpenAI:\n foo"):
            client.predict(
                endpoint="test",
                inputs={
                    "messages": [
                        {"role": "user", "content": "Hello!"},
                    ],
                },
            )
        mock_post.assert_called_once()
