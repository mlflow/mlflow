from unittest import mock

import pytest

from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException


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


@pytest.fixture
def mock_bad_azure_openai_creds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_TYPE", "azure")
    monkeypatch.setenv("OPENAI_API_VERSION", "2023-05-15")
    monkeypatch.setenv("OPENAI_API_BASE", "https://openai-for.openai.azure.com/")


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
        "openai.OpenAI",
    ) as mock_client:
        mock_client().chat.completions.create().model_dump.return_value = mock_resp
        resp = client.predict(
            endpoint="test",
            inputs={
                "messages": [
                    {"role": "user", "content": "Hello!"},
                ],
            },
        )
        mock_client().chat.completions.create.assert_called_with(
            messages=[{"role": "user", "content": "Hello!"}], model="test"
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
    with mock.patch("openai.AzureOpenAI") as mock_client:
        mock_client().chat.completions.create().model_dump.return_value = mock_resp
        resp = client.predict(
            endpoint="test",
            inputs={
                "messages": [
                    {"role": "user", "content": "Hello!"},
                ],
            },
        )
        mock_client().chat.completions.create.assert_called_with(
            messages=[{"role": "user", "content": "Hello!"}],
            model="test",
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


def test_openai_exception(mock_openai_creds):
    client = get_deploy_client("openai")
    with mock.patch("openai.OpenAI") as mock_client:
        mock_client().chat.completions.create.side_effect = (Exception("foo"),)
        with pytest.raises(Exception, match="foo"):
            client.predict(
                endpoint="test",
                inputs={
                    "messages": [
                        {"role": "user", "content": "Hello!"},
                    ],
                },
            )
        mock_client().chat.completions.create.assert_called_once()
