from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import Route, RouteModelInfo
from mlflow.metrics.genai.model_utils import (
    _parse_model_uri,
    score_model_on_payload,
)


@pytest.fixture
def set_envs(monkeypatch):
    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
        }
    )


@pytest.fixture
def set_deployment_envs(monkeypatch):
    monkeypatch.setenvs(
        {
            "MLFLOW_DEPLOYMENTS_TARGET": "databricks",
        }
    )


@pytest.fixture
def set_azure_envs(monkeypatch):
    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_API_TYPE": "azure",
            "OPENAI_API_VERSION": "2023-05-15",
            "OPENAI_API_BASE": "https://openai-for.openai.azure.com/",
            "OPENAI_DEPLOYMENT_NAME": "test-openai",
        }
    )


@pytest.fixture
def set_bad_azure_envs(monkeypatch):
    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_API_TYPE": "azure",
            "OPENAI_API_VERSION": "2023-05-15",
            "OPENAI_API_BASE": "https://openai-for.openai.azure.com/",
        }
    )


def test_parse_model_uri():
    prefix, suffix = _parse_model_uri("openai:/gpt-4o-mini")

    assert prefix == "openai"
    assert suffix == "gpt-4o-mini"

    prefix, suffix = _parse_model_uri("model:/123")

    assert prefix == "model"
    assert suffix == "123"

    prefix, suffix = _parse_model_uri("gateway:/my-route")

    assert prefix == "gateway"
    assert suffix == "my-route"

    prefix, suffix = _parse_model_uri("endpoints:/my-endpoint")

    assert prefix == "endpoints"
    assert suffix == "my-endpoint"


def test_parse_model_uri_throws_for_malformed():
    with pytest.raises(MlflowException, match="Malformed model uri"):
        _parse_model_uri("gpt-4o-mini")


def test_score_model_on_payload_throws_for_invalid():
    with pytest.raises(MlflowException, match="Unknown model uri prefix"):
        score_model_on_payload("myprovider:/gpt-4o-mini", {})


def test_score_model_openai_without_key():
    with pytest.raises(MlflowException, match="OPENAI_API_KEY environment variable not set"):
        score_model_on_payload("openai:/gpt-4o-mini", {})


def test_score_model_openai(set_envs):
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
        "mlflow.openai.api_request_parallel_processor.process_api_requests", return_value=[resp]
    ) as mock_post:
        resp = score_model_on_payload("openai:/gpt-4o-mini", "my prompt", {"temperature": 0.1})
        mock_post.assert_called_once_with(
            [
                {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": "my prompt"}],
                }
            ],
            mock.ANY,
            api_token=mock.ANY,
            throw_original_error=True,
            max_workers=1,
        )


def test_openai_authentication_error(set_envs):
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

    with mock.patch("requests.post", return_value=mock_response) as mock_post:
        with pytest.raises(
            MlflowException, match="Authentication Error for OpenAI. Error response"
        ):
            score_model_on_payload("openai:/gpt-4o-mini", "my prompt", {"temperature": 0.1})
        mock_post.assert_called_once()


def test_openai_other_error(set_envs):
    with mock.patch(
        "mlflow.openai.api_request_parallel_processor.process_api_requests",
        side_effect=Exception("foo"),
    ) as mock_post:
        with pytest.raises(MlflowException, match="Error response from OpenAI"):
            score_model_on_payload("openai:/gpt-4o-mini", "my prompt", {"temperature": 0.1})
        mock_post.assert_called_once()


def test_score_model_azure_openai(set_azure_envs):
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
        "mlflow.openai.api_request_parallel_processor.process_api_requests", return_value=[resp]
    ) as mock_post:
        score_model_on_payload("openai:/gpt-4o-mini", "my prompt", {"temperature": 0.1})
        mock_post.assert_called_once_with(
            [
                {
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": "my prompt"}],
                }
            ],
            mock.ANY,
            api_token=mock.ANY,
            throw_original_error=True,
            max_workers=1,
        )


def test_score_model_azure_openai_bad_envs(set_bad_azure_envs):
    with pytest.raises(
        MlflowException, match="Either engine or deployment_id must be set for Azure OpenAI API"
    ):
        score_model_on_payload("openai:/gpt-4o-mini", "my prompt", {"temperature": 0.1})


def test_score_model_gateway_completions():
    expected_output = {
        "choices": [
            {"text": "man, one giant leap for mankind.", "metadata": {"finish_reason": "stop"}}
        ],
        "metadata": {
            "model": "gpt-4-0613",
            "input_tokens": 13,
            "total_tokens": 21,
            "output_tokens": 8,
            "endpoint_type": "llm/v1/completions",
        },
    }

    with mock.patch(
        "mlflow.gateway.get_route",
        return_value=Route(
            name="my-route",
            route_type="llm/v1/completions",
            model=RouteModelInfo(provider="openai"),
            route_url="my-route",
        ).to_endpoint(),
    ):
        with mock.patch("mlflow.gateway.query", return_value=expected_output):
            response = score_model_on_payload("gateway:/my-route", {})
            assert response == expected_output["choices"][0]["text"]


def test_score_model_gateway_chat():
    expected_output = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The core of the sun is estimated to have a temperature of about "
                    "15 million degrees Celsius (27 million degrees Fahrenheit).",
                },
                "metadata": {"finish_reason": "stop"},
            }
        ],
        "metadata": {
            "input_tokens": 17,
            "output_tokens": 24,
            "total_tokens": 41,
            "model": "gpt-4o-mini",
            "endpoint_type": "llm/v1/chat",
        },
    }

    with mock.patch(
        "mlflow.gateway.get_route",
        return_value=Route(
            name="my-route",
            route_type="llm/v1/chat",
            model=RouteModelInfo(provider="openai"),
            route_url="my-route",
        ).to_endpoint(),
    ):
        with mock.patch("mlflow.gateway.query", return_value=expected_output):
            response = score_model_on_payload("gateway:/my-route", {})
            assert response == expected_output["choices"][0]["message"]["content"]


@pytest.mark.parametrize(
    "endpoint_type_key",
    [
        "task",
        "endpoint_type",
    ],
)
def test_score_model_endpoints_chat(set_deployment_envs, endpoint_type_key):
    openai_response_format = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\nHello there, how may I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }
    expected_output = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\nHello there, how may I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
    }

    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_client = mock.MagicMock()
        mock_get_deploy_client.return_value = mock_client
        # mock out mock_client.get_endpoint() to return chat
        mock_client.get_endpoint.return_value = {
            endpoint_type_key: "llm/v1/chat",
        }
        # mock out mock_client.predict() to return expected_output
        mock_client.predict.return_value = openai_response_format
        response = score_model_on_payload(
            "endpoints:/my-endpoint", {"prompt": "my prompt", "temperature": 0.1}
        )
        assert response == expected_output["choices"][0]["message"]["content"]


@pytest.mark.parametrize(
    "endpoint_type_key",
    [
        "task",
        "endpoint_type",
    ],
)
def test_score_model_endpoints_completions(set_deployment_envs, endpoint_type_key):
    openai_response_format = {
        "id": "cmpl-8PgdiXapPWBN3pyUuHcELH766QgqK",
        "object": "text_completion",
        "created": 1701132798,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "text": "\n\nHi there! How can I assist you today?",
                "index": 0,
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 106, "total_tokens": 108},
    }

    expected_output = {
        "choices": [
            {
                "text": "\n\nHi there! How can I assist you today?",
                "index": 0,
                "finish_reason": "stop",
            },
        ],
    }

    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_client = mock.MagicMock()
        mock_get_deploy_client.return_value = mock_client
        # mock out mock_client.get_endpoint() to return completions
        mock_client.get_endpoint.return_value = {
            endpoint_type_key: "llm/v1/completions",
        }
        # mock out mock_client.predict() to return expected_output
        mock_client.predict.return_value = openai_response_format
        response = score_model_on_payload(
            "endpoints:/my-endpoint", {"prompt": "my prompt", "temperature": 0.1}
        )
        assert response == expected_output["choices"][0]["text"]
