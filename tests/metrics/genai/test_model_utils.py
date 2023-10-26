from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
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


def test_parse_model_uri():
    prefix, suffix = _parse_model_uri("openai:/gpt-3.5-turbo")

    assert prefix == "openai"
    assert suffix == "gpt-3.5-turbo"

    prefix, suffix = _parse_model_uri("model:/123")

    assert prefix == "model"
    assert suffix == "123"

    prefix, suffix = _parse_model_uri("gateway:/my-route")

    assert prefix == "gateway"
    assert suffix == "my-route"


def test_parse_model_uri_throws_for_malformed():
    with pytest.raises(MlflowException, match="Malformed model uri"):
        _parse_model_uri("gpt-3.5-turbo")


def test_score_model_on_payload_throws_for_invalid():
    with pytest.raises(MlflowException, match="Unknown model uri prefix"):
        score_model_on_payload("myprovider:/gpt-3.5-turbo", {})


def test_score_model_openai_without_key():
    with pytest.raises(MlflowException, match="OPENAI_API_KEY environment variable not set"):
        score_model_on_payload("openai:/gpt-3.5-turbo", {})


def test_score_model_openai(set_envs):
    resp = {
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
        "mlflow.openai.api_request_parallel_processor.process_api_requests", return_value=[resp]
    ) as mock_post:
        score_model_on_payload("openai:/gpt-3.5-turbo", {"prompt": "my prompt", "temperature": 0.1})
        mock_post.assert_called_once_with(
            [
                {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": "my prompt"}],
                }
            ],
            mock.ANY,
            api_token=mock.ANY,
            max_requests_per_minute=3_500,
            max_tokens_per_minute=90_000,
        )


def test_score_model_azure_openai(set_azure_envs):
    resp = {
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
        "mlflow.openai.api_request_parallel_processor.process_api_requests", return_value=[resp]
    ) as mock_post:
        score_model_on_payload("openai:/gpt-3.5-turbo", {"prompt": "my prompt", "temperature": 0.1})
        mock_post.assert_called_once_with(
            [
                {
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": "my prompt"}],
                }
            ],
            mock.ANY,
            api_token=mock.ANY,
            max_requests_per_minute=3_500,
            max_tokens_per_minute=90_000,
        )


def test_score_model_gateway():
    expected_output = {
        "candidates": [
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
            "model": "gpt-3.5-turbo-0301",
            "route_type": "llm/v1/chat",
        },
    }

    with mock.patch("mlflow.gateway.query", return_value=expected_output):
        response = score_model_on_payload("gateway:/my-route", {})
        assert response == expected_output
