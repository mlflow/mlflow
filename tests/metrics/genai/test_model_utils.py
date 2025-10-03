import json
import sys
from unittest import mock

import pytest

from mlflow.deployments.server.config import Endpoint
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import RouteModelInfo
from mlflow.metrics.genai.model_utils import (
    _parse_model_uri,
    call_deployments_api,
    get_endpoint_type,
    score_model_on_payload,
)


@pytest.fixture
def set_envs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_TYPE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test")


@pytest.fixture
def set_deployment_envs(monkeypatch):
    monkeypatch.setenv("MLFLOW_DEPLOYMENTS_TARGET", "databricks")


@pytest.fixture
def set_azure_envs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_TYPE", "azure")
    monkeypatch.setenv("OPENAI_API_VERSION", "2023-05-15")
    monkeypatch.setenv("OPENAI_API_BASE", "https://openai-for.openai.azure.com/")
    monkeypatch.setenv("OPENAI_DEPLOYMENT_NAME", "test-openai")


@pytest.fixture(autouse=True)
def force_reload_openai():
    # Force reloading OpenAI module in the next test case. This is because they store
    # configuration like api_key, api_version, at the global variable, which is not
    # updated once set. Even if we reset the environment variable, it will retain the
    # old value and cause unexpected side effects.
    # https://github.com/openai/openai-python/blob/ea049cd0c42e115b90f1b9c7db80b2659a0bb92a/src/openai/__init__.py#L134
    sys.modules.pop("openai", None)


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
        score_model_on_payload("myprovider:/gpt-4o-mini", "")


def test_score_model_openai_without_key():
    with pytest.raises(
        MlflowException, match="OpenAI API key must be set in the ``OPENAI_API_KEY``"
    ):
        score_model_on_payload("openai:/gpt-4o-mini", "")


_OAI_RESPONSE = {
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


def test_score_model_openai(set_envs):
    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=_OAI_RESPONSE
    ) as mock_post:
        resp = score_model_on_payload("openai:/gpt-4o-mini", "my prompt", {"temperature": 0.1})

        assert resp == "\n\nThis is a test!"
        mock_post.assert_called_once_with(
            endpoint="https://api.openai.com/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            payload={
                "messages": [{"role": "user", "content": "my prompt"}],
                "model": "gpt-4o-mini",
                "temperature": 0.1,
            },
        )


def test_score_model_openai_with_custom_header_and_proxy_url(set_envs):
    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=_OAI_RESPONSE
    ) as mock_post:
        resp = score_model_on_payload(
            model_uri="openai:/gpt-4o-mini",
            payload="my prompt",
            eval_parameters={"temperature": 0.1},
            extra_headers={"foo": "bar"},
            proxy_url="https://my-proxy.com/chat",
        )

        assert resp == "\n\nThis is a test!"
        mock_post.assert_called_once_with(
            endpoint="https://my-proxy.com/chat",
            headers={"Authorization": "Bearer test", "foo": "bar"},
            payload={
                "messages": [{"role": "user", "content": "my prompt"}],
                "model": "gpt-4o-mini",
                "temperature": 0.1,
            },
        )


def test_openai_other_error(set_envs):
    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request",
        side_effect=Exception("foo"),
    ):
        with pytest.raises(Exception, match="foo"):
            score_model_on_payload("openai:/gpt-4o-mini", "my prompt", {"temperature": 0.1})


def test_score_model_azure_openai(set_azure_envs):
    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=_OAI_RESPONSE
    ) as mock_post:
        resp = score_model_on_payload("openai:/gpt-4o-mini", "my prompt", {"temperature": 0.1})

        assert resp == "\n\nThis is a test!"
        mock_post.assert_called_once_with(
            endpoint="https://openai-for.openai.azure.com/openai/deployments/test-openai/chat/completions?api-version=2023-05-15",
            headers={"api-key": "test"},
            payload={
                "messages": [{"role": "user", "content": "my prompt"}],
                "temperature": 0.1,
            },
        )


def test_score_model_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    resp = {
        "content": [
            {
                "text": "This is a test!",
                "type": "text",
            }
        ],
        "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
        "model": "claude-3-5-sonnet-20241022",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 2095, "output_tokens": 503},
    }

    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=resp
    ) as mock_request:
        response = score_model_on_payload(
            model_uri="anthropic:/claude-3-5-sonnet-20241022",
            payload="input prompt",
            eval_parameters={"max_tokens": 1000, "top_p": 1},
            extra_headers={"anthropic-version": "2024-10-22"},
        )

    assert response == "This is a test!"
    mock_request.assert_called_once_with(
        endpoint="https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2024-10-22",
        },
        payload={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "input prompt"}],
            "max_tokens": 1000,
            "top_p": 1,
        },
    )


def test_score_model_bedrock(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test-session-token")

    resp = {
        "content": [
            {
                "text": "This is a test!",
                "type": "text",
            }
        ],
        "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
        "model": "claude-3-5-sonnet-20241022",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 2095, "output_tokens": 503},
    }

    mock_bedrock = mock.MagicMock()
    with mock.patch("boto3.Session.client", return_value=mock_bedrock) as mock_session:
        mock_bedrock.invoke_model.return_value = {
            "body": mock.MagicMock(read=mock.MagicMock(return_value=json.dumps(resp).encode()))
        }

        response = score_model_on_payload(
            model_uri="bedrock:/anthropic.claude-3-5-sonnet-20241022-v2:0",
            payload="input prompt",
            eval_parameters={
                "temperature": 0,
                "max_tokens": 1000,
                "anthropic_version": "2023-06-01",
            },
        )

    assert response == "This is a test!"
    mock_session.assert_called_once_with(
        service_name="bedrock-runtime",
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        aws_session_token="test-session-token",
    )
    mock_bedrock.invoke_model.assert_called_once_with(
        # Anthropic models in Bedrock does not accept "model" and "stream" key,
        # and requires "anthropic_version" put within the body not headers.
        body=json.dumps(
            {
                "temperature": 0,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": "input prompt"}],
                "anthropic_version": "2023-06-01",
            }
        ).encode(),
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        accept="application/json",
        contentType="application/json",
    )


def test_score_model_mistral(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    # Mistral AI API is compatible with OpenAI format
    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=_OAI_RESPONSE
    ) as mock_request:
        response = score_model_on_payload(
            model_uri="mistral:/mistral-small-latest",
            payload="input prompt",
            eval_parameters={"temperature": 0.1},
        )

    assert response == "\n\nThis is a test!"
    mock_request.assert_called_once_with(
        endpoint="https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        payload={
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "input prompt"}],
            "temperature": 0.1,
        },
    )


def test_score_model_togetherai(monkeypatch):
    monkeypatch.setenv("TOGETHERAI_API_KEY", "test-key")

    resp = {
        "id": "8448080b880415ea-SJC",
        "choices": [{"message": {"role": "assistant", "content": "This is a test!"}}],
        "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        "created": 1705090115,
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "object": "chat.completion",
    }

    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request", return_value=resp
    ) as mock_request:
        response = score_model_on_payload(
            model_uri="togetherai:/mistralai/Mixtral-8x7B-Instruct-v0.1",
            payload="input prompt",
            eval_parameters={"temperature": 0, "max_tokens": 1000},
        )

    assert response == "This is a test!"
    mock_request.assert_called_once_with(
        endpoint="https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        payload={
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": "input prompt"}],
            "temperature": 0,
            "max_tokens": 1000,
        },
    )


def test_score_model_gateway_completions():
    from mlflow.deployments.mlflow import MlflowDeploymentClient

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

    with (
        mock.patch(
            "mlflow.deployments.MlflowDeploymentClient.get_endpoint",
            return_value=Endpoint(
                name="my-route",
                endpoint_type="llm/v1/completions",
                model=RouteModelInfo(provider="openai"),
                endpoint_url="my-route",
                limit=None,
            ),
        ),
        mock.patch(
            "mlflow.deployments.MlflowDeploymentClient.predict", return_value=expected_output
        ),
        mock.patch(
            "mlflow.deployments.get_deploy_client", return_value=MlflowDeploymentClient("url")
        ),
    ):
        response = score_model_on_payload("gateway:/my-route", "")
        assert response == expected_output["choices"][0]["text"]


def test_score_model_gateway_chat():
    from mlflow.deployments.mlflow import MlflowDeploymentClient

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

    with (
        mock.patch(
            "mlflow.deployments.MlflowDeploymentClient.get_endpoint",
            return_value=Endpoint(
                name="my-route",
                endpoint_type="llm/v1/chat",
                model=RouteModelInfo(provider="openai"),
                endpoint_url="my-route",
                limit=None,
            ),
        ),
        mock.patch(
            "mlflow.deployments.MlflowDeploymentClient.predict", return_value=expected_output
        ),
        mock.patch(
            "mlflow.deployments.get_deploy_client", return_value=MlflowDeploymentClient("url")
        ),
    ):
        response = score_model_on_payload("gateway:/my-route", "")
        assert response == expected_output["choices"][0]["message"]["content"]


@pytest.mark.parametrize(
    ("get_endpoint_response", "expected"),
    [
        ({"task": "llm/v1/completions"}, "llm/v1/completions"),
        ({"endpoint_type": "llm/v1/chat"}, "llm/v1/chat"),
        ({}, None),
    ],
)
def test_get_endpoint_type(get_endpoint_response, expected):
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_client = mock_get_deploy_client.return_value
        mock_client.get_endpoint.return_value = get_endpoint_response
        assert get_endpoint_type("endpoints:/my-endpoint") == expected


_TEST_CHAT_RESPONSE = {
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


def test_score_model_endpoints_chat(set_deployment_envs):
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_get_deploy_client().predict.return_value = _TEST_CHAT_RESPONSE
        response = score_model_on_payload(
            model_uri="endpoints:/my-endpoint",
            payload="my prompt",
            eval_parameters={"temperature": 0.1},
            endpoint_type="llm/v1/chat",
        )
        assert response == "\n\nHello there, how may I assist you today?"


_TEST_COMPLETION_RESPONSE = {
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


def test_score_model_endpoints_completions(set_deployment_envs):
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_get_deploy_client().predict.return_value = _TEST_COMPLETION_RESPONSE
        response = score_model_on_payload(
            model_uri="endpoints:/my-endpoint",
            payload="my prompt",
            eval_parameters={"temperature": 0.1},
            endpoint_type="llm/v1/completions",
        )
        assert response == "\n\nHi there! How can I assist you today?"


@pytest.mark.parametrize(
    "input_data",
    [
        "my prompt",
        {"messages": [{"role": "user", "content": "my prompt"}]},
    ],
)
def test_call_deployments_api_chat(input_data, set_deployment_envs):
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_get_deploy_client().predict.return_value = _TEST_CHAT_RESPONSE
        response = call_deployments_api(
            deployment_uri="my-endpoint",
            input_data=input_data,
            eval_parameters={},
            endpoint_type="llm/v1/chat",
        )
        assert response == "\n\nHello there, how may I assist you today?"


@pytest.mark.parametrize(
    "input_data",
    [
        "my prompt",
        {"prompt": "my prompt"},
    ],
)
def test_call_deployments_api_completion(input_data, set_deployment_envs):
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_get_deploy_client().predict.return_value = _TEST_COMPLETION_RESPONSE
        response = call_deployments_api(
            deployment_uri="my-endpoint",
            input_data=input_data,
            eval_parameters={"temperature": 0.1},
            endpoint_type="llm/v1/completions",
        )
        assert response == "\n\nHi there! How can I assist you today?"


def test_call_deployments_api_no_endpoint_type(set_deployment_envs):
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_get_deploy_client().predict.return_value = {"result": "ok"}
        response = call_deployments_api(
            deployment_uri="my-endpoint",
            input_data={"foo": {"bar": "baz"}},
            eval_parameters={},
            endpoint_type=None,
        )
        assert response == {"result": "ok"}


def test_call_deployments_api_str_input_requires_endpoint_type(set_deployment_envs):
    with pytest.raises(MlflowException, match="If string input is provided,"):
        call_deployments_api("my-endpoint", "my prompt", endpoint_type=None)
