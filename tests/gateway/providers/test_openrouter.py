import json
from typing import Any
from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import PassthroughAction
from mlflow.gateway.providers.openrouter import OpenRouterProvider
from mlflow.gateway.schemas import chat
from mlflow.tracing.constant import TokenUsageKey


class MockAsyncResponse:
    def __init__(self, data: dict[str, Any], status: int = 200):
        self.status = status
        self.headers = data.pop("headers", {"Content-Type": "application/json"})
        self._content = data

    def raise_for_status(self) -> None:
        pass

    async def json(self) -> dict[str, Any]:
        return self._content

    async def text(self) -> str:
        return json.dumps(self._content)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        pass


class MockHttpClient(mock.Mock):
    def __init__(self, mock_response=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.post = mock.Mock(return_value=mock_response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _make_provider(model_name: str = "anthropic/claude-3.5-sonnet") -> OpenRouterProvider:
    endpoint_config = EndpointConfig(
        name="openrouter-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "openrouter",
            "name": model_name,
            "config": {"api_key": "sk-or-test-key"},
        },
    )
    return OpenRouterProvider(endpoint_config)


def _chat_response():
    return {
        "id": "chatcmpl-or-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "anthropic/claude-3.5-sonnet",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from OpenRouter!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }


def test_default_api_base():
    provider = _make_provider()
    assert provider._api_base == "https://openrouter.ai/api/v1"


def test_headers():
    provider = _make_provider()
    assert provider.headers == {"Authorization": "Bearer sk-or-test-key"}


def test_name():
    provider = _make_provider()
    assert provider.DISPLAY_NAME == "OpenRouter"


def _system_one_response():
    return {
        "model": "typesafe/jev-1.13",
        "answers": {"evaluation": {"type": "noul", "noul": 0.95}},
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


@pytest.mark.asyncio
async def test_chat():
    provider = _make_provider()
    mock_client = MockHttpClient(MockAsyncResponse(_chat_response()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Hello"}],
        )
        response = await provider.chat(payload)

    result = jsonable_encoder(response)
    assert result["id"] == "chatcmpl-or-123"
    assert result["choices"][0]["message"]["content"] == "Hello from OpenRouter!"


@pytest.mark.asyncio
async def test_system_one_passthrough_uses_only_endpoint_credentials():
    provider = _make_provider("typesafe/jev-1.13")
    payload = {
        "state": {"inputs": "What is MLflow?", "outputs": "An ML platform."},
        "questions": {"evaluation": {"type": "noul", "instructions": "Is it relevant?"}},
    }
    caller_headers = {
        "authorization": "Bearer caller-secret",
        "cookie": "session=caller-session",
        "x-custom-header": "caller-value",
    }

    with mock.patch(
        "mlflow.gateway.providers.openai_compatible.send_request",
        return_value=_system_one_response(),
    ) as send:
        response = await provider.passthrough(
            PassthroughAction.TYPESAFE_SYSTEM_ONE, payload, headers=caller_headers
        )

    assert response == _system_one_response()
    send.assert_awaited_once_with(
        headers={"Authorization": "Bearer sk-or-test-key"},
        base_url="https://openrouter.ai/api/v1",
        path="systemone",
        payload={"model": "typesafe/jev-1.13", **payload},
    )


@pytest.mark.asyncio
async def test_system_one_passthrough_rejects_streaming():
    provider = _make_provider("typesafe/jev-1.13")

    with pytest.raises(AIGatewayException, match="TypeSafe System One") as exc:
        await provider.passthrough(PassthroughAction.TYPESAFE_SYSTEM_ONE, {"stream": True})

    assert exc.value.status_code == 400
    assert exc.value.detail == "TypeSafe System One does not support streaming."


def test_system_one_passthrough_extracts_typesafe_token_usage():
    provider = _make_provider("typesafe/jev-1.13")

    assert provider._extract_passthrough_token_usage(
        PassthroughAction.TYPESAFE_SYSTEM_ONE, _system_one_response()
    ) == {
        TokenUsageKey.INPUT_TOKENS: 100,
        TokenUsageKey.OUTPUT_TOKENS: 20,
        TokenUsageKey.TOTAL_TOKENS: 120,
    }
