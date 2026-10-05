import json
from typing import Any
from unittest import mock

import pytest

from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import PassthroughAction
from mlflow.gateway.providers.litellm_proxy import LiteLLMProxyProvider


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


def _make_provider(
    model_name: str = "jev-1.13.0", api_base: str | None = "http://my-proxy:4000"
) -> LiteLLMProxyProvider:
    config: dict[str, Any] = {"api_key": "sk-litellm-proxy-key"}
    if api_base is not None:
        config["api_base"] = api_base
    endpoint_config = EndpointConfig(
        name="litellm-proxy-endpoint",
        endpoint_type="llm/v1/chat",
        model={"provider": "litellm_proxy", "name": model_name, "config": config},
    )
    return LiteLLMProxyProvider(endpoint_config)


def _system_one_response():
    return {
        "model": "jev-1.13.0",
        "answers": {"evaluation": {"type": "noul", "noul": 0.95}},
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def test_name():
    assert _make_provider().DISPLAY_NAME == "LiteLLM Proxy"


def test_requires_api_base():
    with pytest.raises(AIGatewayException, match="require 'api_base'") as exc:
        _make_provider(api_base=None)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_system_one_passthrough_uses_endpoint_credentials_only():
    provider = _make_provider()
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
        headers={"Authorization": "Bearer sk-litellm-proxy-key"},
        base_url="http://my-proxy:4000",
        path="typesafe/v1/systemone",
        payload={"model": "jev-1.13.0", **payload},
    )


@pytest.mark.asyncio
async def test_system_one_passthrough_rejects_streaming():
    provider = _make_provider()

    with pytest.raises(AIGatewayException, match="TypeSafe System One") as exc:
        await provider.passthrough(PassthroughAction.TYPESAFE_SYSTEM_ONE, {"stream": True})

    assert exc.value.status_code == 400
    assert exc.value.detail == "TypeSafe System One does not support streaming."
