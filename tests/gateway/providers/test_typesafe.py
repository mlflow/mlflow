from copy import deepcopy
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import EndpointConfig, TypeSafeConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.provider_registry import provider_registry
from mlflow.gateway.providers.base import PassthroughAction
from mlflow.gateway.providers.typesafe import TypeSafeProvider
from mlflow.gateway.schemas import chat
from mlflow.tracing.constant import TokenUsageKey


@pytest.fixture
def provider():
    return TypeSafeProvider(
        EndpointConfig(
            name="jev-evaluator",
            endpoint_type="llm/v1/chat",
            model={
                "provider": "typesafe",
                "name": "jev-1.13.0",
                "config": {"typesafe_api_key": "typesafe-test-key"},
            },
        )
    )


def test_provider_registered():
    assert provider_registry.get("typesafe") is TypeSafeProvider


def test_typesafe_api_key_config(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-env-key")
    monkeypatch.setenv("MLFLOW_GATEWAY_RESOLVE_API_KEY_FROM_ENV", "true")
    assert (
        TypeSafeConfig(typesafe_api_key="$TYPESAFE_API_KEY").typesafe_api_key == "typesafe-env-key"
    )

    monkeypatch.setenv("MLFLOW_GATEWAY_RESOLVE_API_KEY_FROM_ENV", "false")
    assert (
        TypeSafeConfig(typesafe_api_key="$TYPESAFE_API_KEY").typesafe_api_key == "$TYPESAFE_API_KEY"
    )

    with pytest.raises(MlflowException, match="not a string"):
        TypeSafeConfig(typesafe_api_key=None)


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        (
            {"type": "noul", "instructions": "Is the answer relevant?"},
            {"type": "noul", "noul": 0.95},
        ),
        (
            {
                "type": "choice",
                "instructions": "Which team should respond?",
                "criteria": {"billing": None, "support": "Technical support"},
            },
            {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.8, "support": 0.2},
                "confidence": 0.6,
            },
        ),
        (
            {
                "type": "score",
                "instructions": "How relevant is the answer?",
                "criteria": ["Irrelevant", "Partially relevant", "Relevant"],
            },
            {
                "type": "score",
                "score": 1.7,
                "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
                "confidence": 0.7,
                "legend": {"0": "Irrelevant", "1": "Partially relevant", "2": "Relevant"},
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_passthrough(provider, question, answer):
    payload = {
        "model": "caller-cannot-override-model",
        "state": {"inputs": "What is MLflow?", "outputs": "An ML platform."},
        "questions": {"quality": question},
        "stream": False,
    }
    original_payload = deepcopy(payload)
    response = {
        "model": "jev-1.13.0",
        "answers": {"quality": answer},
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }

    with patch("mlflow.gateway.providers.typesafe.send_request", return_value=response) as send:
        result = await provider.passthrough(
            PassthroughAction.TYPESAFE_SYSTEM_ONE,
            payload,
            headers={"authorization": "Basic client-auth", "X-MLflow-Authorization": "secret"},
        )

    assert result == response
    assert payload == original_payload
    send.assert_awaited_once_with(
        headers={"Authorization": "Bearer typesafe-test-key"},
        base_url="https://api.typesafe.ai/v1",
        path="systemone",
        payload={
            "model": "jev-1.13.0",
            "state": payload["state"],
            "questions": payload["questions"],
        },
    )


@pytest.mark.asyncio
async def test_rejects_streaming_and_chat(provider):
    with patch("mlflow.gateway.providers.typesafe.send_request") as send:
        with pytest.raises(AIGatewayException, match="does not support streaming"):
            await provider.passthrough(PassthroughAction.TYPESAFE_SYSTEM_ONE, {"stream": True})
        with pytest.raises(AIGatewayException, match="Unsupported passthrough endpoint"):
            await provider.passthrough(PassthroughAction.OPENAI_CHAT, {})
        with pytest.raises(AIGatewayException, match="chat route is not implemented"):
            await provider.chat(chat.RequestPayload(messages=[{"role": "user", "content": "Hi"}]))
        send.assert_not_called()


@pytest.mark.parametrize("status_code", [401, 422, 429, 529])
@pytest.mark.asyncio
async def test_preserves_upstream_errors(provider, status_code):
    error = HTTPException(status_code=status_code, detail="Provider error")
    with patch("mlflow.gateway.providers.typesafe.send_request", side_effect=error):
        with pytest.raises(HTTPException, match="Provider error") as exc:
            await provider.passthrough(PassthroughAction.TYPESAFE_SYSTEM_ONE, {})
    assert exc.value is error


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ({}, None),
        ({"usage": {}}, None),
        (
            {"usage": {"input_tokens": 100, "output_tokens": 0}},
            {
                TokenUsageKey.INPUT_TOKENS: 100,
                TokenUsageKey.OUTPUT_TOKENS: 0,
                TokenUsageKey.TOTAL_TOKENS: 100,
            },
        ),
    ],
)
def test_token_usage(provider, response, expected):
    assert (
        provider._extract_passthrough_token_usage(PassthroughAction.TYPESAFE_SYSTEM_ONE, response)
        == expected
    )
