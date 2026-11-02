import json
from unittest import mock

import pytest

from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.anthropic import AnthropicProvider
from mlflow.gateway.providers.bedrock import AmazonBedrockProvider
from mlflow.gateway.providers.vertex_ai import VertexAIProvider
from mlflow.gateway.schemas import chat

INVALID_INPUT_USAGE = {
    "input_tokens": "120",
    "cache_read_input_tokens": 100.5,
    "cache_creation_input_tokens": [],
}


@pytest.fixture(params=["anthropic", "amazon-bedrock", "vertex_ai"])
def provider(request):
    match request.param:
        case "anthropic":
            provider_class = AnthropicProvider
            config = {"anthropic_api_key": "test-key"}
        case "amazon-bedrock":
            provider_class = AmazonBedrockProvider
            config = {"aws_config": {"aws_region": "us-east-1"}}
        case "vertex_ai":
            provider_class = VertexAIProvider
            config = {"vertex_project": "test-project", "vertex_location": "us-east5"}
    provider = provider_class(
        EndpointConfig(
            name="test-endpoint",
            endpoint_type="llm/v1/chat",
            model={"provider": request.param, "name": "claude-sonnet-4-5", "config": config},
        )
    )
    return provider._delegate if request.param == "vertex_ai" else provider


@pytest.mark.asyncio
@pytest.mark.parametrize("combine_chunks", [False, True])
@pytest.mark.parametrize(
    ("delta_usage", "expected_input", "expected_read", "expected_creation"),
    [
        ([{}], 140, 25, 15),
        ([{"input_tokens": 120}], 160, 25, 15),
        ([{"cache_read_input_tokens": 40}], 155, 40, 15),
        ([{"cache_creation_input_tokens": 30}], 155, 25, 30),
        ([{"cache_read_input_tokens": 40}, {"input_tokens": 120}], 175, 40, 15),
        ([INVALID_INPUT_USAGE], 140, 25, 15),
        (
            [{"input_tokens": None, "cache_read_input_tokens": None}],
            140,
            25,
            15,
        ),
        (
            [{"input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}],
            0,
            0,
            0,
        ),
        (
            [
                {
                    "input_tokens": 120,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 20,
                }
            ],
            240,
            100,
            20,
        ),
    ],
)
async def test_stream_passthrough_accumulates_raw_usage(
    provider, combine_chunks, delta_usage, expected_input, expected_read, expected_creation
):
    events = [
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 25,
                    "cache_creation_input_tokens": 15,
                }
            },
        },
        *[
            {"type": "message_delta", "usage": {"output_tokens": 50, **usage}}
            for usage in delta_usage
        ],
    ]
    chunks = [f"data: {json.dumps(event)}\n\n".encode() for event in events]
    if combine_chunks:
        chunks = [b"".join(chunks)]

    async def stream():
        for chunk in chunks:
            yield chunk

    with mock.patch.object(provider, "_set_span_token_usage") as set_usage:
        assert [
            chunk async for chunk in provider._stream_passthrough_with_usage(stream())
        ] == chunks

    set_usage.assert_called_once_with({
        "input_tokens": expected_input,
        "output_tokens": 50,
        "total_tokens": expected_input + 50,
        "cache_read_input_tokens": expected_read,
        "cache_creation_input_tokens": expected_creation,
    })


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic"], indirect=True)
@pytest.mark.parametrize(
    ("start_usage", "delta_usage", "expected_prompt"),
    [
        (
            {"input_tokens": 100, "cache_read_input_tokens": 25, "cache_creation_input_tokens": 15},
            INVALID_INPUT_USAGE,
            140,
        ),
        (
            INVALID_INPUT_USAGE,
            {"input_tokens": 120},
            120,
        ),
    ],
)
async def test_chat_stream_ignores_invalid_input_usage(
    provider, start_usage, delta_usage, expected_prompt
):
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "model": provider.config.model.name,
                "usage": start_usage,
            },
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50, **delta_usage},
        },
    ]

    async def stream():
        for event in events:
            yield f"data: {json.dumps(event)}\n\n".encode()

    with mock.patch(
        "mlflow.gateway.providers.anthropic.send_stream_request", return_value=stream()
    ) as send_request:
        chunks = [
            chunk
            async for chunk in provider._chat_stream(
                chat.RequestPayload(messages=[{"role": "user", "content": "Hello"}], stream=True)
            )
        ]

    send_request.assert_called_once()
    assert chunks[-1].usage.prompt_tokens == expected_prompt
    assert chunks[-1].usage.completion_tokens == 50
    assert chunks[-1].usage.total_tokens == expected_prompt + 50
