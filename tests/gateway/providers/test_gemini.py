from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.gemini import GeminiProvider
from mlflow.gateway.schemas import chat, completions, embeddings

from tests.gateway.tools import MockAsyncResponse


def completions_config():
    return {
        "name": "completions",
        "endpoint_type": "llm/v1/completions",
        "model": {
            "provider": "gemini",
            "name": "gemini-2.0-flash",
            "config": {
                "gemini_api_key": "key",
            },
        },
    }


def chat_config():
    return {
        "name": "chat",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "provider": "gemini",
            "name": "gemini-2.0-flash",
            "config": {
                "gemini_api_key": "key",
            },
        },
    }


def embedding_config():
    return {
        "name": "embeddings",
        "endpoint_type": "llm/v1/embeddings",
        "model": {
            "provider": "gemini",
            "name": "text-embedding-004",
            "config": {
                "gemini_api_key": "key",
            },
        },
    }


def fake_single_embedding_response():
    return {"embeddings": [{"values": [0.1, 0.2, 0.3]}]}


def fake_batch_embedding_response():
    return {"embeddings": [{"values": [0.1, 0.2, 0.3]}, {"values": [0.4, 0.5, 0.6]}]}


def fake_completion_response():
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Why did the chicken cross the road? To get to the other side."}
                    ]
                },
                "finishReason": "stop",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 10,
            "totalTokenCount": 15,
        },
    }


def fake_chat_response():
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Why did the chicken cross the road? To get to the other side."}
                    ]
                },
                "finishReason": "stop",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 6,
            "candidatesTokenCount": 12,
            "totalTokenCount": 18,
        },
    }


@pytest.mark.asyncio
async def test_gemini_single_embedding():
    config = embedding_config()
    provider = GeminiProvider(RouteConfig(**config))
    payload = {"input": "This is a test embedding."}

    expected_payload = {"content": {"parts": [{"text": "This is a test embedding."}]}}
    expected_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
    )

    with mock.patch(
        "aiohttp.ClientSession.post",
        return_value=MockAsyncResponse(fake_single_embedding_response()),
    ) as mock_post:
        response = await provider.embeddings(embeddings.RequestPayload(**payload))

    expected_data = [embeddings.EmbeddingObject(embedding=[0.1, 0.2, 0.3], index=0)]
    expected_response = {
        "object": "list",
        "data": jsonable_encoder(expected_data),
        "model": "text-embedding-004",
        "usage": {"prompt_tokens": None, "total_tokens": None},
    }
    assert jsonable_encoder(response) == expected_response

    mock_post.assert_called_once_with(
        expected_url,
        json=expected_payload,
        timeout=mock.ANY,
    )


@pytest.mark.asyncio
async def test_gemini_batch_embedding():
    config = embedding_config()
    provider = GeminiProvider(RouteConfig(**config))
    payload = {"input": ["Test embedding 1.", "Test embedding 2."]}

    expected_payload = {
        "requests": [
            {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": "Test embedding 1."}]},
            },
            {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": "Test embedding 2."}]},
            },
        ]
    }
    expected_url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents"

    with mock.patch(
        "aiohttp.ClientSession.post",
        return_value=MockAsyncResponse(fake_batch_embedding_response()),
    ) as mock_post:
        response = await provider.embeddings(embeddings.RequestPayload(**payload))

    expected_data = [
        embeddings.EmbeddingObject(embedding=[0.1, 0.2, 0.3], index=0),
        embeddings.EmbeddingObject(embedding=[0.4, 0.5, 0.6], index=1),
    ]
    expected_response = {
        "object": "list",
        "data": jsonable_encoder(expected_data),
        "model": "text-embedding-004",
        "usage": {"prompt_tokens": None, "total_tokens": None},
    }

    assert jsonable_encoder(response) == expected_response

    mock_post.assert_called_once_with(
        expected_url,
        json=expected_payload,
        timeout=mock.ANY,
    )


@pytest.mark.asyncio
async def test_gemini_completions():
    config = completions_config()
    provider = GeminiProvider(RouteConfig(**config))
    payload = {
        "prompt": "Tell me a joke",
        "temperature": 0.1,
        "top_p": 1,
        "stop": ["\n"],
        "n": 1,
        "max_tokens": 50,
        "top_k": 40,
    }

    expected_payload = {
        "contents": [{"role": "user", "parts": [{"text": "Tell me a joke"}]}],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 1,
            "stopSequences": ["\n"],
            "candidateCount": 1,
            "maxOutputTokens": 50,
            "topK": 40,
        },
    }
    expected_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    )

    with mock.patch("time.time", return_value=1234567890):
        with mock.patch(
            "aiohttp.ClientSession.post",
            return_value=MockAsyncResponse(fake_completion_response()),
        ) as mock_post:
            response = await provider.completions(completions.RequestPayload(**payload))

    expected_choices = [
        completions.Choice(
            index=0,
            text="Why did the chicken cross the road? To get to the other side.",
            finish_reason="stop",
        )
    ]
    expected_response = {
        "id": None,
        "created": 1234567890,
        "object": "text_completion",
        "model": "gemini-2.0-flash",
        "choices": jsonable_encoder(expected_choices),
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
        },
    }

    assert jsonable_encoder(response) == expected_response
    mock_post.assert_called_once_with(
        expected_url,
        json=expected_payload,
        timeout=mock.ANY,
    )


@pytest.mark.asyncio
async def test_gemini_completions_streaming_not_supported():
    config = completions_config()
    provider = GeminiProvider(RouteConfig(**config))
    payload = {"prompt": "Tell me a joke", "stream": True}

    with pytest.raises(
        AIGatewayException,
        match="Streaming is not yet supported for completions with Gemini AI Gateway",
    ):
        await provider.completions(completions.RequestPayload(**payload))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("override", "exclude_keys", "expected_msg"),
    [
        ({"stopSequences": ["\n"]}, ["stop"], "Invalid parameter stopSequences. Use stop instead."),
        ({"candidateCount": 1}, [], "Invalid parameter candidateCount. Use n instead."),
        ({"maxOutputTokens": 50}, [], "Invalid parameter maxOutputTokens. Use max_tokens instead."),
        ({"topK": 40}, [], "Invalid parameter topK. Use top_k instead."),
        ({"top_p": 1.1}, [], "top_p should be less than or equal to 1"),
    ],
)
async def test_invalid_parameters_completions(override, exclude_keys, expected_msg):
    config = completions_config()
    provider = GeminiProvider(RouteConfig(**config))

    base_payload = {
        "prompt": "Tell me a joke",
        "temperature": 0.1,
        "top_p": 0.9,
        "stop": ["\n"],
        "n": 1,
        "max_tokens": 50,
        "top_k": 40,
    }

    payload = {k: v for k, v in base_payload.items() if k not in exclude_keys}
    payload.update(override)

    with pytest.raises(AIGatewayException, match=expected_msg):
        await provider.completions(completions.RequestPayload(**payload))


@pytest.mark.asyncio
async def test_gemini_chat():
    config = chat_config()
    provider = GeminiProvider(RouteConfig(**config))
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Tell me a joke"},
        ],
        "temperature": 0.1,
        "top_p": 1,
        "stop": ["\n"],
        "n": 1,
        "max_tokens": 100,
        "top_k": 40,
    }

    expected_payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "Tell me a joke"}]},
        ],
        "system_instruction": {"parts": [{"text": "You are a helpful assistant"}]},
        "generationConfig": {
            "temperature": 0.1,
            "topP": 1,
            "stopSequences": ["\n"],
            "candidateCount": 1,
            "maxOutputTokens": 100,
            "topK": 40,
        },
    }
    expected_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    )

    with mock.patch("time.time", return_value=1234567890):
        with mock.patch(
            "aiohttp.ClientSession.post",
            return_value=MockAsyncResponse(fake_chat_response()),
        ) as mock_post:
            response = await provider.chat(chat.RequestPayload(**payload))

    expected_choices = [
        chat.Choice(
            index=0,
            message=chat.ResponseMessage(
                role="assistant",
                content="Why did the chicken cross the road? To get to the other side.",
            ),
            finish_reason="stop",
        )
    ]
    expected_response = {
        "id": "gemini-chat-1234567890",
        "created": 1234567890,
        "object": "chat.completion",
        "model": "gemini-2.0-flash",
        "choices": jsonable_encoder(expected_choices),
        "usage": {
            "prompt_tokens": 6,
            "completion_tokens": 12,
            "total_tokens": 18,
        },
    }

    assert jsonable_encoder(response) == expected_response
    mock_post.assert_called_once_with(
        expected_url,
        json=expected_payload,
        timeout=mock.ANY,
    )


@pytest.mark.asyncio
async def test_gemini_chat_streaming_not_supported():
    config = chat_config()
    provider = GeminiProvider(RouteConfig(**config))
    payload = {
        "messages": [{"role": "user", "content": "Tell me a joke"}],
        "stream": True,
    }

    with pytest.raises(
        AIGatewayException,
        match="Streaming is not yet supported for chat completions with Gemini AI Gateway",
    ):
        await provider.chat(chat.RequestPayload(**payload))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("override", "exclude_keys", "expected_msg"),
    [
        ({"stopSequences": ["\n"]}, ["stop"], "Invalid parameter stopSequences. Use stop instead."),
        ({"candidateCount": 1}, [], "Invalid parameter candidateCount. Use n instead."),
        (
            {"maxOutputTokens": 100},
            [],
            "Invalid parameter maxOutputTokens. Use max_tokens instead.",
        ),
        ({"topK": 40}, [], "Invalid parameter topK. Use top_k instead."),
        ({"top_p": 1.1}, [], "top_p should be less than or equal to 1"),
    ],
)
async def test_invalid_parameters_chat(override, exclude_keys, expected_msg):
    config = chat_config()
    provider = GeminiProvider(RouteConfig(**config))

    base_payload = {
        "messages": [{"role": "user", "content": "Tell me a joke"}],
        "temperature": 0.1,
        "top_p": 0.9,
        "stop": ["\n"],
        "n": 1,
        "max_tokens": 100,
        "top_k": 40,
    }

    payload = {k: v for k, v in base_payload.items() if k not in exclude_keys}
    payload.update(override)

    with pytest.raises(AIGatewayException, match=expected_msg):
        await provider.chat(chat.RequestPayload(**payload))
