from unittest import mock

import pytest

from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.litellm import LiteLLMAdapter, LiteLLMProvider
from mlflow.gateway.schemas import chat, embeddings

TEST_MESSAGE = "This is a test"


def chat_config():
    return {
        "name": "chat",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "provider": "litellm",
            "name": "claude-3-5-sonnet-20241022",
            "config": {
                "litellm_api_key": "test-key",
            },
        },
    }


def chat_config_with_api_base():
    return {
        "name": "chat",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "provider": "litellm",
            "name": "custom-model",
            "config": {
                "litellm_api_key": "test-key",
                "litellm_api_base": "https://custom-api.example.com",
            },
        },
    }


def chat_config_with_provider():
    return {
        "name": "chat",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "provider": "litellm",
            "name": "claude-3-5-sonnet-20241022",
            "config": {
                "litellm_provider": "anthropic",
                "litellm_api_key": "test-key",
            },
        },
    }


def embeddings_config():
    return {
        "name": "embeddings",
        "endpoint_type": "llm/v1/embeddings",
        "model": {
            "provider": "litellm",
            "name": "text-embedding-3-small",
            "config": {
                "litellm_api_key": "test-key",
            },
        },
    }


def mock_litellm_chat_response():
    """Create a mock LiteLLM chat response object."""
    response = mock.MagicMock()
    response.id = "litellm-chat-id"
    response.object = "chat.completion"
    response.created = 1234567890
    response.model = "claude-3-5-sonnet-20241022"

    choice = mock.MagicMock()
    choice.index = 0
    choice.message = mock.MagicMock()
    choice.message.role = "assistant"
    choice.message.content = TEST_MESSAGE
    choice.message.tool_calls = None
    choice.finish_reason = "stop"

    response.choices = [choice]
    response.usage = mock.MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30

    return response


def mock_litellm_embeddings_response():
    """Create a mock LiteLLM embeddings response object."""
    response = mock.MagicMock()
    response.model = "text-embedding-3-small"

    data = mock.MagicMock()
    data.__getitem__ = lambda self, key: [0.1, 0.2, 0.3] if key == "embedding" else None
    response.data = [data]

    response.usage = mock.MagicMock()
    response.usage.prompt_tokens = 5
    response.usage.total_tokens = 5

    return response


@pytest.mark.asyncio
async def test_chat():
    config = chat_config()
    mock_response = mock_litellm_chat_response()

    with mock.patch("litellm.acompletion", return_value=mock_response) as mock_completion:
        provider = LiteLLMProvider(EndpointConfig(**config))
        payload = {
            "messages": [{"role": "user", "content": TEST_MESSAGE}],
            "temperature": 0.7,
            "max_tokens": 100,
        }
        response = await provider.chat(chat.RequestPayload(**payload))

        assert response.id == "litellm-chat-id"
        assert response.object == "chat.completion"
        assert response.model == "claude-3-5-sonnet-20241022"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == TEST_MESSAGE
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30

        # Verify litellm was called with correct parameters
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert call_kwargs["messages"] == [{"role": "user", "content": TEST_MESSAGE}]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["api_key"] == "test-key"


@pytest.mark.asyncio
async def test_chat_with_api_base():
    config = chat_config_with_api_base()
    mock_response = mock_litellm_chat_response()

    with mock.patch("litellm.acompletion", return_value=mock_response) as mock_completion:
        provider = LiteLLMProvider(EndpointConfig(**config))
        payload = {"messages": [{"role": "user", "content": TEST_MESSAGE}]}
        await provider.chat(chat.RequestPayload(**payload))

        # Verify API base is passed
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_base"] == "https://custom-api.example.com"


@pytest.mark.asyncio
async def test_chat_with_provider_prefix():
    config = chat_config_with_provider()
    mock_response = mock_litellm_chat_response()

    with mock.patch("litellm.acompletion", return_value=mock_response) as mock_completion:
        provider = LiteLLMProvider(EndpointConfig(**config))
        payload = {"messages": [{"role": "user", "content": TEST_MESSAGE}]}
        await provider.chat(chat.RequestPayload(**payload))

        # Verify model name includes provider prefix
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-3-5-sonnet-20241022"


@pytest.mark.asyncio
async def test_chat_stream():
    config = chat_config()

    # Create mock streaming chunks
    async def mock_stream():
        chunk1 = mock.MagicMock()
        chunk1.id = "chunk-1"
        chunk1.object = "chat.completion.chunk"
        chunk1.created = 1234567890
        chunk1.model = "claude-3-5-sonnet-20241022"
        choice1 = mock.MagicMock()
        choice1.index = 0
        choice1.delta = mock.MagicMock(spec=["role", "content"])
        choice1.delta.role = "assistant"
        choice1.delta.content = "Hello"
        choice1.finish_reason = None
        chunk1.choices = [choice1]
        yield chunk1

        chunk2 = mock.MagicMock()
        chunk2.id = "chunk-2"
        chunk2.object = "chat.completion.chunk"
        chunk2.created = 1234567890
        chunk2.model = "claude-3-5-sonnet-20241022"
        choice2 = mock.MagicMock()
        choice2.index = 0
        choice2.delta = mock.MagicMock(spec=["content"])
        choice2.delta.content = " world"
        choice2.finish_reason = "stop"
        chunk2.choices = [choice2]
        yield chunk2

    with mock.patch("litellm.acompletion", return_value=mock_stream()) as mock_completion:
        provider = LiteLLMProvider(EndpointConfig(**config))
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        chunks = [chunk async for chunk in provider.chat_stream(chat.RequestPayload(**payload))]

        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " world"
        assert chunks[1].choices[0].finish_reason == "stop"

        # Verify stream parameter was set
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["stream"] is True


@pytest.mark.asyncio
async def test_embeddings():
    config = embeddings_config()
    mock_response = mock_litellm_embeddings_response()

    with mock.patch("litellm.aembedding", return_value=mock_response) as mock_embedding:
        provider = LiteLLMProvider(EndpointConfig(**config))
        payload = {"input": "Hello world"}
        response = await provider.embeddings(embeddings.RequestPayload(**payload))

        assert response.model == "text-embedding-3-small"
        assert len(response.data) == 1
        assert response.data[0].embedding == [0.1, 0.2, 0.3]
        assert response.usage.prompt_tokens == 5
        assert response.usage.total_tokens == 5

        # Verify litellm was called with correct parameters
        mock_embedding.assert_called_once()
        call_kwargs = mock_embedding.call_args[1]
        assert call_kwargs["model"] == "text-embedding-3-small"
        assert call_kwargs["input"] == "Hello world"
        assert call_kwargs["api_key"] == "test-key"


@pytest.mark.asyncio
async def test_embeddings_batch():
    config = embeddings_config()

    # Create mock response for batch
    response = mock.MagicMock()
    response.model = "text-embedding-3-small"

    data1 = mock.MagicMock()
    data1.__getitem__ = lambda self, key: [0.1, 0.2, 0.3] if key == "embedding" else None
    data2 = mock.MagicMock()
    data2.__getitem__ = lambda self, key: [0.4, 0.5, 0.6] if key == "embedding" else None
    response.data = [data1, data2]

    response.usage = mock.MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.total_tokens = 10

    with mock.patch("litellm.aembedding", return_value=response):
        provider = LiteLLMProvider(EndpointConfig(**config))
        payload = {"input": ["Hello", "World"]}
        response_payload = await provider.embeddings(embeddings.RequestPayload(**payload))

        assert len(response_payload.data) == 2
        assert response_payload.data[0].embedding == [0.1, 0.2, 0.3]
        assert response_payload.data[1].embedding == [0.4, 0.5, 0.6]


def test_adapter_chat_to_model():
    config = EndpointConfig(**chat_config())
    payload = {
        "messages": [{"role": "user", "content": TEST_MESSAGE}],
        "temperature": 0.7,
    }

    result = LiteLLMAdapter.chat_to_model(payload, config)

    assert result["model"] == "claude-3-5-sonnet-20241022"
    assert result["messages"] == [{"role": "user", "content": TEST_MESSAGE}]
    assert result["temperature"] == 0.7


def test_adapter_embeddings_to_model():
    config = EndpointConfig(**embeddings_config())
    payload = {"input": TEST_MESSAGE}

    result = LiteLLMAdapter.embeddings_to_model(payload, config)

    assert result["model"] == "text-embedding-3-small"
    assert result["input"] == TEST_MESSAGE


def test_adapter_chat_to_model_with_provider():
    config = EndpointConfig(**chat_config_with_provider())
    payload = {
        "messages": [{"role": "user", "content": TEST_MESSAGE}],
        "temperature": 0.7,
    }

    result = LiteLLMAdapter.chat_to_model(payload, config)

    assert result["model"] == "anthropic/claude-3-5-sonnet-20241022"
    assert result["messages"] == [{"role": "user", "content": TEST_MESSAGE}]
    assert result["temperature"] == 0.7


def test_adapter_model_to_chat():
    config = EndpointConfig(**chat_config())
    resp = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": TEST_MESSAGE, "tool_calls": None},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    result = LiteLLMAdapter.model_to_chat(resp, config)

    assert result.id == "test-id"
    assert result.model == "test-model"
    assert len(result.choices) == 1
    assert result.choices[0].message.content == TEST_MESSAGE
    assert result.usage.prompt_tokens == 10


def test_adapter_model_to_embeddings():
    config = EndpointConfig(**embeddings_config())
    resp = {
        "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
        "model": "test-model",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }

    result = LiteLLMAdapter.model_to_embeddings(resp, config)

    assert result.model == "test-model"
    assert len(result.data) == 1
    assert result.data[0].embedding == [0.1, 0.2, 0.3]
    assert result.data[0].index == 0
    assert result.usage.prompt_tokens == 5
