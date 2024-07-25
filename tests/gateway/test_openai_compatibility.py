from unittest import mock

import openai
import pytest

from mlflow.gateway.providers.openai import OpenAIProvider

from tests.gateway.tools import (
    UvicornGateway,
    save_yaml,
)


@pytest.fixture(scope="module")
def config():
    return {
        "routes": [
            {
                "name": "chat",
                "route_type": "llm/v1/chat",
                "model": {
                    "name": "gpt-4o-mini",
                    "provider": "openai",
                    "config": {"openai_api_key": "test"},
                },
            },
            {
                "name": "completions",
                "route_type": "llm/v1/completions",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {"openai_api_key": "test"},
                },
            },
            {
                "name": "embeddings",
                "route_type": "llm/v1/embeddings",
                "model": {
                    "provider": "openai",
                    "name": "text-embedding-ada-002",
                    "config": {
                        "openai_api_key": "test",
                    },
                },
            },
        ]
    }


@pytest.fixture
def server(config, tmp_path):
    conf = tmp_path / "config.yaml"
    save_yaml(conf, config)
    with UvicornGateway(conf) as g:
        yield g


@pytest.fixture
def client(server) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{server.url}/v1", api_key="test")


def test_chat(client):
    async def mock_chat(self, payload):
        return {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "test",
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 7,
                "total_tokens": 20,
            },
        }

    with mock.patch.object(OpenAIProvider, "chat", mock_chat):
        chat = client.chat.completions.create(
            model="chat", messages=[{"role": "user", "content": "hello"}]
        )
        assert chat.choices[0].message.content == "test"


def test_chat_invalid_endpoint(client):
    with pytest.raises(openai.BadRequestError, match="is not a chat endpoint"):
        client.chat.completions.create(
            model="completions", messages=[{"role": "user", "content": "hello"}]
        )


def test_completions(client):
    async def mock_completions(self, payload):
        return {
            "id": "cmpl-abc123",
            "object": "text_completion",
            "created": 1677858242,
            "model": "gpt-4",
            "choices": [
                {
                    "finish_reason": "length",
                    "index": 0,
                    "logprobs": None,
                    "text": "test",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 11},
        }

    with mock.patch.object(OpenAIProvider, "completions", mock_completions):
        completions = client.completions.create(
            model="completions",
            prompt="hello",
        )
        assert completions.choices[0].text == "test"


def test_completions_invalid_endpoint(client):
    with pytest.raises(openai.BadRequestError, match="is not a completions endpoint"):
        client.completions.create(model="chat", prompt="hello")


def test_embeddings(client):
    async def mock_embeddings(self, payload):
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [
                        0.1,
                        0.2,
                        0.3,
                    ],
                    "index": 0,
                }
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }

    with mock.patch.object(OpenAIProvider, "embeddings", mock_embeddings):
        embeddings = client.embeddings.create(model="embeddings", input="hello")
        assert embeddings.data[0].embedding == [0.1, 0.2, 0.3]


def test_embeddings_invalid_endpoint(client):
    with pytest.raises(openai.BadRequestError, match="is not an embeddings endpoint"):
        client.embeddings.create(model="chat", input="hello")
