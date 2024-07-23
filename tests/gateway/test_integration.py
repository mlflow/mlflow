import os
from unittest.mock import patch

import pytest
import requests

import mlflow
import mlflow.gateway.utils
from mlflow.exceptions import MlflowException
from mlflow.gateway import MlflowGatewayClient, get_route, query, set_gateway_uri
from mlflow.gateway.config import Route
from mlflow.gateway.providers.ai21labs import AI21LabsProvider
from mlflow.gateway.providers.anthropic import AnthropicProvider
from mlflow.gateway.providers.bedrock import AmazonBedrockProvider
from mlflow.gateway.providers.cohere import CohereProvider
from mlflow.gateway.providers.huggingface import HFTextGenerationInferenceServerProvider
from mlflow.gateway.providers.mistral import MistralProvider
from mlflow.gateway.providers.mlflow import MlflowModelServingProvider
from mlflow.gateway.providers.mosaicml import MosaicMLProvider
from mlflow.gateway.providers.openai import OpenAIProvider
from mlflow.gateway.providers.palm import PaLMProvider
from mlflow.gateway.providers.togetherai import TogetherAIProvider
from mlflow.utils.request_utils import _cached_get_request_session

from tests.gateway.tools import (
    UvicornGateway,
    log_completions_transformers_model,
    log_sentence_transformers_model,
    save_yaml,
    start_mlflow_server,
    stop_mlflow_server,
)


@pytest.fixture
def basic_config_dict():
    return {
        "routes": [
            {
                "name": "chat-openai",
                "route_type": "llm/v1/chat",
                "model": {
                    "name": "gpt-4o-mini",
                    "provider": "openai",
                    "config": {"openai_api_key": "$OPENAI_API_KEY"},
                },
            },
            {
                "name": "completions-openai",
                "route_type": "llm/v1/completions",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {"openai_api_key": "$OPENAI_API_KEY"},
                },
            },
            {
                "name": "embeddings-openai",
                "route_type": "llm/v1/embeddings",
                "model": {
                    "provider": "openai",
                    "name": "text-embedding-ada-002",
                    "config": {
                        "openai_api_base": "https://api.openai.com/v1",
                        "openai_api_key": "$OPENAI_API_KEY",
                    },
                },
            },
            {
                "name": "completions-anthropic",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "anthropic",
                    "name": "claude-instant-1.1",
                    "config": {
                        "anthropic_api_key": "$ANTHROPIC_API_KEY",
                    },
                },
            },
            {
                "name": "completions-ai21labs",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "ai21labs",
                    "name": "j2-ultra",
                    "config": {
                        "ai21labs_api_key": "$AI21LABS_API_KEY",
                    },
                },
            },
            {
                "name": "completions-cohere",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "cohere",
                    "name": "command",
                    "config": {
                        "cohere_api_key": "$COHERE_API_KEY",
                    },
                },
            },
            {
                "name": "completions-mosaicml",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "mosaicml",
                    "name": "mpt-7b-instruct",
                    "config": {
                        "mosaicml_api_key": "$MOSAICML_API_KEY",
                    },
                },
            },
            {
                "name": "completions-palm",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "palm",
                    "name": "text-bison-001",
                    "config": {
                        "palm_api_key": "$PALM_API_KEY",
                    },
                },
            },
            {
                "name": "chat-palm",
                "route_type": "llm/v1/chat",
                "model": {
                    "provider": "palm",
                    "name": "chat-bison-001",
                    "config": {
                        "palm_api_key": "$PALM_API_KEY",
                    },
                },
            },
            {
                "name": "chat-mosaicml",
                "route_type": "llm/v1/chat",
                "model": {
                    "provider": "mosaicml",
                    "name": "llama2-70b-chat",
                    "config": {
                        "mosaicml_api_key": "$MOSAICML_API_KEY",
                    },
                },
            },
            {
                "name": "embeddings-cohere",
                "route_type": "llm/v1/embeddings",
                "model": {
                    "provider": "cohere",
                    "name": "embed-english-v2.0",
                    "config": {
                        "cohere_api_key": "$COHERE_API_KEY",
                    },
                },
            },
            {
                "name": "embeddings-mosaicml",
                "route_type": "llm/v1/embeddings",
                "model": {
                    "provider": "mosaicml",
                    "name": "instructor-large",
                    "config": {
                        "mosaicml_api_key": "$MOSAICML_API_KEY",
                    },
                },
            },
            {
                "name": "embeddings-palm",
                "route_type": "llm/v1/embeddings",
                "model": {
                    "provider": "palm",
                    "name": "embedding-gecko-001",
                    "config": {
                        "palm_api_key": "$PALM_API_KEY",
                    },
                },
            },
            {
                "name": "chat-oss",
                "route_type": "llm/v1/chat",
                "model": {
                    "provider": "mlflow-model-serving",
                    "name": "mpt-chatbot",
                    "config": {"model_server_url": "http://127.0.0.1:5000"},
                },
            },
            {
                "name": "completions-oss",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "mlflow-model-serving",
                    "name": "completion-model",
                    "config": {"model_server_url": "http://127.0.0.1:6000"},
                },
            },
            {
                "name": "embeddings-oss",
                "route_type": "llm/v1/embeddings",
                "model": {
                    "provider": "mlflow-model-serving",
                    "name": "sentence-transformers",
                    "config": {"model_server_url": "http://127.0.0.1:5002"},
                },
            },
            {
                "name": "completions-huggingface",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "huggingface-text-generation-inference",
                    "name": "hf-falcon-7b-instruct",
                    "config": {"hf_server_url": "http://127.0.0.1:5000"},
                },
            },
            {
                "name": "completions-bedrock",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "bedrock",
                    "name": "amazon.titan-tg1-large",
                    "config": {"aws_config": {"aws_region": "us-east-1"}},
                },
            },
            {
                "name": "completions-mistral",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "mistral",
                    "name": "mistral-tiny",
                    "config": {
                        "mistral_api_key": "$MISTRAL_API_KEY",
                    },
                },
            },
            {
                "name": "embeddings-mistral",
                "route_type": "llm/v1/embeddings",
                "model": {
                    "provider": "mistral",
                    "name": "mistral-embed",
                    "config": {
                        "mistral_api_key": "$MISTRAL_API_KEY",
                    },
                },
            },
            {
                "name": "completions-togetherai",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "togetherai",
                    "name": "mistralai/Mixtral-8x7B-v0.1",
                    "config": {
                        "togetherai_api_key": "$TOGETHERAI_API_KEY",
                    },
                },
            },
            {
                "name": "chat-togetherai",
                "route_type": "llm/v1/chat",
                "model": {
                    "provider": "togetherai",
                    "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "config": {
                        "togetherai_api_key": "$TOGETHERAI_API_KEY",
                    },
                },
            },
            {
                "name": "embeddings-togetherai",
                "route_type": "llm/v1/embeddings",
                "model": {
                    "provider": "togetherai",
                    "name": "togethercomputer/m2-bert-80M-8k-retrieval",
                    "config": {
                        "togetherai_api_key": "$TOGETHERAI_API_KEY",
                    },
                },
            },
        ]
    }


@pytest.fixture(autouse=True)
def clear_uri():
    mlflow.gateway.utils._gateway_uri = None


@pytest.fixture
def gateway(basic_config_dict, tmp_path):
    conf = tmp_path / "config.yaml"
    save_yaml(conf, basic_config_dict)
    with UvicornGateway(conf) as g:
        yield g


@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")
    monkeypatch.setenv("AI21LABS_API_KEY", "test_ai21labs_key")
    monkeypatch.setenv("MOSAICML_API_KEY", "test_mosaicml_key")
    monkeypatch.setenv("PALM_API_KEY", "test_palm_key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test_mistral_key")
    monkeypatch.setenv("TOGETHERAI_API_KEY", "test_togetherai_key")


@pytest.fixture
def serve_embeddings_model():
    model_uri = log_sentence_transformers_model()
    server = start_mlflow_server(port=5002, model_uri=model_uri)
    yield server.url
    stop_mlflow_server(server.pid)


@pytest.fixture
def serve_completions_model():
    model_uri = log_completions_transformers_model()
    server = start_mlflow_server(port=6000, model_uri=model_uri)
    yield server.url
    stop_mlflow_server(server.pid)


def test_create_gateway_client_with_declared_url(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)
    assert gateway_client.gateway_uri == gateway.url
    assert isinstance(gateway_client.get_route("chat-openai"), Route)
    routes = gateway_client.search_routes()
    assert len(routes) == 23
    assert all(isinstance(route, Route) for route in routes)


def test_openai_chat(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    route = get_route("chat-openai")
    expected_output = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "\n\nThis is a test!",
                    "tool_calls": None,
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

    data = {"messages": [{"role": "user", "content": "test"}]}

    async def mock_chat(self, payload):
        return expected_output

    with patch.object(OpenAIProvider, "chat", mock_chat):
        response = query(route=route.name, data=data)
    assert response == expected_output


def test_openai_completions(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("completions-openai")
    expected_output = {
        "id": "chatcmpl-abc123",
        "object": "text_completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [{"text": "test.", "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 11},
    }

    data = {"prompt": "test", "max_tokens": 50}

    async def mock_completions(self, payload):
        return expected_output

    with patch.object(OpenAIProvider, "completions", mock_completions):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_openai_embeddings(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("embeddings-openai")
    expected_output = {
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

    data = {"input": "mock me and my test"}

    async def mock_embeddings(self, payload):
        return expected_output

    with patch.object(OpenAIProvider, "embeddings", mock_embeddings):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_anthropic_completions(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    route = get_route("completions-anthropic")
    expected_output = {
        "id": None,
        "object": "text_completion",
        "created": 1677858242,
        "model": "claude-instant-1.1",
        "choices": [
            {
                "text": "test",
                "index": 0,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }

    data = {
        "prompt": "test",
        "max_tokens": 500,
        "temperature": 0.3,
    }

    async def mock_completions(self, payload):
        return expected_output

    with patch.object(AnthropicProvider, "completions", mock_completions):
        response = query(route=route.name, data=data)
    assert response == expected_output


def test_ai21labs_completions(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("completions-ai21labs")
    expected_output = {
        "id": None,
        "object": "text_completion",
        "created": 1677858242,
        "model": "j2-ultra",
        "choices": [{"text": "mock using MagicMock please", "index": 0, "finish_reason": "length"}],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }

    data = {"prompt": "mock my test", "max_tokens": 50}

    async def mock_completions(self, payload):
        return expected_output

    with patch.object(AI21LabsProvider, "completions", mock_completions):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_cohere_completions(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("completions-cohere")
    expected_output = {
        "id": None,
        "object": "text_completion",
        "created": 1677858242,
        "model": "command",
        "choices": [
            {
                "text": "mock using MagicMock please",
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }

    data = {"prompt": "mock my test", "max_tokens": 50}

    async def mock_completions(self, payload):
        return expected_output

    with patch.object(CohereProvider, "completions", mock_completions):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_mosaicml_completions(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("completions-mosaicml")
    expected_output = {
        "id": None,
        "object": "text_completion",
        "created": 1677858242,
        "model": "mpt-7b-instruct",
        "choices": [{"text": "mock using MagicMock please", "index": 0, "finish_reason": None}],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }

    data = {"prompt": "mock my test", "max_tokens": 50}

    async def mock_completions(self, payload):
        return expected_output

    with patch.object(MosaicMLProvider, "completions", mock_completions):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_mosaicml_chat(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("chat-mosaicml")
    expected_output = {
        "id": None,
        "created": 1700242674,
        "object": "chat.completion",
        "model": "llama2-70b-chat",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a test",
                    "tool_calls": None,
                },
                "finish_reason": None,
                "index": 0,
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }

    data = {"messages": [{"role": "user", "content": "test"}]}

    async def mock_chat(self, payload):
        return expected_output

    with patch.object(MosaicMLProvider, "chat", mock_chat):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_palm_completions(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("completions-palm")
    expected_output = {
        "id": None,
        "object": "text_completion",
        "created": 1677858242,
        "model": "text-bison-001",
        "choices": [
            {
                "text": "mock using MagicMock please",
                "index": 0,
                "finish_reason": None,
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }

    data = {"prompt": "mock my test", "max_tokens": 50}

    async def mock_completions(self, payload):
        return expected_output

    with patch.object(PaLMProvider, "completions", mock_completions):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_palm_chat(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("chat-palm")
    expected_output = {
        "id": None,
        "created": 1700242674,
        "object": "chat.completion",
        "model": "chat-bison",
        "choices": [
            {
                "message": {
                    "role": "1",
                    "content": "Hi there! How can I help you today?",
                    "tool_calls": None,
                },
                "finish_reason": None,
                "index": 0,
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }

    data = {"messages": [{"role": "user", "content": "test"}]}

    async def mock_chat(self, payload):
        return expected_output

    with patch.object(PaLMProvider, "chat", mock_chat):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_cohere_embeddings(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("embeddings-cohere")
    expected_output = {
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
        "model": "embed-english-v2.0",
        "usage": {"prompt_tokens": None, "total_tokens": None},
    }

    data = {"input": "mock me and my test"}

    async def mock_embeddings(self, payload):
        return expected_output

    with patch.object(CohereProvider, "embeddings", mock_embeddings):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_mosaicml_embeddings(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("embeddings-mosaicml")
    expected_output = {
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
        "model": "instructor-large",
        "usage": {"prompt_tokens": None, "total_tokens": None},
    }

    data = {"input": "mock me and my test"}

    async def mock_embeddings(self, payload):
        return expected_output

    with patch.object(MosaicMLProvider, "embeddings", mock_embeddings):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_palm_embeddings(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("embeddings-palm")
    expected_output = {
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
        "model": "embedding-gecko-001",
        "usage": {"prompt_tokens": None, "total_tokens": None},
    }

    data = {"input": "mock me and my test"}

    async def mock_embeddings(self, payload):
        return expected_output

    with patch.object(PaLMProvider, "embeddings", mock_embeddings):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_invalid_response_structure_raises(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    route = get_route("chat-openai")
    expected_output = {
        "embeddings": [[0.0, 1.0]],
        "metadata": {
            "input_tokens": 17,
            "output_tokens": 24,
            "total_tokens": 41,
            "model": "gpt-4o-mini",
            "route_type": "llm/v1/chat",
        },
    }

    data = {"messages": [{"role": "user", "content": "invalid test"}]}

    async def mock_chat(self, payload):
        return expected_output

    def _mock_request_session(
        max_retries,
        backoff_factor,
        backoff_jitter,
        retry_codes,
        raise_on_status,
        respect_retry_after_header,
    ):
        return _cached_get_request_session(1, 1, 0.5, retry_codes, True, os.getpid(), True)

    with patch(
        "mlflow.utils.request_utils._get_request_session", _mock_request_session
    ), patch.object(OpenAIProvider, "chat", mock_chat), pytest.raises(
        MlflowException, match=".*Max retries exceeded.*"
    ):
        query(route=route.name, data=data)


def test_invalid_response_structure_no_raises(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    route = get_route("chat-openai")
    expected_output = {
        "embeddings": [[0.0, 1.0]],
        "metadata": {
            "input_tokens": 17,
            "output_tokens": 24,
            "total_tokens": 41,
            "model": "gpt-4o-mini",
            "route_type": "llm/v1/chat",
        },
    }

    data = {"messages": [{"role": "user", "content": "invalid test"}]}

    async def mock_chat(self, payload):
        return expected_output

    def _mock_request_session(
        max_retries,
        backoff_factor,
        backoff_jitter,
        retry_codes,
        raise_on_status,
        respect_retry_after_header,
    ):
        return _cached_get_request_session(0, 1, 0.5, retry_codes, False, os.getpid(), True)

    with patch(
        "mlflow.utils.request_utils._get_request_session", _mock_request_session
    ), patch.object(OpenAIProvider, "chat", mock_chat), pytest.raises(
        requests.exceptions.HTTPError, match=".*Internal Server Error.*"
    ):
        query(route=route.name, data=data)


def test_invalid_query_request_raises(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    route = get_route("chat-openai")
    expected_output = {
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
            "prompt_tokens": 17,
            "completion_tokens": 24,
            "total_tokens": 41,
        },
    }

    data = {"text": "this is invalid"}

    async def mock_chat(self, payload):
        return expected_output

    def _mock_request_session(
        max_retries,
        backoff_factor,
        backoff_jitter,
        retry_codes,
        raise_on_status,
        respect_retry_after_header,
    ):
        return _cached_get_request_session(2, 1, 0.5, retry_codes, True, os.getpid(), True)

    with patch(
        "mlflow.utils.request_utils._get_request_session", _mock_request_session
    ), patch.object(OpenAIProvider, "chat", new=mock_chat), pytest.raises(
        requests.exceptions.HTTPError, match="Unprocessable Entity for"
    ):
        query(route=route.name, data=data)


def test_mlflow_chat(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("chat-oss")
    expected_output = {
        "id": None,
        "created": 1700242674,
        "object": "chat.completion",
        "model": "chat-bot-9000",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "It is a test",
                    "tool_calls": None,
                },
                "finish_reason": None,
                "index": 0,
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }

    data = {"messages": [{"role": "user", "content": "test"}]}

    with patch.object(MlflowModelServingProvider, "chat", return_value=expected_output):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_mlflow_completions(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("completions-oss")
    expected_output = {
        "id": None,
        "object": "text_completion",
        "created": 1677858242,
        "model": "completion-model",
        "choices": [
            {
                "text": "test",
                "index": 0,
                "finish_reason": None,
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }

    data = {"prompt": "this is a test"}

    with patch.object(MlflowModelServingProvider, "completions", return_value=expected_output):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_mlflow_embeddings(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    route = get_route("embeddings-oss")
    expected_output = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [
                    0.001,
                    -0.001,
                ],
                "index": 0,
            },
            {
                "object": "embedding",
                "embedding": [
                    0.002,
                    -0.002,
                ],
                "index": 1,
            },
        ],
        "model": "sentence-transformers",
        "usage": {"prompt_tokens": None, "total_tokens": None},
    }

    data = {"input": ["test1", "test2"]}

    with patch.object(MlflowModelServingProvider, "embeddings", return_value=expected_output):
        response = query(route=route.name, data=data)
    assert response == expected_output


def test_gateway_query_mlflow_embeddings_model(serve_embeddings_model, gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    route = get_route("embeddings-oss")

    data = {"input": ["test1", "test2"]}

    response = query(route=route.name, data=data)
    assert response["model"] == "sentence-transformers"

    embeddings_response = response["data"]

    assert isinstance(embeddings_response, list)
    assert len(embeddings_response) == 2

    usage_response = response["usage"]

    assert not usage_response["prompt_tokens"]
    assert not usage_response["total_tokens"]


def test_gateway_query_mlflow_completions_model(serve_completions_model, gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("completions-oss")

    data = {"prompt": "test [MASK]"}

    response = client.query(route=route.name, data=data)
    assert response["model"] == "completion-model"

    completions_response = response["choices"]

    assert isinstance(completions_response, list)
    assert isinstance(completions_response[0]["text"], str)
    assert len(completions_response) == 1

    metadata_response = response["usage"]
    assert not metadata_response["prompt_tokens"]
    assert not metadata_response["completion_tokens"]


def test_huggingface_completions(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("completions-huggingface")
    expected_output = {
        "id": None,
        "object": "text_completion",
        "created": 1677858242,
        "model": "llm/v1/completions",
        "choices": [
            {
                "text": "mock using MagicMock please",
                "index": 0,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    }

    data = {"prompt": "mock my test", "max_tokens": 50}

    async def mock_completions(self, payload):
        return expected_output

    with patch.object(HFTextGenerationInferenceServerProvider, "completions", mock_completions):
        response = client.query(route=route.name, data=data)

    assert response == expected_output


def test_bedrock_completions(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    route = get_route("completions-bedrock")
    expected_output = {
        "id": None,
        "object": "text_completion",
        "created": 1677858242,
        "model": "amazon.titan-tg1-large",
        "choices": [
            {
                "text": "\nThis is a test",
                "index": 0,
                "finish_reason": None,
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }

    data = {
        "prompt": "test",
        "max_tokens": 500,
        "temperature": 0.3,
    }

    async def mock_completions(self, payload):
        return expected_output

    with patch.object(AmazonBedrockProvider, "completions", mock_completions):
        response = query(route=route.name, data=data)

    assert response == expected_output


def test_mistral_completions(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("completions-mistral")
    expected_output = {
        "id": None,
        "object": "text_completion",
        "created": 1677858242,
        "model": "mistral-tiny",
        "choices": [
            {
                "index": 0,
                "text": "mock using MagicMock please",
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }

    data = {"prompt": "mock my test", "max_tokens": 50}

    async def mock_completions(self, payload):
        return expected_output

    with patch.object(MistralProvider, "completions", mock_completions):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_mistral_embeddings(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("embeddings-mistral")
    expected_output = {
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
        "model": "mistral-embed",
        "usage": {"prompt_tokens": None, "total_tokens": None},
    }

    data = {"input": "mock me and my test"}

    async def mock_embeddings(self, payload):
        return expected_output

    with patch.object(MistralProvider, "embeddings", mock_embeddings):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_togetherai_completions(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("completions-togetherai")

    expected_output = {
        "id": "8782fbf478f2ee87-ATH",
        "object": "text_completion",
        "created": 1713761335,
        "model": "mistralai/Mixtral-8x7B-v0.1",
        "choices": [
            {
                "index": 0,
                "text": (
                    "\n\nGeralt of Rivia"
                    "\nGeralt of Rivia is the protagonist of The Witcher 3: Wild Hunt.\n\n"
                    "## Is Geralt a good guy?\n\nGeralt is a good guy, but he’s not a hero."
                    "He’s a monster hunter, "
                    "and he’s not afraid to kill."
                    "He’s also not afraid to get his hands dirty. He’s not a hero, "
                    "but he’s not a villain either.\n\n"
                    "## Is Geralt a good person?\n\nGeralt is a good person. "
                    "He’s a monster hunter, but he’s also "
                    "a good person. He’s a good person because he’s a monster hunter. "
                    "He’s a good person because he’s "
                    "a good person.\n\n"
                    "## Is Geralt a good person?\n\nGeralt is a good person. "
                    "He’s a monster hunter, but he"
                ),
                "finish_reason": None,
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 200, "total_tokens": 215},
    }

    data = {
        "prompt": "Who's the protagonist in the Witcher 3 Wild Hunt?",
        "temperature": 0.1,
        "max_tokens": 200,
    }

    async def mock_completions(self, payload):
        return expected_output

    with patch.object(TogetherAIProvider, "completions", mock_completions):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_togetherai_chat(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("chat-togetherai")
    expected_output = {
        "id": "8782fc033a48eea0-ATH",
        "object": "chat.completion",
        "created": 1713761337,
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "I am not Alexander, but I can help you with your request. "
                        "To tell someone to get out of the sunlight, you could say:\n\n"
                        '* "Step out of the sun, please."\n'
                        '* "Can you move to the shade? The sun is quite strong."\n'
                        '* "The sun is too intense, do you mind finding some shade?"\n\n'
                        "I hope this helps! If you have any other questions or need further "
                        "clarification, just let me know."
                    ),
                    "tool_calls": None,
                },
                "finish_reason": None,
            }
        ],
        "usage": {"prompt_tokens": 19, "completion_tokens": 97, "total_tokens": 116},
    }

    data = {
        "messages": [{"role": "user", "content": "Get out of the sunlight's way Alexander!"}],
    }

    async def mock_chat(self, payload):
        return expected_output

    with patch.object(TogetherAIProvider, "chat", mock_chat):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_togetherai_embeddings(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("embeddings-togetherai")
    expected_output = {
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
        "model": "togethercomputer/m2-bert-80M-8k-retrieval",
        "usage": {"prompt_tokens": 7, "total_tokens": 20},
    }

    data = {"input": "Please I need embeddings!", "max_tokens": 50}

    async def mock_embeddings(self, payload):
        return expected_output

    with patch.object(TogetherAIProvider, "embeddings", mock_embeddings):
        response = client.query(route=route.name, data=data)
    assert response == expected_output
