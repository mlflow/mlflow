from unittest import mock

import pytest
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.providers.huggingface import HFTextGenerationInferenceServerProvider
from mlflow.gateway.schemas import chat, completions, embeddings

from tests.gateway.tools import MockAsyncResponse


def completions_config():
    return {
        "name": "completions",
        "route_type": "llm/v1/completions",
        "model": {
            "provider": "huggingface-text-generation-inference",
            "name": "hf-tgi",
            "config": {"hf_server_url": "url"},
        },
    }


def embedding_config():
    return {
        "name": "embeddings",
        "route_type": "llm/v1/embeddings",
        "model": {
            "provider": "huggingface-text-generation-inference",
            "name": "hf-tgi",
            "config": {"hf_server_url": "url"},
        },
    }


def chat_config():
    return {
        "name": "chat",
        "route_type": "llm/v1/chat",
        "model": {
            "provider": "huggingface-text-generation-inference",
            "name": "hf-tgi",
            "config": {"hf_server_url": "url"},
        },
    }


def completions_response():
    return {
        "generated_text": "this is a test response",
        "details": {
            "finish_reason": "length",
            "generated_tokens": 5,
            "seed": 0,
            "prefill": [{"text": "This"}, {"text": "is"}, {"text": "a"}, {"text": "test"}],
        },
    }


@pytest.mark.asyncio
async def test_completions():
    resp = completions_response()
    config = completions_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = HFTextGenerationInferenceServerProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "candidates": [
                {
                    "text": "this is a test response",
                    "metadata": {"finish_reason": "length", "seed": "0"},
                }
            ],
            "metadata": {
                "input_tokens": 4,
                "output_tokens": 5,
                "total_tokens": 9,
                "model": "hf-tgi",
                "route_type": "llm/v1/completions",
            },
        }
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_completion_fails_with_multiple_candidates():
    config = chat_config()
    provider = HFTextGenerationInferenceServerProvider(RouteConfig(**config))
    payload = {"prompt": "This is a test", "candidate_count": 2}
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert (
        "'candidate_count' must be '1' for the Text Generation Inference provider."
        in e.value.detail
    )
    assert e.value.status_code == 422


@pytest.mark.asyncio
async def test_chat_is_not_supported_for_tgi():
    config = chat_config()
    provider = HFTextGenerationInferenceServerProvider(RouteConfig(**config))
    payload = {"messages": [{"role": "user", "content": "TGI, can you chat with me? I'm lonely."}]}

    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.chat(chat.RequestPayload(**payload))
    assert (
        "The chat route is not available for the Text Generation Inference provider."
        in e.value.detail
    )
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_embeddings_are_not_supported_for_tgi():
    config = embedding_config()
    provider = HFTextGenerationInferenceServerProvider(RouteConfig(**config))
    payload = {"text": "give me that sweet, sweet vector, please."}

    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.embeddings(embeddings.RequestPayload(**payload))
    assert (
        "The embedding route is not available for the Text Generation Inference provider."
        in e.value.detail
    )
    assert e.value.status_code == 404
