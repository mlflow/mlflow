from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
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
            "config": {"hf_server_url": "https://testserverurl.com"},
        },
    }


def embedding_config():
    return {
        "name": "embeddings",
        "route_type": "llm/v1/embeddings",
        "model": {
            "provider": "huggingface-text-generation-inference",
            "name": "hf-tgi",
            "config": {"hf_server_url": "https://testserverurl.com"},
        },
    }


def chat_config():
    return {
        "name": "chat",
        "route_type": "llm/v1/chat",
        "model": {
            "provider": "huggingface-text-generation-inference",
            "name": "hf-tgi",
            "config": {"hf_server_url": "https://testserverurl.com"},
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
    with mock.patch("time.time", return_value=1677858242), mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = HFTextGenerationInferenceServerProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
            "n": 1,
            "max_tokens": 1000,
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": None,
            "object": "text_completion",
            "created": 1677858242,
            "model": "hf-tgi",
            "choices": [
                {
                    "text": "this is a test response",
                    "index": 0,
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
        }
        mock_post.assert_called_once_with(
            "https://testserverurl.com/generate",
            json={
                "inputs": "This is a test",
                "parameters": {
                    "temperature": 0.001,
                    "max_new_tokens": 1000,
                    "details": True,
                    "decoder_input_details": True,
                },
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.asyncio
async def test_completions_temperature_is_scaled_correctly():
    resp = completions_response()
    config = completions_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = HFTextGenerationInferenceServerProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
            "temperature": 0.5,
        }
        await provider.completions(completions.RequestPayload(**payload))
        assert mock_post.call_args[1]["json"]["parameters"]["temperature"] == 0.5 * 50


@pytest.mark.asyncio
async def test_completion_fails_with_multiple_candidates():
    config = chat_config()
    provider = HFTextGenerationInferenceServerProvider(RouteConfig(**config))
    payload = {"prompt": "This is a test", "n": 2}
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "'n' must be '1' for the Text Generation Inference provider." in e.value.detail
    assert e.value.status_code == 422


@pytest.mark.asyncio
async def test_chat_is_not_supported_for_tgi():
    config = chat_config()
    provider = HFTextGenerationInferenceServerProvider(RouteConfig(**config))
    payload = {"messages": [{"role": "user", "content": "TGI, can you chat with me? I'm lonely."}]}

    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.chat(chat.RequestPayload(**payload))
    assert (
        "The chat route is not implemented for Hugging Face Text Generation Inference models."
        in e.value.detail
    )
    assert e.value.status_code == 501


@pytest.mark.asyncio
async def test_embeddings_are_not_supported_for_tgi():
    config = embedding_config()
    provider = HFTextGenerationInferenceServerProvider(RouteConfig(**config))
    payload = {"input": "give me that sweet, sweet vector, please."}

    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.embeddings(embeddings.RequestPayload(**payload))
    assert (
        "The embeddings route is not implemented for Hugging Face Text Generation Inference models."
        in e.value.detail
    )
    assert e.value.status_code == 501
