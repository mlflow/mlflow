from unittest import mock

import pytest

from mlflow.gateway.providers.openai import OpenAIProvider


@pytest.mark.asyncio
async def test_chat():
    resp = {
        "id": "chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve",
        "object": "chat.completion",
        "created": 1677649420,
        "model": "gpt-3.5-turbo",
        "usage": {"prompt_tokens": 56, "completion_tokens": 31, "total_tokens": 87},
        "choices": [
            {
                "message": {"role": "user", "content": "LLM"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }
    with mock.patch("openai.ChatCompletion.acreate", return_value=resp) as mock_acreate:
        provider = OpenAIProvider(config={...})
        payload = {
            "messages": [
                {"role": "user", "content": "This is a a test"},
            ],
        }
        response = await provider.chat(payload)
        assert response == {...}
        mock_acreate.assert_called_once()


@pytest.mark.asyncio
async def test_completions():
    resp = {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "text-davinci-003",
        "choices": [
            {
                "text": "\n\nThis is indeed a test",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    with mock.patch("openai.Completion.acreate", return_value=resp) as mock_acreate:
        provider = OpenAIProvider(config={...})
        payload = {
            "prompt": "This is a test",
        }
        response = await provider.completions(payload)
        assert response == {...}
        mock_acreate.assert_called_once()


@pytest.mark.asyncio
async def test_embeddings():
    resp = {
        "data": [
            {
                "embedding": [
                    -0.006929283495992422,
                    -0.005336422007530928,
                    -4.547132266452536e-05,
                    -0.024047505110502243,
                ],
                "index": 0,
                "object": "embedding",
            }
        ],
        "model": "text-davinci-003",
        "object": "list",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }
    with mock.patch("openai.Embedding.acreate", return_value=resp) as mock_acreate:
        payload = {
            "input": "This is a test",
        }
        provider = OpenAIProvider(config={...})
        response = await provider.embeddings(payload)
        assert response == {...}
        mock_acreate.assert_called_once()
