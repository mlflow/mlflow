import json
from contextlib import contextmanager
from unittest import mock

import requests

import mlflow

TEST_CONTENT = "test"


class _MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self.content = json.dumps(json_data).encode()
        self.headers = {"Content-Type": "application/json"}
        self.text = mlflow.__version__


def _chat_completion_json_sample(content):
    # https://platform.openai.com/docs/api-reference/chat/create
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "text": content,
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }


def _models_retrieve_json_sample():
    # https://platform.openai.com/docs/api-reference/models/retrieve
    return {
        "id": "gpt-3.5-turbo",
        "object": "model",
        "owned_by": "openai",
        "permission": [],
    }


def _mock_chat_completion_response(content=TEST_CONTENT):
    return _MockResponse(200, _chat_completion_json_sample(content))


def _mock_embeddings_response(num_texts):
    return _MockResponse(
        200,
        {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [
                        0.0,
                    ],
                    "index": i,
                }
                for i in range(num_texts)
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        },
    )


def _mock_models_retrieve_response():
    return _MockResponse(200, _models_retrieve_json_sample())


@contextmanager
def _mock_request(**kwargs):
    with mock.patch("requests.Session.request", **kwargs) as m:
        yield m


def _mock_openai_request():
    original = requests.Session.request

    def request(*args, **kwargs):
        if len(args) > 2:
            url = args[2]
        else:
            url = kwargs.get("url")

        if url.endswith("/chat/completions"):
            messages = json.loads(kwargs.get("data")).get("messages")
            return _mock_chat_completion_response(content=json.dumps(messages))
        elif url.endswith("/embeddings"):
            inp = json.loads(kwargs.get("data")).get("input")
            return _mock_embeddings_response(len(inp) if isinstance(inp, list) else 1)
        else:
            return original(*args, **kwargs)

    return _mock_request(new=request)
