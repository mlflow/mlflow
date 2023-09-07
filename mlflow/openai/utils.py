import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict
from unittest import mock

import requests

import mlflow
from mlflow.models.signature import ParamSchema

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


def _completion_json_sample(content):
    return {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 1589478378,
        "model": "text-davinci-003",
        "choices": [{"text": content, "index": 0, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
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


def _mock_completion_response(content=TEST_CONTENT):
    return _MockResponse(200, _completion_json_sample(content))


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
        elif url.endswith("/completions"):
            prompt = json.loads(kwargs.get("data")).get("prompt")
            return _mock_completion_response(content=json.dumps(prompt))
        elif url.endswith("/embeddings"):
            inp = json.loads(kwargs.get("data")).get("input")
            return _mock_embeddings_response(len(inp) if isinstance(inp, list) else 1)
        else:
            return original(*args, **kwargs)

    return _mock_request(new=request)


def _get_task_model_params(model: Dict[str, Any], params: Dict[str, Any] = None):
    model_dict = model.copy()
    task = model_dict.pop("task", None)
    if params:
        if any(params.keys() - model_dict.keys()):
            raise mlflow.MlflowException.invalid_parameter_value(
                f"Providing any of {list(model_dict.keys())} as parameters is not allowed.",
            )

    return task, model_dict, params


def _validate_task_model_params(model: Dict[str, Any], schema: ParamSchema = None):
    if schema:
        if any(param.name in model.keys() for param in schema.params):
            raise mlflow.MlflowException.invalid_parameter_value(
                f"Providing any of {','.join(model.keys())} as parameters is not allowed.",
            )


class _OAITokenHolder:
    def __init__(self, api_type):
        import openai

        self.api_type = api_type
        self.api_token = None

        if self.api_type in ("azure_ad", "azuread"):
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError:
                raise mlflow.MlflowException(
                    "Using API type ``azure_ad`` or ``azuread`` requires the package"
                    " ``azure-identity`` to be installed."
                )
            self.credential = DefaultAzureCredential()
            self.openai_token = None
        else:
            self.openai_key_available = openai.api_key or "OPENAI_API_KEY" in os.environ

    def validate(self, logger=None):
        """
        Validates the token or API key configured for accessing the OpenAI resource.
        """
        if self.api_type in ("azure_ad", "azuread"):
            if not self.api_token or self.api_token.expires_on < time.time() + 60:
                import openai

                if logger:
                    logger.debug(
                        "Token for Azure AD is either expired or invalid. New token to acquire."
                    )
                self.api_token = self.credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                )
                if not self.api_token:
                    raise mlflow.MlflowException(
                        "Unable to acquire a valid Azure AD token for the resource."
                    )
                openai.api_key = self.api_token.token
            if logger:
                logger.debug("Token or key validated correctly.")
        else:
            if not self.openai_key_available:
                raise mlflow.MlflowException(
                    "OpenAI API key must be set in the ``OPENAI_API_KEY`` environment variable."
                )
