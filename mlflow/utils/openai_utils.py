import json
import os
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional
from unittest import mock

import requests

import mlflow

TEST_CONTENT = "test"

TEST_SOURCE_DOCUMENTS = [
    {
        "page_content": "We see the unity among leaders ...",
        "metadata": {"source": "tests/langchain/state_of_the_union.txt"},
    },
]
TEST_INTERMEDIATE_STEPS = (
    [
        {
            "tool": "Search",
            "tool_input": "High temperature in SF yesterday",
            "log": " I need to find the temperature first...",
            "result": "San Francisco...",
        },
    ],
)

REQUEST_URL_CHAT = "https://api.openai.com/v1/chat/completions"
REQUEST_URL_COMPLETIONS = "https://api.openai.com/v1/completions"
REQUEST_URL_EMBEDDINGS = "https://api.openai.com/v1/embeddings"

REQUEST_FIELDS_CHAT = {
    "model",
    "messages",
    "frequency_penalty",
    "logit_bias",
    "max_tokens",
    "n",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "stream",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "user",
    "function_call",
    "functions",
}
REQUEST_FIELDS_COMPLETIONS = {
    "model",
    "prompt",
    "best_of",
    "echo",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_tokens",
    "n",
    "presence_penalty",
    "seed",
    "stop",
    "stream",
    "suffix",
    "temperature",
    "top_p",
    "user",
}
REQUEST_FIELDS_EMBEDDINGS = {"input", "model", "encoding_format", "user"}
REQUEST_FIELDS = REQUEST_FIELDS_CHAT | REQUEST_FIELDS_COMPLETIONS | REQUEST_FIELDS_EMBEDDINGS


class _MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self.content = json.dumps(json_data).encode()
        self.headers = {"Content-Type": "application/json"}
        self.text = mlflow.__version__
        self.json_data = json_data

    def json(self):
        return self.json_data


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


@contextmanager
def _mock_request_post(**kwargs):
    with mock.patch("requests.post", **kwargs) as m:
        yield m


def _mock_openai_request():
    original = requests.post

    def request(*args, **kwargs):
        url = kwargs.get("url")
        for key in kwargs.get("json"):
            assert key in REQUEST_FIELDS, f"'{key}' is not a valid request field"

        if "/chat/completions" in url:
            messages = kwargs.get("json").get("messages")
            return _mock_chat_completion_response(content=json.dumps(messages))
        elif "/completions" in url:
            prompt = kwargs.get("json").get("prompt")
            return _mock_completion_response(content=json.dumps(prompt))
        elif "/embeddings" in url:
            inp = kwargs.get("json").get("input")
            return _mock_embeddings_response(len(inp) if isinstance(inp, list) else 1)
        else:
            return original(*args, **kwargs)

    return _mock_request_post(new=request)


def _validate_model_params(task, model, params):
    if not params:
        return

    if any(key in model for key in params):
        raise mlflow.MlflowException.invalid_parameter_value(
            f"Providing any of {list(model.keys())} as parameters in the signature is not "
            "allowed because they were indicated as part of the OpenAI model. Either remove "
            "the argument when logging the model or remove the parameter from the signature.",
        )
    if "batch_size" in params and task == "chat.completions":
        raise mlflow.MlflowException.invalid_parameter_value(
            "Parameter `batch_size` is not supported for task `chat.completions`"
        )


def _exclude_params_from_envs(params, envs):
    """
    params passed at inference time should override envs.
    """
    return {k: v for k, v in envs.items() if k not in params} if params else envs


class _OAITokenHolder:
    def __init__(self, api_type):
        self._api_token = None
        self._credential = None
        self._is_azure_ad = api_type in ("azure_ad", "azuread")
        self._key_configured = "OPENAI_API_KEY" in os.environ

        if self._is_azure_ad and not self._key_configured:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError:
                raise mlflow.MlflowException(
                    "Using API type `azure_ad` or `azuread` requires the package"
                    " `azure-identity` to be installed."
                )
            self._credential = DefaultAzureCredential()

    def validate(self, logger=None):
        """
        Validates the token or API key configured for accessing the OpenAI resource.
        """
        if self._key_configured:
            return

        if self._is_azure_ad:
            if not self._api_token or self._api_token.expires_on < time.time() + 60:
                from azure.core.exceptions import ClientAuthenticationError

                if logger:
                    logger.debug(
                        "Token for Azure AD is either expired or unset. Attempting to "
                        "acquire a new token."
                    )
                try:
                    self._api_token = self._credential.get_token(
                        "https://cognitiveservices.azure.com/.default"
                    )
                except ClientAuthenticationError as err:
                    raise mlflow.MlflowException(
                        "Unable to acquire a valid Azure AD token for the resource due to "
                        f"the following error: {err.message}"
                    ) from err
                os.environ["OPENAI_API_KEY"] = self._api_token.token
            if logger:
                logger.debug("Token refreshed successfully")
        else:
            raise mlflow.MlflowException(
                "OpenAI API key must be set in the ``OPENAI_API_KEY`` environment variable."
            )


class _OpenAIApiConfig(NamedTuple):
    api_type: str
    batch_size: int
    max_requests_per_minute: int
    max_tokens_per_minute: int
    api_version: Optional[str]
    api_base: str
    engine: Optional[str]
    deployment_id: Optional[str]


# See https://github.com/openai/openai-python/blob/cf03fe16a92cd01f2a8867537399c12e183ba58e/openai/__init__.py#L30-L38
# for the list of environment variables that openai-python uses
class _OpenAIEnvVar(str, Enum):
    OPENAI_API_TYPE = "OPENAI_API_TYPE"
    OPENAI_API_BASE = "OPENAI_API_BASE"
    OPENAI_API_KEY = "OPENAI_API_KEY"
    OPENAI_API_KEY_PATH = "OPENAI_API_KEY_PATH"
    OPENAI_API_VERSION = "OPENAI_API_VERSION"
    OPENAI_ORGANIZATION = "OPENAI_ORGANIZATION"
    OPENAI_ENGINE = "OPENAI_ENGINE"
    # use deployment_name instead of deployment_id to be
    # consistent with gateway
    OPENAI_DEPLOYMENT_NAME = "OPENAI_DEPLOYMENT_NAME"

    @property
    def secret_key(self):
        return self.value.lower()

    @classmethod
    def read_environ(cls):
        env_vars = {}
        for e in _OpenAIEnvVar:
            if value := os.getenv(e.value):
                env_vars[e.value] = value
        return env_vars
