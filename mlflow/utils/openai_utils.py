import json
import os
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional
from unittest import mock
from unittest.mock import AsyncMock

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


class _MockStreamResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self.iter_lines = lambda: iter([f"data: {json.dumps(json_data)}".encode()])
        self.headers = {"Content-Type": "text/event-stream"}


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


def _mock_chat_completion_response(content=TEST_CONTENT):
    return _MockResponse(200, _chat_completion_json_sample(content))


def _chat_completion_stream_chunk(content):
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": content,
                    "function_call": None,
                    "role": None,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }


def _mock_chat_completion_stream_response(content=TEST_CONTENT):
    return _MockStreamResponse(200, _chat_completion_stream_chunk(content))


@contextmanager
def _mock_request(**kwargs):
    with mock.patch("requests.Session.request", **kwargs) as m:
        yield m


@contextmanager
def _mock_openai_arequest(stream=False):
    if stream:
        side_effect = _mock_async_chat_completion_stream_response
    else:
        side_effect = _mock_async_chat_completion_response

    with mock.patch("aiohttp.ClientSession.request", side_effect=side_effect) as mock_request:
        yield mock_request


async def _mock_async_chat_completion_response(content=TEST_CONTENT, **kwargs):
    resp = AsyncMock()
    json_data = _chat_completion_json_sample(content)
    resp.status = 200
    resp.content = json.dumps(json_data).encode()
    resp.headers = {"Content-Type": "application/json"}
    resp.text = mlflow.__version__
    resp.json_data = json_data
    resp.json.return_value = json_data
    resp.read.return_value = resp.content
    return resp


async def _mock_async_chat_completion_stream_response(content=TEST_CONTENT, **kwargs):
    resp = AsyncMock()
    json_data = _chat_completion_stream_chunk(content)
    resp.status = 200

    class DummyAsyncIter:
        def __init__(self):
            self._content = [f"data: {json.dumps(json_data)}".encode()]
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index < len(self._content):
                self._index += 1
                return self._content[self._index - 1]
            else:
                raise StopAsyncIteration

    # OpenAI uses content instead of iter_lines for async stream parsing
    resp.content = DummyAsyncIter()
    resp.headers = {"Content-Type": "text/event-stream"}
    return resp


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


class _OAITokenHolder:
    def __init__(self, api_type):
        self._credential = None
        self._api_type = api_type
        self._is_azure_ad = api_type in ("azure_ad", "azuread")
        self._azure_ad_token = None
        self._api_token_env = os.environ.get("OPENAI_API_KEY")

        if self._is_azure_ad and not self._api_token_env:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError:
                raise mlflow.MlflowException(
                    "Using API type `azure_ad` or `azuread` requires the package"
                    " `azure-identity` to be installed."
                )
            self._credential = DefaultAzureCredential()

    @property
    def token(self):
        return self._api_token_env or self._azure_ad_token.token

    def auth_headers(self):
        if self._api_type == "azure":
            # For Azure OpenAI API keys, the `api-key` header must be used:
            # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#authentication
            return {"api-key": self.token}
        else:
            return {"Authorization": f"Bearer {self.token}"}

    def refresh(self, logger=None):
        """Validates the token or API key configured for accessing the OpenAI resource."""

        if self._api_token_env:
            return

        if self._is_azure_ad:
            if not self._azure_ad_token or self._azure_ad_token.expires_on < time.time() + 60:
                from azure.core.exceptions import ClientAuthenticationError

                if logger:
                    logger.debug(
                        "Token for Azure AD is either expired or unset. Attempting to "
                        "acquire a new token."
                    )
                try:
                    self._azure_ad_token = self._credential.get_token(
                        "https://cognitiveservices.azure.com/.default"
                    )
                except ClientAuthenticationError as err:
                    raise mlflow.MlflowException(
                        "Unable to acquire a valid Azure AD token for the resource due to "
                        f"the following error: {err.message}"
                    ) from err

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
