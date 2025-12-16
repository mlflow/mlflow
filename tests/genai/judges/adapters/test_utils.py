from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    DatabricksManagedJudgeAdapter,
)
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    DatabricksServingEndpointAdapter,
)
from mlflow.genai.judges.adapters.gateway_adapter import GatewayAdapter
from mlflow.genai.judges.adapters.litellm_adapter import LiteLLMAdapter
from mlflow.genai.judges.adapters.utils import get_adapter
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.types.llm import ChatMessage


@pytest.fixture
def string_prompt():
    return "This is a test prompt"


@pytest.fixture
def list_prompt():
    return [
        ChatMessage(role="system", content="You are a helpful assistant"),
        ChatMessage(role="user", content="Please evaluate this"),
    ]


@pytest.mark.parametrize(
    ("model_uri", "prompt_type", "expected_adapter"),
    [
        # Databricks adapters
        (_DATABRICKS_DEFAULT_JUDGE_MODEL, "string", DatabricksManagedJudgeAdapter),
        (_DATABRICKS_DEFAULT_JUDGE_MODEL, "list", DatabricksManagedJudgeAdapter),
        ("databricks:/my-endpoint", "string", DatabricksServingEndpointAdapter),
        ("databricks:/my-endpoint", "list", DatabricksServingEndpointAdapter),
        ("endpoints:/my-endpoint", "string", DatabricksServingEndpointAdapter),
        ("endpoints:/my-endpoint", "list", DatabricksServingEndpointAdapter),
        # Gateway adapter
        ("openai:/gpt-4", "string", GatewayAdapter),
        ("anthropic:/claude-3-5-sonnet-20241022", "string", GatewayAdapter),
        ("bedrock:/anthropic.claude-3-5-sonnet-20241022-v2:0", "string", GatewayAdapter),
    ],
)
def test_get_adapter_without_litellm(
    model_uri, prompt_type, expected_adapter, string_prompt, list_prompt
):
    prompt = string_prompt if prompt_type == "string" else list_prompt
    with mock.patch(
        "mlflow.genai.judges.adapters.litellm_adapter._is_litellm_available",
        return_value=False,
    ):
        adapter = get_adapter(model_uri, prompt)
        assert isinstance(adapter, expected_adapter)


@pytest.mark.parametrize(
    ("model_uri", "prompt_type", "expected_adapter"),
    [
        # Databricks adapters (take priority over litellm)
        (_DATABRICKS_DEFAULT_JUDGE_MODEL, "string", DatabricksManagedJudgeAdapter),
        (_DATABRICKS_DEFAULT_JUDGE_MODEL, "list", DatabricksManagedJudgeAdapter),
        ("databricks:/my-endpoint", "string", DatabricksServingEndpointAdapter),
        ("databricks:/my-endpoint", "list", DatabricksServingEndpointAdapter),
        ("endpoints:/my-endpoint", "string", DatabricksServingEndpointAdapter),
        ("endpoints:/my-endpoint", "list", DatabricksServingEndpointAdapter),
        # LiteLLM adapter
        ("openai:/gpt-4", "string", LiteLLMAdapter),
        ("openai:/gpt-4", "list", LiteLLMAdapter),
        ("anthropic:/claude-3-5-sonnet-20241022", "string", LiteLLMAdapter),
        ("anthropic:/claude-3-5-sonnet-20241022", "list", LiteLLMAdapter),
    ],
)
def test_get_adapter_with_litellm(
    model_uri, prompt_type, expected_adapter, string_prompt, list_prompt
):
    prompt = string_prompt if prompt_type == "string" else list_prompt
    with mock.patch(
        "mlflow.genai.judges.adapters.litellm_adapter._is_litellm_available",
        return_value=True,
    ):
        adapter = get_adapter(model_uri, prompt)
        assert isinstance(adapter, expected_adapter)


@pytest.mark.parametrize(
    ("model_uri", "expected_error"),
    [
        ("openai:/gpt-4", "No suitable adapter found for model_uri='openai:/gpt-4'"),
        (
            "vertex_ai:/gemini-pro",
            "No suitable adapter found for model_uri='vertex_ai:/gemini-pro'",
        ),
    ],
)
def test_get_adapter_unsupported_with_list(model_uri, expected_error, list_prompt):
    with mock.patch(
        "mlflow.genai.judges.adapters.litellm_adapter._is_litellm_available",
        return_value=False,
    ):
        with pytest.raises(MlflowException, match=expected_error):
            get_adapter(model_uri, list_prompt)
