from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    DatabricksManagedJudgeAdapter,
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
        # Gateway adapter (used when litellm is unavailable)
        ("openai:/gpt-4", "string", GatewayAdapter),
        ("anthropic:/claude-3-5-sonnet-20241022", "string", GatewayAdapter),
        ("gemini:/gemini-2.5-flash", "string", GatewayAdapter),
        ("mistral:/mistral-large", "string", GatewayAdapter),
        # endpoints with string prompt
        ("endpoints:/my-endpoint", "string", GatewayAdapter),
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
        # Databricks managed judge (takes top priority)
        (_DATABRICKS_DEFAULT_JUDGE_MODEL, "string", DatabricksManagedJudgeAdapter),
        (_DATABRICKS_DEFAULT_JUDGE_MODEL, "list", DatabricksManagedJudgeAdapter),
        # Gateway adapter (takes priority over LiteLLM for supported providers)
        ("databricks:/my-endpoint", "string", GatewayAdapter),
        ("databricks:/my-endpoint", "list", GatewayAdapter),
        ("endpoints:/my-endpoint", "string", GatewayAdapter),
        # endpoints + list: Gateway rejects, falls through to LiteLLM
        ("endpoints:/my-endpoint", "list", LiteLLMAdapter),
        ("openai:/gpt-4", "string", GatewayAdapter),
        ("openai:/gpt-4", "list", GatewayAdapter),
        ("anthropic:/claude-3-5-sonnet-20241022", "string", GatewayAdapter),
        ("anthropic:/claude-3-5-sonnet-20241022", "list", GatewayAdapter),
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
    ("model_uri", "prompt_type"),
    [
        # Completely unknown providers
        ("unknown_provider:/some-model", "string"),
        ("unknown_provider:/some-model", "list"),
        # endpoints with list prompt — Gateway rejects, LiteLLM not available
        ("endpoints:/my-endpoint", "list"),
    ],
)
def test_get_adapter_unsupported_without_litellm(
    model_uri, prompt_type, string_prompt, list_prompt
):
    prompt = string_prompt if prompt_type == "string" else list_prompt
    with mock.patch(
        "mlflow.genai.judges.adapters.litellm_adapter._is_litellm_available",
        return_value=False,
    ):
        with pytest.raises(
            MlflowException, match=f"No suitable adapter found for model_uri='{model_uri}'"
        ):
            get_adapter(model_uri, prompt)
