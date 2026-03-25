from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    DatabricksManagedJudgeAdapter,
)
from mlflow.genai.judges.adapters.gateway_adapter import GatewayAdapter
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
        # Gateway adapter for native providers
        ("openai:/gpt-4", "string", GatewayAdapter),
        ("openai:/gpt-4", "list", GatewayAdapter),
        ("anthropic:/claude-3-5-sonnet-20241022", "string", GatewayAdapter),
        ("anthropic:/claude-3-5-sonnet-20241022", "list", GatewayAdapter),
        ("gemini:/gemini-2.5-flash", "string", GatewayAdapter),
        ("mistral:/mistral-large", "string", GatewayAdapter),
        # endpoints with string prompt
        ("endpoints:/my-endpoint", "string", GatewayAdapter),
    ],
)
def test_get_adapter(model_uri, prompt_type, expected_adapter, string_prompt, list_prompt):
    prompt = string_prompt if prompt_type == "string" else list_prompt
    adapter = get_adapter(model_uri, prompt)
    assert isinstance(adapter, expected_adapter)


@pytest.mark.parametrize(
    ("model_uri", "prompt_type"),
    [
        # endpoints with list prompt is not supported
        ("endpoints:/my-endpoint", "list"),
        # Unsupported providers
        ("vertex_ai:/gemini-pro", "list"),
        ("bedrock:/anthropic.claude-3-5-sonnet-20241022-v2:0", "list"),
        ("bedrock:/anthropic.claude-3-5-sonnet-20241022-v2:0", "string"),
        ("cohere:/command-r", "string"),
    ],
)
def test_get_adapter_unsupported(model_uri, prompt_type, string_prompt, list_prompt):
    prompt = string_prompt if prompt_type == "string" else list_prompt
    with pytest.raises(
        MlflowException, match=f"No suitable adapter found for model_uri='{model_uri}'"
    ):
        get_adapter(model_uri, prompt)
