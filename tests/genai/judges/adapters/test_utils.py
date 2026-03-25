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
        # Gateway adapter
        ("openai:/gpt-4", "string", GatewayAdapter),
        ("anthropic:/claude-3-5-sonnet-20241022", "string", GatewayAdapter),
    ],
)
def test_get_adapter(model_uri, prompt_type, expected_adapter, string_prompt, list_prompt):
    prompt = string_prompt if prompt_type == "string" else list_prompt
    adapter = get_adapter(model_uri, prompt)
    assert isinstance(adapter, expected_adapter)


def test_get_adapter_gateway_with_list(list_prompt):
    adapter = get_adapter("openai:/gpt-4", list_prompt)
    assert isinstance(adapter, GatewayAdapter)


@pytest.mark.parametrize(
    ("model_uri", "prompt_type"),
    [
        ("vertex_ai:/gemini-pro", "list"),
        ("vertex_ai:/gemini-pro", "string"),
    ],
)
def test_get_adapter_unsupported_provider(model_uri, prompt_type, string_prompt, list_prompt):
    prompt = string_prompt if prompt_type == "string" else list_prompt
    with pytest.raises(
        MlflowException, match=f"No suitable adapter found for model_uri='{model_uri}'"
    ):
        get_adapter(model_uri, prompt)
