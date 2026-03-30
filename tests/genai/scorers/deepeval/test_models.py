from unittest.mock import Mock, patch

import pytest

from mlflow.genai.scorers.deepeval.models import DatabricksDeepEvalLLM, create_deepeval_model
from mlflow.genai.utils.gateway_utils import GatewayLiteLLMConfig


@pytest.fixture
def mock_call_chat_completions():
    with patch("mlflow.genai.scorers.deepeval.models.call_chat_completions") as mock:
        result = Mock()
        result.output = "Test output"
        mock.return_value = result
        yield mock


def test_databricks_deepeval_llm_generate(mock_call_chat_completions):
    llm = DatabricksDeepEvalLLM()
    result = llm.generate("Test prompt")

    assert result == "Test output"
    mock_call_chat_completions.assert_called_once_with(
        user_prompt="Test prompt",
        system_prompt="",
    )


def test_create_deepeval_model_gateway():
    mock_config = GatewayLiteLLMConfig(
        api_base="http://localhost:5000/gateway/mlflow/v1/",
        api_key="mlflow-gateway-auth",
        model="openai/my-endpoint",
        extra_headers=None,
    )
    with patch(
        "mlflow.genai.scorers.deepeval.models.get_gateway_litellm_config",
        return_value=mock_config,
    ) as mock_get_config:
        model = create_deepeval_model("gateway:/my-endpoint")

    mock_get_config.assert_called_once_with("my-endpoint")
    assert model.__class__.__name__ == "LiteLLMModel"


def test_create_deepeval_model_gateway_sets_api_base_and_key():
    mock_config = GatewayLiteLLMConfig(
        api_base="http://localhost:5000/gateway/mlflow/v1/",
        api_key="mlflow-gateway-auth",
        model="openai/my-endpoint",
        extra_headers=None,
    )
    with (
        patch(
            "mlflow.genai.scorers.deepeval.models.get_gateway_litellm_config",
            return_value=mock_config,
        ),
        patch("mlflow.genai.scorers.deepeval.models.LiteLLMModel") as mock_litellm_model,
    ):
        create_deepeval_model("gateway:/my-endpoint")

    mock_litellm_model.assert_called_once_with(
        model="openai/my-endpoint",
        base_url="http://localhost:5000/gateway/mlflow/v1/",
        api_key="mlflow-gateway-auth",
        generation_kwargs={"drop_params": True},
    )


@pytest.mark.parametrize(
    ("model_uri", "expected_model_arg"),
    [
        ("openai:/gpt-4", "openai/gpt-4"),
        ("databricks:/my-endpoint", "databricks/my-endpoint"),
    ],
)
def test_create_deepeval_model_non_gateway(model_uri, expected_model_arg):
    with patch("mlflow.genai.scorers.deepeval.models.LiteLLMModel") as mock_litellm_model:
        create_deepeval_model(model_uri)

    mock_litellm_model.assert_called_once_with(
        model=expected_model_arg,
        generation_kwargs={"drop_params": True},
    )
