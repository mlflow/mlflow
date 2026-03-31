from unittest.mock import Mock, patch

import phoenix.evals as phoenix_evals
import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.phoenix.models import (
    DatabricksPhoenixModel,
    GatewayPhoenixModel,
    create_phoenix_model,
)
from mlflow.genai.utils.gateway_utils import GatewayLiteLLMConfig


@pytest.fixture
def mock_call_chat_completions():
    with patch("mlflow.genai.scorers.phoenix.models.call_chat_completions") as mock:
        result = Mock()
        result.output = "Test output"
        mock.return_value = result
        yield mock


def test_databricks_phoenix_model_call(mock_call_chat_completions):
    model = DatabricksPhoenixModel()
    result = model("Test prompt")

    assert result == "Test output"
    mock_call_chat_completions.assert_called_once_with(
        user_prompt="Test prompt",
        system_prompt="",
    )


def test_databricks_phoenix_model_get_model_name():
    model = DatabricksPhoenixModel()
    assert model.get_model_name() == "databricks"


def test_create_phoenix_model_databricks():
    model = create_phoenix_model("databricks")
    assert isinstance(model, DatabricksPhoenixModel)
    assert model.get_model_name() == "databricks"


def test_create_phoenix_model_databricks_endpoint():
    model = create_phoenix_model("databricks:/my-endpoint")
    assert isinstance(model, GatewayPhoenixModel)
    assert model.get_model_name() == "databricks/my-endpoint"


def test_create_phoenix_model_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = create_phoenix_model("openai:/gpt-4")
    assert isinstance(model, GatewayPhoenixModel)


def test_create_phoenix_model_invalid_format():
    with pytest.raises(MlflowException, match="Malformed model uri"):
        create_phoenix_model("gpt-4")


def test_create_phoenix_model_gateway(monkeypatch):
    mock_config = GatewayLiteLLMConfig(
        api_base="http://localhost:5000/gateway/mlflow/v1/",
        api_key="mlflow-gateway-auth",
        model="openai/my-endpoint",
        extra_headers=None,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "mlflow-gateway-auth")
    with patch(
        "mlflow.genai.scorers.phoenix.models.get_gateway_litellm_config",
        return_value=mock_config,
    ) as mock_get_config:
        model = create_phoenix_model("gateway:/my-endpoint")

    mock_get_config.assert_called_once_with("my-endpoint")
    assert isinstance(model, phoenix_evals.LiteLLMModel)
    assert model.model == "openai/my-endpoint"
    assert model.model_kwargs["api_base"] == "http://localhost:5000/gateway/mlflow/v1/"
    assert model.model_kwargs["api_key"] == "mlflow-gateway-auth"


def test_create_phoenix_model_gateway_sets_api_base_and_key():
    mock_config = GatewayLiteLLMConfig(
        api_base="http://localhost:5000/gateway/mlflow/v1/",
        api_key="mlflow-gateway-auth",
        model="openai/my-endpoint",
        extra_headers=None,
    )
    mock_litellm_model = Mock()
    with (
        patch(
            "mlflow.genai.scorers.phoenix.models.get_gateway_litellm_config",
            return_value=mock_config,
        ),
        patch("phoenix.evals.LiteLLMModel", mock_litellm_model),
    ):
        create_phoenix_model("gateway:/my-endpoint")

    mock_litellm_model.assert_called_once_with(
        model="openai/my-endpoint",
        model_kwargs={
            "api_base": "http://localhost:5000/gateway/mlflow/v1/",
            "api_key": "mlflow-gateway-auth",
            "drop_params": True,
        },
    )


def test_gateway_phoenix_model_call():
    model = GatewayPhoenixModel("openai", "gpt-4")

    with patch(
        "mlflow.genai.scorers.phoenix.models._call_llm_provider_api",
        return_value="The answer is 42.",
    ) as mock_call:
        result = model("What is the answer?")

    assert result == "The answer is 42."
    mock_call.assert_called_once_with("openai", "gpt-4", input_data="What is the answer?")


def test_gateway_phoenix_model_converts_non_string_prompt():
    model = GatewayPhoenixModel("openai", "gpt-4")

    with patch(
        "mlflow.genai.scorers.phoenix.models._call_llm_provider_api",
        return_value="response",
    ) as mock_call:
        model(12345)

    mock_call.assert_called_once_with("openai", "gpt-4", input_data="12345")


def test_gateway_phoenix_model_get_model_name():
    model = GatewayPhoenixModel("anthropic", "claude-3")
    assert model.get_model_name() == "anthropic/claude-3"


@pytest.mark.parametrize(
    ("model_uri", "env_var"),
    [
        ("openai:/gpt-4", "OPENAI_API_KEY"),
        ("anthropic:/claude-3", "ANTHROPIC_API_KEY"),
    ],
)
def test_create_phoenix_model_uses_gateway_for_supported_providers(model_uri, env_var, monkeypatch):
    monkeypatch.setenv(env_var, "test-key")
    model = create_phoenix_model(model_uri)
    assert isinstance(model, GatewayPhoenixModel)
