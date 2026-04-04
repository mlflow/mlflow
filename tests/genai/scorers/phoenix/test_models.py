from unittest.mock import Mock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.phoenix.models import (
    DatabricksPhoenixModel,
    GatewayPhoenixModel,
    create_phoenix_model,
)


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


def test_create_phoenix_model_gateway_uses_native_provider():
    with patch(
        "mlflow.genai.scorers.phoenix.models._get_provider_instance",
    ):
        model = create_phoenix_model("gateway:/my-endpoint")

    assert isinstance(model, GatewayPhoenixModel)
    assert model.get_model_name() == "gateway/my-endpoint"


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


def test_create_phoenix_model_falls_back_to_litellm_for_unsupported_provider():
    mock_litellm_cls = Mock()
    with patch("phoenix.evals.LiteLLMModel", mock_litellm_cls):
        model = create_phoenix_model("some_unknown:/model")

    assert model is mock_litellm_cls.return_value
    mock_litellm_cls.assert_called_once_with(
        model="some_unknown/model",
        model_kwargs={"drop_params": True},
    )


@pytest.mark.parametrize("provider", ["cohere", "mosaicml", "palm"])
def test_create_phoenix_model_registered_but_unsupported_falls_back_to_litellm(provider):
    mock_litellm_cls = Mock()
    with patch("phoenix.evals.LiteLLMModel", mock_litellm_cls):
        model = create_phoenix_model(f"{provider}:/my-model")

    assert model is mock_litellm_cls.return_value
    mock_litellm_cls.assert_called_once_with(
        model=f"{provider}/my-model",
        model_kwargs={"drop_params": True},
    )
