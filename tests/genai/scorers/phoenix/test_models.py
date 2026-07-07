from unittest.mock import Mock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.phoenix.models import (
    MlflowPhoenixModel,
    create_phoenix_model,
)


@pytest.fixture
def mock_call_chat_completions():
    with patch(
        "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.call_chat_completions",
    ) as mock:
        result = Mock()
        result.output = "Test output"
        mock.return_value = result
        yield mock


def test_databricks_phoenix_model_call(mock_call_chat_completions):
    from mlflow.genai.scorers.llm_backend import ScorerLLMClient

    backend = ScorerLLMClient("databricks")
    model = MlflowPhoenixModel(backend)
    result = model("Test prompt")

    assert result == "Test output"
    mock_call_chat_completions.assert_called_once_with(
        user_prompt="Test prompt",
        system_prompt="",
        model="databricks",
    )


def test_databricks_phoenix_model_get_model_name():
    from mlflow.genai.scorers.llm_backend import ScorerLLMClient

    backend = ScorerLLMClient("databricks")
    model = MlflowPhoenixModel(backend)
    assert model.get_model_name() == "databricks"


def test_create_phoenix_model_databricks():
    model = create_phoenix_model("databricks")
    assert isinstance(model, MlflowPhoenixModel)
    assert model.get_model_name() == "databricks"


def test_create_phoenix_model_databricks_endpoint():
    model = create_phoenix_model("databricks:/my-endpoint")
    assert isinstance(model, MlflowPhoenixModel)
    assert model.get_model_name() == "databricks/my-endpoint"


def test_create_phoenix_model_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = create_phoenix_model("openai:/gpt-4")
    assert isinstance(model, MlflowPhoenixModel)


def test_create_phoenix_model_invalid_format():
    with pytest.raises(MlflowException, match="Malformed model uri"):
        create_phoenix_model("gpt-4")


def test_create_phoenix_model_gateway_uses_native_provider():
    with patch(
        "mlflow.genai.scorers.llm_backend._get_provider_instance",
    ) as mock_get_provider:
        model = create_phoenix_model("gateway:/my-endpoint")

    mock_get_provider.assert_called_once()
    assert isinstance(model, MlflowPhoenixModel)
    assert model.get_model_name() == "gateway/my-endpoint"


def test_gateway_phoenix_model_call():
    from mlflow.genai.scorers.llm_backend import ScorerLLMClient

    with patch("mlflow.genai.scorers.llm_backend._get_provider_instance") as mock_get_provider:
        backend = ScorerLLMClient("openai:/gpt-4")
    mock_get_provider.assert_called_once()
    model = MlflowPhoenixModel(backend)

    with patch(
        "mlflow.genai.scorers.llm_backend._call_llm_provider_api",
        return_value="The answer is 42.",
    ) as mock_call:
        result = model("What is the answer?")

    assert result == "The answer is 42."
    mock_call.assert_called_once()


def test_gateway_phoenix_model_converts_non_string_prompt():
    from mlflow.genai.scorers.llm_backend import ScorerLLMClient

    with patch("mlflow.genai.scorers.llm_backend._get_provider_instance") as mock_get_provider:
        backend = ScorerLLMClient("openai:/gpt-4")
    mock_get_provider.assert_called_once()
    model = MlflowPhoenixModel(backend)

    with patch(
        "mlflow.genai.scorers.llm_backend._call_llm_provider_api",
        return_value="response",
    ) as mock_call:
        model(12345)

    mock_call.assert_called_once()


def test_gateway_phoenix_model_get_model_name():
    from mlflow.genai.scorers.llm_backend import ScorerLLMClient

    with patch("mlflow.genai.scorers.llm_backend._get_provider_instance") as mock_get_provider:
        backend = ScorerLLMClient("anthropic:/claude-3")
    mock_get_provider.assert_called_once()
    model = MlflowPhoenixModel(backend)
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
    assert isinstance(model, MlflowPhoenixModel)


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
