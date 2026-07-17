from unittest.mock import Mock, patch

import pytest

from mlflow.genai.scorers.deepeval.models import MlflowDeepEvalLLM, create_deepeval_model


@pytest.fixture
def mock_call_chat_completions():
    with patch(
        "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.call_chat_completions",
    ) as mock:
        result = Mock()
        result.output = "Test output"
        mock.return_value = result
        yield mock


def test_databricks_model_generate(mock_call_chat_completions):
    model = create_deepeval_model("databricks")
    assert isinstance(model, MlflowDeepEvalLLM)
    result = model.generate("Test prompt")
    assert result == "Test output"
    mock_call_chat_completions.assert_called_once()


def test_create_deepeval_model_gateway_uses_native_provider():
    with patch("mlflow.genai.scorers.llm_backend._get_provider_instance") as mock_get_provider:
        model = create_deepeval_model("gateway:/my-endpoint")
    mock_get_provider.assert_called_once()

    assert isinstance(model, MlflowDeepEvalLLM)
    assert model.get_model_name() == "gateway/my-endpoint"


def test_create_deepeval_model_supported_provider_uses_native(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = create_deepeval_model("openai:/gpt-4")
    assert isinstance(model, MlflowDeepEvalLLM)
    assert model.get_model_name() == "openai/gpt-4"


def test_native_model_passes_model_kwargs(mock_call_chat_completions):
    model = create_deepeval_model("databricks", model_kwargs={"temperature": 0.2})
    assert isinstance(model, MlflowDeepEvalLLM)
    assert model._model_kwargs == {"temperature": 0.2}


def test_native_model_forwards_kwargs_to_complete_prompt():
    with patch(
        "mlflow.genai.scorers.llm_backend.ScorerLLMClient.complete_prompt",
        return_value="output",
    ) as mock_complete:
        model = create_deepeval_model(
            "databricks", model_kwargs={"temperature": 0.0, "max_tokens": 512}
        )
        model.generate("test prompt")
        mock_complete.assert_called_once_with("test prompt", temperature=0.0, max_tokens=512)


def test_litellm_model_merges_model_kwargs(monkeypatch):
    monkeypatch.setenv("UNSUPPORTED_API_KEY", "test-key")
    with patch("mlflow.genai.scorers.llm_backend._get_provider_instance", side_effect=Exception):
        model = create_deepeval_model(
            "unsupported:/some-model",
            model_kwargs={"temperature": 0.5, "max_tokens": 1024},
        )
    from deepeval.models import LiteLLMModel

    assert isinstance(model, LiteLLMModel)
    assert model.generation_kwargs["drop_params"] is True
    assert model.generation_kwargs["max_tokens"] == 1024
    assert model.temperature == 0.5


def test_create_deepeval_model_no_kwargs_default():
    with patch(
        "mlflow.genai.scorers.llm_backend.ScorerLLMClient.complete_prompt",
        return_value="output",
    ) as mock_complete:
        model = create_deepeval_model("databricks")
        model.generate("test prompt")
        mock_complete.assert_called_once_with("test prompt")


def test_create_deepeval_model_does_not_mutate_input_dict():
    kwargs = {"temperature": 0.5, "max_tokens": 100}
    with patch(
        "mlflow.genai.scorers.llm_backend._get_provider_instance",
        side_effect=Exception,
    ):
        create_deepeval_model("unsupported:/model", model_kwargs=kwargs)
    assert "temperature" in kwargs
