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
