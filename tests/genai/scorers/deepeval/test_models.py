from unittest.mock import Mock, patch

import pytest

from mlflow.genai.scorers.deepeval.models import DatabricksDeepEvalLLM, create_deepeval_model


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


def test_create_deepeval_model_gateway_uses_native_provider():
    from mlflow.genai.scorers.deepeval.models import GatewayDeepEvalLLM

    with patch("mlflow.genai.scorers.deepeval.models._get_provider_instance"):
        model = create_deepeval_model("gateway:/my-endpoint")

    assert isinstance(model, GatewayDeepEvalLLM)
    assert model.get_model_name() == "gateway/my-endpoint"


def test_create_deepeval_model_supported_provider_uses_gateway(monkeypatch):
    from mlflow.genai.scorers.deepeval.models import GatewayDeepEvalLLM

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = create_deepeval_model("openai:/gpt-4")
    assert isinstance(model, GatewayDeepEvalLLM)
    assert model.get_model_name() == "openai/gpt-4"
