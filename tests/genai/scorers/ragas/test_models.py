from unittest.mock import Mock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.ragas.models import DatabricksRagasLLM, create_ragas_model


@pytest.fixture
def mock_call_chat_completions():
    with patch("mlflow.genai.scorers.ragas.models.call_chat_completions") as mock:
        result = Mock()
        result.output = "Test output"
        mock.return_value = result
        yield mock


def test_databricks_ragas_llm_generate_text(mock_call_chat_completions):
    llm = DatabricksRagasLLM()
    result = llm.generate_text(prompt="Test prompt")

    assert result == "Test output"
    mock_call_chat_completions.assert_called_once_with(
        user_prompt="Test prompt",
        system_prompt="",
    )


def test_create_ragas_model_databricks():
    model = create_ragas_model("databricks")
    assert model.__class__.__name__ == "DatabricksRagasLLM"


def test_create_ragas_model_databricks_serving_endpoint():
    model = create_ragas_model("databricks:/my-endpoint")
    assert model.__class__.__name__ == "DatabricksServingEndpointRagasLLM"


def test_create_ragas_model_openai():
    model = create_ragas_model("openai:/gpt-4")
    assert model.__class__.__name__ == "MlflowLiteLLM"


def test_create_ragas_model_with_provider_no_slash():
    model = create_ragas_model("openai:gpt-4")
    assert model.__class__.__name__ == "MlflowLiteLLM"


def test_create_ragas_model_rejects_model_name_only():
    with pytest.raises(MlflowException, match="Invalid model_uri format"):
        create_ragas_model("gpt-4")
