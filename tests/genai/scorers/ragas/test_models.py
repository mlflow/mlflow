from unittest.mock import Mock, patch

import pytest

from mlflow.genai.scorers.ragas.models import DatabricksRagasLLM


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
