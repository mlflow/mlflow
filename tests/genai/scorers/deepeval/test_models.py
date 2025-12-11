from unittest.mock import Mock, patch

import pytest

from mlflow.genai.scorers.deepeval.models import DatabricksDeepEvalLLM


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
