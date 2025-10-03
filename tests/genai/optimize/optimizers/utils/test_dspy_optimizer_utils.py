from unittest.mock import patch

import pytest

pytest.importorskip("dspy", minversion="2.6.0")

from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.optimizers.utils.dspy_optimizer_utils import (
    format_dspy_prompt,
    parse_model_name,
)


@pytest.mark.parametrize("convert_to_single_text", [True, False])
def test_format_dspy_prompt(convert_to_single_text):
    import dspy

    mock_program = dspy.Predict("input_text, language -> translation")

    with dspy.context(adapter=dspy.JSONAdapter()):
        with patch("dspy.JSONAdapter.format") as mock_format:
            mock_format.return_value = [
                {"role": "system", "content": "You are a translator"},
                {"role": "user", "content": "Input Text: {{input_text}}, Language: {{language}}"},
            ]
            result = format_dspy_prompt(mock_program, convert_to_single_text)

    if convert_to_single_text:
        expected = (
            "<system>\nYou are a translator\n</system>\n\n"
            "<user>\nInput Text: {{input_text}}, Language: {{language}}\n</user>"
        )
    else:
        expected = [
            {"role": "system", "content": "You are a translator"},
            {"role": "user", "content": "Input Text: {{input_text}}, Language: {{language}}"},
        ]

    assert result == expected
    mock_format.assert_called_once()


def test_parse_model_name():
    # Test URI format conversion
    assert parse_model_name("openai:/gpt-4") == "openai/gpt-4"
    assert parse_model_name("anthropic:/claude-3") == "anthropic/claude-3"
    assert parse_model_name("mistral:/mistral-7b") == "mistral/mistral-7b"

    # Test that already formatted names are unchanged
    assert parse_model_name("openai/gpt-4") == "openai/gpt-4"
    assert parse_model_name("anthropic/claude-3") == "anthropic/claude-3"

    # Test invalid formats raise errors
    with pytest.raises(MlflowException, match="Invalid model name format"):
        parse_model_name("invalid-model-name")

    with pytest.raises(MlflowException, match="Model name cannot be empty"):
        parse_model_name("")

    with pytest.raises(MlflowException, match="Invalid model name format"):
        parse_model_name("openai:")

    with pytest.raises(MlflowException, match="Invalid model name format"):
        parse_model_name("openai/")
