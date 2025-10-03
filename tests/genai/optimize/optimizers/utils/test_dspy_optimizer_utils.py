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


@pytest.mark.parametrize(
    ("input_name", "expected_output"),
    [
        # URI format conversion
        ("openai:/gpt-4", "openai/gpt-4"),
        ("anthropic:/claude-3", "anthropic/claude-3"),
        ("mistral:/mistral-7b", "mistral/mistral-7b"),
        # Already formatted names are unchanged
        ("openai/gpt-4", "openai/gpt-4"),
        ("anthropic/claude-3", "anthropic/claude-3"),
    ],
)
def test_parse_model_name_valid(input_name, expected_output):
    assert parse_model_name(input_name) == expected_output


@pytest.mark.parametrize(
    ("input_name", "error_match"),
    [
        ("invalid-model-name", "Invalid model name format"),
        ("", "Model name cannot be empty"),
        ("openai:", "Invalid model name format"),
        ("openai/", "Invalid model name format"),
    ],
)
def test_parse_model_name_invalid(input_name, error_match):
    with pytest.raises(MlflowException, match=error_match):
        parse_model_name(input_name)
