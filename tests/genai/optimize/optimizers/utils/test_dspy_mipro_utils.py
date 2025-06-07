from unittest.mock import patch

import pytest

pytest.importorskip("dspy", minversion="2.6.0")

from mlflow.genai.optimize.optimizers.utils.dspy_mipro_utils import format_optimized_prompt


def test_format_optimized_prompt():
    import dspy

    mock_program = dspy.Predict("input_text, language -> translation")
    input_fields = {"input_text": str, "language": str}

    with dspy.context(adapter=dspy.JSONAdapter()):
        with patch("dspy.JSONAdapter.format") as mock_format:
            mock_format.return_value = [
                {"role": "system", "content": "You are a translator"},
                {"role": "user", "content": "Input Text: {{input_text}}, Language: {{language}}"},
            ]
            result = format_optimized_prompt(mock_program, input_fields)

    expected = (
        "<system>\nYou are a translator\n</system>\n\n"
        "<user>\nInput Text: {{input_text}}, Language: {{language}}\n</user>"
    )

    assert result == expected
    mock_format.assert_called_once()
