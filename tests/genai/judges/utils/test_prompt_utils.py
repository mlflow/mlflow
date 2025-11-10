import pytest

from mlflow.genai.judges.base import Judge
from mlflow.genai.judges.utils.prompt_utils import add_output_format_instructions
from mlflow.genai.prompts.utils import format_prompt


def test_add_output_format_instructions():
    output_fields = Judge.get_output_fields()

    simple_prompt = "Evaluate this response"
    formatted = add_output_format_instructions(simple_prompt, output_fields=output_fields)

    assert simple_prompt in formatted
    assert "JSON format" in formatted
    assert '"result"' in formatted
    assert '"rationale"' in formatted
    assert "no markdown" in formatted.lower()
    assert "The evaluation rating/result" in formatted
    assert "Detailed explanation for the evaluation" in formatted

    complex_prompt = "This is a multi-line\nprompt with various\ninstruction details"
    formatted = add_output_format_instructions(complex_prompt, output_fields=output_fields)

    assert complex_prompt in formatted
    assert formatted.startswith(complex_prompt)
    assert formatted.endswith("}")

    assert formatted.index(complex_prompt) < formatted.index("JSON format")
    assert formatted.index(complex_prompt) < formatted.index('"result"')
    assert formatted.index(complex_prompt) < formatted.index('"rationale"')


@pytest.mark.parametrize(
    ("prompt_template", "values", "expected"),
    [
        # Test with Unicode escape-like sequences
        (
            "User input: {{ user_text }}",
            {"user_text": r"Path is C:\users\john"},
            r"User input: Path is C:\users\john",
        ),
        # Test with newlines and tabs
        (
            "Data: {{ data }}",
            {"data": "Line1\\nLine2\\tTabbed"},
            "Data: Line1\\nLine2\\tTabbed",
        ),
        # Test with multiple variables
        (
            "Path: {{ path }}, Command: {{ cmd }}",
            {"path": r"C:\temp", "cmd": r"echo \u0041"},
            r"Path: C:\temp, Command: echo \u0041",
        ),
    ],
)
def test_format_prompt_with_backslashes(
    prompt_template: str, values: dict[str, str], expected: str
) -> None:
    result = format_prompt(prompt_template, **values)
    assert result == expected
