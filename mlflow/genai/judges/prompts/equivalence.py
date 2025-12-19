from mlflow.genai.prompts.utils import format_prompt

# NB: User-facing name for the equivalence assessment.
EQUIVALENCE_FEEDBACK_NAME = "equivalence"


EQUIVALENCE_PROMPT_INSTRUCTIONS = """\
Compare the following actual output against the expected output. You must determine whether they \
are semantically equivalent or convey the same meaning, and if the output format matches the \
expected format (e.g., JSON structure, list format, sentence structure).

<actual_output>{{output}}</actual_output>
<expected_output>{{expected_output}}</expected_output>\
"""

EQUIVALENCE_PROMPT_OUTPUT = """

Please indicate whether the actual output is equivalent to the expected output using only the following json format. Do not use any markdown formatting or output additional lines.
{
  "rationale": "Reason for the assessment. Explain whether the outputs are semantically equivalent and whether the format matches. Start each rationale with `Let's think step by step`",
  "result": "yes|no"
}\
"""  # noqa: E501

EQUIVALENCE_PROMPT = EQUIVALENCE_PROMPT_INSTRUCTIONS + EQUIVALENCE_PROMPT_OUTPUT


def get_prompt(
    output: str,
    expected_output: str,
) -> str:
    """Generate output equivalence evaluation prompt.

    Args:
        output: The actual output to evaluate
        expected_output: The expected output to compare against

    Returns:
        Formatted prompt string
    """
    return format_prompt(
        EQUIVALENCE_PROMPT,
        output=output,
        expected_output=expected_output,
    )
