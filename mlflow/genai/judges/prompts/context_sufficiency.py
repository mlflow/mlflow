from typing import Any

from mlflow.genai.prompts.utils import format_prompt

# NB: User-facing name for the is_context_sufficient assessment.
CONTEXT_SUFFICIENCY_FEEDBACK_NAME = "context_sufficiency"


CONTEXT_SUFFICIENCY_PROMPT_INSTRUCTIONS = """\
Consider the following claim and document. You must determine whether claim is supported by the \
document. Do not focus on the correctness or completeness of the claim. Do not make assumptions, \
approximations, or bring in external knowledge.

<claim>
  <question>{{input}}</question>
  <answer>{{ground_truth}}</answer>
</claim>
<document>{{retrieval_context}}</document>\
"""

CONTEXT_SUFFICIENCY_PROMPT_OUTPUT = """

Please indicate whether each statement in the claim is supported by the document using only the following json format. Do not use any markdown formatting or output additional lines.
{
  "rationale": "Reason for the assessment. If the claim is not fully supported by the document, state which parts are not supported. Start each rationale with `Let's think step by step`",
  "result": "yes|no"
}\
"""  # noqa: E501

CONTEXT_SUFFICIENCY_PROMPT = (
    CONTEXT_SUFFICIENCY_PROMPT_INSTRUCTIONS + CONTEXT_SUFFICIENCY_PROMPT_OUTPUT
)


def get_prompt(
    request: str,
    context: Any,
    expected_response: str | None = None,
    expected_facts: list[str] | None = None,
) -> str:
    """Generate context sufficiency evaluation prompt.

    Args:
        request: The input question/request
        context: The retrieval context to evaluate sufficiency of
        expected_response: Expected response (optional)
        expected_facts: List of expected facts (optional, converted to expected_response)

    Returns:
        Formatted prompt string
    """
    # Convert expected_facts to expected_response format if provided
    ground_truth = expected_response
    if expected_facts and not expected_response:
        ground_truth = _convert_expected_facts_to_expected_response(expected_facts)
    elif not ground_truth:
        ground_truth = ""

    return format_prompt(
        CONTEXT_SUFFICIENCY_PROMPT,
        input=request,
        ground_truth=ground_truth,
        retrieval_context=str(context),
    )


def _convert_expected_facts_to_expected_response(expected_facts: list[str] | None) -> str:
    if not expected_facts:
        return ""
    rendered_facts = "\n    - ".join([""] + expected_facts)
    return f"  {rendered_facts.strip()}" if rendered_facts else ""
