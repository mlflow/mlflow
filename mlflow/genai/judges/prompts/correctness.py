from typing import List, Optional

from mlflow.genai.judges.utils import format_prompt

# NB: User-facing name for the is_correct assessment.
CORRECTNESS_FEEDBACK_NAME = "correctness"


CORRECTNESS_PROMPT = """\
Consider the following question, claim and document. You must determine whether the claim is supported by the document in the context of the question. Do not focus on the correctness or completeness of the claim. Do not make assumptions, approximations, or bring in external knowledge.

<question>{{input}}</question>
<claim>{{ground_truth}}</claim>
<document>{{input}} - {{output}}</document>

Please indicate whether each statement in the claim is supported by the document in the context of the question using the json format:
{
  "rationale": "Reason for the assessment. If the claim is not fully supported by the document in the context of the question, state which parts are not supported. Start each rationale with `Let's think step by step`",
  "result": "yes|no"
}\
"""

# This suffix is only shown when expected facts are provided to squeeze out better judge quality.
CORRECTNESS_PROMPT_SUFFIX = """

If the claim is fully supported by the document in the context of the question, you must say "The response is correct" in the rationale. If the claim is not fully supported by the document in the context of the question, you must say "The response is not correct"."""


def convert_expected_facts_to_expected_response(expected_facts: Optional[List[str]]) -> str:
    """Convert expected facts list to a formatted string for the correctness prompt."""
    if not expected_facts:
        return ""
    return "\n- ".join([""] + expected_facts)


def get_prompt(
    request: str,
    response: str,
    expected_response: Optional[str] = None,
    expected_facts: Optional[List[str]] = None
) -> str:
    """Generate correctness evaluation prompt.

    Args:
        request: The input question/request
        response: The actual response to evaluate
        expected_response: Expected response (optional)
        expected_facts: List of expected facts (optional, converted to expected_response)

    Returns:
        Formatted prompt string
    """
    # Convert expected_facts to expected_response format if provided
    ground_truth = expected_response
    if expected_facts and not expected_response:
        ground_truth = convert_expected_facts_to_expected_response(expected_facts)
    elif not ground_truth:
        ground_truth = ""

    prompt = format_prompt(
        CORRECTNESS_PROMPT,
        input=request,
        output=response,
        ground_truth=ground_truth,
    )

    # Add suffix when expected facts are provided (not expected_response)
    if expected_facts and not expected_response:
        prompt += CORRECTNESS_PROMPT_SUFFIX

    return prompt