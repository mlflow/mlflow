from mlflow.genai.prompts.utils import format_prompt

# NB: User-facing name for the is_correct assessment.
CORRECTNESS_FEEDBACK_NAME = "correctness"

CORRECTNESS_BASE_INSTRUCTIONS = """\
Consider the following question, claim and document. You must determine whether the claim is \
supported by the document in the context of the question. Do not focus on the correctness or \
completeness of the claim. Do not make assumptions, approximations, or bring in external \
knowledge."""

CORRECTNESS_PROMPT_INSTRUCTIONS = f"""{CORRECTNESS_BASE_INSTRUCTIONS}

<question>{{{{input}}}}</question>
<claim>{{{{ground_truth}}}}</claim>
<document>{{{{input}}}} - {{{{output}}}}</document>\
"""

CORRECTNESS_PROMPT_OUTPUT = """

Please indicate whether each statement in the claim is supported by the document in the context of the question using only the following json format. Do not use any markdown formatting or output additional lines.
{
  "rationale": "Reason for the assessment. If the claim is not fully supported by the document in the context of the question, state which parts are not supported. Start each rationale with `Let's think step by step`",
  "result": "yes|no"
}\
"""  # noqa: E501

CORRECTNESS_PROMPT = CORRECTNESS_PROMPT_INSTRUCTIONS + CORRECTNESS_PROMPT_OUTPUT

# This suffix is only shown when expected facts are provided to squeeze out better judge quality.
CORRECTNESS_PROMPT_SUFFIX = """

If the claim is fully supported by the document in the context of the question, you must say "The response is correct" in the rationale. If the claim is not fully supported by the document in the context of the question, you must say "The response is not correct"."""  # noqa: E501

CORRECTNESS_TRACE_FALLBACK = f"""{CORRECTNESS_BASE_INSTRUCTIONS}

Extract the question, expected response/facts, and actual response from the following trace and \
evaluate correctness:
<trace>
{{{{trace}}}}
</trace>
"""


def get_prompt(
    request: str,
    response: str,
    expected_response: str | None = None,
    expected_facts: list[str] | None = None,
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
        ground_truth = "\n- ".join([""] + expected_facts) if expected_facts else ""
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


def get_trace_fallback_prompt() -> str:
    """Get the trace-based fallback prompt for correctness evaluation.

    Returns:
        Complete prompt for trace-based correctness evaluation with trace placeholder.
    """
    return CORRECTNESS_TRACE_FALLBACK + CORRECTNESS_PROMPT_OUTPUT
