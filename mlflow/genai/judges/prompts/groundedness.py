from typing import Any

from mlflow.genai.prompts.utils import format_prompt

# NB: User-facing name for the is_grounded assessment.
GROUNDEDNESS_FEEDBACK_NAME = "groundedness"


GROUNDEDNESS_PROMPT_INSTRUCTIONS = """\
Consider the following claim and document. You must determine whether claim is supported by the \
document. Do not focus on the correctness or completeness of the claim. Do not make assumptions, \
approximations, or bring in external knowledge.

<claim>
  <question>{{input}}</question>
  <answer>{{output}}</answer>
</claim>
<document>{{retrieval_context}}</document>\
"""

GROUNDEDNESS_PROMPT_OUTPUT = """

Please indicate whether each statement in the claim is supported by the document using only the following json format. Do not use any markdown formatting or output additional lines.
{
  "rationale": "Reason for the assessment. If the claim is not fully supported by the document, state which parts are not supported. Start each rationale with `Let's think step by step`",
  "result": "yes|no"
}\
"""  # noqa: E501

GROUNDEDNESS_PROMPT = GROUNDEDNESS_PROMPT_INSTRUCTIONS + GROUNDEDNESS_PROMPT_OUTPUT


def get_prompt(request: str, response: str, context: Any) -> str:
    """Generate groundedness evaluation prompt.

    Args:
        request: The input question/request
        response: The response to evaluate for groundedness
        context: The retrieval context to check groundedness against

    Returns:
        Formatted prompt string
    """
    return format_prompt(
        GROUNDEDNESS_PROMPT,
        input=request,
        output=response,
        retrieval_context=str(context),
    )
