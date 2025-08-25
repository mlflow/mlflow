from typing import Any

from mlflow.genai.judges.utils import format_prompt

# NB: User-facing name for the is_grounded assessment.
GROUNDEDNESS_FEEDBACK_NAME = "groundedness"


GROUNDEDNESS_PROMPT = """\
Consider the following claim and document. You must determine whether claim is supported by the document. Do not focus on the correctness or completeness of the claim. Do not make assumptions, approximations, or bring in external knowledge.

<claim>
  <question>{{input}}</question>
  <answer>{{output}}</answer>
</claim>
<document>{{retrieval_context}}</document>

Please indicate whether each statement in the claim is supported by the document using the json format:
{
  "rationale": "Reason for the assessment. If the claim is not fully supported by the document, state which parts are not supported. Start each rationale with `Let's think step by step`",
  "result": "yes|no"
}
Do not output additional lines.\
"""  # noqa: E501


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
