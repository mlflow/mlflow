from mlflow.genai.prompts.utils import format_prompt

# NB: User-facing name for the is_context_relevant assessment.
RELEVANCE_TO_QUERY_ASSESSMENT_NAME = "relevance_to_context"

RELEVANCE_TO_QUERY_BASE_INSTRUCTIONS = """\
Consider the following question and answer. You must determine whether the answer provides \
information that is (fully or partially) relevant to the question. Do not focus on the correctness \
or completeness of the answer. Do not make assumptions, approximations, or bring in external \
knowledge."""

RELEVANCE_TO_QUERY_PROMPT_INSTRUCTIONS = f"""{RELEVANCE_TO_QUERY_BASE_INSTRUCTIONS}

<question>{{{{input}}}}</question>
<answer>{{{{output}}}}</answer>\
"""

RELEVANCE_TO_QUERY_PROMPT_OUTPUT = """

Please indicate whether the answer contains information that is relevant to the question using only the following json format. Do not use any markdown formatting or output additional lines.
{
  "rationale": "Reason for the assessment. If the answer does not provide any information that is relevant to the question then state which parts are not relevant. Start each rationale with `Let's think step by step`",
  "result": "yes|no"
}
`result` must only be `yes` or `no`."""  # noqa: E501

RELEVANCE_TO_QUERY_PROMPT = (
    RELEVANCE_TO_QUERY_PROMPT_INSTRUCTIONS + RELEVANCE_TO_QUERY_PROMPT_OUTPUT
)

# Trace-based fallback template when extraction fails
RELEVANCE_TO_QUERY_TRACE_FALLBACK_INSTRUCTIONS = f"""{RELEVANCE_TO_QUERY_BASE_INSTRUCTIONS}

Extract the question and answer from the trace {{{{{{ trace }}}}}} and evaluate relevance.
"""


def get_prompt(request: str, context: str) -> str:
    return format_prompt(
        RELEVANCE_TO_QUERY_PROMPT,
        input=request,
        output=context,
    )


def get_trace_fallback_prompt() -> str:
    """Get the trace-based fallback prompt for relevance evaluation."""
    return RELEVANCE_TO_QUERY_TRACE_FALLBACK_INSTRUCTIONS + RELEVANCE_TO_QUERY_PROMPT_OUTPUT
