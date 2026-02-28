_DEFAULT_TRIAGE_SAMPLE_SIZE = 100
_MAX_SUMMARIES_FOR_CLUSTERING = 50
_MIN_FREQUENCY_THRESHOLD = 0.01
_MIN_CONFIDENCE = 75
_MIN_EXAMPLES = 2

# Truncation limits for trace summaries shown to the analysis LLM.
# These are generous — modern LLMs have large context windows, and
# aggressive truncation causes false-positive "truncation" issues.
_TRACE_IO_CHAR_LIMIT = 5000
_SPAN_IO_CHAR_LIMIT = 2000
_ERROR_CHAR_LIMIT = 1000
_TRIM_MARKER = " [..TRIMMED BY ANALYSIS TOOL]"

_DEFAULT_JUDGE_MODEL = "openai:/gpt-5-mini"
_DEFAULT_ANALYSIS_MODEL = "openai:/gpt-5"
_DEFAULT_SCORER_NAME = "_issue_discovery_judge"

_TEMPLATE_VARS = {
    "{{ trace }}",
    "{{ inputs }}",
    "{{ outputs }}",
    "{{ conversation }}",
    "{{ expectations }}",
}

# ---- Satisfaction scorer instructions ----

_SATISFACTION_INSTRUCTIONS_PREAMBLE = """\
Follow all the steps below VERY CAREFULLY AND PRECISELY to determine if the user's goals \
were achieved efficiently.

A goal is an outcome the user was trying to accomplish through their interaction with the AI \
assistant. A goal is NOT simply the topic of the {context_noun} or the specific question(s) the \
user asked! Correcting for an assistant's mistakes or shortcomings is also NOT a user goal. \
Goals should always be independent of the agent's behavior.\
"""

_SATISFACTION_INSTRUCTIONS_BODY = """
{extra_goal_context}\
Thoroughly analyze the {template_var} between a user and an AI assistant to identify if \
the user's goals were achieved.

1. First, determine what the user was trying to accomplish (identify all relevant goals)

2. Assess whether those goals were achieved efficiently by the assistant using the *user's* \
messages as the source of truth. If the user did not indicate dissatisfaction, express \
frustration, have to ask for unnecessary follow-up information or clarifications that should \
have been provided initially, have to rephrase their request unnecessarily, resolve confusion \
or inconsistency caused by a poor response from the assistant, encounter inconsistent or \
conflicting information from the assistant, or encounter repetitive or redundant responses \
from the assistant that were not requested explicitly, then consider the goals achieved \
efficiently. If you are unsure, then also consider the goals achieved efficiently.

3. If not achieved (or achieved poorly), identify ALL likely *user* expectations that were \
violated. An expectation is something the user expected the assistant to do or a property that \
the assistant should have exhibited.

**CRITICAL** - DO NOT:
- Include goals about correcting the assistant's mistakes as user goals
- Infer whether goals were achieved based on anything EXCEPT the user's messages
- Verify factual correctness UNLESS the user's messages indicate a potential issue
- Consider lack of acknowledgement at the end as an indication of failure
- Consider the user ending the {context_noun} as an indication of failure\
{extra_donts}

Return True if the user's goals were achieved efficiently, False otherwise.

In your rationale, explain:
- What the user wanted to achieve (list all goals)
- Whether they were achieved efficiently
- If not, list each violated expectation with the observable behavior that demonstrates the issue\
"""


def _build_satisfaction_instructions(*, use_conversation: bool) -> str:
    if use_conversation:
        preamble = _SATISFACTION_INSTRUCTIONS_PREAMBLE.format(context_noun="conversation")
        body = _SATISFACTION_INSTRUCTIONS_BODY.format(
            extra_goal_context=(
                " Agent responses may give users new "
                "information or context that leads to new goals, but these goals are driven "
                "by the user's knowledge, context, and motivations external to the assistant.\n\n"
            ),
            template_var="{{ conversation }}",
            context_noun="conversation",
            extra_donts=(
                "\n- Consider abrupt topic changes as failure "
                "unless preceding messages indicate unmet expectations"
                "\n- Interpret off-topic user messages as an indication of failure"
            ),
        )
    else:
        preamble = _SATISFACTION_INSTRUCTIONS_PREAMBLE.format(context_noun="interaction")
        body = _SATISFACTION_INSTRUCTIONS_BODY.format(
            extra_goal_context="\n",
            template_var="{{ trace }}",
            context_noun="interaction",
            extra_donts="",
        )
    return preamble + body


# ---- Deep analysis prompt ----

_DEEP_ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert at diagnosing AI application failures. "
    "Given enriched trace summaries with span-level detail, analyze each "
    "failing trace individually.\n\n"
    "IMPORTANT: Fields ending with '[..TRIMMED BY ANALYSIS TOOL]' were "
    "shortened for this analysis — do NOT treat this as evidence of "
    "truncation in the original application response.\n\n"
    "For each trace, identify:\n"
    "- The failure category (tool_error, hallucination, latency, "
    "incomplete_response, error_propagation, wrong_tool_use, "
    "context_loss, or other)\n"
    "- A brief failure summary\n"
    "- A root cause hypothesis based on the span evidence\n"
    "- Which spans are most relevant to the failure\n"
    "- Severity (1=minor, 3=moderate, 5=critical)"
)

# ---- Clustering prompt ----

_CLUSTERING_SYSTEM_PROMPT = (
    "You are an expert at analyzing AI application failures. "
    "Given per-trace analyses with failure categories and root causes, "
    "group them into at most {max_issues} distinct issue categories.\n\n"
    "For each issue provide:\n"
    "- A snake_case name\n"
    "- A clear description\n"
    "- The root cause\n"
    "- Indices of example traces from the input\n"
    "- A confidence score 0-100 indicating how confident you are this "
    "is a real, distinct issue (0=not confident, 50=moderate, "
    "75=highly confident, 100=certain). Be rigorous — only score "
    "75+ if multiple traces clearly demonstrate the same pattern."
)

# ---- Scorer generation prompt ----

_SCORER_GENERATION_SYSTEM_PROMPT = (
    "You are an expert at writing detection instructions for AI quality judges. "
    "Given an issue description and example failures, write concise instructions "
    "that a judge can use to detect this issue in a trace.\n\n"
    "IMPORTANT: The judge returns yes/no (pass/fail). A passing trace (yes) "
    "means the trace is FREE of this issue. A failing trace (no) means "
    "the issue WAS detected. Write instructions so that 'yes' = clean/good "
    "and 'no' = issue found.\n\n"
    "CRITICAL RULE ON SPLITTING SCORERS:\n"
    "Each scorer MUST test exactly ONE criterion. If the issue involves "
    "multiple independent criteria, you MUST split them into separate scorers. "
    "Indicators that you need to split:\n"
    "- The word 'and' joining two distinct checks "
    "(e.g. 'is slow AND hallucinates')\n"
    "- The word 'or' joining two distinct checks "
    "(e.g. 'truncates OR omits data')\n"
    "- Multiple failure modes that could occur independently\n"
    "For example, 'response is truncated and uses wrong API' should become TWO "
    "scorers: one for truncation and one for wrong API usage.\n\n"
    "CRITICAL: Each scorer's detection_instructions MUST contain the literal text "
    "'{{ trace }}' (with double curly braces) as a template variable — "
    "this is how the judge receives the trace data.\n"
    "Example: 'Analyze the {{ trace }} to determine if...'"
)
