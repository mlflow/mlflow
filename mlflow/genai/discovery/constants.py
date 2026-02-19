_DEFAULT_SAMPLE_SIZE = 100
_MAX_SUMMARIES_FOR_CLUSTERING = 50
_MIN_FREQUENCY_THRESHOLD = 0.01
_MIN_CONFIDENCE = 75
_MIN_EXAMPLES = 2

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

_SESSION_SATISFACTION_INSTRUCTIONS = """\
Follow all the steps below VERY CAREFULLY AND PRECISELY to determine if the user's goals \
were achieved efficiently.

A goal is an outcome the user was trying to accomplish through their interaction with the AI \
assistant. A goal is NOT simply the topic of the conversation or the specific question(s) the \
user asked! Correcting for an assistant's mistakes or shortcomings is also NOT a user goal. \
Goals should always be independent of the agent's behavior. Agent responses may give users new \
information or context that leads to new goals, but these goals are driven by the user's \
knowledge, context, and motivations external to the assistant.

Thoroughly analyze the {{ conversation }} between a user and an AI assistant to identify if \
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
- Consider the user ending the conversation as an indication of failure
- Consider abrupt topic changes as failure unless preceding messages indicate unmet expectations
- Interpret off-topic user messages as an indication of failure

Return True if the user's goals were achieved efficiently, False otherwise.

In your rationale, explain:
- What the user wanted to achieve (list all goals)
- Whether they were achieved efficiently
- If not, list each violated expectation with the observable behavior that demonstrates the issue\
"""

_TRACE_SATISFACTION_INSTRUCTIONS = """\
Follow all the steps below VERY CAREFULLY AND PRECISELY to determine if the user's goals \
were achieved efficiently.

A goal is an outcome the user was trying to accomplish through their interaction with the AI \
assistant. A goal is NOT simply the topic of the interaction or the specific question(s) the \
user asked! Correcting for an assistant's mistakes or shortcomings is also NOT a user goal. \
Goals should always be independent of the agent's behavior.

Thoroughly analyze the {{ trace }} between a user and an AI assistant to identify if the \
user's goals were achieved.

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
- Consider the user ending the interaction as an indication of failure

Return True if the user's goals were achieved efficiently, False otherwise.

In your rationale, explain:
- What the user wanted to achieve (list all goals)
- Whether they were achieved efficiently
- If not, list each violated expectation with the observable behavior that demonstrates the issue\
"""
