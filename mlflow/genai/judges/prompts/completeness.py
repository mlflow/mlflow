# NB: User-facing name for the completeness assessment.
COMPLETENESS_ASSESSMENT_NAME = "completeness"

COMPLETENESS_PROMPT = """\
Consider the following user prompt and assistant response.
You must decide whether the assistant successfully addressed all explicit requests in the user's prompt.
Output only "yes" or "no" based on whether the conversation is complete or incomplete according to the criteria below.

First, list all explicit user requests made in the user prompt.
Second, for each request, determine whether it was addressed by the assistant response.
Do not evaluate factual correctness, style, or usefulness beyond whether each request was directly handled.
If the assistant refuses but gives a clear and explicit explanation for the refusal, treat the response as complete;
if it refuses without providing any reasoning, treat it as incomplete.
If the assistant indicates it is missing information and asks the user for the necessary details instead of answering, treat this as complete.
If any explicit request in the user prompt is ignored, or handled in a way that does not match the user's instructions, treat the response as incomplete.
Do not make assumptions or bring in external knowledge.

<question>{{inputs}}</question>
<answer>{{outputs}}</answer>
"""  # noqa: E501
