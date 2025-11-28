# NB: User-facing name for the conversation completeness assessment.
CONVERSATION_COMPLETENESS_ASSESSMENT_NAME = "conversation_completeness"

CONVERSATION_COMPLETENESS_PROMPT = """\
Consider the following conversation history between a user and an assistant.
Your task is to output exactly one label: "yes" or "no" based on the criteria below.

First, list all explicit user requests made throughout the conversation in the rationale section.
Second, for each request, determine whether it was addressed by the assistant by the end of the conversation,\
and **quote** the assistant's explicit response in the rationale section if you judge the request as addressed.
If there is no explicit response to a request—or the response can only be inferred from context—mark that request as incomplete.
Requests may be satisfied at any point in the dialogue as long as they are resolved by the final turn.
A refusal counts as addressed only if the assistant provides a clear and explicit explanation; refusals without reasoning should be marked incomplete.
Do not assume completeness merely because the user seems satisfied; evaluate solely whether each identified request was actually fulfilled.
Output "no" only if one or more user requests remain unaddressed in the final state. Output "yes" if all requests were addressed.
Base your judgment strictly on information explicitly stated or strongly implied in the conversation, without using outside assumptions.

<conversation>{{ conversation }}</conversation>
"""  # noqa: E501
