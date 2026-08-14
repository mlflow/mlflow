# NB: User-facing name for the user frustration assessment.
USER_FRUSTRATION_ASSESSMENT_NAME = "user_frustration"

USER_FRUSTRATION_PROMPT = """\
Consider the following conversation history between a user and an assistant. Your task is to
determine the user's emotional trajectory and output exactly one of the following labels:
"none", "resolved", or "unresolved".\

Return "none" when the user **never** expresses frustration at any point in the conversation;
Return "unresolved" when the user is frustrated near the end or leaves without clear satisfaction.
Only return "resolved" when the user **is frustrated at some point** in the conversation but clearly ends the conversation satisfied or reassured;
    - Do not assume the user is satisfied just because the assistant's final response is helpful, constructive, or polite;
    - Only label a conversation as "resolved" if the user explicitly or strongly implies satisfaction, relief, or acceptance in their own final turns.

Base your decision only on explicit or strongly implied signals in the conversation and do not
use outside knowledge or assumptions.

<conversation>{{ conversation }}</conversation>
"""  # noqa: E501
