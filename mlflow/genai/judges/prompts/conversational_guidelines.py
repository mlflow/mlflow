CONVERSATIONAL_GUIDELINES_ASSESSMENT_NAME = "conversational_guidelines"

CONVERSATIONAL_GUIDELINES_PROMPT = """\
Consider the following conversation history between a user and an assistant.
Your task is to evaluate whether the assistant's responses throughout the conversation comply with
the provided guidelines and output exactly one label: "yes" or "no".

<guidelines>
{{ guidelines }}
</guidelines>

Evaluation criteria:
- Assess whether EVERY assistant response in the conversation follows ALL the provided guidelines.
- Focus on judging only the assistant's responses, not the user's messages.
- Only focus on the provided guidelines and not the correctness, relevance, or effectiveness of the responses.
- A guideline violation at ANY point in the conversation means the entire conversation fails.
- If none of the guidelines apply to the given conversation, the result must be "yes".

Output "yes" if all assistant responses comply with all guidelines throughout the entire conversation.
Output "no" if any assistant response violates any guideline at any point in the conversation.

<conversation>{{ conversation }}</conversation>
"""  # noqa: E501
