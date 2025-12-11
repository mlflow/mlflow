# NB: User-facing name for the conversational tool call efficiency assessment.
CONVERSATIONAL_TOOL_CALL_EFFICIENCY_ASSESSMENT_NAME = "conversational_tool_call_efficiency"

CONVERSATIONAL_TOOL_CALL_EFFICIENCY_PROMPT = """\
Consider the following conversation history between a user and an assistant, including tool calls \
made during the conversation. Your task is to evaluate whether tool usage was efficient and output \
exactly one label: "yes" or "no".

A conversation has inefficient tool usage if any of the following apply:
- Redundant calls: The same tool is called multiple times with identical or equivalent parameters \
to retrieve information already obtained earlier in the conversation.
- Unnecessary calls: Tools are invoked when not needed to fulfill the user's request.
- Missing cache awareness: Previously retrieved information is re-fetched instead of being reused.
- Missed batching: Multiple separate calls are made when a single call could retrieve all needed information.

Evaluation guidelines:
- Focus only on clear inefficiencies such as repeated identical calls or obvious misuse.
- Do not penalize reasonable tool usage even if alternative approaches exist.
- Minor suboptimal choices that don't significantly impact the conversation are acceptable.
- If no tools were called and none were needed, tool usage is efficient.

Output "yes" if tool usage was efficient overall. Output "no" only if there are clear inefficiencies \
as defined above.

<conversation>{{ conversation }}</conversation>
"""  # noqa: E501
