TASK_SUCCESS_ASSESSMENT_NAME = "task_success"

TASK_SUCCESS_PROMPT = """\
Consider the following user request and assistant response.
You must decide whether the assistant successfully executed the task the user asked for.
Output only "yes" or "no" based on the criteria below.

The task is successful if:
- The assistant executed the requested action \
(e.g., submitted a form, generated a report, wrote the code, called a tool).
- The assistant produced the requested deliverable.
- The assistant clearly explained why the task cannot be performed and offered an alternative.

The task is NOT successful if:
- The assistant only described how to do the task instead of doing it.
- The assistant started the task but did not finish it.
- The assistant ignored the task and responded with unrelated content.
- The assistant refused the task without providing any explanation.

Do not make assumptions or bring in external knowledge.

<request>{{inputs}}</request>
<response>{{outputs}}</response>
"""

TASK_SUCCESS_PROMPT_WITH_TRACE = """\
Consider the user's request and the agent's execution trace ({{ trace }}).
You must decide whether the agent successfully executed the task the user asked for.
Output only "yes" or "no" based on the criteria below.

The task is successful if:
- The agent executed the requested action \
(e.g., submitted a form, generated a report, wrote the code, called a tool).
- The agent produced the requested deliverable.
- The agent clearly explained why the task cannot be performed and offered an alternative.

The task is NOT successful if:
- The agent only described how to do the task instead of doing it.
- The agent started the task but did not finish it.
- The agent ignored the task and responded with unrelated content.
- The agent refused the task without providing any explanation.
- The agent's response claims an action was performed, but the trace contains no evidence \
(e.g., a corresponding tool call) that the action actually happened.

Verify claims of completed actions against the trace: a response saying "done" is not \
sufficient evidence on its own; look for tool calls or intermediate steps that show the \
action was actually executed.

Do not make assumptions or bring in external knowledge.
"""
