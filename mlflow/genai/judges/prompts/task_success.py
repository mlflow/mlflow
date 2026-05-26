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
