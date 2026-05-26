TASK_SUCCESS_ASSESSMENT_NAME = "task_success"

TASK_SUCCESS_PROMPT = """\
Consider the following user request and assistant response.
You must decide whether the assistant completed the task described in the user's request.
Output only "yes" or "no" based on whether the task was completed according to the criteria below.

Focus only on whether the requested task was carried out to completion.
Do not evaluate factual correctness, response quality, or writing style \
beyond whether the task was actually performed.

A task is considered complete if:
- The assistant performed the action the user requested \
(e.g., submitted a form, generated a report, wrote the code, made the API call).
- The assistant provided a complete deliverable that fulfills the user's request.
- The assistant clearly explained why the task cannot be completed and offered an alternative.

A task is NOT complete if:
- The assistant only described how to do the task without actually doing it, \
when the user asked for the task to be performed.
- The assistant started the task but did not finish it.
- The assistant ignored the task and responded with unrelated content.
- The assistant refused the task without providing any explanation.

Do not make assumptions or bring in external knowledge.

<request>{{inputs}}</request>
<response>{{outputs}}</response>
"""
