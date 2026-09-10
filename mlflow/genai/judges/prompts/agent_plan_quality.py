# NB: User-facing name for the agent plan quality assessment.
AGENT_PLAN_QUALITY_ASSESSMENT_NAME = "agent_plan_quality"

AGENT_PLAN_QUALITY_PROMPT = """\
Consider the following conversation history between a user and an agent, including any tool calls \
and intermediate reasoning steps. Your task is to evaluate the quality of the agent's action \
planning and reasoning across the session and output exactly one label: "yes" or "no".

Evaluate the agent's planning for the following quality criteria:
- Goal decomposition: The agent breaks the user's request into well-defined, achievable subgoals \
that together cover the request, rather than jumping straight to execution without a plan.
- Logical ordering: Steps and tool calls are sequenced in a reasonable order, with prerequisites \
satisfied before dependent steps, and without obvious out-of-order actions.
- Step relevance: Each planned step or tool call clearly advances the user's goal; the agent does \
not invoke unrelated tools, perform busy work, or pursue tangents that do not contribute to the outcome.
- Adaptivity: When a step fails, returns unexpected results, or reveals new information, the agent \
updates its plan accordingly instead of mechanically continuing the original sequence.
- Termination: The agent stops once the goal is achieved, or explicitly surfaces the remaining \
uncertainty to the user; it does not loop indefinitely or keep acting after the task is done.

Evaluation guidelines:
- Focus on the agent's planning and reasoning behavior, not on surface phrasing.
- A plan is acceptable even if implicit, as long as the sequence of actions demonstrates the \
criteria above.
- Do not penalize the agent for exploratory steps that are justified by missing information, \
provided the exploration is bounded and on-topic.
- Tool failures, invalid arguments, or external errors are acceptable if the agent recognizes and \
recovers from them; they are not acceptable if the agent ignores them and proceeds as if they \
succeeded.
- Ask-for-clarification is valid planning behavior when the request is genuinely ambiguous.

Output "yes" if the agent's planning is sound overall across the session. Output "no" only if \
there is at least one clear planning failure as defined above.

<conversation>{{ conversation }}</conversation>
"""  # noqa: E501
