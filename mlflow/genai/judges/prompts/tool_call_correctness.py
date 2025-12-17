from typing import TYPE_CHECKING

from mlflow.genai.judges.utils.formatting_utils import (
    format_available_tools,
    format_tools_called,
)
from mlflow.genai.prompts.utils import format_prompt

if TYPE_CHECKING:
    from mlflow.genai.utils.type import FunctionCall
    from mlflow.types.chat import ChatTool

# NB: User-facing name for the is_tool_call_correct assessment.
TOOL_CALL_CORRECTNESS_FEEDBACK_NAME = "tool_call_correctness"

TOOL_CALL_CORRECTNESS_PROMPT_INSTRUCTIONS = """\
Consider whether the agent selected appropriate tools and called with the correct arguments for the
task.

Given the user's request, the available tools (including their described capabilities/constraints),
and the sequence of tool calls made by the agent, evaluate if the agent chose suitable tools and
used them in a reasonable way.

Focus only on the choice of tools and the arguments passed to them. Do NOT judge whether the tools'
outputs or implementations are correct.

Evaluate:

1) Need for tools
- Was using any tool necessary or helpful for this request?
- Did the agent fail to use an obviously appropriate tool that was available?

2) Tool selection
- For each step, is the chosen tool a good match for the subtask, given the tool descriptions?
- Did the agent avoid tools that are clearly irrelevant, overpowered, or disallowed for the request?

3) Arguments and intent alignment
- Do the arguments match the tool's schema?
- Are the arguments clearly grounded in the user's request and the tool's documented purpose?
- Across calls, are key parameters provided in ways that logically follow from prior tool outputs
or user messages, rather than arbitrary changes?

4) Tool flow and combinations
- When multiple tools are used, is the overall sequence of tool choices logically sound?
- Are follow-up tool calls justified by what the agent appears to be trying to achieve with respect
to the user's request?

<request>
{{request}}
</request>

<available_tools>
{{available_tools}}
</available_tools>

<tools_called>
{{tools_called}}
</tools_called>"""

TOOL_CALL_CORRECTNESS_PROMPT_OUTPUT = """

Please evaluate whether the agent's tool calls and their arguments are correct and reasonable using only the following json format. Return "yes" if the tool calls and arguments are correct and reasonable, otherwise return "no".
Do not use any markdown formatting or output additional lines.
{
  "rationale": "Reason for the assessment. If incorrect or unreasonable tool calls are found, identify which specific calls or arguments are problematic and explain why. If all tool calls and arguments are correct, explain why they are appropriate. Start each rationale with `Let's think step by step`",
  "result": "yes|no"
}\
"""  # noqa: E501

TOOL_CALL_CORRECTNESS_PROMPT = (
    TOOL_CALL_CORRECTNESS_PROMPT_INSTRUCTIONS + TOOL_CALL_CORRECTNESS_PROMPT_OUTPUT
)


def get_prompt(
    request: str, tools_called: list["FunctionCall"], available_tools: list["ChatTool"]
) -> str:
    """Generate tool call correctness evaluation prompt.

    Args:
        request: The original user request that the agent is trying to fulfill
        tools_called: The sequence of tools that were called by the agent.
            Each element should be a FunctionCall object.
        available_tools: The set of available tools

    Returns:
        Formatted prompt string
    """
    available_tools_str = format_available_tools(available_tools)
    tools_called_str = format_tools_called(tools_called)

    return format_prompt(
        TOOL_CALL_CORRECTNESS_PROMPT,
        request=request,
        available_tools=available_tools_str,
        tools_called=tools_called_str,
    )
