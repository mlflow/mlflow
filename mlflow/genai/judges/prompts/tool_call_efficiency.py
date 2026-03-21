from typing import TYPE_CHECKING

from mlflow.genai.judges.utils.formatting_utils import (
    format_available_tools,
    format_tools_called,
)
from mlflow.genai.prompts.utils import format_prompt

if TYPE_CHECKING:
    from mlflow.genai.utils.type import FunctionCall
    from mlflow.types.chat import ChatTool

# NB: User-facing name for the is_tool_call_efficient assessment.
TOOL_CALL_EFFICIENCY_FEEDBACK_NAME = "tool_call_efficiency"

TOOL_CALL_EFFICIENCY_PROMPT_INSTRUCTIONS = """\
Consider the agent's tool usage for redundancy and inefficiency.

Given the user's request, the available tools, and the sequence of tools called by the agent, \
determine whether any tool calls were unnecessary or could have been made more efficient. In your \
analysis, treat retries caused by temporary tool failures (e.g., timeouts, transient errors) as \
efficient and not redundant.

Consider in particular:

Calls to the same tool with identical or very similar arguments

Repeated calls to the same tool with the same parameters

Multiple calls that could reasonably have been consolidated into a single call

<request>
{{request}}
</request>

<available_tools>
{{available_tools}}
</available_tools>

<tools_called>
{{tools_called}}
</tools_called>"""

TOOL_CALL_EFFICIENCY_PROMPT_OUTPUT = """

Please evaluate whether the agent's tool usage is efficient and free of redundancy using only the following json format. Return "yes" if the tool usage is efficient and free of redundancy, otherwise return "no".
Do not use any markdown formatting or output additional lines.
{
  "rationale": "Reason for the assessment. If redundant tool calls are found, identify which specific calls are redundant and explain why. If no redundancy is found, explain why the tool usage is efficient. Start each rationale with `Let's think step by step`",
  "result": "yes|no"
}\
"""  # noqa: E501

TOOL_CALL_EFFICIENCY_PROMPT = (
    TOOL_CALL_EFFICIENCY_PROMPT_INSTRUCTIONS + TOOL_CALL_EFFICIENCY_PROMPT_OUTPUT
)


def get_prompt(
    request: str,
    tools_called: list["FunctionCall"],
    available_tools: list["ChatTool"],
) -> str:
    """Generate tool call efficiency evaluation prompt.

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
        TOOL_CALL_EFFICIENCY_PROMPT,
        request=request,
        available_tools=available_tools_str,
        tools_called=tools_called_str,
    )
