import logging
from typing import TYPE_CHECKING

from mlflow.genai.prompts.utils import format_prompt

_logger = logging.getLogger(__name__)

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


def _format_available_tools(available_tools: list["ChatTool"]) -> str:
    """Format available tools with descriptions and parameters.

    Args:
        available_tools: The set of available tools

    Returns:
        Formatted string representation of available tools

    Example:
        >>> # Output format:
        >>> # - search: Search for information on the web
        >>> #     - query (required): string - The search query to execute
        >>> #     - max_results (optional): integer - Maximum number of results
        >>> # - translate: Translate text to another language
        >>> #     - text (required): string - The text to translate
        >>> #     - target (required): string - Target language code
    """
    available_tools_parts = []
    for tool in available_tools:
        if not tool.function:
            _logger.warning(f"Skipping tool with missing function definition: {tool}")
            continue

        tool_str = f"- {tool.function.name}"
        if tool.function.description:
            tool_str += f": {tool.function.description}"

        if tool.function.parameters and tool.function.parameters.properties:
            params = tool.function.parameters
            required_params = set(params.required or [])
            param_lines = []

            for param_name, param_prop in params.properties.items():
                is_required = param_name in required_params
                required_marker = " (required)" if is_required else " (optional)"
                param_line = f"    - {param_name}{required_marker}"

                if hasattr(param_prop, "type") and param_prop.type:
                    param_line += f": {param_prop.type}"

                if param_prop.description:
                    param_line += f" - {param_prop.description}"

                param_lines.append(param_line)

            if param_lines:
                tool_str += "\n" + "\n".join(param_lines)

        available_tools_parts.append(tool_str)
    return "\n\n".join(available_tools_parts) if available_tools_parts else "No tools available"


def _format_tools_called(tools_called: list["FunctionCall"]) -> str:
    """Format tools called with step numbers, arguments, and outputs.

    Args:
        tools_called: The sequence of tools that were called by the agent.
            Each element should be a FunctionCall object.

    Returns:
        Formatted string representation of tools called

    Example:
        >>> # Output format:
        >>> # Tool Call 1: search
        >>> #   Input Arguments: {"query": "capital of France"}
        >>> #   Output: Paris
        >>> #
        >>> # Tool Call 2: translate
        >>> #   Input Arguments: {"text": "Paris", "target": "es"}
        >>> #   Output: París
    """
    tools_called_parts = []
    for idx, tool in enumerate(tools_called, start=1):
        tool_name = tool.name
        tool_args = tool.arguments or {}
        tool_output = tool.outputs or "(no output)"

        tool_str = f"Tool Call {idx}: {tool_name}\n"
        tool_str += f"  Input Arguments: {tool_args}\n"
        tool_str += f"  Output: {tool_output}\n"
        if tool.exception:
            tool_str += f"  Exception: {tool.exception}"
        tools_called_parts.append(tool_str)
    return "\n\n".join(tools_called_parts) if tools_called_parts else "No tools called"


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
    available_tools_str = _format_available_tools(available_tools)
    tools_called_str = _format_tools_called(tools_called)

    return format_prompt(
        TOOL_CALL_EFFICIENCY_PROMPT,
        request=request,
        available_tools=available_tools_str,
        tools_called=tools_called_str,
    )
