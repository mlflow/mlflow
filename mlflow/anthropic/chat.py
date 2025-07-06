from typing import Any

from mlflow.types.chat import (
    ChatTool,
    FunctionToolDefinition,
)


def convert_tool_to_mlflow_chat_tool(tool: dict[str, Any]) -> ChatTool:
    """
    Convert Anthropic tool definition into MLflow's standard format (OpenAI compatible).

    Ref: https://docs.anthropic.com/en/docs/build-with-claude/tool-use

    Args:
        tool: A dictionary represents a single tool definition in the input request.

    Returns:
        ChatTool: MLflow's standard tool definition object.
    """
    return ChatTool(
        type="function",
        function=FunctionToolDefinition(
            name=tool.get("name"),
            description=tool.get("description"),
            parameters=tool.get("input_schema"),
        ),
    )
