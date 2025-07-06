from typing import Any

from mlflow.types.chat import (
    ChatTool,
    FunctionToolDefinition,
)


def convert_tool_to_mlflow_chat_tool(tool: dict[str, Any]) -> ChatTool:
    """
    Convert Mistral AI tool definition into MLflow's standard format (OpenAI compatible).

    Ref: https://docs.mistral.ai/capabilities/function_calling/#tools

    Args:
        tool: A dictionary represents a single tool definition in the input request.

    Returns:
        ChatTool: MLflow's standard tool definition object.
    """
    function = tool["function"]
    return ChatTool(
        type="function",
        function=FunctionToolDefinition(
            name=function["name"],
            description=function.get("description"),
            parameters=function["parameters"],
        ),
    )
