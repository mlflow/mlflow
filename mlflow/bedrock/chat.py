from typing import Any

from mlflow.types.chat import ChatTool, FunctionToolDefinition


def convert_tool_to_mlflow_chat_tool(tool: dict[str, Any]) -> ChatTool:
    """
    Convert Bedrock tool definition into MLflow's standard format (OpenAI compatible).

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Tool.html

    Args:
        tool: A dictionary represents a single tool definition in the input request.

    Returns:
        ChatTool: MLflow's standard tool definition object.
    """
    tool_spec = tool["toolSpec"]
    return ChatTool(
        type="function",
        function=FunctionToolDefinition(
            name=tool_spec["name"],
            description=tool_spec.get("description"),
            parameters=tool_spec["inputSchema"].get("json"),
        ),
    )
