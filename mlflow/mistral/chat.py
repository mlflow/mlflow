from typing import Any

from pydantic import BaseModel

from mlflow.types.chat import (
    ChatTool,
    FunctionToolDefinition,
)


def convert_tool_to_mlflow_chat_tool(tool: dict[str, Any] | BaseModel) -> ChatTool:
    """
    Convert Mistral AI tool definition into MLflow's standard format (OpenAI compatible).

    Ref: https://docs.mistral.ai/capabilities/function_calling/#tools

    Args:
        tool: A single tool definition in the input request, as a dict or a Mistral SDK model.

    Returns:
        ChatTool: MLflow's standard tool definition object.
    """
    if isinstance(tool, BaseModel):
        tool = tool.model_dump()
    function = tool["function"]
    return ChatTool(
        type="function",
        function=FunctionToolDefinition(
            name=function["name"],
            description=function.get("description"),
            parameters=function["parameters"],
        ),
    )
