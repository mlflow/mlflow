import json
from typing import Union

from pydantic import BaseModel

from mlflow.exceptions import MlflowException
from mlflow.types.chat import (
    ChatMessage,
    ChatTool,
    Function,
    FunctionToolDefinition,
    ImageContentPart,
    ImageUrl,
    TextContentPart,
    ToolCall,
)
from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER


def _to_dict(obj: BaseModel):
    if IS_PYDANTIC_V2_OR_NEWER:
        return obj.model_dump()
    return obj.dict()


def convert_message_to_mlflow_chat(message: Union[BaseModel, dict]) -> ChatMessage:
    """
    Convert Mistral AI message object into MLflow's standard format (OpenAI compatible).

    Ref: https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post

    Args:
        message: Mistral AI message object or a dictionary representing the message.

    Returns:
        ChatMessage: MLflow's standard chat message object.
    """
    if isinstance(message, dict):
        content = message.get("content")
        role = message.get("role")
        tool_calls = message.get("tool_calls")
        tool_call_id = message.get("tool_call_id")
    elif isinstance(message, BaseModel):
        content = message.content
        role = message.role
        # tool_calls is available if message is an AssistantMessage object
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            tool_calls = [_to_dict(tool_call) for tool_call in tool_calls]
        # tool_call_id is available if message is a ToolMessage object
        tool_call_id = getattr(message, "tool_call_id", None)
    else:
        raise MlflowException.invalid_parameter_value(
            f"Message must be either a dict or a Message object, but got: {type(message)}."
        )

    if tool_calls:
        tool_calls = [
            ToolCall(
                id=tool_call["id"],
                function=Function(
                    name=tool_call["function"]["name"],
                    arguments=json.dumps(tool_call["function"]["arguments"]),
                ),
                type="function",
            )
            for tool_call in tool_calls
        ]

    if isinstance(content, str):
        return ChatMessage(
            role=role, content=content, tool_calls=tool_calls, tool_call_id=tool_call_id
        )

    elif isinstance(content, list):
        contents = []
        tool_calls = []
        tool_call_id = None
        for content_chunk in content:
            if isinstance(content_chunk, BaseModel):
                content_chunk = _to_dict(content_chunk)
            contents.append(_parse_content(content_chunk))

        return ChatMessage(
            role=role, content=contents, tool_calls=tool_calls, tool_call_id=tool_call_id
        )

    else:
        raise MlflowException.invalid_parameter_value(
            f"Invalid content type. Must be either a string or a list, but got: {type(content)}."
        )


def _parse_content(content: Union[str, dict]) -> Union[TextContentPart, ImageContentPart]:
    if isinstance(content, str):
        return TextContentPart(text=content, type="text")

    content_type = content.get("type")
    if content_type == "text":
        return TextContentPart(text=content["text"], type="text")
    elif content_type == "image_url":
        return ImageContentPart(
            image_url=ImageUrl(url=content["image_url"], detail="auto"),
            type="image_url",
        )
    else:
        raise MlflowException.invalid_parameter_value(
            f"Unknown content type: {content_type['type']}. Please make sure the message "
            "is a valid Mistral AI message object. If it is a valid type, contact to the "
            "MLflow maintainer via https://github.com/mlflow/mlflow/issues/new/choose for "
            "requesting support for a new message type."
        )


def convert_tool_to_mlflow_chat_tool(tool: dict) -> ChatTool:
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
