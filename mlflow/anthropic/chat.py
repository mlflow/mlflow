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


def convert_message_to_mlflow_chat(message: Union[BaseModel, dict]) -> ChatMessage:
    """
    Convert Anthropic message object into MLflow's standard format (OpenAI compatible).

    Ref: https://docs.anthropic.com/en/api/messages#body-messages

    Args:
        message: Anthropic message object or a dictionary representing the message.

    Returns:
        ChatMessage: MLflow's standard chat message object.
    """
    if isinstance(message, dict):
        content = message.get("content")
        role = message.get("role")
    elif isinstance(message, BaseModel):
        content = message.content
        role = message.role
    else:
        raise MlflowException.invalid_parameter_value(
            f"Message must be either a dict or a Message object, but got: {type(message)}."
        )

    if isinstance(content, str):
        return ChatMessage(role=role, content=content)

    elif isinstance(content, list):
        contents = []
        tool_calls = []
        tool_call_id = None
        for content_block in content:
            if isinstance(content_block, BaseModel):
                if IS_PYDANTIC_V2_OR_NEWER:
                    content_block = content_block.model_dump()
                else:
                    content_block = content_block.dict()

            content_type = content_block.get("type")
            if content_type == "tool_use":
                # Anthropic response contains tool calls in the content block
                # Ref: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#example-api-response-with-a-tool-use-content-block
                tool_calls.append(
                    ToolCall(
                        id=content_block["id"],
                        function=Function(
                            name=content_block["name"], arguments=json.dumps(content_block["input"])
                        ),
                        type="function",
                    )
                )
            elif content_type == "tool_result":
                # In Anthropic, the result of tool execution is returned as a special content type
                # "tool_result" with "user" role, which corresponds to the "tool" role in OpenAI.
                role = "tool"
                tool_call_id = content_block["tool_use_id"]
                if result_content := content_block.get("content"):
                    contents.append(_parse_content(result_content))
                else:
                    contents.append(TextContentPart(text="", type="text"))
            else:
                contents.append(_parse_content(content_block))

        message = ChatMessage(role=role, content=contents)
        # Only set tool_calls field when it is present
        if tool_calls:
            message.tool_calls = tool_calls
        if tool_call_id:
            message.tool_call_id = tool_call_id
        return message

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
    elif content_type == "image":
        source = content["source"]
        return ImageContentPart(
            image_url=ImageUrl(
                url=f"data:{source['media_type']};{source['type']},{source['data']}"
            ),
            type="image_url",
        )
    else:
        raise MlflowException.invalid_parameter_value(
            f"Unknown content type: {content_type['type']}. Please make sure the message "
            "is a valid Anthropic message object. If it is a valid type, contact to the "
            "MLflow maintainer via https://github.com/mlflow/mlflow/issues/new/choose for "
            "requesting support for a new message type."
        )


def convert_tool_to_mlflow_chat_tool(tool: dict) -> ChatTool:
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
