import base64
import json
import logging
from typing import Optional, Union

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

_logger = logging.getLogger(__name__)


def convert_message_to_mlflow_chat(message: dict) -> ChatMessage:
    """
    Convert Bedrock Converse API's message object into MLflow's standard format (OpenAI compatible).

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Message.html

    Args:
        message: Bedrock Converse API's message object.

    Returns:
        ChatMessage: MLflow's standard chat message object.
    """
    role = message["role"]
    contents = []
    tool_calls = []
    tool_call_id = None
    for content in message["content"]:
        if tool_call := content.get("toolUse"):
            input = tool_call.get("input")
            tool_calls.append(
                ToolCall(
                    id=tool_call["toolUseId"],
                    function=Function(
                        name=tool_call["name"],
                        arguments=input if isinstance(input, str) else json.dumps(input),
                    ),
                    type="function",
                )
            )
        elif tool_result := content.get("toolResult"):
            tool_call_id = tool_result["toolUseId"]
            # "tool_result" content corresponds to the "tool" message in OpenAI.
            # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultContentBlock.html
            role = "tool"
            for content in tool_result["content"]:
                parsed_content = _parse_content(content)
                if parsed_content:
                    contents.append(parsed_content)
        else:
            parsed_content = _parse_content(content)
            if parsed_content:
                contents.append(parsed_content)

    message = ChatMessage(role=role, content=contents)
    if tool_calls:
        message.tool_calls = tool_calls
    if tool_call_id:
        message.tool_call_id = tool_call_id
    return message


def _parse_content(content: dict) -> Optional[Union[TextContentPart, ImageContentPart]]:
    """
    Parse a single content block in the Bedrock message object.

    Some content types like video and document are not supported by OpenAI's spec. This
    function returns None for those content types.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
    """
    if text := content.get("text"):
        return TextContentPart(text=text, type="text")
    elif json_content := content.get("json"):
        return TextContentPart(text=json.dumps(json_content), type="text")
    elif image := content.get("image"):
        # Bedrock support passing images in both raw bytes and base64 encoded strings.
        # OpenAI spec only supports base64 encoded images, so we encode the raw bytes to base64.
        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageBlock.html
        image_bytes = image["source"]["bytes"]
        if isinstance(image_bytes, bytes):
            data = base64.b64encode(image_bytes).decode("utf-8")
        else:
            data = image_bytes
        format = "image/" + image["format"]
        image_url = ImageUrl(url=f"data:{format};base64,{data}", detail="auto")
        return ImageContentPart(type="image_url", image_url=image_url)
    # NB: Video and Document content type are not supported by OpenAI's spec, so recording as text.
    else:
        _logger.debug(f"Received an unsupported content type: {list(content.keys())[0]}")
        return None


def convert_tool_to_mlflow_chat_tool(tool: dict) -> ChatTool:
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
