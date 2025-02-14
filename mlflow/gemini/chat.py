import json
import logging
from typing import TYPE_CHECKING, Optional, Union

from mlflow.types.chat import (
    ChatMessage,
    ChatTool,
    Function,
    FunctionParams,
    FunctionToolDefinition,
    ImageContentPart,
    ImageUrl,
    ParamProperty,
    TextContentPart,
    ToolCall,
)

if TYPE_CHECKING:
    from google import genai

_logger = logging.getLogger(__name__)


def convert_gemini_func_to_mlflow_chat_tool(
    function_def: "genai.types.FunctionDeclaration",
) -> ChatTool:
    """
    Convert Gemini function definition into MLflow's standard format (OpenAI compatible).
    Ref: https://ai.google.dev/gemini-api/docs/function-calling

    Args:
        function_def: A genai.types.FunctionDeclaration or genai.protos.FunctionDeclaration object
                      representing a function definition.

    Returns:
        ChatTool: MLflow's standard tool definition object.
    """
    return ChatTool(
        type="function",
        function=FunctionToolDefinition(
            name=function_def.name,
            description=function_def.description,
            parameters=_convert_gemini_function_param_to_mlflow_function_param(
                function_def.parameters
            ),
        ),
    )


def convert_gemini_func_call_to_mlflow_tool_call(
    func_call: "genai.types.FunctionCall",
) -> ToolCall:
    """
    Convert Gemini function call into MLflow's standard format (OpenAI compatible).
    Ref: https://ai.google.dev/gemini-api/docs/function-calling

    Args:
        func_call: A genai.types.FunctionCall or genai.protos.FunctionCall object
                   representing a single func call.

    Returns:
        ToolCall: MLflow's standard tool call object.
    """
    # original args object is not json serializable
    args = func_call.args or {}

    return ToolCall(
        # Gemini does not have func call id
        id=func_call.name,
        type="function",
        function=Function(name=func_call.name, arguments=json.dumps(dict(args))),
    )


def parse_gemini_content_to_mlflow_chat_messages(
    content: "genai.types.ContentsType",
) -> list[ChatMessage]:
    """
    Convert a gemini content to chat messages.

    Args:
        content: A genai.types.ContentsType object representing the model content.

    Returns:
        list[ChatMessage]: A list of MLflow's standard chat messages.
    """
    if isinstance(content, str):
        # Assume str content is used only for user input
        return [
            ChatMessage(
                role="user",
                content=content,
            )
        ]
    elif isinstance(content, list):
        # either list of user inputs or multi-turn conversation
        if not content:
            return []
        # when chat history is passed, parse content recursively
        if hasattr(content[0], "parts"):
            return sum(
                [
                    parse_gemini_content_to_mlflow_chat_messages(content_block)
                    for content_block in content
                ],
                [],
            )

        # when multiple contents are passed by user
        return [_construct_chat_message(content, "user")]
    elif hasattr(content, "parts"):
        # eigher genai.types.Content or ContentDict

        # This could be unset for single turn conversation even if this is content proto
        # https://github.com/googleapis/googleapis/blob/9e966149c59f47f6305d66c98e2a9e7d9c26a2eb/google/ai/generativelanguage/v1beta/content.proto#L64
        role = getattr(content, "role", "model") or "model"
        # we normalize role and use assistant
        if role == "model":
            role = "assistant"

        return [_construct_chat_message(content.parts, role)]
    else:
        _logger.debug(f"Received an unsupported content type: {content.__class__}")
        return []


def _construct_chat_message(parts: list["genai.types.PartType"], role: str) -> ChatMessage:
    tool_calls = []
    content_parts = []
    for content_part in parts:
        part = _parse_content_part(content_part)
        if isinstance(part, (TextContentPart, ImageContentPart)):
            content_parts.append(part)
        elif isinstance(part, ToolCall):
            tool_calls.append(part)
    chat_message = ChatMessage(
        role=role,
        content=content_parts or None,
    )

    if tool_calls:
        chat_message.tool_calls = tool_calls

    return chat_message


def _parse_content_part(part: "genai.types.PartType") -> Optional[Union[TextContentPart, ToolCall]]:
    """
    Convert Gemini part type into MLflow's standard format (OpenAI compatible).
    Ref: https://ai.google.dev/gemini-api/docs/function-calling

    Args:
        part: A genai.types.PartType object representing a part of content.

    Returns:
        Optional[Union[TextContentPart, ToolCall]]: MLflow's standard content part.
    """
    # The schema of the Part proto is available at https://github.com/googleapis/googleapis/blob/9e966149c59f47f6305d66c98e2a9e7d9c26a2eb/google/ai/generativelanguage/v1beta/content.proto#L76
    if function_call := getattr(part, "function_call", None):
        # FunctionCall part: https://github.com/googleapis/googleapis/blob/9e966149c59f47f6305d66c98e2a9e7d9c26a2eb/google/ai/generativelanguage/v1beta/content.proto#L316
        return convert_gemini_func_call_to_mlflow_tool_call(function_call)
    elif function_response := getattr(part, "function_response", None):
        # FunctionResponse part: https://github.com/googleapis/googleapis/blob/9e966149c59f47f6305d66c98e2a9e7d9c26a2eb/google/ai/generativelanguage/v1beta/content.proto#L332
        if hasattr(function_response, "json"):
            # genai
            return TextContentPart(text=function_response.json(), type="text")
        # generativeai
        return TextContentPart(
            text=str(type(function_response).to_dict(function_response)), type="text"
        )
    elif blob := getattr(part, "inline_data", None):
        # Blob part: https://github.com/googleapis/googleapis/blob/9e966149c59f47f6305d66c98e2a9e7d9c26a2eb/google/ai/generativelanguage/v1beta/content.proto#L109C9-L109C13
        return ImageContentPart(
            image_url=ImageUrl(
                url=f"data:{blob.mime_type};base64,{blob.data}",
                detail="auto",
            ),
            type="image_url",
        )
    elif file := getattr(part, "file_data", None):
        # FileData part: https://github.com/googleapis/googleapis/blob/9e966149c59f47f6305d66c98e2a9e7d9c26a2eb/google/ai/generativelanguage/v1beta/content.proto#L124
        return ImageContentPart(
            image_url=ImageUrl(
                url=file.file_uri,
                detail="auto",
            ),
            type="image_url",
        )
    elif hasattr(part, "mime_type"):
        # Blob part or FileData part
        url = (
            part.file_uri
            if hasattr(part, "file_uri")
            else f"data:{part.mime_type};base64,{part.data}"
        )
        return ImageContentPart(
            image_url=ImageUrl(url=url, detail="auto"),
            type="image_url",
        )
    elif isinstance(part, dict):
        if "mime_type" in part:
            # genai.types.BlobDict
            return ImageContentPart(
                image_url=ImageUrl(
                    url=f"data:{part['mime_type']};base64,{part['data']}", detail="auto"
                ),
                type="image_url",
            )
        elif "text" in part:
            return TextContentPart(text=part["text"], type="text")
    elif text := getattr(part, "text", None):
        # Text part
        return TextContentPart(text=text, type="text")
    elif isinstance(part, str):
        return TextContentPart(text=part, type="text")
    # TODO: Gemini supports more types. Consider including unsupported types (e.g. PIL image)
    _logger.debug(f"Received an unsupported content block type: {part.__class__}")


def _convert_gemini_param_property_to_mlflow_param_property(param_property) -> ParamProperty:
    """
    Convert Gemini parameter property definition into MLflow's standard format (OpenAI compatible).
    Ref: https://ai.google.dev/gemini-api/docs/function-calling

    Args:
        param_property: A genai.types.Schema or genai.protos.Schema object
                        representing a parameter property.

    Returns:
        ParamProperty: MLflow's standard param property object.
    """
    type_name = param_property.type
    type_name = type_name.name.lower() if hasattr(type_name, "name") else type_name.lower()
    return ParamProperty(
        description=param_property.description,
        enum=param_property.enum,
        type=type_name,
    )


def _convert_gemini_function_param_to_mlflow_function_param(
    function_params: "genai.types.Schema",
) -> FunctionParams:
    """
    Convert Gemini function parameter definition into MLflow's standard format (OpenAI compatible).
    Ref: https://ai.google.dev/gemini-api/docs/function-calling

    Args:
        function_params: A genai.types.Schema or genai.protos.Schema object
                         representing function parameters.

    Returns:
        FunctionParams: MLflow's standard function parameter object.
    """
    return FunctionParams(
        properties={
            k: _convert_gemini_param_property_to_mlflow_param_property(v)
            for k, v in function_params.properties.items()
        },
        required=function_params.required,
    )
