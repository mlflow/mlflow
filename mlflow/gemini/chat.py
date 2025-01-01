import json
import logging
from typing import TYPE_CHECKING

from mlflow.types.chat import (
    ChatMessage,
    ChatTool,
    Function,
    FunctionParams,
    FunctionToolDefinition,
    ParamProperty,
    TextContentPart,
    ToolCall,
)

if TYPE_CHECKING:
    import google.generativeai as genai

_logger = logging.getLogger(__name__)


def convert_gemini_param_property_to_mlflow_param_property(param_property) -> ParamProperty:
    """
    Convert Gemini parameter property definition into MLflow's standard format (OpenAI compatible).
    Ref: https://ai.google.dev/gemini-api/docs/function-calling

    Args:
        param_property: A genai.protos.Schema object representing a parameter property.

    Returns:
        ParamProperty: MLflow's standard param property object.
    """
    return ParamProperty(
        description=param_property.description,
        enum=param_property.enum,
        type=param_property.type.name.lower(),
    )


def convert_gemini_function_param_to_mlflow_function_param(
    function_params: "genai.protos.Schema",
) -> FunctionParams:
    """
    Convert Gemini function parameter definition into MLflow's standard format (OpenAI compatible).
    Ref: https://ai.google.dev/gemini-api/docs/function-calling

    Args:
        function_params: A genai.protos.Schema object representing function parameters.

    Returns:
        FunctionParams: MLflow's standard function parameter object.
    """
    return FunctionParams(
        properties={
            k: convert_gemini_param_property_to_mlflow_param_property(v)
            for k, v in function_params.properties.items()
        },
        required=function_params.required,
    )


def convert_gemini_func_to_mlflow_chat_tool(
    function_def: "genai.protos.FunctionDeclaration",
) -> ChatTool:
    """
    Convert Gemini function definition into MLflow's standard format (OpenAI compatible).
    Ref: https://ai.google.dev/gemini-api/docs/function-calling

    Args:
        function_def: A genai.protos.FunctionDeclaration object representing a function definition.

    Returns:
        ChatTool: MLflow's standard tool definition object.
    """
    return ChatTool(
        type="function",
        function=FunctionToolDefinition(
            name=function_def.name,
            description=function_def.description,
            parameters=convert_gemini_function_param_to_mlflow_function_param(
                function_def.parameters
            ),
        ),
    )


def convert_gemini_func_call_to_mlflow_tool_call(
    func_call: "genai.protos.FunctionCall",
) -> ToolCall:
    """
    Convert Gemini function call into MLflow's standard format (OpenAI compatible).
    Ref: https://ai.google.dev/gemini-api/docs/function-calling

    Args:
        func_call: A genai.protos.FunctionCall object representing a single func call.

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
    content: "genai.content_types.ContentsType",
) -> list[ChatMessage]:
    """
    Convert a gemini content to chat messages.

    Args:
        content: A genai.content_types.ContentsType object representing the model content.

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
        content_parts = []
        for content_block in content:
            if isinstance(content_block, str):
                content_parts.append(TextContentPart(text=content_block, type="text"))
            else:
                # TODO: support more content types (e.g. PIL image)
                _logger.debug(
                    f"Received an unsupported content block type: {content_block.__class__}"
                )
        return [
            ChatMessage(
                role="user",
                content=content_parts,
            )
        ]
    elif hasattr(content, "parts"):
        # eigher protos.Content or ContentDict

        # This could be unset for single turn conversation even if this is content proto
        # https://github.com/googleapis/googleapis/blob/9e966149c59f47f6305d66c98e2a9e7d9c26a2eb/google/ai/generativelanguage/v1beta/content.proto#L64
        role = getattr(content, "role", "model") or "model"
        tool_calls = []
        chat_content = None
        for part in content.parts:
            if function_call := getattr(part, "function_call", None):
                tool_calls.append(convert_gemini_func_call_to_mlflow_tool_call(function_call))
            elif text := getattr(part, "text", None):
                chat_content = text
            elif isinstance(part, str):
                chat_content = part

        chat_message = ChatMessage(
            role=role,
            content=chat_content,
        )

        if tool_calls:
            chat_message.tool_calls = tool_calls

        return [chat_message]
    else:
        _logger.debug(f"Received an unsupported content type: {content.__class__}")
        return []
