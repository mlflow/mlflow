import json
import logging
from typing import Any, Optional, Union

from pydantic import BaseModel

from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.exceptions import MlflowException
from mlflow.tracing import set_span_chat_messages, set_span_chat_tools
from mlflow.types.chat import (
    ChatMessage,
    ChatTool,
    ContentType,
    Function,
    FunctionToolDefinition,
    ImageContentPart,
    ImageUrl,
    TextContentPart,
    ToolCall,
)

_logger = logging.getLogger(__name__)


_RESPONSE_API_BUILT_IN_TOOLS = {
    "file_search",
    "computer_use_preview",
    "web_search_preview",
}


def set_span_chat_attributes(span: LiveSpan, inputs: dict[str, Any], output: Any):
    if span.span_type not in (SpanType.CHAT_MODEL, SpanType.LLM):
        return

    messages = _parse_inputs_output(inputs, output)
    try:
        set_span_chat_messages(span, messages)
    except MlflowException:
        _logger.debug(
            "Failed to set chat messages on span",
            exc_info=True,
        )

    if tools := _parse_tools(inputs):
        try:
            set_span_chat_tools(span, tools)
        except MlflowException:
            _logger.debug("Failed to set chat tools on span", exc_info=True)


def _parse_inputs_output(inputs: dict[str, Any], output: Any) -> list[ChatMessage]:
    from openai.types.chat import ChatCompletion

    try:
        from openai.types.responses import Response

        if isinstance(output, Response):
            messages = []
            if _input := inputs.get("input"):
                if isinstance(_input, str):
                    messages.append(ChatMessage(role="user", content=_input))
                elif isinstance(_input, list):
                    for item in _input:
                        messages.extend(_parse_response_item(item, messages))

            for output in output.output:
                output_dict = output.model_dump(exclude_unset=True)
                messages.extend(_parse_response_item(output_dict, messages))

            return messages
    except ImportError:
        pass

    messages = []
    if "messages" in inputs:
        messages.extend(inputs["messages"])

    if isinstance(output, ChatCompletion):
        messages.extend([output.choices[0].message.to_dict(exclude_unset=True)])
    elif isinstance(output, str):
        messages.extend([{"role": "assistant", "content": output}])

    return messages


def _parse_response_item(
    item: Union[dict[str, Any], BaseModel],
    past_messages: list[ChatMessage],
) -> list[ChatMessage]:
    """Parse Response API output into MLflow standard chat messages"""
    if isinstance(item, BaseModel):
        item = item.model_dump()

    item_type = item.get("type", "message")
    if item_type == "message":
        content, refusal = _parse_message_content(item["content"], past_messages)
        message = ChatMessage(role=item["role"])
        if content:
            message.content = content
        if refusal:
            message.refusal = refusal
        return [message]

    elif item_type == "function_call":
        return [
            _get_tool_call_message(
                tool_id=item["call_id"], tool_name=item["name"], arguments=item["arguments"]
            )
        ]

    elif item_type == "function_call_output":
        return [
            ChatMessage(
                role="tool",
                content=item["output"],
                tool_call_id=item["call_id"],
            )
        ]

    elif item_type == "file_search_call":
        return [
            _get_tool_call_message(
                tool_id=item["id"],
                tool_name=item["type"],
                arguments=json.dumps({"queries": item["queries"]}),
            ),
            ChatMessage(role="tool", tool_call_id=item.get("id"), content=item_type),
        ]

    elif item_type == "web_search_call":
        return [
            _get_tool_call_message(tool_id=item["id"], tool_name=item["type"], arguments=""),
            ChatMessage(role="tool", tool_call_id=item.get("id"), content=item_type),
        ]

    elif item_type == "computer_call":
        return [
            _get_tool_call_message(
                tool_id=item["call_id"],
                tool_name=item["type"],
                arguments=json.dumps({"action": item["action"]}),
            )
        ]
    elif item_type == "computer_call_output":
        output = item["output"]
        return [
            ChatMessage(
                role="tool",
                content=[
                    # Screenshot of the computer after taking the action
                    ImageContentPart(
                        image_url=ImageUrl(url=output.get("image_url")),
                        type="image_url",
                    ),
                ],
                tool_call_id=item["call_id"],
            )
        ]

    elif item_type == "reasoning":
        summary = item["summary"][0]["text"] if item["summary"] else None
        return [ChatMessage(role="assistant", content=summary)]

    raise MlflowException(f"Unknown output type: {type(item)}")


def _parse_message_content(
    content: Union[str, list[dict[str, Any]]], past_messages: Optional[list[ChatMessage]] = None
) -> tuple[ContentType, Optional[str]]:
    if isinstance(content, str):
        return content, None

    if not isinstance(content, list):
        raise MlflowException(f"Invalid content type: {type(content)}")

    parsed_contents = []
    refusal = None
    for item in content:
        content_type = item.get("type")
        if content_type == "input_text":
            parsed_contents.append(TextContentPart(text=item.get("text"), type="text"))

        elif content_type == "input_image":
            parsed_contents.append(
                ImageContentPart(
                    image_url=ImageUrl(
                        url=item["image_url"],
                        detail=item.get("detail"),
                    ),
                    type="image_url",
                )
            )

        elif content_type == "input_file":
            # TODO: MLflow chat schema doesn't support file content yet. Even if it does,
            #   including the full file data in an attribute is not a good idea.
            parsed_contents.append(
                TextContentPart(
                    text=f"{item.get('file_id')}:{item.get('file_name')}",
                    type="text",
                )
            )

        elif content_type == "output_text":
            if annotations := item.get("annotations"):
                _populate_tool_result_message(annotations, past_messages)
            parsed_contents.append(TextContentPart(text=item.get("text"), type="text"))

        elif content_type == "refusal":
            refusal = item.get("refusal")

        else:
            raise MlflowException(f"Unknown content type: {content_type}")

    return parsed_contents, refusal


def _get_tool_call_message(tool_id: str, tool_name: str, arguments: str) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id=tool_id,
                type="function",
                function=Function(name=tool_name, arguments=arguments),
            )
        ],
    )


def _populate_tool_result_message(
    annotations: dict[str, Any], messages: list[ChatMessage]
) -> ChatMessage:
    """
    Parses annotations from the Response API output into MLflow standard chat spec.

    In OpenAI spec, annotations are used for populating information from file/web
    search tools in the Responses API response. When converting to MLflow standard
    chat spec, it should be presented as a tool result message.
    """
    file_citations = [a for a in annotations if a.get("type") == "file_citation"]
    web_citations = [a for a in annotations if a.get("type") == "url_citation"]

    if file_search_result_msg := _find_tool_message(messages, "file_search_call"):
        file_search_result_msg.content = json.dumps(file_citations)

    if web_search_result_msg := _find_tool_message(messages, "web_search_call"):
        web_search_result_msg.content = json.dumps(web_citations)


def _find_tool_message(messages, tool_type):
    return next((msg for msg in messages if msg.role == "tool" and msg.content == tool_type), None)


def _parse_tools(inputs: dict[str, Any]) -> list[ChatTool]:
    tools = inputs.get("tools", [])

    parsed_tools = []
    for tool in tools:
        tool_type = tool.get("type", "function")
        if tool_type == "function":
            if "function" in tool:
                # ChatCompletion API style
                parsed_tools.append(ChatTool(**tool))
            else:
                # Responses API style
                definition = {k: v for k, v in tool.items() if k != "type"}
                parsed_tools.append(
                    ChatTool(
                        type="function",
                        function=FunctionToolDefinition(**definition),
                    )
                )
        elif tool_type in _RESPONSE_API_BUILT_IN_TOOLS:
            parsed_tools.append(
                ChatTool(
                    type="function",
                    function=FunctionToolDefinition(
                        name=tool_type,
                    ),
                )
            )
        else:
            raise MlflowException(f"Unknown tool type: {tool_type}")

    return parsed_tools
