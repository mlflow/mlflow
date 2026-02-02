import json
from itertools import tee
from typing import Any, Generator, Iterator
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, model_validator

from mlflow.types.agent import ChatContext
from mlflow.types.responses_helpers import (
    BaseRequestPayload,
    Message,
    OutputItem,
    Response,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextAnnotationDeltaEvent,
    ResponseTextDeltaEvent,
)

__all__ = [
    "ResponsesAgentRequest",
    "ResponsesAgentResponse",
    "ResponsesAgentStreamEvent",
]

from mlflow.types.schema import Schema
from mlflow.types.type_hints import _infer_schema_from_type_hint
from mlflow.utils.autologging_utils.logging_and_warnings import (
    MlflowEventsAndWarningsBehaviorGlobally,
)


class ResponsesAgentRequest(BaseRequestPayload):
    """Request object for ResponsesAgent.

    Args:
        input: List of simple `role` and `content` messages or output items. See examples at
            https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro#testing-out-your-agent
            and
            https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro#creating-agent-output.
        custom_inputs (Dict[str, Any]): An optional param to provide arbitrary additional context
            to the model. The dictionary values must be JSON-serializable.
            **Optional** defaults to ``None``
        context (:py:class:`mlflow.types.agent.ChatContext`): The context to be used in the chat
            endpoint. Includes conversation_id and user_id. **Optional** defaults to ``None``
    """

    input: list[Message | OutputItem]
    custom_inputs: dict[str, Any] | None = None
    context: ChatContext | None = None


class ResponsesAgentResponse(Response):
    """Response object for ResponsesAgent.

    Args:
        output: List of output items. See examples at
            https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro#creating-agent-output.
        reasoning: Reasoning parameters
        usage: Usage information
        custom_outputs (Dict[str, Any]): An optional param to provide arbitrary additional context
            from the model. The dictionary values must be JSON-serializable. **Optional**, defaults
            to ``None``
    """

    custom_outputs: dict[str, Any] | None = None


class ResponsesAgentStreamEvent(BaseModel):
    """Stream event for ResponsesAgent.
    See examples at https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro#streaming-agent-output

    Args:
        type (str): Type of the stream event
        custom_outputs (Dict[str, Any]): An optional param to provide arbitrary additional context
            from the model. The dictionary values must be JSON-serializable. **Optional**, defaults
            to ``None``
    """

    model_config = ConfigDict(extra="allow")
    type: str
    custom_outputs: dict[str, Any] | None = None

    @model_validator(mode="after")
    def check_type(self) -> "ResponsesAgentStreamEvent":
        type = self.type
        if type == "response.output_item.done":
            ResponseOutputItemDoneEvent(**self.model_dump())
        elif type == "response.output_text.delta":
            ResponseTextDeltaEvent(**self.model_dump())
        elif type == "response.output_text.annotation.added":
            ResponseTextAnnotationDeltaEvent(**self.model_dump())
        elif type == "error":
            ResponseErrorEvent(**self.model_dump())
        elif type == "response.completed":
            ResponseCompletedEvent(**self.model_dump())
        """
        unvalidated types: {
            "response.created",
            "response.in_progress",
            "response.completed",
            "response.failed",
            "response.incomplete",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.done",
            "response.output_item.added",
            "response.refusal.delta",
            "response.refusal.done",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.file_search_call.in_progress",
            "response.file_search_call.searching",
            "response.file_search_call.completed",
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
            "response.error",
        }
        """
        return self


with MlflowEventsAndWarningsBehaviorGlobally(
    reroute_warnings=False,
    disable_event_logs=True,
    disable_warnings=True,
):
    properties = _infer_schema_from_type_hint(ResponsesAgentRequest).to_dict()[0]["properties"]
    formatted_properties = [{**prop, "name": name} for name, prop in properties.items()]
    RESPONSES_AGENT_INPUT_SCHEMA = Schema.from_json(json.dumps(formatted_properties))
    RESPONSES_AGENT_OUTPUT_SCHEMA = _infer_schema_from_type_hint(ResponsesAgentResponse)
RESPONSES_AGENT_INPUT_EXAMPLE = {"input": [{"role": "user", "content": "Hello!"}]}

try:
    from langchain_core.messages import BaseMessage

    _HAS_LANGCHAIN_BASE_MESSAGE = True
except ImportError:
    _HAS_LANGCHAIN_BASE_MESSAGE = False


def responses_agent_output_reducer(
    chunks: list[ResponsesAgentStreamEvent | dict[str, Any]],
):
    """Output reducer for ResponsesAgent streaming."""
    output_items = []
    for chunk in chunks:
        # Handle both dict and pydantic object formats
        if isinstance(chunk, dict):
            chunk_type = chunk.get("type")
            if chunk_type == "response.output_item.done":
                output_items.append(chunk.get("item"))
        else:
            # Pydantic object (ResponsesAgentStreamEvent)
            if hasattr(chunk, "type") and chunk.type == "response.output_item.done":
                output_items.append(chunk.item)

    return ResponsesAgentResponse(output=output_items).model_dump(exclude_none=True)


def create_text_delta(delta: str, item_id: str) -> dict[str, Any]:
    """Helper method to create a dictionary conforming to the text delta schema for
    streaming.

    Read more at https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro#streaming-agent-output.
    """
    return {
        "type": "response.output_text.delta",
        "item_id": item_id,
        "delta": delta,
    }


def create_annotation_added(
    item_id: str, annotation: dict[str, Any], annotation_index: int | None = 0
) -> dict[str, Any]:
    """Helper method to create annotation added event."""
    return {
        "type": "response.output_text.annotation.added",
        "item_id": item_id,
        "annotation_index": annotation_index,
        "annotation": annotation,
    }


def create_text_output_item(
    text: str, id: str, annotations: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Helper method to create a dictionary conforming to the text output item schema.

    Read more at https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro#creating-agent-output.

    Args:
        text (str): The text to be outputted.
        id (str): The id of the output item.
        annotations (Optional[list[dict]]): The annotations of the output item.
    """
    content_item = {
        "text": text,
        "type": "output_text",
    }
    if annotations is not None:
        content_item["annotations"] = annotations
    return {
        "id": id,
        "content": [content_item],
        "role": "assistant",
        "type": "message",
    }


def create_reasoning_item(id: str, reasoning_text: str) -> dict[str, Any]:
    """Helper method to create a dictionary conforming to the reasoning item schema.

    Read more at https://www.mlflow.org/docs/latest/llms/responses-agent-intro/#creating-agent-output.
    """
    return {
        "type": "reasoning",
        "summary": [
            {
                "type": "summary_text",
                "text": reasoning_text,
            }
        ],
        "id": id,
    }


def create_function_call_item(id: str, call_id: str, name: str, arguments: str) -> dict[str, Any]:
    """Helper method to create a dictionary conforming to the function call item schema.

    Read more at https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro#creating-agent-output.

    Args:
        id (str): The id of the output item.
        call_id (str): The id of the function call.
        name (str): The name of the function to be called.
        arguments (str): The arguments to be passed to the function.
    """
    return {
        "type": "function_call",
        "id": id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def create_function_call_output_item(call_id: str, output: str) -> dict[str, Any]:
    """Helper method to create a dictionary conforming to the function call output item
    schema.

    Read more at https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro#creating-agent-output.

    Args:
        call_id (str): The id of the function call.
        output (str): The output of the function call.
    """
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }


def create_mcp_approval_request_item(
    id: str, arguments: str, name: str, server_label: str
) -> dict[str, Any]:
    """Helper method to create a dictionary conforming to the MCP approval request item schema.

    Read more at https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro#creating-agent-output.

    Args:
        id (str): The unique id of the approval request.
        arguments (str): A JSON string of arguments for the tool.
        name (str): The name of the tool to run.
        server_label (str): The label of the MCP server making the request.
    """
    return {
        "type": "mcp_approval_request",
        "id": id,
        "arguments": arguments,
        "name": name,
        "server_label": server_label,
    }


def create_mcp_approval_response_item(
    id: str,
    approval_request_id: str,
    approve: bool,
    reason: str | None = None,
) -> dict[str, Any]:
    """Helper method to create a dictionary conforming to the MCP approval response item schema.

    Read more at https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro#creating-agent-output.

    Args:
        id (str): The unique id of the approval response.
        approval_request_id (str): The id of the approval request being answered.
        approve (bool): Whether the request was approved.
        reason (Optional[str]): The reason for the approval.
    """
    return {
        "type": "mcp_approval_response",
        "id": id,
        "approval_request_id": approval_request_id,
        "approve": approve,
        "reason": reason,
    }


def responses_to_cc(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert from a Responses API output item to a list of ChatCompletion messages."""
    msg_type = message.get("type")
    if msg_type == "function_call":
        return [
            {
                "role": "assistant",
                "content": "tool call",  # empty content is not supported by claude models
                "tool_calls": [
                    {
                        "id": message["call_id"],
                        "type": "function",
                        "function": {
                            "arguments": message.get("arguments") or "{}",
                            "name": message["name"],
                        },
                    }
                ],
            }
        ]
    elif msg_type == "message" and isinstance(message.get("content"), list):
        return [
            {"role": message["role"], "content": content["text"]} for content in message["content"]
        ]
    elif msg_type == "reasoning":
        return [{"role": "assistant", "content": json.dumps(message["summary"])}]
    elif msg_type == "function_call_output":
        return [
            {
                "role": "tool",
                "content": message["output"],
                "tool_call_id": message["call_id"],
            }
        ]
    elif msg_type == "mcp_approval_request":
        return [
            {
                "role": "assistant",
                "content": "mcp approval request",
                "tool_calls": [
                    {
                        "id": message["id"],
                        "type": "function",
                        "function": {
                            "arguments": message.get("arguments") or "{}",
                            "name": message["name"],
                        },
                    }
                ],
            }
        ]
    elif msg_type == "mcp_approval_response":
        return [
            {
                "role": "tool",
                "content": str(message["approve"]),
                "tool_call_id": message["approval_request_id"],
            }
        ]
    compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
    filtered = {k: v for k, v in message.items() if k in compatible_keys}
    return [filtered] if filtered else []


def to_chat_completions_input(
    responses_input: list[dict[str, Any] | Message | OutputItem],
) -> list[dict[str, Any]]:
    """Convert from Responses input items to ChatCompletion dictionaries."""
    cc_msgs = []
    for msg in responses_input:
        if isinstance(msg, BaseModel):
            cc_msgs.extend(responses_to_cc(msg.model_dump()))
        else:
            cc_msgs.extend(responses_to_cc(msg))
    return cc_msgs


def output_to_responses_items_stream(
    chunks: Iterator[dict[str, Any]],
    aggregator: list[dict[str, Any]] | None = None,
) -> Generator[ResponsesAgentStreamEvent, None, None]:
    """
    For streaming, convert from various message format dicts to Responses output items,
    returning a generator of ResponsesAgentStreamEvent objects.

    If `aggregator` is provided, it will be extended with the aggregated output item dicts.

    Handles an iterator of ChatCompletion chunks or LangChain BaseMessage objects.
    """
    peeking_iter, chunks = tee(chunks)
    first_chunk = next(peeking_iter)
    if _HAS_LANGCHAIN_BASE_MESSAGE and isinstance(first_chunk, BaseMessage):
        yield from _langchain_message_stream_to_responses_stream(chunks, aggregator)
    else:
        yield from _cc_stream_to_responses_stream(chunks, aggregator)


if _HAS_LANGCHAIN_BASE_MESSAGE:

    def _langchain_message_stream_to_responses_stream(
        chunks: Iterator[BaseMessage],
        aggregator: list[dict[str, Any]] | None = None,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Convert from a stream of LangChain BaseMessage objects to a stream of
        ResponsesAgentStreamEvent objects. Skips user or human messages.
        """
        for chunk in chunks:
            message = chunk.model_dump()
            role = message["type"]
            if role == "ai":
                if message.get("content"):
                    text_output_item = create_text_output_item(
                        text=message["content"],
                        id=message.get("id") or str(uuid4()),
                    )
                    if aggregator is not None:
                        aggregator.append(text_output_item)
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done", item=text_output_item
                    )
                if tool_calls := message.get("tool_calls"):
                    for tool_call in tool_calls:
                        function_call_item = create_function_call_item(
                            id=message.get("id") or str(uuid4()),
                            call_id=tool_call["id"],
                            name=tool_call["name"],
                            arguments=json.dumps(tool_call["args"]),
                        )
                        if aggregator is not None:
                            aggregator.append(function_call_item)
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done", item=function_call_item
                        )

            elif role == "tool":
                function_call_output_item = create_function_call_output_item(
                    call_id=message["tool_call_id"],
                    output=message["content"],
                )
                if aggregator is not None:
                    aggregator.append(function_call_output_item)
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done", item=function_call_output_item
                )
            elif role == "user" or "human":
                continue


def _cc_stream_to_responses_stream(
    chunks: Iterator[dict[str, Any]],
    aggregator: list[dict[str, Any]] | None = None,
) -> Generator[ResponsesAgentStreamEvent, None, None]:
    """
    Convert from stream of ChatCompletion chunks to a stream of
    ResponsesAgentStreamEvent objects.
    """
    llm_content = ""
    reasoning_content = ""
    tool_calls = []
    msg_id = None
    for chunk in chunks:
        if chunk.get("choices") is None or len(chunk["choices"]) == 0:
            continue
        delta = chunk["choices"][0]["delta"]
        msg_id = chunk.get("id", None)
        content = delta.get("content", None)
        if tc := delta.get("tool_calls"):
            if not tool_calls:  # only accommodate for single tool call right now
                tool_calls = tc
            else:
                tool_calls[0]["function"]["arguments"] += tc[0]["function"]["arguments"]
        elif content is not None:
            # logic for content item format
            # https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference#contentitem
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "reasoning":
                            reasoning_content += item.get("summary", [])[0].get("text", "")
                        if item.get("type") == "text" and item.get("text"):
                            llm_content += item["text"]
                            yield ResponsesAgentStreamEvent(
                                **create_text_delta(item["text"], item_id=msg_id)
                            )
            elif reasoning_content != "":
                # reasoning content is done streaming
                reasoning_item = create_reasoning_item(msg_id, reasoning_content)
                if aggregator is not None:
                    aggregator.append(reasoning_item)
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=reasoning_item,
                )
                reasoning_content = ""

            if isinstance(content, str):
                llm_content += content
                yield ResponsesAgentStreamEvent(**create_text_delta(content, item_id=msg_id))

    # yield an `output_item.done` `output_text` event that aggregates the stream
    # this enables tracing and payload logging
    if llm_content:
        text_output_item = create_text_output_item(llm_content, msg_id)
        if aggregator is not None:
            aggregator.append(text_output_item)
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=text_output_item,
        )

    for tool_call in tool_calls:
        function_call_output_item = create_function_call_item(
            msg_id,
            tool_call["id"],
            tool_call["function"]["name"],
            tool_call["function"]["arguments"],
        )
        if aggregator is not None:
            aggregator.append(function_call_output_item)
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=function_call_output_item,
        )
