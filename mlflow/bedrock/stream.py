import json
import logging
from typing import Any, Optional

from botocore.eventstream import EventStream

from mlflow.bedrock.chat import convert_message_to_mlflow_chat
from mlflow.bedrock.utils import (
    capture_exception,
    parse_token_usage_from_response,
)
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils import set_span_chat_messages

_logger = logging.getLogger(__name__)


class BaseEventStreamWrapper:
    """
    A wrapper class for a event stream to record events and accumulated response
    in an MLflow span if possible.

    A span should be ended when the stream is exhausted rather than when it is created.

    Args:
        stream: The original event stream to wrap.
        span: The span to record events and response in.
        inputs: The inputs to the converse API.
    """

    def __init__(
        self,
        stream: EventStream,
        span: LiveSpan,
        inputs: Optional[dict[str, Any]] = None,
    ):
        self._stream = stream
        self._span = span
        self._inputs = inputs

    def __iter__(self):
        for event in self._stream:
            self._handle_event(self._span, event)
            yield event

        # End the span when the stream is exhausted
        self._close()

    def __getattr__(self, attr):
        """Delegate all other attributes to the original stream."""
        return getattr(self._stream, attr)

    def _handle_event(self, span, event):
        """Process a single event from the stream."""
        raise NotImplementedError

    def _close(self):
        """End the span and run any finalization logic."""
        raise NotImplementedError

    @capture_exception("Failed to handle event for the stream")
    def _end_span(self):
        """End the span."""
        self._span.end()


def _extract_token_usage_from_chunk(chunk: dict[str, Any]) -> dict[str, int]:
    """Extract token usage values for each key from a streaming chunk, even if only partial."""
    usage = (
        chunk.get("message", {}).get("usage")
        if chunk.get("type") == "message_start"
        else chunk.get("usage")
    )
    if isinstance(usage, dict):
        result = parse_token_usage_from_response(usage, require_full_usage=False)
        return result if result is not None else {}
    return {}


class InvokeModelStreamWrapper(BaseEventStreamWrapper):
    """A wrapper class for a event stream returned by the InvokeModelWithResponseStream API.

    This wrapper intercepts streaming events from Bedrock's invoke_model_with_response_stream
    API and accumulates token usage information across multiple chunks. It buffers partial
    token usage data as it arrives and sets the final aggregated usage on the span when
    the stream is exhausted.

    Attributes:
        _usage_buffer (dict): Internal buffer to accumulate token usage data from
            streaming chunks. Uses TokenUsageKey constants as keys.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._usage_buffer = {}

    def _buffer_token_usage_from_chunk(self, chunk: dict[str, Any]):
        """Buffer token usage information from streaming chunks.

        Extracts token usage data from a streaming chunk and stores it in the internal
        buffer. If the same token type (input/output/total) appears in multiple chunks,
        the latest value overwrites the previous one.

        Args:
            chunk (dict[str, Any]): A streaming chunk from Bedrock API containing
                potential token usage information in the 'usage' field.
        """
        usage_dict = _extract_token_usage_from_chunk(chunk)
        for token_key, token_value in usage_dict.items():
            self._usage_buffer[token_key] = token_value

    @capture_exception("Failed to handle event for the stream")
    def _handle_event(self, span, event):
        """Process a single streaming event from the InvokeModelWithResponseStream API.

        Parses the event chunk, records it as a span event, and extracts any token
        usage information for buffering.

        Args:
            span: The MLflow span to record events in.
            event: Raw event from the Bedrock streaming API.
        """
        chunk = json.loads(event["chunk"]["bytes"])
        self._span.add_event(SpanEvent(name=chunk["type"], attributes={"json": json.dumps(chunk)}))

        # Buffer usage information from streaming chunks
        self._buffer_token_usage_from_chunk(chunk)

    def _close(self):
        """Finalize the streaming span with accumulated token usage data.

        Builds a standardized token usage dictionary from the buffered data and
        sets it as a span attribute. This method is called when the stream is
        exhausted.
        """
        # Build a standardized usage dict from buffered data using the utility function
        usage_dict = parse_token_usage_from_response(self._usage_buffer, require_full_usage=True)

        if usage_dict:
            self._span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)

        self._end_span()


class ConverseStreamWrapper(BaseEventStreamWrapper):
    """A wrapper class for event streams returned by the ConverseStream API.

    This wrapper intercepts streaming events from Bedrock's converse_stream API and
    accumulates the complete response. It handles the structured event format of the
    Converse API, including message content, tool usage, and token usage information.
    The wrapper builds the final response incrementally and sets it on the span when
    the stream is exhausted.

    Attributes:
        _response_builder (_ConverseMessageBuilder): Helper class to accumulate
            streaming response chunks into a complete message structure.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._response_builder = _ConverseMessageBuilder()

    def __getattr__(self, attr):
        """Delegate all other attributes to the original stream.

        Args:
            attr: Attribute name to delegate to the underlying stream.

        Returns:
            The attribute value from the original stream.
        """
        return getattr(self._stream, attr)

    @capture_exception("Failed to handle event for the stream")
    def _handle_event(self, span, event):
        """Process a single event from the ConverseStream API.

        Parses the structured event format from Bedrock's converse_stream API and
        accumulates the response data. Each event is also recorded as a span event
        for debugging and observability.

        For detailed event format documentation, see:
        https://boto3.amazonaws.com/v1/documentation/api/1.35.8/reference/services/bedrock-runtime/client/converse_stream.html

        Args:
            span: The MLflow span to record events in.
            event: Raw event from the Bedrock ConverseStream API.
        """
        event_name = list(event.keys())[0]
        self._response_builder.process_event(event_name, event[event_name])
        # Record raw event as a span event
        self._span.add_event(
            SpanEvent(name=event_name, attributes={"json": json.dumps(event[event_name])})
        )

    @capture_exception("Failed to record the accumulated response in the span")
    def _close(self):
        """Finalize the streaming span with complete response and token usage data.

        Builds the final response from accumulated streaming chunks, extracts token
        usage information, and sets both the response and usage data on the span.
        Also records chat message attributes in MLflow's standard format.
        """
        # Build a standardized usage dict and set it on the span if valid
        converse_response = self._response_builder.build()
        self._span.set_outputs(converse_response)

        usage_data = converse_response.get("usage")
        usage_dict = None
        if isinstance(usage_data, dict):
            usage_dict = parse_token_usage_from_response(usage_data, require_full_usage=True)

        if usage_dict:
            self._span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)

        # Record the chat message attributes in the MLflow's standard format
        messages = self._inputs.get("messages", []) + [converse_response["output"]["message"]]
        mlflow_messages = [convert_message_to_mlflow_chat(m) for m in messages]
        set_span_chat_messages(self._span, mlflow_messages)

        self._end_span()


class _ConverseMessageBuilder:
    """A helper class to accumulate the chunks of a streaming Converse API response."""

    def __init__(self):
        self._role = "assistant"
        self._text_content_buffer = ""
        self._tool_use = {}
        self._response = {}

    def process_event(self, event_name: str, event_attr: dict[str, Any]):
        if event_name == "messageStart":
            self._role = event_attr["role"]
        elif event_name == "contentBlockStart":
            # ContentBlockStart event is only used for tool usage. It carries the tool id
            # and the name, but not the input arguments.
            self._tool_use = {
                # In streaming, input is always string
                "input": "",
                **event_attr["start"]["toolUse"],
            }
        elif event_name == "contentBlockDelta":
            delta = event_attr["delta"]
            if text := delta.get("text"):
                self._text_content_buffer += text
            if tool_use := delta.get("toolUse"):
                self._tool_use["input"] += tool_use["input"]
        elif event_name == "contentBlockStop":
            pass
        elif event_name == "messageStop" or event_name == "metadata":
            self._response.update(event_attr)
        else:
            _logger.debug(f"Unknown event, skipping: {event_name}")

    def build(self) -> dict[str, Any]:
        message = {
            "role": self._role,
            "content": [{"text": self._text_content_buffer}],
        }
        if self._tool_use:
            message["content"].append({"toolUse": self._tool_use})

        self._response.update({"output": {"message": message}})

        return self._response
