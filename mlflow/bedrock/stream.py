import json
import logging
from typing import Any

from botocore.eventstream import EventStream

from mlflow.bedrock.utils import (
    capture_exception,
    parse_complete_token_usage_from_response,
    parse_partial_token_usage_from_response,
)
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.tracing.constant import SpanAttributeKey

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
        inputs: dict[str, Any] | None = None,
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


def _extract_token_usage_from_chunk(chunk: dict[str, Any]) -> dict[str, int] | None:
    """Extract partial token usage from streaming chunk.

    Args:
        chunk: A single streaming chunk from Bedrock API.

    Returns:
        Token usage dictionary with standardized keys, or None if no usage found.
    """
    try:
        usage = (
            chunk.get("message", {}).get("usage")
            if chunk.get("type") == "message_start"
            else chunk.get("usage")
        )
        if isinstance(usage, dict):
            return parse_partial_token_usage_from_response(usage)
        return None
    except (KeyError, TypeError, AttributeError) as e:
        _logger.debug(f"Failed to extract token usage from chunk: {e}")
        return None


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
        """Buffer token usage from streaming chunk."""
        if usage_data := _extract_token_usage_from_chunk(chunk):
            for token_key, token_value in usage_data.items():
                self._usage_buffer[token_key] = token_value

    @capture_exception("Failed to handle event for the stream")
    def _handle_event(self, span, event):
        """Process streaming event and buffer token usage."""
        chunk = json.loads(event["chunk"]["bytes"])
        self._span.add_event(SpanEvent(name=chunk["type"], attributes={"json": json.dumps(chunk)}))

        # Buffer usage information from streaming chunks
        self._buffer_token_usage_from_chunk(chunk)

    def _close(self):
        """Set accumulated token usage on span and end it."""
        # Build a standardized usage dict from buffered data using the utility function
        if usage_data := parse_complete_token_usage_from_response(self._usage_buffer):
            self._span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_data)

        self._end_span()


class ConverseStreamWrapper(BaseEventStreamWrapper):
    """A wrapper class for a event stream returned by the ConverseStream API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._response_builder = _ConverseMessageBuilder()

    def __getattr__(self, attr):
        """Delegate all other attributes to the original stream."""
        return getattr(self._stream, attr)

    @capture_exception("Failed to handle event for the stream")
    def _handle_event(self, span, event):
        """
        Process a single event from the stream.

        Refer to the following documentation for the event format:
        https://boto3.amazonaws.com/v1/documentation/api/1.35.8/reference/services/bedrock-runtime/client/converse_stream.html
        """
        event_name = list(event.keys())[0]
        self._response_builder.process_event(event_name, event[event_name])
        # Record raw event as a span event
        self._span.add_event(
            SpanEvent(name=event_name, attributes={"json": json.dumps(event[event_name])})
        )

    @capture_exception("Failed to record the accumulated response in the span")
    def _close(self):
        """Set final response and token usage on span and end it."""
        # Build a standardized usage dict and set it on the span if valid
        converse_response = self._response_builder.build()
        self._span.set_outputs(converse_response)

        raw_usage_data = converse_response.get("usage")
        if isinstance(raw_usage_data, dict):
            if usage_data := parse_complete_token_usage_from_response(raw_usage_data):
                self._span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_data)

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
