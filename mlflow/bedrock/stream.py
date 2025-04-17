import json
import logging
from typing import Any, Optional

from botocore.eventstream import EventStream

from mlflow.bedrock.chat import convert_message_to_mlflow_chat
from mlflow.bedrock.utils import capture_exception
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
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


class InvokeModelStreamWrapper(BaseEventStreamWrapper):
    """A wrapper class for a event stream returned by the InvokeModelWithResponseStream API."""

    @capture_exception("Failed to handle event for the stream")
    def _handle_event(self, span, event):
        chunk = json.loads(event["chunk"]["bytes"])
        self._span.add_event(SpanEvent(name=chunk["type"], attributes={"json": json.dumps(chunk)}))

    def _close(self):
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
        # Record the accumulated response as the output of the span
        converse_response = self._response_builder.build()
        self._span.set_outputs(converse_response)

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

    def process_event(self, event_name: str, event_attr: dict):
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
