from unittest.mock import patch

import anthropic
from anthropic.types import Message, TextBlock, Usage

import mlflow.anthropic
from mlflow.entities.span import SpanType

from tests.tracing.helper import get_traces

DUMMY_CREATE_MESSAGE_REQUEST = {
    "model": "test_model",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "test message"}],
}

DUMMY_CREATE_MESSAGE_RESPONSE = Message(
    id="test_id",
    content=[TextBlock(text="test answer", type="text")],
    model="test_model",
    role="assistant",
    stop_reason="end_turn",
    stop_sequence=None,
    type="message",
    usage=Usage(input_tokens=10, output_tokens=18),
)


def create(self, max_tokens, model, messages):
    return DUMMY_CREATE_MESSAGE_RESPONSE


def test_messages_autolog():
    with patch("anthropic.resources.Messages.create", new=create):
        mlflow.anthropic.autolog()
        client = anthropic.Anthropic(api_key="test_key")
        client.messages.create(**DUMMY_CREATE_MESSAGE_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Messages.create"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == DUMMY_CREATE_MESSAGE_REQUEST
    assert span.outputs == DUMMY_CREATE_MESSAGE_RESPONSE.to_dict()

    with patch("anthropic.resources.Messages.create", new=create):
        mlflow.anthropic.autolog(disable=True)
        client = anthropic.Anthropic(api_key="test_key")
        client.messages.create(**DUMMY_CREATE_MESSAGE_REQUEST)

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1
