import time
from typing import Any
from unittest import mock

import pytest

import mlflow
from mlflow.entities.trace_state import TraceState
from mlflow.gateway.providers.base import BaseProvider, PassthroughAction
from mlflow.gateway.schemas import chat, embeddings
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracking.fluent import _get_experiment_id


def get_traces():
    return TracingClient().search_traces(locations=[_get_experiment_id()])


class MockProvider(BaseProvider):
    """Mock provider for testing tracing functionality built into BaseProvider."""

    NAME = "MockProvider"

    class MockConfig:
        pass

    CONFIG_TYPE = MockConfig

    def __init__(self, enable_tracing: bool = False):
        self.config = mock.MagicMock()
        self.config.model.name = "mock-model"
        self._enable_tracing = enable_tracing
        # These will be set by tests to control behavior
        self._chat_response = None
        self._chat_stream_chunks = None
        self._chat_error = None
        self._embeddings_response = None
        self._passthrough_response = None
        self._passthrough_error = None

    async def _chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        if self._chat_error:
            raise self._chat_error
        return self._chat_response

    async def _chat_stream(self, payload: chat.RequestPayload):
        for chunk in self._chat_stream_chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def _embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        return self._embeddings_response

    async def _passthrough(
        self,
        action: PassthroughAction,
        payload: dict[str, Any],
        headers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._passthrough_error:
            raise self._passthrough_error
        return self._passthrough_response


@pytest.fixture
def mock_provider():
    return MockProvider(enable_tracing=True)


async def _collect_chunks(async_gen):
    return [chunk async for chunk in async_gen]


@pytest.mark.asyncio
async def test_chat_stream_captures_usage_from_final_chunk(mock_provider):
    # Set up mock chunks - final chunk has usage
    mock_provider._chat_stream_chunks = [
        chat.StreamResponsePayload(
            id="1",
            created=int(time.time()),
            model="mock-model",
            choices=[chat.StreamChoice(index=0, delta=chat.StreamDelta(content="Hello"))],
        ),
        chat.StreamResponsePayload(
            id="2",
            created=int(time.time()),
            model="mock-model",
            choices=[chat.StreamChoice(index=0, delta=chat.StreamDelta(content=" world"))],
        ),
        chat.StreamResponsePayload(
            id="3",
            created=int(time.time()),
            model="mock-model",
            choices=[chat.StreamChoice(index=0, delta=chat.StreamDelta(content="!"))],
            usage=chat.ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ),
    ]

    # Create a parent trace context so the provider creates spans
    @mlflow.trace
    async def traced_operation():
        payload = chat.RequestPayload(messages=[chat.RequestMessage(role="user", content="Hi")])
        return await _collect_chunks(mock_provider.chat_stream(payload))

    chunks = await traced_operation()

    # Verify all chunks were yielded
    assert len(chunks) == 3

    # Get traces and verify
    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.state == TraceState.OK

    # Find the provider span (child of the root span)
    span_name_to_span = {span.name: span for span in trace.data.spans}
    assert "traced_operation" in span_name_to_span
    assert "provider/MockProvider/mock-model" in span_name_to_span

    provider_span = span_name_to_span["provider/MockProvider/mock-model"]
    assert provider_span.attributes.get(SpanAttributeKey.MODEL_PROVIDER) == "MockProvider"
    assert provider_span.attributes.get(SpanAttributeKey.MODEL) == "mock-model"
    assert provider_span.attributes.get("method") == "chat_stream"
    assert provider_span.attributes.get("streaming") is True

    # Verify usage was captured
    token_usage = provider_span.attributes.get(SpanAttributeKey.CHAT_USAGE)
    assert token_usage is not None
    assert token_usage[TokenUsageKey.INPUT_TOKENS] == 10
    assert token_usage[TokenUsageKey.OUTPUT_TOKENS] == 5
    assert token_usage[TokenUsageKey.TOTAL_TOKENS] == 15


@pytest.mark.asyncio
async def test_chat_stream_without_usage(mock_provider):
    mock_provider._chat_stream_chunks = [
        chat.StreamResponsePayload(
            id="1",
            created=int(time.time()),
            model="mock-model",
            choices=[chat.StreamChoice(index=0, delta=chat.StreamDelta(content="Hello"))],
        ),
        chat.StreamResponsePayload(
            id="2",
            created=int(time.time()),
            model="mock-model",
            choices=[chat.StreamChoice(index=0, delta=chat.StreamDelta(content=" world"))],
        ),
    ]

    @mlflow.trace
    async def traced_operation():
        payload = chat.RequestPayload(messages=[chat.RequestMessage(role="user", content="Hi")])
        return await _collect_chunks(mock_provider.chat_stream(payload))

    chunks = await traced_operation()
    assert len(chunks) == 2

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.state == TraceState.OK

    span_name_to_span = {span.name: span for span in trace.data.spans}
    provider_span = span_name_to_span["provider/MockProvider/mock-model"]

    # Verify no usage attributes were set
    assert provider_span.attributes.get(SpanAttributeKey.CHAT_USAGE) is None


@pytest.mark.asyncio
async def test_chat_stream_no_active_span(mock_provider):
    mock_provider._chat_stream_chunks = [
        chat.StreamResponsePayload(
            id="1",
            created=int(time.time()),
            model="mock-model",
            choices=[chat.StreamChoice(index=0, delta=chat.StreamDelta(content="Hello"))],
        ),
        chat.StreamResponsePayload(
            id="2",
            created=int(time.time()),
            model="mock-model",
            choices=[chat.StreamChoice(index=0, delta=chat.StreamDelta(content=" world"))],
        ),
    ]

    # Call without a parent trace context
    payload = chat.RequestPayload(messages=[chat.RequestMessage(role="user", content="Hi")])
    chunks = await _collect_chunks(mock_provider.chat_stream(payload))

    assert len(chunks) == 2

    # No traces should be created
    traces = get_traces()
    assert len(traces) == 0


@pytest.mark.asyncio
async def test_chat_stream_handles_error(mock_provider):
    mock_provider._chat_stream_chunks = [
        chat.StreamResponsePayload(
            id="1",
            created=int(time.time()),
            model="mock-model",
            choices=[chat.StreamChoice(index=0, delta=chat.StreamDelta(content="Hello"))],
        ),
        ValueError("Stream error"),  # Error will be raised
    ]

    @mlflow.trace
    async def traced_operation():
        payload = chat.RequestPayload(messages=[chat.RequestMessage(role="user", content="Hi")])
        return await _collect_chunks(mock_provider.chat_stream(payload))

    with pytest.raises(ValueError, match="Stream error"):
        await traced_operation()

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    # The root span should have error status because the exception propagated
    assert trace.info.state == TraceState.ERROR

    span_name_to_span = {span.name: span for span in trace.data.spans}
    provider_span = span_name_to_span["provider/MockProvider/mock-model"]

    # Verify error was captured as an exception event
    exception_events = [e for e in provider_span.events if e.name == "exception"]
    assert len(exception_events) == 1
    assert exception_events[0].attributes["exception.message"] == "Stream error"
    assert exception_events[0].attributes["exception.type"] == "ValueError"


@pytest.mark.asyncio
async def test_chat_stream_partial_usage(mock_provider):
    mock_provider._chat_stream_chunks = [
        chat.StreamResponsePayload(
            id="1",
            created=int(time.time()),
            model="mock-model",
            choices=[chat.StreamChoice(index=0, delta=chat.StreamDelta(content="!"))],
            usage=chat.ChatUsage(prompt_tokens=10, completion_tokens=None, total_tokens=None),
        ),
    ]

    @mlflow.trace
    async def traced_operation():
        payload = chat.RequestPayload(messages=[chat.RequestMessage(role="user", content="Hi")])
        return await _collect_chunks(mock_provider.chat_stream(payload))

    await traced_operation()

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    span_name_to_span = {span.name: span for span in trace.data.spans}
    provider_span = span_name_to_span["provider/MockProvider/mock-model"]

    # Verify only input_tokens was set (partial usage)
    token_usage = provider_span.attributes.get(SpanAttributeKey.CHAT_USAGE)
    assert token_usage is not None
    assert token_usage[TokenUsageKey.INPUT_TOKENS] == 10
    assert TokenUsageKey.OUTPUT_TOKENS not in token_usage
    assert TokenUsageKey.TOTAL_TOKENS not in token_usage


@pytest.mark.asyncio
async def test_chat_non_streaming(mock_provider):
    mock_provider._chat_response = chat.ResponsePayload(
        id="1",
        created=int(time.time()),
        model="mock-model",
        choices=[
            chat.Choice(
                index=0,
                message=chat.ResponseMessage(role="assistant", content="Hello!"),
                finish_reason="stop",
            )
        ],
        usage=chat.ChatUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )

    @mlflow.trace
    async def traced_operation():
        payload = chat.RequestPayload(messages=[chat.RequestMessage(role="user", content="Hi")])
        return await mock_provider.chat(payload)

    result = await traced_operation()
    assert result.choices[0].message.content == "Hello!"

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.state == TraceState.OK

    span_name_to_span = {span.name: span for span in trace.data.spans}
    provider_span = span_name_to_span["provider/MockProvider/mock-model"]

    assert provider_span.attributes.get(SpanAttributeKey.MODEL_PROVIDER) == "MockProvider"
    assert provider_span.attributes.get(SpanAttributeKey.MODEL) == "mock-model"
    assert provider_span.attributes.get("method") == "chat"
    # Non-streaming should not have streaming attribute
    assert provider_span.attributes.get("streaming") is None

    # Verify usage was captured
    token_usage = provider_span.attributes.get(SpanAttributeKey.CHAT_USAGE)
    assert token_usage is not None
    assert token_usage[TokenUsageKey.INPUT_TOKENS] == 5
    assert token_usage[TokenUsageKey.OUTPUT_TOKENS] == 3
    assert token_usage[TokenUsageKey.TOTAL_TOKENS] == 8


@pytest.mark.asyncio
async def test_chat_non_streaming_error(mock_provider):
    mock_provider._chat_error = RuntimeError("Method failed")

    @mlflow.trace
    async def traced_operation():
        payload = chat.RequestPayload(messages=[chat.RequestMessage(role="user", content="Hi")])
        return await mock_provider.chat(payload)

    with pytest.raises(RuntimeError, match="Method failed"):
        await traced_operation()

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    span_name_to_span = {span.name: span for span in trace.data.spans}
    provider_span = span_name_to_span["provider/MockProvider/mock-model"]

    # Verify error was captured as an exception event
    exception_events = [e for e in provider_span.events if e.name == "exception"]
    assert len(exception_events) == 1
    assert exception_events[0].attributes["exception.message"] == "Method failed"
    assert exception_events[0].attributes["exception.type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_embeddings(mock_provider):
    mock_provider._embeddings_response = embeddings.ResponsePayload(
        data=[embeddings.EmbeddingObject(embedding=[0.1, 0.2, 0.3], index=0)],
        model="mock-model",
        usage=embeddings.EmbeddingsUsage(prompt_tokens=4, total_tokens=4),
    )

    @mlflow.trace
    async def traced_operation():
        payload = embeddings.RequestPayload(input="Hello")
        return await mock_provider.embeddings(payload)

    result = await traced_operation()
    assert len(result.data) == 1
    assert result.data[0].embedding == [0.1, 0.2, 0.3]

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.state == TraceState.OK

    span_name_to_span = {span.name: span for span in trace.data.spans}
    provider_span = span_name_to_span["provider/MockProvider/mock-model"]

    assert provider_span.attributes.get(SpanAttributeKey.MODEL_PROVIDER) == "MockProvider"
    assert provider_span.attributes.get(SpanAttributeKey.MODEL) == "mock-model"
    assert provider_span.attributes.get("method") == "embeddings"


@pytest.mark.asyncio
async def test_passthrough_with_tracing(mock_provider):
    mock_provider._passthrough_response = {"id": "1", "result": "success"}

    result = await mock_provider.passthrough(
        action=PassthroughAction.OPENAI_CHAT,
        payload={"messages": [{"role": "user", "content": "Hi"}]},
    )

    assert result == {"id": "1", "result": "success"}

    # Passthrough with @mlflow.trace creates its own trace
    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.state == TraceState.OK

    # The span should have provider attributes and action
    span_name_to_span = {span.name: span for span in trace.data.spans}
    assert "provider/MockProvider/mock-model" in span_name_to_span

    passthrough_span = span_name_to_span["provider/MockProvider/mock-model"]
    assert passthrough_span.attributes.get(SpanAttributeKey.MODEL_PROVIDER) == "MockProvider"
    assert passthrough_span.attributes.get(SpanAttributeKey.MODEL) == "mock-model"
    assert passthrough_span.attributes.get("action") == "openai_chat"


@pytest.mark.asyncio
async def test_passthrough_without_tracing():
    provider = MockProvider(enable_tracing=False)
    provider._passthrough_response = {"id": "1", "result": "success"}

    result = await provider.passthrough(
        action=PassthroughAction.OPENAI_CHAT,
        payload={"messages": [{"role": "user", "content": "Hi"}]},
    )

    assert result == {"id": "1", "result": "success"}

    # No traces should be created when tracing is disabled
    traces = get_traces()
    assert len(traces) == 0


@pytest.mark.asyncio
async def test_passthrough_error_with_tracing(mock_provider):
    mock_provider._passthrough_error = RuntimeError("Passthrough failed")

    with pytest.raises(RuntimeError, match="Passthrough failed"):
        await mock_provider.passthrough(
            action=PassthroughAction.OPENAI_CHAT,
            payload={"messages": [{"role": "user", "content": "Hi"}]},
        )

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.state == TraceState.ERROR
