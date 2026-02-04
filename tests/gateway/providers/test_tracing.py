from unittest import mock

import pytest

import mlflow
from mlflow.entities.trace_state import TraceState
from mlflow.gateway.providers.base import BaseProvider
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.types.chat import ChatUsage


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


class MockChunk:
    """Mock streaming chunk with optional usage."""

    def __init__(self, content: str, usage: ChatUsage | None = None):
        self.content = content
        self.usage = usage


@pytest.fixture
def mock_provider():
    return MockProvider(enable_tracing=True)


async def _collect_chunks(async_gen):
    return [chunk async for chunk in async_gen]


@pytest.mark.asyncio
async def test_maybe_trace_stream_method_captures_usage_from_final_chunk(mock_provider):
    async def mock_stream(*args, **kwargs):
        yield MockChunk("Hello")
        yield MockChunk(" world")
        usage = ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        yield MockChunk("!", usage=usage)

    # Create a parent trace context so the wrapper creates spans
    @mlflow.trace
    async def traced_operation():
        return await _collect_chunks(
            mock_provider._maybe_trace_stream_method("test_stream", mock_stream)
        )

    chunks = await traced_operation()

    # Verify all chunks were yielded
    assert len(chunks) == 3
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"
    assert chunks[2].content == "!"

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
    assert provider_span.attributes.get("method") == "test_stream"
    assert provider_span.attributes.get("streaming") is True

    # Verify usage was captured in mlflow.chat.tokenUsage format
    token_usage = provider_span.attributes.get(SpanAttributeKey.CHAT_USAGE)
    assert token_usage is not None
    assert token_usage[TokenUsageKey.INPUT_TOKENS] == 10
    assert token_usage[TokenUsageKey.OUTPUT_TOKENS] == 5
    assert token_usage[TokenUsageKey.TOTAL_TOKENS] == 15


@pytest.mark.asyncio
async def test_maybe_trace_stream_method_without_usage(mock_provider):
    async def mock_stream(*args, **kwargs):
        yield MockChunk("Hello")
        yield MockChunk(" world")

    @mlflow.trace
    async def traced_operation():
        return await _collect_chunks(
            mock_provider._maybe_trace_stream_method("test_stream", mock_stream)
        )

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
async def test_maybe_trace_stream_method_no_active_span(mock_provider):
    async def mock_stream(*args, **kwargs):
        yield MockChunk("Hello")
        yield MockChunk(" world")

    # Call without a parent trace context
    chunks = await _collect_chunks(
        mock_provider._maybe_trace_stream_method("test_stream", mock_stream)
    )

    assert len(chunks) == 2
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"

    # No traces should be created
    traces = get_traces()
    assert len(traces) == 0


@pytest.mark.asyncio
async def test_maybe_trace_stream_method_handles_error(mock_provider):
    async def mock_stream(*args, **kwargs):
        yield MockChunk("Hello")
        raise ValueError("Stream error")

    @mlflow.trace
    async def traced_operation():
        return await _collect_chunks(
            mock_provider._maybe_trace_stream_method("test_stream", mock_stream)
        )

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
async def test_maybe_trace_stream_method_partial_usage(mock_provider):
    async def mock_stream(*args, **kwargs):
        usage = ChatUsage(prompt_tokens=10, completion_tokens=None, total_tokens=None)
        yield MockChunk("!", usage=usage)

    @mlflow.trace
    async def traced_operation():
        return await _collect_chunks(
            mock_provider._maybe_trace_stream_method("test_stream", mock_stream)
        )

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
async def test_maybe_trace_method_non_streaming(mock_provider):
    async def mock_method(*args, **kwargs):
        return "result"

    @mlflow.trace
    async def traced_operation():
        return await mock_provider._maybe_trace_method("test_method", mock_method)

    result = await traced_operation()
    assert result == "result"

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.state == TraceState.OK

    span_name_to_span = {span.name: span for span in trace.data.spans}
    provider_span = span_name_to_span["provider/MockProvider/mock-model"]

    assert provider_span.attributes.get(SpanAttributeKey.MODEL_PROVIDER) == "MockProvider"
    assert provider_span.attributes.get(SpanAttributeKey.MODEL) == "mock-model"
    assert provider_span.attributes.get("method") == "test_method"
    # Non-streaming should not have streaming attribute
    assert provider_span.attributes.get("streaming") is None


@pytest.mark.asyncio
async def test_maybe_trace_method_non_streaming_error(mock_provider):
    async def mock_method(*args, **kwargs):
        raise RuntimeError("Method failed")

    @mlflow.trace
    async def traced_operation():
        return await mock_provider._maybe_trace_method("test_method", mock_method)

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
