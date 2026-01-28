import time
from unittest import mock

import pytest

import mlflow
from mlflow.entities.trace_state import TraceState
from mlflow.gateway.providers.base import TracingProviderWrapper
from mlflow.tracing.client import TracingClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.types.chat import ChatUsage


def get_traces():
    """Get all traces from the current experiment."""
    return TracingClient().search_traces(locations=[_get_experiment_id()])


def purge_traces():
    """Delete all traces from the current experiment."""
    traces = get_traces()
    if len(traces) == 0:
        return

    TracingClient().delete_traces(
        experiment_id=_get_experiment_id(),
        max_traces=1000,
        max_timestamp_millis=int(time.time() * 1000),
    )


@pytest.fixture(autouse=True)
def clean_traces():
    """Clean up traces before and after each test."""
    purge_traces()
    yield
    purge_traces()


class MockProvider:
    """Mock provider for testing."""

    NAME = "MockProvider"

    def __init__(self):
        self.config = mock.MagicMock()
        self.config.model.name = "mock-model"


class MockChunk:
    """Mock streaming chunk with optional usage."""

    def __init__(self, content: str, usage: ChatUsage | None = None):
        self.content = content
        self.usage = usage


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def tracing_wrapper(mock_provider):
    return TracingProviderWrapper(mock_provider)


async def _collect_chunks(async_gen):
    return [chunk async for chunk in async_gen]


@pytest.mark.asyncio
async def test_trace_stream_method_captures_usage_from_final_chunk(tracing_wrapper):
    async def mock_stream(*args, **kwargs):
        yield MockChunk("Hello")
        yield MockChunk(" world")
        usage = ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        yield MockChunk("!", usage=usage)

    # Create a parent trace context so the wrapper creates spans
    @mlflow.trace
    async def traced_operation():
        return await _collect_chunks(
            tracing_wrapper._trace_stream_method("test_stream", mock_stream)
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
    assert provider_span.attributes.get("provider") == "MockProvider"
    assert provider_span.attributes.get("model") == "mock-model"
    assert provider_span.attributes.get("method") == "test_stream"
    assert provider_span.attributes.get("streaming") is True

    # Verify usage was captured
    assert provider_span.attributes.get("prompt_tokens") == 10
    assert provider_span.attributes.get("completion_tokens") == 5
    assert provider_span.attributes.get("total_tokens") == 15


@pytest.mark.asyncio
async def test_trace_stream_method_without_usage(tracing_wrapper):
    async def mock_stream(*args, **kwargs):
        yield MockChunk("Hello")
        yield MockChunk(" world")

    @mlflow.trace
    async def traced_operation():
        return await _collect_chunks(
            tracing_wrapper._trace_stream_method("test_stream", mock_stream)
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
    assert provider_span.attributes.get("prompt_tokens") is None
    assert provider_span.attributes.get("completion_tokens") is None
    assert provider_span.attributes.get("total_tokens") is None


@pytest.mark.asyncio
async def test_trace_stream_method_no_active_span(tracing_wrapper):
    async def mock_stream(*args, **kwargs):
        yield MockChunk("Hello")
        yield MockChunk(" world")

    # Call without a parent trace context
    chunks = await _collect_chunks(tracing_wrapper._trace_stream_method("test_stream", mock_stream))

    assert len(chunks) == 2
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"

    # No traces should be created
    traces = get_traces()
    assert len(traces) == 0


@pytest.mark.asyncio
async def test_trace_stream_method_handles_error(tracing_wrapper):
    async def mock_stream(*args, **kwargs):
        yield MockChunk("Hello")
        raise ValueError("Stream error")

    @mlflow.trace
    async def traced_operation():
        return await _collect_chunks(
            tracing_wrapper._trace_stream_method("test_stream", mock_stream)
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

    # Verify error was captured
    assert provider_span.attributes.get("error") == "Stream error"


@pytest.mark.asyncio
async def test_trace_stream_method_partial_usage(tracing_wrapper):
    async def mock_stream(*args, **kwargs):
        usage = ChatUsage(prompt_tokens=10, completion_tokens=None, total_tokens=None)
        yield MockChunk("!", usage=usage)

    @mlflow.trace
    async def traced_operation():
        return await _collect_chunks(
            tracing_wrapper._trace_stream_method("test_stream", mock_stream)
        )

    await traced_operation()

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    span_name_to_span = {span.name: span for span in trace.data.spans}
    provider_span = span_name_to_span["provider/MockProvider/mock-model"]

    # Verify only prompt_tokens was set
    assert provider_span.attributes.get("prompt_tokens") == 10
    assert provider_span.attributes.get("completion_tokens") is None
    assert provider_span.attributes.get("total_tokens") is None


@pytest.mark.asyncio
async def test_trace_method_non_streaming(tracing_wrapper):
    async def mock_method(*args, **kwargs):
        return "result"

    @mlflow.trace
    async def traced_operation():
        return await tracing_wrapper._trace_method("test_method", mock_method)

    result = await traced_operation()
    assert result == "result"

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.state == TraceState.OK

    span_name_to_span = {span.name: span for span in trace.data.spans}
    provider_span = span_name_to_span["provider/MockProvider/mock-model"]

    assert provider_span.attributes.get("provider") == "MockProvider"
    assert provider_span.attributes.get("model") == "mock-model"
    assert provider_span.attributes.get("method") == "test_method"
    # Non-streaming should not have streaming attribute
    assert provider_span.attributes.get("streaming") is None


@pytest.mark.asyncio
async def test_trace_method_non_streaming_error(tracing_wrapper):
    async def mock_method(*args, **kwargs):
        raise RuntimeError("Method failed")

    @mlflow.trace
    async def traced_operation():
        return await tracing_wrapper._trace_method("test_method", mock_method)

    with pytest.raises(RuntimeError, match="Method failed"):
        await traced_operation()

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    span_name_to_span = {span.name: span for span in trace.data.spans}
    provider_span = span_name_to_span["provider/MockProvider/mock-model"]

    assert provider_span.attributes.get("error") == "Method failed"
