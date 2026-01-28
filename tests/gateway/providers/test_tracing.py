from unittest import mock

import pytest

from mlflow.gateway.providers.base import TracingProviderWrapper
from mlflow.types.chat import ChatUsage


class MockProvider:
    """Mock provider for testing."""

    NAME = "mock"

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
        # First chunk - no usage
        yield MockChunk("Hello")
        # Second chunk - no usage
        yield MockChunk(" world")
        # Final chunk with usage
        usage = ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        yield MockChunk("!", usage=usage)

    # Mock active span
    mock_span = mock.MagicMock()

    with (
        mock.patch("mlflow.get_current_active_span", return_value=mock_span),
        mock.patch("mlflow.start_span") as mock_start_span,
    ):
        mock_start_span.return_value = mock_span

        chunks = await _collect_chunks(
            tracing_wrapper._trace_stream_method("test_stream", mock_stream)
        )

        # Verify all chunks were yielded
        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[2].content == "!"

        # Verify usage was captured from the final chunk
        mock_span.set_attribute.assert_any_call("prompt_tokens", 10)
        mock_span.set_attribute.assert_any_call("completion_tokens", 5)
        mock_span.set_attribute.assert_any_call("total_tokens", 15)
        mock_span.set_status.assert_called_with("OK")
        mock_span.end.assert_called_once()


@pytest.mark.asyncio
async def test_trace_stream_method_without_usage(tracing_wrapper):
    async def mock_stream(*args, **kwargs):
        yield MockChunk("Hello")
        yield MockChunk(" world")

    mock_span = mock.MagicMock()

    with (
        mock.patch("mlflow.get_current_active_span", return_value=mock_span),
        mock.patch("mlflow.start_span") as mock_start_span,
    ):
        mock_start_span.return_value = mock_span

        chunks = await _collect_chunks(
            tracing_wrapper._trace_stream_method("test_stream", mock_stream)
        )

        assert len(chunks) == 2
        mock_span.set_status.assert_called_with("OK")
        mock_span.end.assert_called_once()

        # Verify no usage attributes were set
        usage_calls = [
            call
            for call in mock_span.set_attribute.call_args_list
            if call[0][0] in ("prompt_tokens", "completion_tokens", "total_tokens")
        ]
        assert len(usage_calls) == 0


@pytest.mark.asyncio
async def test_trace_stream_method_no_active_span(tracing_wrapper):
    async def mock_stream(*args, **kwargs):
        yield MockChunk("Hello")
        yield MockChunk(" world")

    with mock.patch("mlflow.get_current_active_span", return_value=None):
        chunks = await _collect_chunks(
            tracing_wrapper._trace_stream_method("test_stream", mock_stream)
        )
        assert len(chunks) == 2


@pytest.mark.asyncio
async def test_trace_stream_method_handles_error(tracing_wrapper):
    async def mock_stream(*args, **kwargs):
        yield MockChunk("Hello")
        raise ValueError("Stream error")

    mock_span = mock.MagicMock()

    with (
        mock.patch("mlflow.get_current_active_span", return_value=mock_span),
        mock.patch("mlflow.start_span") as mock_start_span,
    ):
        mock_start_span.return_value = mock_span

        with pytest.raises(ValueError, match="Stream error"):
            await _collect_chunks(tracing_wrapper._trace_stream_method("test_stream", mock_stream))

        mock_span.set_status.assert_called_with("ERROR")
        mock_span.set_attribute.assert_any_call("error", "Stream error")
        mock_span.end.assert_called_once()


@pytest.mark.asyncio
async def test_trace_stream_method_partial_usage(tracing_wrapper):
    async def mock_stream(*args, **kwargs):
        # Final chunk with only some usage fields
        usage = ChatUsage(prompt_tokens=10, completion_tokens=None, total_tokens=None)
        yield MockChunk("!", usage=usage)

    mock_span = mock.MagicMock()

    with (
        mock.patch("mlflow.get_current_active_span", return_value=mock_span),
        mock.patch("mlflow.start_span") as mock_start_span,
    ):
        mock_start_span.return_value = mock_span

        await _collect_chunks(tracing_wrapper._trace_stream_method("test_stream", mock_stream))

        # Verify only prompt_tokens was set
        set_attr_calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
        assert set_attr_calls.get("prompt_tokens") == 10
        assert (
            "completion_tokens" not in set_attr_calls
            or set_attr_calls.get("completion_tokens") is None
        )
        assert "total_tokens" not in set_attr_calls or set_attr_calls.get("total_tokens") is None
