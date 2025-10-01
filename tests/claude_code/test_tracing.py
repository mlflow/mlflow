"""Simplified tests for mlflow.claude_code.tracing module."""

import asyncio
import json
import logging

import pytest

from mlflow.claude_code.hooks import sdk_stop_hook_handler
from mlflow.claude_code.tracing import (
    CLAUDE_TRACING_LEVEL,
    get_hook_response,
    parse_timestamp_to_ns,
    process_transcript,
    setup_logging,
)
from mlflow.entities.span import SpanType

from tests.tracing.helper import get_traces

# ============================================================================
# TIMESTAMP PARSING TESTS
# ============================================================================


def test_parse_timestamp_to_ns_iso_string():
    """Test parsing ISO timestamp string to nanoseconds."""
    iso_timestamp = "2024-01-15T10:30:45.123456Z"
    result = parse_timestamp_to_ns(iso_timestamp)

    # Verify it returns an integer (nanoseconds)
    assert isinstance(result, int)
    assert result > 0


def test_parse_timestamp_to_ns_unix_seconds():
    """Test parsing Unix timestamp (seconds) to nanoseconds."""
    unix_timestamp = 1705312245.123456
    result = parse_timestamp_to_ns(unix_timestamp)

    # Should convert seconds to nanoseconds
    expected = int(unix_timestamp * 1_000_000_000)
    assert result == expected


def test_parse_timestamp_to_ns_large_number():
    """Test parsing large timestamp numbers."""
    large_timestamp = 1705312245123
    result = parse_timestamp_to_ns(large_timestamp)

    # Function treats large numbers as seconds and converts to nanoseconds
    # Just verify we get a reasonable nanosecond value
    assert isinstance(result, int)
    assert result > 0


# ============================================================================
# LOGGING TESTS
# ============================================================================


def test_setup_logging_creates_logger(monkeypatch, tmp_path):
    """Test that setup_logging returns a logger."""
    monkeypatch.chdir(tmp_path)
    logger = setup_logging()

    # Verify logger was created
    assert logger is not None
    assert logger.name == "mlflow.claude_code.tracing"

    # Verify log directory was created
    log_dir = tmp_path / ".claude" / "mlflow"
    assert log_dir.exists()
    assert log_dir.is_dir()


def test_custom_logging_level():
    """Test that custom claude_tracing logging level is configured."""
    assert CLAUDE_TRACING_LEVEL > logging.INFO
    assert CLAUDE_TRACING_LEVEL < logging.WARNING
    assert logging.getLevelName(CLAUDE_TRACING_LEVEL) == "CLAUDE_TRACING"


def test_logger_has_claude_tracing_method(monkeypatch, tmp_path):
    """Test that logger has claude_tracing method."""
    monkeypatch.chdir(tmp_path)
    logger = setup_logging()
    assert hasattr(logger, "claude_tracing")


# ============================================================================
# HOOK RESPONSE TESTS
# ============================================================================


def test_get_hook_response_success():
    """Test get_hook_response returns success response."""
    response = get_hook_response()
    assert response == {"continue": True}


def test_get_hook_response_with_error():
    """Test get_hook_response returns error response."""
    response = get_hook_response(error="Test error")
    assert response == {"continue": False, "stopReason": "Test error"}


def test_get_hook_response_with_additional_fields():
    """Test get_hook_response accepts additional fields."""
    response = get_hook_response(custom_field="value")
    assert response == {"continue": True, "custom_field": "value"}


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

# Sample Claude Code transcript for testing
DUMMY_TRANSCRIPT = [
    {
        "type": "user",
        "message": {"role": "user", "content": "What is 2 + 2?"},
        "timestamp": "2025-01-15T10:00:00.000Z",
        "sessionId": "test-session-123",
    },
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Let me calculate that for you."}],
        },
        "timestamp": "2025-01-15T10:00:01.000Z",
    },
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "Bash",
                    "input": {"command": "echo $((2 + 2))"},
                }
            ],
        },
        "timestamp": "2025-01-15T10:00:02.000Z",
    },
    {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool_123", "content": "4"}],
        },
        "timestamp": "2025-01-15T10:00:03.000Z",
    },
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "The answer is 4."}],
        },
        "timestamp": "2025-01-15T10:00:04.000Z",
    },
]


@pytest.fixture
def mock_transcript_file(tmp_path):
    """Create a mock transcript file."""
    transcript_path = tmp_path / "transcript.jsonl"
    with open(transcript_path, "w") as f:
        for entry in DUMMY_TRANSCRIPT:
            f.write(json.dumps(entry) + "\n")
    return str(transcript_path)


def test_sdk_stop_hook_handler_creates_trace(mock_transcript_file):
    """Test that sdk_stop_hook_handler creates an MLflow trace."""
    from mlflow.claude_code.hooks import sdk_stop_hook_handler

    async def test():
        input_data = {
            "session_id": "test-session-123",
            "transcript_path": mock_transcript_file,
        }

        result = await sdk_stop_hook_handler(input_data, None, None)
        assert result["continue"] is True

        # Verify trace was created
        traces = get_traces()
        assert len(traces) == 1
        trace = traces[0]

        # Verify trace has spans
        spans = list(trace.search_spans())
        assert len(spans) > 0

        # Verify root span and metadata
        root_span = trace.data.spans[0]
        assert root_span.name == "claude_code_conversation"
        assert root_span.span_type == SpanType.AGENT
        assert trace.info.trace_metadata.get("mlflow.trace.session") == "test-session-123"

    asyncio.run(test())


def test_process_transcript_creates_spans(mock_transcript_file):
    """Test that process_transcript creates proper span structure."""
    trace = process_transcript(mock_transcript_file, "test-session-123")

    assert trace is not None

    # Verify trace has spans
    spans = list(trace.search_spans())
    assert len(spans) > 0

    # Find LLM and tool spans
    llm_spans = [s for s in spans if s.span_type == SpanType.LLM]
    tool_spans = [s for s in spans if s.span_type == SpanType.TOOL]

    assert len(llm_spans) == 2
    assert len(tool_spans) == 1

    # Verify tool span has proper attributes
    tool_span = tool_spans[0]
    assert tool_span.name == "tool_Bash"


def test_sdk_stop_hook_handler_handles_missing_transcript():
    """Test that sdk_stop_hook_handler handles missing transcript gracefully."""

    async def test():
        input_data = {
            "session_id": "test-session-123",
            "transcript_path": "/nonexistent/path/transcript.jsonl",
        }

        result = await sdk_stop_hook_handler(input_data, None, None)
        assert result["continue"] is False
        assert "stopReason" in result

    asyncio.run(test())
