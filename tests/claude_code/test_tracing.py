import json
import logging

import pytest

import mlflow
from mlflow.claude_code.tracing import (
    CLAUDE_TRACING_LEVEL,
    get_hook_response,
    parse_timestamp_to_ns,
    process_transcript,
    setup_logging,
)
from mlflow.entities.span import SpanType
from mlflow.tracing.constant import TraceMetadataKey

# ============================================================================
# TIMESTAMP PARSING TESTS
# ============================================================================


def test_parse_timestamp_to_ns_iso_string():
    iso_timestamp = "2024-01-15T10:30:45.123456Z"
    result = parse_timestamp_to_ns(iso_timestamp)

    # Verify it returns an integer (nanoseconds)
    assert isinstance(result, int)
    assert result > 0


def test_parse_timestamp_to_ns_unix_seconds():
    unix_timestamp = 1705312245.123456
    result = parse_timestamp_to_ns(unix_timestamp)

    # Should convert seconds to nanoseconds
    expected = int(unix_timestamp * 1_000_000_000)
    assert result == expected


def test_parse_timestamp_to_ns_large_number():
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
    assert CLAUDE_TRACING_LEVEL > logging.INFO
    assert CLAUDE_TRACING_LEVEL < logging.WARNING
    assert logging.getLevelName(CLAUDE_TRACING_LEVEL) == "CLAUDE_TRACING"


# ============================================================================
# HOOK RESPONSE TESTS
# ============================================================================


def test_get_hook_response_success():
    response = get_hook_response()
    assert response == {"continue": True}


def test_get_hook_response_with_error():
    response = get_hook_response(error="Test error")
    assert response == {"continue": False, "stopReason": "Test error"}


def test_get_hook_response_with_additional_fields():
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
    transcript_path = tmp_path / "transcript.jsonl"
    with open(transcript_path, "w") as f:
        for entry in DUMMY_TRANSCRIPT:
            f.write(json.dumps(entry) + "\n")
    return str(transcript_path)


def test_process_transript_creates_trace(mock_transcript_file):
    trace = process_transcript(mock_transcript_file, "test-session-123")

    # Verify trace was created
    assert trace is not None

    # Verify trace has spans
    spans = list(trace.search_spans())
    assert len(spans) > 0

    # Verify root span and metadata
    root_span = trace.data.spans[0]
    assert root_span.name == "claude_code_conversation"
    assert root_span.span_type == SpanType.AGENT
    assert trace.info.trace_metadata.get("mlflow.trace.session") == "test-session-123"


def test_process_transcript_creates_spans(mock_transcript_file):
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


def test_process_transcript_returns_none_for_nonexistent_file():
    result = process_transcript("/nonexistent/path/transcript.jsonl", "test-session-123")
    assert result is None


def test_process_transcript_links_trace_to_run(mock_transcript_file):
    with mlflow.start_run() as run:
        trace = process_transcript(mock_transcript_file, "test-session-123")

        assert trace is not None
        assert trace.info.trace_metadata.get(TraceMetadataKey.SOURCE_RUN) == run.info.run_id
