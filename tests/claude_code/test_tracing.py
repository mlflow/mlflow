import importlib
import json
import logging
from pathlib import Path

import pytest
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

import mlflow
import mlflow.claude_code.tracing as tracing_module
from mlflow.claude_code.tracing import (
    CLAUDE_TRACING_LEVEL,
    get_hook_response,
    parse_timestamp_to_ns,
    process_sdk_messages,
    process_transcript,
    setup_logging,
)
from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey

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
    setup_logging()

    assert CLAUDE_TRACING_LEVEL > logging.INFO
    assert CLAUDE_TRACING_LEVEL < logging.WARNING
    assert logging.getLevelName(CLAUDE_TRACING_LEVEL) == "CLAUDE_TRACING"


def test_get_logger_lazy_initialization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    # Force reload to reset the module state
    importlib.reload(tracing_module)

    log_dir = tmp_path / ".claude" / "mlflow"

    # Before calling get_logger(), the log directory should NOT exist
    assert not log_dir.exists()

    # Call get_logger() for the first time - this should trigger initialization
    logger1 = tracing_module.get_logger()

    # After calling get_logger(), the log directory SHOULD exist
    assert log_dir.exists()
    assert log_dir.is_dir()

    # Verify logger was created properly
    assert logger1 is not None
    assert logger1.name == "mlflow.claude_code.tracing"

    # Call get_logger() again - should return the same logger instance
    logger2 = tracing_module.get_logger()
    assert logger2 is logger1


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


# Sample Claude Code transcript with token usage for testing
DUMMY_TRANSCRIPT_WITH_USAGE = [
    {
        "type": "user",
        "message": {"role": "user", "content": "Hello Claude!"},
        "timestamp": "2025-01-15T10:00:00.000Z",
        "sessionId": "test-session-usage",
    },
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello! How can I help you today?"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 150, "output_tokens": 25},
        },
        "timestamp": "2025-01-15T10:00:01.000Z",
    },
]


@pytest.fixture
def mock_transcript_file_with_usage(tmp_path):
    transcript_path = tmp_path / "transcript_with_usage.jsonl"
    with open(transcript_path, "w") as f:
        for entry in DUMMY_TRANSCRIPT_WITH_USAGE:
            f.write(json.dumps(entry) + "\n")
    return str(transcript_path)


def test_process_transcript_tracks_token_usage(mock_transcript_file_with_usage):
    trace = process_transcript(mock_transcript_file_with_usage, "test-session-usage")

    assert trace is not None

    # Find the LLM span
    spans = list(trace.search_spans())
    llm_spans = [s for s in spans if s.span_type == SpanType.LLM]

    assert len(llm_spans) == 1
    llm_span = llm_spans[0]

    # Verify token usage is tracked using the standardized CHAT_USAGE attribute
    token_usage = llm_span.get_attribute(SpanAttributeKey.CHAT_USAGE)
    assert token_usage is not None
    assert token_usage["input_tokens"] == 150
    assert token_usage["output_tokens"] == 25
    assert token_usage["total_tokens"] == 175

    # Verify trace-level token usage aggregation works
    assert trace.info.token_usage is not None
    assert trace.info.token_usage["input_tokens"] == 150
    assert trace.info.token_usage["output_tokens"] == 25
    assert trace.info.token_usage["total_tokens"] == 175


# ============================================================================
# SDK MESSAGE PROCESSING TESTS
# ============================================================================


def test_process_sdk_messages_empty_list():
    assert process_sdk_messages([]) is None


def test_process_sdk_messages_no_user_prompt():
    messages = [
        AssistantMessage(
            content=[TextBlock(text="Hello!")],
            model="claude-sonnet-4-20250514",
        ),
    ]
    assert process_sdk_messages(messages) is None


def test_process_sdk_messages_simple_conversation():
    messages = [
        UserMessage(content="What is 2 + 2?"),
        AssistantMessage(
            content=[TextBlock(text="The answer is 4.")],
            model="claude-sonnet-4-20250514",
        ),
        ResultMessage(
            subtype="success",
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=1,
            session_id="test-sdk-session",
            usage={"input_tokens": 100, "output_tokens": 20},
        ),
    ]

    trace = process_sdk_messages(messages, "test-sdk-session")

    assert trace is not None
    spans = list(trace.search_spans())

    root_span = trace.data.spans[0]
    assert root_span.name == "claude_code_conversation"
    assert root_span.span_type == SpanType.AGENT

    # LLM span should have conversation context as input
    llm_spans = [s for s in spans if s.span_type == SpanType.LLM]
    assert len(llm_spans) == 1
    assert llm_spans[0].name == "llm_call_1"
    assert llm_spans[0].inputs["model"] == "claude-sonnet-4-20250514"
    assert llm_spans[0].inputs["messages"] == [{"role": "user", "content": "What is 2 + 2?"}]

    # Token usage from ResultMessage should be on the root span and trace level
    token_usage = root_span.get_attribute(SpanAttributeKey.CHAT_USAGE)
    assert token_usage is not None
    assert token_usage["input_tokens"] == 100
    assert token_usage["output_tokens"] == 20
    assert token_usage["total_tokens"] == 120

    assert trace.info.token_usage is not None
    assert trace.info.token_usage["input_tokens"] == 100
    assert trace.info.token_usage["output_tokens"] == 20
    assert trace.info.token_usage["total_tokens"] == 120

    # Duration should reflect ResultMessage.duration_ms (1000ms = 1s)
    duration_ns = root_span.end_time_ns - root_span.start_time_ns
    assert abs(duration_ns - 1_000_000_000) < 1_000_000  # within 1ms tolerance

    assert trace.info.trace_metadata.get("mlflow.trace.session") == "test-sdk-session"
    assert trace.info.request_preview == "What is 2 + 2?"
    assert trace.info.response_preview == "The answer is 4."


def test_process_sdk_messages_multiple_tools():
    messages = [
        UserMessage(content="Read two files"),
        AssistantMessage(
            content=[
                ToolUseBlock(id="tool_1", name="Read", input={"path": "a.py"}),
                ToolUseBlock(id="tool_2", name="Read", input={"path": "b.py"}),
            ],
            model="claude-sonnet-4-20250514",
        ),
        UserMessage(
            content=[
                ToolResultBlock(tool_use_id="tool_1", content="content of a"),
                ToolResultBlock(tool_use_id="tool_2", content="content of b"),
            ],
            tool_use_result={"tool_use_id": "tool_1"},
        ),
        AssistantMessage(
            content=[TextBlock(text="Here are the contents.")],
            model="claude-sonnet-4-20250514",
        ),
        ResultMessage(
            subtype="success",
            duration_ms=2000,
            duration_api_ms=1500,
            is_error=False,
            num_turns=2,
            session_id="multi-tool-session",
        ),
    ]

    trace = process_sdk_messages(messages, "multi-tool-session")

    assert trace is not None
    spans = list(trace.search_spans())

    tool_spans = [s for s in spans if s.span_type == SpanType.TOOL]
    assert len(tool_spans) == 2
    assert all(s.name == "tool_Read" for s in tool_spans)
    tool_results = {s.outputs["result"] for s in tool_spans}
    assert tool_results == {"content of a", "content of b"}


def test_process_sdk_messages_cache_tokens():
    """Cache tokens (cache_creation + cache_read) are included in input token count."""
    messages = [
        UserMessage(content="Hello"),
        AssistantMessage(
            content=[TextBlock(text="Hi!")],
            model="claude-sonnet-4-20250514",
        ),
        ResultMessage(
            subtype="success",
            duration_ms=5000,
            duration_api_ms=4000,
            is_error=False,
            num_turns=1,
            session_id="cache-session",
            usage={
                "input_tokens": 36,
                "cache_creation_input_tokens": 23554,
                "cache_read_input_tokens": 139035,
                "output_tokens": 3344,
            },
        ),
    ]

    trace = process_sdk_messages(messages, "cache-session")

    assert trace is not None
    root_span = trace.data.spans[0]

    # input_tokens should include cache tokens: 36 + 23554 + 139035 = 162625
    token_usage = root_span.get_attribute(SpanAttributeKey.CHAT_USAGE)
    assert token_usage["input_tokens"] == 162625
    assert token_usage["output_tokens"] == 3344
    assert token_usage["total_tokens"] == 162625 + 3344

    # Trace-level aggregation should match
    assert trace.info.token_usage["input_tokens"] == 162625
    assert trace.info.token_usage["output_tokens"] == 3344
