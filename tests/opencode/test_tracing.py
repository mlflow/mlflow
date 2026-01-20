import importlib
import logging
from pathlib import Path

import pytest

import mlflow
import mlflow.opencode.tracing as tracing_module
from mlflow.entities.span import SpanType
from mlflow.opencode.tracing import (
    OPENCODE_TRACING_LEVEL,
    extract_assistant_response,
    extract_user_prompt,
    find_last_user_message_index,
    process_session,
    setup_logging,
    timestamp_to_ns,
)
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey


def test_timestamp_to_ns_milliseconds():
    # Opencode uses milliseconds timestamps
    ms_timestamp = 1705312245123
    result = timestamp_to_ns(ms_timestamp)

    # Should convert milliseconds to nanoseconds
    # Use approximate comparison due to floating point precision
    expected = int(ms_timestamp * 1_000_000)
    assert abs(result - expected) < 1000  # Allow small precision difference


def test_timestamp_to_ns_none():
    result = timestamp_to_ns(None)
    assert result is None


def test_timestamp_to_ns_float():
    ms_timestamp = 1705312245123.456
    result = timestamp_to_ns(ms_timestamp)

    # Should convert to nanoseconds
    assert isinstance(result, int)
    assert result > 0


def test_setup_logging_creates_logger(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    logger = setup_logging()

    # Verify logger was created
    assert logger is not None
    assert logger.name == "mlflow.opencode.tracing"

    # Verify log directory was created
    log_dir = tmp_path / ".opencode" / "mlflow"
    assert log_dir.exists()
    assert log_dir.is_dir()


def test_custom_logging_level():
    setup_logging()

    assert OPENCODE_TRACING_LEVEL > logging.INFO
    assert OPENCODE_TRACING_LEVEL < logging.WARNING
    assert logging.getLevelName(OPENCODE_TRACING_LEVEL) == "OPENCODE_TRACING"


def test_get_logger_lazy_initialization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    # Force reload to reset the module state
    importlib.reload(tracing_module)

    log_dir = tmp_path / ".opencode" / "mlflow"

    # Before calling get_logger(), the log directory should NOT exist
    assert not log_dir.exists()

    # Call get_logger() for the first time - this should trigger initialization
    logger1 = tracing_module.get_logger()

    # After calling get_logger(), the log directory SHOULD exist
    assert log_dir.exists()
    assert log_dir.is_dir()

    # Verify logger was created properly
    assert logger1 is not None
    assert logger1.name == "mlflow.opencode.tracing"

    # Call get_logger() again - should return the same logger instance
    logger2 = tracing_module.get_logger()
    assert logger2 is logger1


# Sample Opencode session data for testing
DUMMY_MESSAGES = [
    {
        "info": {
            "id": "msg_1",
            "sessionID": "session_123",
            "role": "user",
            "time": {"created": 1705312200000},
            "agent": "build",
            "model": {"providerID": "anthropic", "modelID": "claude-sonnet-4-20250514"},
        },
        "parts": [{"id": "part_1", "type": "text", "text": "What is 2 + 2?"}],
    },
    {
        "info": {
            "id": "msg_2",
            "sessionID": "session_123",
            "role": "assistant",
            "time": {"created": 1705312201000, "completed": 1705312202000},
            "parentID": "msg_1",
            "modelID": "claude-sonnet-4-20250514",
            "providerID": "anthropic",
            "mode": "build",
            "path": {"cwd": "/test", "root": "/test"},
            "cost": 0.001,
            "tokens": {
                "input": 150,
                "output": 25,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
        },
        "parts": [{"id": "part_2", "type": "text", "text": "Let me calculate that for you."}],
    },
    {
        "info": {
            "id": "msg_3",
            "sessionID": "session_123",
            "role": "assistant",
            "time": {"created": 1705312202000, "completed": 1705312203000},
            "parentID": "msg_1",
            "modelID": "claude-sonnet-4-20250514",
            "providerID": "anthropic",
            "mode": "build",
            "path": {"cwd": "/test", "root": "/test"},
            "cost": 0.002,
            "tokens": {
                "input": 200,
                "output": 50,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
        },
        "parts": [
            {
                "id": "part_3",
                "type": "tool",
                "tool": "Bash",
                "callID": "call_123",
                "state": {
                    "status": "completed",
                    "input": {"command": "echo $((2 + 2))"},
                    "output": "4",
                    "title": "Bash command",
                    "metadata": {},
                    "time": {"start": 1705312202500, "end": 1705312202800},
                },
            }
        ],
    },
    {
        "info": {
            "id": "msg_4",
            "sessionID": "session_123",
            "role": "assistant",
            "time": {"created": 1705312203000, "completed": 1705312204000},
            "parentID": "msg_1",
            "modelID": "claude-sonnet-4-20250514",
            "providerID": "anthropic",
            "mode": "build",
            "path": {"cwd": "/test", "root": "/test"},
            "cost": 0.001,
            "tokens": {
                "input": 100,
                "output": 20,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
        },
        "parts": [{"id": "part_4", "type": "text", "text": "The answer is 4."}],
    },
]

DUMMY_SESSION_INFO = {
    "id": "session_123",
    "projectID": "project_1",
    "directory": "/test/project",
    "title": "Math calculation",
    "version": "1.0.0",
    "time": {"created": 1705312200000, "updated": 1705312204000},
}


def test_extract_user_prompt():
    prompt = extract_user_prompt(DUMMY_MESSAGES)
    assert prompt == "What is 2 + 2?"


def test_extract_assistant_response():
    response = extract_assistant_response(DUMMY_MESSAGES)
    assert response == "The answer is 4."


def test_find_last_user_message_index():
    idx = find_last_user_message_index(DUMMY_MESSAGES)
    assert idx == 0  # First message is the user message


def test_find_last_user_message_index_none():
    # Messages with no user message
    assistant_only = [{"info": {"role": "assistant"}, "parts": []}]
    idx = find_last_user_message_index(assistant_only)
    assert idx is None


def test_process_session_creates_trace():
    trace = process_session("session_123", DUMMY_SESSION_INFO, DUMMY_MESSAGES)

    # Verify trace was created
    assert trace is not None

    # Verify trace has spans
    spans = list(trace.search_spans())
    assert len(spans) > 0

    # Verify root span and metadata
    root_span = trace.data.spans[0]
    assert root_span.name == "opencode_conversation"
    assert root_span.span_type == SpanType.AGENT
    assert trace.info.trace_metadata.get("mlflow.trace.session") == "session_123"


def test_process_session_creates_spans():
    trace = process_session("session_123", DUMMY_SESSION_INFO, DUMMY_MESSAGES)

    assert trace is not None

    # Verify trace has spans
    spans = list(trace.search_spans())
    assert len(spans) > 0

    # Find LLM and tool spans
    llm_spans = [s for s in spans if s.span_type == SpanType.LLM]
    tool_spans = [s for s in spans if s.span_type == SpanType.TOOL]

    # Should have LLM spans for text responses (3 assistant messages, but only 2 have text)
    assert len(llm_spans) >= 2

    # Should have 1 tool span
    assert len(tool_spans) == 1

    # Verify tool span has proper attributes
    tool_span = tool_spans[0]
    assert tool_span.name == "tool_Bash"


def test_process_session_returns_none_for_empty_messages():
    result = process_session("session_123", DUMMY_SESSION_INFO, [])
    assert result is None


def test_process_session_returns_none_for_no_user_message():
    assistant_only = [{"info": {"role": "assistant"}, "parts": []}]
    result = process_session("session_123", DUMMY_SESSION_INFO, assistant_only)
    assert result is None


def test_process_session_links_trace_to_run():
    with mlflow.start_run() as run:
        trace = process_session("session_123", DUMMY_SESSION_INFO, DUMMY_MESSAGES)

        assert trace is not None
        assert trace.info.trace_metadata.get(TraceMetadataKey.SOURCE_RUN) == run.info.run_id


def test_process_session_tracks_token_usage():
    trace = process_session("session_123", DUMMY_SESSION_INFO, DUMMY_MESSAGES)

    assert trace is not None

    # Find an LLM span
    spans = list(trace.search_spans())
    llm_spans = [s for s in spans if s.span_type == SpanType.LLM]

    assert len(llm_spans) >= 1
    llm_span = llm_spans[0]

    # Verify token usage is tracked using the standardized CHAT_USAGE attribute
    token_usage = llm_span.get_attribute(SpanAttributeKey.CHAT_USAGE)
    assert token_usage is not None
    assert "input_tokens" in token_usage
    assert "output_tokens" in token_usage
    assert "total_tokens" in token_usage


def test_process_session_with_reasoning():
    messages_with_reasoning = [
        {
            "info": {
                "id": "msg_1",
                "sessionID": "session_456",
                "role": "user",
                "time": {"created": 1705312200000},
                "agent": "build",
                "model": {"providerID": "anthropic", "modelID": "claude-sonnet-4-20250514"},
            },
            "parts": [{"id": "part_1", "type": "text", "text": "Solve this problem"}],
        },
        {
            "info": {
                "id": "msg_2",
                "sessionID": "session_456",
                "role": "assistant",
                "time": {"created": 1705312201000, "completed": 1705312203000},
                "parentID": "msg_1",
                "modelID": "claude-sonnet-4-20250514",
                "providerID": "anthropic",
                "mode": "build",
                "path": {"cwd": "/test", "root": "/test"},
                "cost": 0.005,
                "tokens": {
                    "input": 100,
                    "output": 50,
                    "reasoning": 200,
                    "cache": {"read": 10, "write": 20},
                },
            },
            "parts": [
                {
                    "id": "part_r1",
                    "type": "reasoning",
                    "text": "Let me think about this...",
                    "time": {"start": 1705312201000, "end": 1705312202000},
                },
                {"id": "part_t1", "type": "text", "text": "Here is the solution."},
            ],
        },
    ]

    session_info = {
        "id": "session_456",
        "projectID": "project_1",
        "directory": "/test/project",
        "title": "Problem solving",
        "version": "1.0.0",
        "time": {"created": 1705312200000, "updated": 1705312203000},
    }

    trace = process_session("session_456", session_info, messages_with_reasoning)
    assert trace is not None

    # Check token usage includes reasoning tokens
    spans = list(trace.search_spans())
    llm_spans = [s for s in spans if s.span_type == SpanType.LLM]
    assert len(llm_spans) >= 1

    token_usage = llm_spans[0].get_attribute(SpanAttributeKey.CHAT_USAGE)
    assert token_usage is not None
    # Total should include reasoning tokens
    assert token_usage["total_tokens"] == 100 + 50 + 200
