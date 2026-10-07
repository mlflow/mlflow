import importlib
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
    _get_current_user,
    get_hook_response,
    process_sdk_messages,
    setup_logging,
)
from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey

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


def test_get_current_user_falls_back_to_username(monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.setenv("USERNAME", "windows-user")

    assert _get_current_user() == "windows-user"


def test_get_current_user_prefers_user(monkeypatch):
    monkeypatch.setenv("USER", "unix-user")
    monkeypatch.setenv("USERNAME", "windows-user")

    assert _get_current_user() == "unix-user"


def test_get_current_user_returns_empty_string_when_user_lookup_fails(monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("USERNAME", raising=False)
    assert _get_current_user() == ""


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
# ASYNC TRACE LOGGING UTILITY TESTS
# ============================================================================


def test_flush_trace_async_logging_calls_flush(monkeypatch):
    mock_exporter = type("MockExporter", (), {"_async_queue": True})()
    monkeypatch.setattr(tracing_module, "_get_trace_exporter", lambda: mock_exporter)
    flushed = []
    monkeypatch.setattr(mlflow, "flush_trace_async_logging", lambda: flushed.append(True))
    tracing_module._flush_trace_async_logging()
    assert len(flushed) == 1


def test_flush_trace_async_logging_skips_without_async_queue(monkeypatch):
    mock_exporter = object()  # no _async_queue attribute
    monkeypatch.setattr(tracing_module, "_get_trace_exporter", lambda: mock_exporter)
    flushed = []
    monkeypatch.setattr(mlflow, "flush_trace_async_logging", lambda: flushed.append(True))
    tracing_module._flush_trace_async_logging()
    assert len(flushed) == 0


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


def test_process_sdk_messages_simple_conversation(monkeypatch):
    monkeypatch.delenv("LOGNAME", raising=False)
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LNAME", raising=False)
    monkeypatch.setenv("USERNAME", "windows-user")

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

    # LLM span should have conversation context as input in Anthropic format
    llm_spans = [s for s in spans if s.span_type == SpanType.LLM]
    assert len(llm_spans) == 1
    assert llm_spans[0].name == "llm"
    assert llm_spans[0].inputs["model"] == "claude-sonnet-4-20250514"
    assert llm_spans[0].inputs["messages"] == [{"role": "user", "content": "What is 2 + 2?"}]
    assert llm_spans[0].get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "anthropic"

    # Output should be in Anthropic response format
    outputs = llm_spans[0].outputs
    assert outputs["type"] == "message"
    assert outputs["role"] == "assistant"
    assert outputs["content"] == [{"type": "text", "text": "The answer is 4."}]

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
    assert trace.info.trace_metadata.get(TraceMetadataKey.TRACE_USER) == "windows-user"
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

    # input_tokens is the non-cached input the Anthropic API reports, matching
    # mlflow.anthropic.autolog. Cache fields are exposed as separate keys so
    # consumers can compute cache hit rate without scraping transcripts.
    token_usage = root_span.get_attribute(SpanAttributeKey.CHAT_USAGE)
    assert token_usage["input_tokens"] == 36
    assert token_usage["output_tokens"] == 3344
    assert token_usage["total_tokens"] == 36 + 3344
    assert token_usage["cache_read_input_tokens"] == 139035
    assert token_usage["cache_creation_input_tokens"] == 23554

    # Trace-level aggregation should match
    assert trace.info.token_usage["input_tokens"] == 36
    assert trace.info.token_usage["output_tokens"] == 3344
