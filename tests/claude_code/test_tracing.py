import importlib
import json
import logging
from pathlib import Path
from typing import Any

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
    METADATA_KEY_CLAUDE_CODE_VERSION,
    find_last_user_message_index,
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

    # Verify LLM spans have MESSAGE_FORMAT set to "anthropic" for Chat UI rendering
    for llm_span in llm_spans:
        assert llm_span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "anthropic"

    # Verify LLM span outputs are in Anthropic response format
    first_llm = llm_spans[0]
    outputs = first_llm.outputs
    assert outputs["type"] == "message"
    assert outputs["role"] == "assistant"
    assert isinstance(outputs["content"], list)

    # Verify LLM span inputs contain messages in Anthropic format
    inputs = first_llm.inputs
    assert "messages" in inputs
    messages = inputs["messages"]
    assert any(m["role"] == "user" for m in messages)


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


def test_process_transcript_preserves_cache_tokens(tmp_path):
    """Verify cache_read/cache_creation fields from Anthropic usage survive on the
    CHAT_USAGE span attribute so prompt-cache hit rate is observable.
    """
    transcript_entries = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Cached prompt"},
            "timestamp": "2025-01-15T10:00:00.000Z",
            "sessionId": "cache-transcript-session",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Answer using cache."}],
                "model": "claude-sonnet-4-20250514",
                "usage": {
                    "input_tokens": 36,
                    "cache_creation_input_tokens": 23554,
                    "cache_read_input_tokens": 139035,
                    "output_tokens": 3344,
                },
            },
            "timestamp": "2025-01-15T10:00:01.000Z",
        },
    ]

    transcript_path = tmp_path / "transcript_cache.jsonl"
    with open(transcript_path, "w") as f:
        for entry in transcript_entries:
            f.write(json.dumps(entry) + "\n")

    trace = process_transcript(str(transcript_path), "cache-transcript-session")

    assert trace is not None
    llm_spans = [s for s in trace.search_spans() if s.span_type == SpanType.LLM]
    assert len(llm_spans) == 1

    # input_tokens is the non-cached input the Anthropic API reports, matching
    # mlflow.anthropic.autolog. Cache fields are exposed as separate keys so
    # consumers can compute cache hit rate.
    token_usage = llm_spans[0].get_attribute(SpanAttributeKey.CHAT_USAGE)
    assert token_usage["input_tokens"] == 36
    assert token_usage["output_tokens"] == 3344
    assert token_usage["total_tokens"] == 36 + 3344
    assert token_usage["cache_read_input_tokens"] == 139035
    assert token_usage["cache_creation_input_tokens"] == 23554


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


# ============================================================================
# FIND LAST USER MESSAGE INDEX TESTS
# ============================================================================


def test_find_last_user_message_skips_skill_injection():
    transcript = [
        {"type": "queue-operation"},
        {"type": "queue-operation"},
        # Entry 2: actual user prompt
        {
            "type": "user",
            "message": {"role": "user", "content": "Enable tracing on the agent."},
            "timestamp": "2025-01-01T00:00:00Z",
        },
        # Entry 3: assistant thinking
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "Let me use the skill."}],
            },
            "timestamp": "2025-01-01T00:00:01Z",
        },
        # Entry 4: assistant invokes Skill tool
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_abc123",
                        "name": "Skill",
                        "input": {"skill": "instrumenting-with-mlflow-tracing"},
                    }
                ],
            },
            "timestamp": "2025-01-01T00:00:02Z",
        },
        # Entry 5: tool result with commandName (correctly skipped by toolUseResult check)
        {
            "type": "user",
            "toolUseResult": {
                "success": True,
                "commandName": "instrumenting-with-mlflow-tracing",
            },
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc123",
                        "content": "Launching skill: instrumenting-with-mlflow-tracing",
                    }
                ],
            },
            "timestamp": "2025-01-01T00:00:03Z",
        },
        # Entry 6: skill content injection (BUG: not flagged as tool result)
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Base directory for this skill: /path/to/skill\n\n"
                            "# MLflow Tracing Guide\n\n...(full skill content)..."
                        ),
                    }
                ],
            },
            "timestamp": "2025-01-01T00:00:04Z",
        },
        # Entry 7: assistant continues
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "Now let me implement tracing."}],
            },
            "timestamp": "2025-01-01T00:00:05Z",
        },
        # Entry 8: assistant text response
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "I've enabled tracing on the agent."}],
            },
            "timestamp": "2025-01-01T00:00:06Z",
        },
    ]

    idx = find_last_user_message_index(transcript)

    # Should return index 2 (actual user prompt), not 6 (skill injection)
    assert idx == 2
    assert transcript[idx]["message"]["content"] == "Enable tracing on the agent."


def test_find_last_user_message_index_basic():
    transcript = [
        {"type": "queue-operation"},
        {
            "type": "user",
            "message": {"role": "user", "content": "First question"},
            "timestamp": "2025-01-01T00:00:00Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "First answer"}],
            },
            "timestamp": "2025-01-01T00:00:01Z",
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "Second question"},
            "timestamp": "2025-01-01T00:00:02Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Second answer"}],
            },
            "timestamp": "2025-01-01T00:00:03Z",
        },
    ]

    idx = find_last_user_message_index(transcript)

    assert idx == 3
    assert transcript[idx]["message"]["content"] == "Second question"


def test_find_last_user_message_skips_consecutive_skill_injections():
    transcript = [
        # Entry 0: actual user prompt
        {
            "type": "user",
            "message": {"role": "user", "content": "Do the thing."},
            "timestamp": "2025-01-01T00:00:00Z",
        },
        # Entry 1: assistant invokes first Skill
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Skill",
                        "input": {"skill": "skill-one"},
                    }
                ],
            },
            "timestamp": "2025-01-01T00:00:01Z",
        },
        # Entry 2: first skill tool result
        {
            "type": "user",
            "toolUseResult": {"success": True, "commandName": "skill-one"},
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "Launching skill: skill-one",
                    }
                ],
            },
            "timestamp": "2025-01-01T00:00:02Z",
        },
        # Entry 3: first skill content injection
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Base directory: /skill-one\n# Skill One"}],
            },
            "timestamp": "2025-01-01T00:00:03Z",
        },
        # Entry 4: assistant invokes second Skill
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "Skill",
                        "input": {"skill": "skill-two"},
                    }
                ],
            },
            "timestamp": "2025-01-01T00:00:04Z",
        },
        # Entry 5: second skill tool result
        {
            "type": "user",
            "toolUseResult": {"success": True, "commandName": "skill-two"},
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_2",
                        "content": "Launching skill: skill-two",
                    }
                ],
            },
            "timestamp": "2025-01-01T00:00:05Z",
        },
        # Entry 6: second skill content injection
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Base directory: /skill-two\n# Skill Two"}],
            },
            "timestamp": "2025-01-01T00:00:06Z",
        },
        # Entry 7: assistant response
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
            },
            "timestamp": "2025-01-01T00:00:07Z",
        },
    ]

    idx = find_last_user_message_index(transcript)

    # Should skip both skill injections (entries 3 and 6) and return entry 0
    assert idx == 0
    assert transcript[idx]["message"]["content"] == "Do the thing."


def test_process_transcript_captures_claude_code_version(tmp_path):
    transcript = [
        {
            "type": "queue-operation",
            "operation": "dequeue",
            "timestamp": "2025-01-15T09:59:59.000Z",
            "sessionId": "test-version-session",
        },
        {
            "type": "user",
            "version": "2.1.34",
            "message": {"role": "user", "content": "Hello!"},
            "timestamp": "2025-01-15T10:00:00.000Z",
        },
        {
            "type": "assistant",
            "version": "2.1.34",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}],
            },
            "timestamp": "2025-01-15T10:00:01.000Z",
        },
    ]

    transcript_path = tmp_path / "version_transcript.jsonl"
    transcript_path.write_text("\n".join(json.dumps(entry) for entry in transcript) + "\n")
    trace = process_transcript(str(transcript_path), "test-version-session")

    assert trace is not None
    assert trace.info.trace_metadata.get(METADATA_KEY_CLAUDE_CODE_VERSION) == "2.1.34"


def test_process_transcript_no_version_field(mock_transcript_file):
    trace = process_transcript(mock_transcript_file, "test-session-no-version")

    assert trace is not None
    assert METADATA_KEY_CLAUDE_CODE_VERSION not in trace.info.trace_metadata


def test_process_transcript_includes_steer_messages(tmp_path):
    transcript = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Tell me about Python."},
            "timestamp": "2025-01-15T10:00:00.000Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Python is a programming language."}],
            },
            "timestamp": "2025-01-15T10:00:01.000Z",
        },
        {
            "type": "queue-operation",
            "operation": "enqueue",
            "content": "also tell me about Java",
            "timestamp": "2025-01-15T10:00:02.000Z",
            "sessionId": "test-steer-session",
        },
        {
            "type": "queue-operation",
            "operation": "remove",
            "timestamp": "2025-01-15T10:00:03.000Z",
            "sessionId": "test-steer-session",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Java is also a programming language."}],
            },
            "timestamp": "2025-01-15T10:00:04.000Z",
        },
    ]

    transcript_path = tmp_path / "steer_transcript.jsonl"
    transcript_path.write_text("\n".join(json.dumps(entry) for entry in transcript) + "\n")
    trace = process_transcript(str(transcript_path), "test-steer-session")
    assert trace is not None

    spans = list(trace.search_spans())
    llm_spans = [s for s in spans if s.span_type == SpanType.LLM]
    assert len(llm_spans) == 2

    # The second LLM span should include the steer message in its inputs
    second_llm = llm_spans[1]
    input_messages = second_llm.inputs["messages"]
    steer_messages = [m for m in input_messages if m.get("content") == "also tell me about Java"]
    assert len(steer_messages) == 1


# ============================================================================
# SKILL ATTRIBUTE PROPAGATION TESTS
# ============================================================================
#
# Canonical Skill wire format (used by every test below).
#
# When the user invokes a Skill, the Claude Code CLI emits TWO distinct
# user-role stream entries on the happy path:
#
#   [assistant] tool_use(id=X, name="Skill", input={"skill": "<name>"})
#   [user]      tool_result(tool_use_id=X)               ← success result
#                 + toolUseResult.commandName = "<name>"
#                 + toolUseResult.success    = True
#   [user]      text "<skill body>"                      ← synthetic body injection
#   [assistant] ... continuation (work inside the skill body) ...
#
# At the API layer the SDK merges the two user entries into one message with
# two content blocks (tool_result + text) before sending. At the stream/
# transcript layer they remain as two consecutive user entries, which is what
# this file's test fixtures should reflect.
#
# On the FAILED path (is_error=True), the CLI does NOT emit a body injection
# — the skill never bootstraps — so the failed fixtures stop at the
# tool_result and the assistant's next turn is recovery work:
#
#   [assistant] tool_use(id=X, name="Skill", ...)
#   [user]      tool_result(tool_use_id=X, is_error=True)
#                 + toolUseResult.commandName = "<name>"
#                 + toolUseResult.success    = False
#   [assistant] "recovering..."                          ← NO body injection
#
# Tests below follow this convention. If you're adding a new happy-path
# skill test and the body-injection user entry seems redundant for your
# assertion, add it anyway — keeping every fixture wire-format-accurate
# saves the next contributor from wondering whether they or the test is wrong.


def _skill_transcript_with_child_work() -> list[dict[Any, Any]]:
    """Transcript where a Skill invocation is followed by a child LLM turn and
    a child Bash tool call — both should inherit mlflow.skill.name = "my-skill".

    A pre-skill Bash call is included to verify it is NOT tagged (it happened
    before the skill was invoked).
    """
    return [
        {
            "type": "user",
            "message": {"role": "user", "content": "Do the thing."},
            "timestamp": "2025-01-15T10:00:00.000Z",
        },
        # Pre-skill tool call — must NOT inherit the skill name.
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "pre_bash", "name": "Bash", "input": {}}],
            },
            "timestamp": "2025-01-15T10:00:01.000Z",
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "pre_bash", "content": "ok"}],
            },
            "timestamp": "2025-01-15T10:00:02.000Z",
        },
        # Skill invocation.
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "skill_tool",
                        "name": "Skill",
                        "input": {"skill": "my-skill"},
                    }
                ],
            },
            "timestamp": "2025-01-15T10:00:03.000Z",
        },
        {
            "type": "user",
            "toolUseResult": {"success": True, "commandName": "my-skill"},
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "skill_tool",
                        "content": "Launching skill",
                    }
                ],
            },
            "timestamp": "2025-01-15T10:00:04.000Z",
        },
        # Synthetic body injection emitted by the CLI on the happy path.
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "<my-skill body>"}]},
            "timestamp": "2025-01-15T10:00:04.500Z",
        },
        # Child LLM turn after the skill — should inherit the skill name.
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Thinking inside the skill."}],
                "model": "claude-sonnet-4-20250514",
            },
            "timestamp": "2025-01-15T10:00:05.000Z",
        },
        # Child tool call after the skill — should also inherit the skill name.
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "child_bash", "name": "Bash", "input": {}}],
            },
            "timestamp": "2025-01-15T10:00:06.000Z",
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "child_bash", "content": "done"}
                ],
            },
            "timestamp": "2025-01-15T10:00:07.000Z",
        },
    ]


def test_process_transcript_propagates_skill_to_child_spans(tmp_path):
    transcript_path = tmp_path / "skill_transcript.jsonl"
    transcript_path.write_text(
        "\n".join(json.dumps(e) for e in _skill_transcript_with_child_work()) + "\n"
    )

    trace = process_transcript(str(transcript_path), "skill-prop-session")
    assert trace is not None

    spans = list(trace.search_spans())

    # Pre-skill Bash is NOT tagged.
    pre_bash = [s for s in spans if s.attributes.get("tool_id") == "pre_bash"]
    assert len(pre_bash) == 1
    assert pre_bash[0].get_attribute(SpanAttributeKey.SKILL_NAME) is None

    # Skill TOOL span is tagged with its own command name.
    skill_spans = [s for s in spans if s.attributes.get("tool_id") == "skill_tool"]
    assert len(skill_spans) == 1
    assert skill_spans[0].get_attribute(SpanAttributeKey.SKILL_NAME) == "my-skill"

    # Post-skill LLM span inherits the skill name.
    post_skill_llm = [
        s
        for s in spans
        if s.span_type == SpanType.LLM
        and s.outputs.get("content", [{}])[0].get("text") == "Thinking inside the skill."
    ]
    assert len(post_skill_llm) == 1
    assert post_skill_llm[0].get_attribute(SpanAttributeKey.SKILL_NAME) == "my-skill"

    # Child Bash tool inherits the skill name.
    child_bash = [s for s in spans if s.attributes.get("tool_id") == "child_bash"]
    assert len(child_bash) == 1
    assert child_bash[0].get_attribute(SpanAttributeKey.SKILL_NAME) == "my-skill"


def test_process_sdk_messages_propagates_skill_to_child_spans():
    messages = [
        UserMessage(content="Do the thing."),
        # Skill invocation.
        AssistantMessage(
            content=[ToolUseBlock(id="skill_tool", name="Skill", input={"skill": "my-skill"})],
            model="claude-sonnet-4-20250514",
        ),
        UserMessage(
            content=[ToolResultBlock(tool_use_id="skill_tool", content="Launching skill")],
            tool_use_result={"success": True, "commandName": "my-skill"},
        ),
        # Synthetic body injection emitted by the CLI on the happy path.
        UserMessage(content="<my-skill body>", tool_use_result=None),
        # Child LLM turn — should inherit.
        AssistantMessage(
            content=[TextBlock(text="Thinking inside the skill.")],
            model="claude-sonnet-4-20250514",
        ),
        # Child Bash tool — should inherit.
        AssistantMessage(
            content=[ToolUseBlock(id="child_bash", name="Bash", input={})],
            model="claude-sonnet-4-20250514",
        ),
        UserMessage(content=[ToolResultBlock(tool_use_id="child_bash", content="done")]),
        ResultMessage(
            subtype="success",
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=3,
            session_id="sdk-skill-prop",
        ),
    ]

    trace = process_sdk_messages(messages, "sdk-skill-prop")
    assert trace is not None

    spans = list(trace.search_spans())

    skill_spans = [s for s in spans if s.attributes.get("tool_id") == "skill_tool"]
    assert len(skill_spans) == 1
    assert skill_spans[0].get_attribute(SpanAttributeKey.SKILL_NAME) == "my-skill"

    post_skill_llm = [
        s
        for s in spans
        if s.span_type == SpanType.LLM
        and "Thinking inside the skill." in str(s.outputs.get("content", ""))
    ]
    assert len(post_skill_llm) == 1
    assert post_skill_llm[0].get_attribute(SpanAttributeKey.SKILL_NAME) == "my-skill"

    child_bash = [s for s in spans if s.attributes.get("tool_id") == "child_bash"]
    assert len(child_bash) == 1
    assert child_bash[0].get_attribute(SpanAttributeKey.SKILL_NAME) == "my-skill"


def _skill_transcript_with_failed_skill() -> list[dict[Any, Any]]:
    """Transcript where a Skill invocation fails (is_error=True). Subsequent
    spans should NOT inherit the skill name — a failed skill never injected
    its body, so attributing later work to it inflates cost.
    """
    return [
        {
            "type": "user",
            "message": {"role": "user", "content": "Do the thing."},
            "timestamp": "2025-01-15T10:00:00.000Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "skill_tool",
                        "name": "Skill",
                        "input": {"skill": "broken-skill"},
                    }
                ],
            },
            "timestamp": "2025-01-15T10:00:01.000Z",
        },
        {
            "type": "user",
            "toolUseResult": {"success": False, "commandName": "broken-skill"},
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "skill_tool",
                        "content": "Skill failed to launch",
                        "is_error": True,
                    }
                ],
            },
            "timestamp": "2025-01-15T10:00:02.000Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "I'll try a different approach."}],
                "model": "claude-sonnet-4-20250514",
            },
            "timestamp": "2025-01-15T10:00:03.000Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "post_bash", "name": "Bash", "input": {}}],
            },
            "timestamp": "2025-01-15T10:00:04.000Z",
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "post_bash", "content": "ok"}],
            },
            "timestamp": "2025-01-15T10:00:05.000Z",
        },
    ]


def test_process_transcript_failed_skill_does_not_propagate(tmp_path):
    transcript_path = tmp_path / "failed_skill_transcript.jsonl"
    transcript_path.write_text(
        "\n".join(json.dumps(e) for e in _skill_transcript_with_failed_skill()) + "\n"
    )

    trace = process_transcript(str(transcript_path), "failed-skill-session")
    assert trace is not None
    spans = list(trace.search_spans())

    # The failed Skill's OWN span IS stamped with its commandName so
    # operators can attribute the failure to a specific skill.
    skill_spans = [s for s in spans if s.attributes.get("tool_id") == "skill_tool"]
    assert len(skill_spans) == 1
    assert skill_spans[0].get_attribute(SpanAttributeKey.SKILL_NAME) == "broken-skill"

    # Post-skill LLM and tool spans must NOT be tagged with broken-skill.
    post_llm = [s for s in spans if s.span_type == SpanType.LLM]
    for s in post_llm:
        assert s.get_attribute(SpanAttributeKey.SKILL_NAME) is None

    post_bash = [s for s in spans if s.attributes.get("tool_id") == "post_bash"]
    assert len(post_bash) == 1
    assert post_bash[0].get_attribute(SpanAttributeKey.SKILL_NAME) is None


def test_process_sdk_messages_failed_skill_does_not_propagate():
    messages = [
        UserMessage(content="Do the thing."),
        AssistantMessage(
            content=[ToolUseBlock(id="skill_tool", name="Skill", input={"skill": "broken"})],
            model="claude-sonnet-4-20250514",
        ),
        UserMessage(
            content=[ToolResultBlock(tool_use_id="skill_tool", content="failed", is_error=True)],
            tool_use_result={"success": False, "commandName": "broken"},
        ),
        AssistantMessage(
            content=[TextBlock(text="trying again")],
            model="claude-sonnet-4-20250514",
        ),
        ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=2,
            session_id="failed-sdk-skill",
        ),
    ]

    trace = process_sdk_messages(messages, "failed-sdk-skill")
    assert trace is not None
    spans = list(trace.search_spans())

    # The failed Skill's OWN span IS stamped with its commandName.
    skill_spans = [s for s in spans if s.attributes.get("tool_id") == "skill_tool"]
    assert len(skill_spans) == 1
    assert skill_spans[0].get_attribute(SpanAttributeKey.SKILL_NAME) == "broken"

    # Post-failure LLM span must NOT be tagged with the failed skill name.
    post_llm = [s for s in spans if s.span_type == SpanType.LLM]
    assert len(post_llm) == 1
    assert post_llm[0].get_attribute(SpanAttributeKey.SKILL_NAME) is None


def test_build_tool_result_map_tolerates_missing_tool_use_result_attr():
    # Older claude_agent_sdk versions did not expose UserMessage.tool_use_result.
    # _build_tool_result_map uses getattr with a default so the function is
    # safe regardless of whether the attribute exists on the class. We can't
    # easily strip the attribute from the real UserMessage dataclass, so this
    # test verifies the explicit-None case (the bulk of the real-world risk).
    from mlflow.claude_code.tracing import _build_tool_result_map

    msg = UserMessage(
        content=[ToolResultBlock(tool_use_id="t1", content="ok")],
        tool_use_result=None,
    )
    result = _build_tool_result_map([msg])
    assert "t1" in result
    assert result["t1"].command_name is None
    assert result["t1"].is_error is False


def test_process_sdk_messages_skill_does_not_bleed_across_user_prompts():
    # ClaudeSDKClient autolog accumulates messages across query() calls and
    # invokes process_sdk_messages with the full list. Turn 1 uses a Skill;
    # Turn 2 is unrelated. The Skill must NOT tag Turn 2's spans.
    messages = [
        UserMessage(content="Turn 1: write a PRD."),
        AssistantMessage(
            content=[ToolUseBlock(id="skill_tool", name="Skill", input={"skill": "prd-writer"})],
            model="claude-sonnet-4-20250514",
        ),
        UserMessage(
            content=[ToolResultBlock(tool_use_id="skill_tool", content="Launching")],
            tool_use_result={"success": True, "commandName": "prd-writer"},
        ),
        # Synthetic body injection emitted by the CLI on the happy path.
        UserMessage(content="<prd-writer body>", tool_use_result=None),
        AssistantMessage(
            content=[TextBlock(text="PRD body produced under the skill.")],
            model="claude-sonnet-4-20250514",
        ),
        # Turn 2: fresh user prompt, no skill. The skill scope must reset here.
        UserMessage(content="Turn 2: refactor my code."),
        AssistantMessage(
            content=[TextBlock(text="Refactoring (unrelated to the skill).")],
            model="claude-sonnet-4-20250514",
        ),
        AssistantMessage(
            content=[ToolUseBlock(id="turn2_bash", name="Bash", input={})],
            model="claude-sonnet-4-20250514",
        ),
        UserMessage(content=[ToolResultBlock(tool_use_id="turn2_bash", content="ok")]),
        ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=4,
            session_id="multi-turn-skill",
        ),
    ]

    trace = process_sdk_messages(messages, "multi-turn-skill")
    assert trace is not None
    spans = list(trace.search_spans())

    # Turn 1 LLM (inside skill scope) IS tagged.
    turn1_llm = [
        s
        for s in spans
        if s.span_type == SpanType.LLM
        and "PRD body produced under the skill." in str(s.outputs.get("content", ""))
    ]
    assert len(turn1_llm) == 1
    assert turn1_llm[0].get_attribute(SpanAttributeKey.SKILL_NAME) == "prd-writer"

    # Turn 2 LLM is NOT tagged — skill scope was cleared at the new user prompt.
    turn2_llm = [
        s
        for s in spans
        if s.span_type == SpanType.LLM
        and "Refactoring (unrelated to the skill)." in str(s.outputs.get("content", ""))
    ]
    assert len(turn2_llm) == 1
    assert turn2_llm[0].get_attribute(SpanAttributeKey.SKILL_NAME) is None

    # Turn 2 child TOOL is NOT tagged either.
    turn2_bash = [s for s in spans if s.attributes.get("tool_id") == "turn2_bash"]
    assert len(turn2_bash) == 1
    assert turn2_bash[0].get_attribute(SpanAttributeKey.SKILL_NAME) is None


def test_process_sdk_messages_skill_body_injection_does_not_reset_scope():
    # If the SDK ever surfaces a skill-body injection as a UserMessage with
    # tool_use_result=None immediately after the Skill tool-result, the reset
    # logic must NOT clear active_skill_name — the prior message's commandName
    # marks it as a skill body, not a fresh user prompt.
    messages = [
        UserMessage(content="Run the skill."),
        AssistantMessage(
            content=[ToolUseBlock(id="skill_tool", name="Skill", input={"skill": "my-skill"})],
            model="claude-sonnet-4-20250514",
        ),
        UserMessage(
            content=[ToolResultBlock(tool_use_id="skill_tool", content="Launching")],
            tool_use_result={"success": True, "commandName": "my-skill"},
        ),
        # Injected skill body — looks like a fresh prompt but must NOT reset.
        UserMessage(content="<the skill's prompt body>", tool_use_result=None),
        AssistantMessage(
            content=[TextBlock(text="Skill body output.")],
            model="claude-sonnet-4-20250514",
        ),
        ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=2,
            session_id="skill-body-injection",
        ),
    ]

    trace = process_sdk_messages(messages, "skill-body-injection")
    assert trace is not None
    spans = list(trace.search_spans())

    body_llm = [
        s
        for s in spans
        if s.span_type == SpanType.LLM and "Skill body output." in str(s.outputs.get("content", ""))
    ]
    assert len(body_llm) == 1
    assert body_llm[0].get_attribute(SpanAttributeKey.SKILL_NAME) == "my-skill"


def _skill_transcript_with_prior_skill_then_failed_skill() -> list[dict[Any, Any]]:
    """Successful skill A → failed skill B → recovery work. The recovery
    spans must NOT carry skill A's name; the failed skill B IS stamped on its
    own span.
    """
    return [
        {
            "type": "user",
            "message": {"role": "user", "content": "Do the thing."},
            "timestamp": "2025-01-15T10:00:00.000Z",
        },
        # Skill A — succeeds.
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "skill_a",
                        "name": "Skill",
                        "input": {"skill": "skill-a"},
                    }
                ],
            },
            "timestamp": "2025-01-15T10:00:01.000Z",
        },
        {
            "type": "user",
            "toolUseResult": {"success": True, "commandName": "skill-a"},
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "skill_a",
                        "content": "Skill A launched",
                    }
                ],
            },
            "timestamp": "2025-01-15T10:00:02.000Z",
        },
        # Synthetic body injection for skill-a's happy-path boot.
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "<skill-a body>"}]},
            "timestamp": "2025-01-15T10:00:02.500Z",
        },
        # Skill B — fails.
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "skill_b",
                        "name": "Skill",
                        "input": {"skill": "skill-b"},
                    }
                ],
            },
            "timestamp": "2025-01-15T10:00:03.000Z",
        },
        {
            "type": "user",
            "toolUseResult": {"success": False, "commandName": "skill-b"},
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "skill_b",
                        "content": "Skill B failed",
                        "is_error": True,
                    }
                ],
            },
            "timestamp": "2025-01-15T10:00:04.000Z",
        },
        # Recovery LLM + tool — must NOT inherit skill-a.
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "recovering"}],
                "model": "claude-sonnet-4-20250514",
            },
            "timestamp": "2025-01-15T10:00:05.000Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "recovery_bash", "name": "Bash", "input": {}}
                ],
            },
            "timestamp": "2025-01-15T10:00:06.000Z",
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "recovery_bash", "content": "ok"}
                ],
            },
            "timestamp": "2025-01-15T10:00:07.000Z",
        },
    ]


def test_process_transcript_failed_skill_clears_prior_skill_scope(tmp_path):
    transcript_path = tmp_path / "prior_then_failed.jsonl"
    transcript_path.write_text(
        "\n".join(json.dumps(e) for e in _skill_transcript_with_prior_skill_then_failed_skill())
        + "\n"
    )

    trace = process_transcript(str(transcript_path), "prior-then-failed")
    assert trace is not None
    spans = list(trace.search_spans())

    # Skill A's own span carries skill-a.
    skill_a = [s for s in spans if s.attributes.get("tool_id") == "skill_a"]
    assert len(skill_a) == 1
    assert skill_a[0].get_attribute(SpanAttributeKey.SKILL_NAME) == "skill-a"

    # Skill B's own span carries skill-b.
    skill_b = [s for s in spans if s.attributes.get("tool_id") == "skill_b"]
    assert len(skill_b) == 1
    assert skill_b[0].get_attribute(SpanAttributeKey.SKILL_NAME) == "skill-b"

    # Recovery work after the failure must NOT inherit either skill name.
    recovery_llm = [s for s in spans if s.span_type == SpanType.LLM]
    assert len(recovery_llm) == 1
    assert recovery_llm[0].get_attribute(SpanAttributeKey.SKILL_NAME) is None

    recovery_bash = [s for s in spans if s.attributes.get("tool_id") == "recovery_bash"]
    assert len(recovery_bash) == 1
    assert recovery_bash[0].get_attribute(SpanAttributeKey.SKILL_NAME) is None


def test_process_sdk_messages_failed_skill_clears_prior_skill_scope():
    messages = [
        UserMessage(content="Do the thing."),
        # Skill A — succeeds.
        AssistantMessage(
            content=[ToolUseBlock(id="skill_a", name="Skill", input={"skill": "skill-a"})],
            model="claude-sonnet-4-20250514",
        ),
        UserMessage(
            content=[ToolResultBlock(tool_use_id="skill_a", content="launched")],
            tool_use_result={"success": True, "commandName": "skill-a"},
        ),
        # Synthetic body injection for skill-a's happy-path boot.
        UserMessage(content="<skill-a body>", tool_use_result=None),
        # Skill B — fails.
        AssistantMessage(
            content=[ToolUseBlock(id="skill_b", name="Skill", input={"skill": "skill-b"})],
            model="claude-sonnet-4-20250514",
        ),
        UserMessage(
            content=[ToolResultBlock(tool_use_id="skill_b", content="failed", is_error=True)],
            tool_use_result={"success": False, "commandName": "skill-b"},
        ),
        # Recovery LLM + Bash. Must NOT inherit skill-a.
        AssistantMessage(
            content=[TextBlock(text="recovering")],
            model="claude-sonnet-4-20250514",
        ),
        AssistantMessage(
            content=[ToolUseBlock(id="recovery_bash", name="Bash", input={})],
            model="claude-sonnet-4-20250514",
        ),
        UserMessage(content=[ToolResultBlock(tool_use_id="recovery_bash", content="ok")]),
        ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=3,
            session_id="prior-then-failed-sdk",
        ),
    ]

    trace = process_sdk_messages(messages, "prior-then-failed-sdk")
    assert trace is not None
    spans = list(trace.search_spans())

    skill_a = [s for s in spans if s.attributes.get("tool_id") == "skill_a"]
    assert len(skill_a) == 1
    assert skill_a[0].get_attribute(SpanAttributeKey.SKILL_NAME) == "skill-a"

    skill_b = [s for s in spans if s.attributes.get("tool_id") == "skill_b"]
    assert len(skill_b) == 1
    assert skill_b[0].get_attribute(SpanAttributeKey.SKILL_NAME) == "skill-b"

    recovery_llm = [s for s in spans if s.span_type == SpanType.LLM]
    assert len(recovery_llm) == 1
    assert recovery_llm[0].get_attribute(SpanAttributeKey.SKILL_NAME) is None

    recovery_bash = [s for s in spans if s.attributes.get("tool_id") == "recovery_bash"]
    assert len(recovery_bash) == 1
    assert recovery_bash[0].get_attribute(SpanAttributeKey.SKILL_NAME) is None


def _nested_skill_transcript() -> list[dict[Any, Any]]:
    """3-level nesting: Skill A's body invokes Skill B; B's body invokes Skill C.

    The user-invoked skill is A. Ideally every span produced while A is running
    (including B's and C's own TOOL spans and the LLM reasoning under them)
    should be attributed to skill-A — B and C are implementation details of A.
    The single-slot state machine doesn't preserve that outer scope, which is
    what this test documents.
    """
    return [
        {
            "type": "user",
            "message": {"role": "user", "content": "use skill A"},
            "timestamp": "2026-06-05T10:00:00Z",
        },
        # Skill A
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "ta", "name": "Skill", "input": {"skill": "A"}}
                ],
            },
            "timestamp": "2026-06-05T10:00:01Z",
        },
        {
            "type": "user",
            "toolUseResult": {"success": True, "commandName": "skill-A"},
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "ta", "content": "launch A"}],
            },
            "timestamp": "2026-06-05T10:00:02Z",
        },
        # A's body injection
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "A body"}]},
            "timestamp": "2026-06-05T10:00:03Z",
        },
        # A reasoning
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "A reasoning"}],
            },
            "timestamp": "2026-06-05T10:00:04Z",
        },
        # Skill B (invoked from inside A)
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tb", "name": "Skill", "input": {"skill": "B"}}
                ],
            },
            "timestamp": "2026-06-05T10:00:05Z",
        },
        {
            "type": "user",
            "toolUseResult": {"success": True, "commandName": "skill-B"},
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tb", "content": "launch B"}],
            },
            "timestamp": "2026-06-05T10:00:06Z",
        },
        # B's body injection
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "B body"}]},
            "timestamp": "2026-06-05T10:00:07Z",
        },
        # B reasoning
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "B reasoning"}],
            },
            "timestamp": "2026-06-05T10:00:08Z",
        },
        # Skill C (invoked from inside B)
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tc", "name": "Skill", "input": {"skill": "C"}}
                ],
            },
            "timestamp": "2026-06-05T10:00:09Z",
        },
        {
            "type": "user",
            "toolUseResult": {"success": True, "commandName": "skill-C"},
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tc", "content": "launch C"}],
            },
            "timestamp": "2026-06-05T10:00:10Z",
        },
        # C's body injection
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "C body"}]},
            "timestamp": "2026-06-05T10:00:11Z",
        },
        # Deepest: C reasoning
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "C reasoning (deepest)"}],
            },
            "timestamp": "2026-06-05T10:00:12Z",
        },
        # Unwinding: C done, back to B's continuation (semantically)
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "back to B"}],
            },
            "timestamp": "2026-06-05T10:00:13Z",
        },
        # Unwinding: B done, back to A
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "back to A"}],
            },
            "timestamp": "2026-06-05T10:00:14Z",
        },
        # Unwinding: A done, final reply to user
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "final reply"}],
            },
            "timestamp": "2026-06-05T10:00:15Z",
        },
    ]


def test_process_transcript_nested_skills_outer_scope_lost(tmp_path):
    """KNOWN LIMITATION (documented): The single-slot state machine doesn't
    preserve the outer skill's scope across nested invocations.

    Ideal attribution: the user invoked skill-A, so every span produced while
    A is running — B's TOOL span, C's TOOL span, and all of B/C's reasoning
    LLMs — should be tagged skill-A. B and C are implementation details of A.

    Actual behavior: each nested Skill overwrites the active scope, so B's
    work is tagged skill-B, C's work is tagged skill-C, and the trailing
    unwind LLMs keep skill-C because there's no stack to pop.

    Set a breakpoint in `_create_llm_and_tool_spans` at the
    `active_skill_name = ...` assignment to step through this.
    """
    transcript_path = tmp_path / "nested.jsonl"
    transcript_path.write_text("\n".join(json.dumps(e) for e in _nested_skill_transcript()) + "\n")

    trace = process_transcript(str(transcript_path), "nested-skill-test")
    assert trace is not None
    spans = list(trace.search_spans())

    # Each Skill's OWN TOOL span is stamped with its own commandName via the
    # own-span pass (identification). Ideally these should also report skill-A
    # as the outer scope, but the single slot can only hold one value.
    assert (
        next(s for s in spans if s.attributes.get("tool_id") == "ta").get_attribute(
            SpanAttributeKey.SKILL_NAME
        )
        == "skill-A"
    )
    # ↓ THE LIMITATION (own-span): tb/tc are tagged with themselves; the outer
    #   skill-A scope is lost. Ideally both would be skill-A.
    assert (
        next(s for s in spans if s.attributes.get("tool_id") == "tb").get_attribute(
            SpanAttributeKey.SKILL_NAME
        )
        == "skill-B"
    )
    assert (
        next(s for s in spans if s.attributes.get("tool_id") == "tc").get_attribute(
            SpanAttributeKey.SKILL_NAME
        )
        == "skill-C"
    )

    # LLM spans, ordered by their text content (proxy for chronological order).
    def llm_with_text(text: str):
        return next(
            s
            for s in spans
            if s.span_type == SpanType.LLM and text in str(s.outputs.get("content", ""))
        )

    # Inside-A reasoning: tagged skill-A (correct).
    assert llm_with_text("A reasoning").get_attribute(SpanAttributeKey.SKILL_NAME) == "skill-A"

    # ↓ THE LIMITATION (propagation): once a nested skill is invoked, the slot
    #   is overwritten and all descendants get the inner skill's name. Ideally
    #   every LLM below should still be tagged skill-A — A is the user-invoked
    #   skill and B/C are implementation details of A.
    assert llm_with_text("B reasoning").get_attribute(SpanAttributeKey.SKILL_NAME) == "skill-B"
    assert llm_with_text("C reasoning").get_attribute(SpanAttributeKey.SKILL_NAME) == "skill-C"
    assert llm_with_text("back to B").get_attribute(SpanAttributeKey.SKILL_NAME) == "skill-C"
    assert llm_with_text("back to A").get_attribute(SpanAttributeKey.SKILL_NAME) == "skill-C"
    assert llm_with_text("final reply").get_attribute(SpanAttributeKey.SKILL_NAME) == "skill-C"
