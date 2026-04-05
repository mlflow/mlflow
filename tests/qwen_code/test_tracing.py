import json

import mlflow.qwen_code.tracing as tracing_module
from mlflow.codex.tracing import get_hook_response
from mlflow.entities.span import SpanType
from mlflow.qwen_code.tracing import (
    QWEN_TRACING_LEVEL,
    parse_timestamp_to_ns,
    process_transcript,
    setup_logging,
)


def test_parse_timestamp_to_ns_iso_string():
    result = parse_timestamp_to_ns("2024-01-15T10:30:45.123456Z")
    assert isinstance(result, int)
    assert result > 0


def test_parse_timestamp_to_ns_unix_seconds():
    unix_timestamp = 1705312245.123456
    result = parse_timestamp_to_ns(unix_timestamp)
    expected = int(unix_timestamp * 1_000_000_000)
    assert result == expected


def test_parse_timestamp_to_ns_none():
    assert parse_timestamp_to_ns(None) is None


def test_setup_logging_creates_logger(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    logger = setup_logging()

    assert logger is not None
    assert logger.name == "mlflow.qwen_code.tracing"

    log_dir = tmp_path / ".qwen" / "mlflow"
    assert log_dir.exists()


def test_setup_logging_custom_level(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    logger = setup_logging()
    assert logger.level == QWEN_TRACING_LEVEL


def test_get_hook_response_success():
    response = get_hook_response()
    assert response == {"continue": True}


def test_get_hook_response_error():
    response = get_hook_response(error="something went wrong")
    assert response == {"continue": False, "stopReason": "something went wrong"}


def test_process_transcript_with_tree_format(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tracing_module._MODULE_LOGGER = None

    # Qwen Code tree-structured transcript:
    # user (root) → assistant (tool call result) → assistant (text response)
    transcript = [
        {
            "uuid": "user-001",
            "parentUuid": None,
            "sessionId": "session-abc",
            "type": "user",
            "message": "Write a hello world function",
            "timestamp": "2024-01-15T10:30:00Z",
        },
        {
            "uuid": "tool-001",
            "parentUuid": "user-001",
            "sessionId": "session-abc",
            "type": "assistant",
            "message": "",
            "timestamp": "2024-01-15T10:30:01Z",
            "model": "qwen3-coder",
            "toolCallResult": {
                "name": "write_file",
                "input": {"path": "hello.py", "content": "print('hello')"},
                "output": "File written successfully",
            },
        },
        {
            "uuid": "assistant-001",
            "parentUuid": "user-001",
            "sessionId": "session-abc",
            "type": "assistant",
            "message": "I've created hello.py with a hello world function.",
            "timestamp": "2024-01-15T10:30:02Z",
            "model": "qwen3-coder",
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150,
            },
        },
    ]

    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text("\n".join(json.dumps(record) for record in transcript))

    trace = process_transcript(str(transcript_path), session_id="test-session")

    assert trace is not None
    spans = trace.data.spans
    assert len(spans) == 3  # agent + tool + llm

    root_span = next(s for s in spans if s.parent_id is None)
    assert root_span.name == "qwen_code_conversation"
    assert root_span.span_type == SpanType.AGENT

    tool_spans = [s for s in spans if s.span_type == SpanType.TOOL]
    assert len(tool_spans) == 1
    assert tool_spans[0].name == "tool_write_file"

    llm_spans = [s for s in spans if s.span_type == SpanType.LLM]
    assert len(llm_spans) == 1


def test_process_transcript_empty(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tracing_module._MODULE_LOGGER = None

    transcript_path = tmp_path / "empty.jsonl"
    transcript_path.write_text("")

    trace = process_transcript(str(transcript_path))
    assert trace is None


def test_process_transcript_no_user_message(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tracing_module._MODULE_LOGGER = None

    transcript = [
        {
            "uuid": "assistant-001",
            "type": "assistant",
            "message": "Hello",
            "timestamp": "2024-01-15T10:30:00Z",
        },
    ]

    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text(json.dumps(transcript[0]))

    trace = process_transcript(str(transcript_path))
    assert trace is None


def test_process_transcript_none_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tracing_module._MODULE_LOGGER = None

    trace = process_transcript(None)
    assert trace is None
