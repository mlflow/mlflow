import json

import mlflow.codex.tracing as tracing_module
from mlflow.codex.tracing import (
    CODEX_TRACING_LEVEL,
    get_hook_response,
    parse_timestamp_to_ns,
    process_transcript,
    setup_logging,
)
from mlflow.entities.span import SpanType


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
    assert logger.name == "mlflow.codex.tracing"

    log_dir = tmp_path / ".codex" / "mlflow"
    assert log_dir.exists()
    assert log_dir.is_dir()


def test_setup_logging_custom_level(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    logger = setup_logging()
    assert logger.level == CODEX_TRACING_LEVEL


def test_get_hook_response_success():
    response = get_hook_response()
    assert response == {"continue": True}


def test_get_hook_response_error():
    response = get_hook_response(error="something went wrong")
    assert response == {"continue": False, "stopReason": "something went wrong"}


def _make_rollout_transcript(*, with_tool_call: bool = False) -> list[dict[str, object]]:
    """Build a realistic Codex rollout transcript in the RolloutLine format.

    Mirrors the actual format from codex-rs/protocol/src/protocol.rs:
    - session_meta (first line)
    - event_msg task_started (turn boundary)
    - response_item messages, function_calls, function_call_outputs
    - event_msg token_count
    - event_msg task_complete (turn boundary)
    """
    records = [
        {
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "session_meta",
            "payload": {
                "id": "test-session-001",
                "timestamp": "2024-01-15T10:30:00Z",
                "cwd": "/tmp/test",
                "originator": "codex-tui",
                "cli_version": "0.118.0",
                "source": "cli",
            },
        },
        {
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "event_msg",
            "payload": {"type": "task_started"},
        },
        {
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Write a hello world function in Python"}
                ],
            },
        },
    ]

    if with_tool_call:
        records.extend([
            {
                "timestamp": "2024-01-15T10:30:01Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I'll create the file for you."}],
                },
            },
            {
                "timestamp": "2024-01-15T10:30:02Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "call_abc123",
                    "arguments": '{"cmd": "cat > hello.py << EOF\\nprint(\'hello\')\\nEOF"}',
                },
            },
            {
                "timestamp": "2024-01-15T10:30:03Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_abc123",
                    "output": "File written successfully",
                },
            },
            {
                "timestamp": "2024-01-15T10:30:04Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Created hello.py with hello world."}
                    ],
                },
            },
        ])
    else:
        records.append({
            "timestamp": "2024-01-15T10:30:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "```python\ndef hello_world():\n    print('Hello, world!')\n```",
                    }
                ],
            },
        })

    records.extend([
        {
            "timestamp": "2024-01-15T10:30:05Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150,
                    }
                },
            },
        },
        {
            "timestamp": "2024-01-15T10:30:05Z",
            "type": "event_msg",
            "payload": {"type": "task_complete"},
        },
    ])

    return records


def test_process_transcript_text_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tracing_module._MODULE_LOGGER = None

    transcript = _make_rollout_transcript(with_tool_call=False)
    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text("\n".join(json.dumps(r) for r in transcript))

    trace = process_transcript(str(transcript_path), session_id="test-session")

    assert trace is not None
    spans = trace.data.spans
    # agent root + 1 LLM span
    assert len(spans) == 2

    root_span = next(s for s in spans if s.parent_id is None)
    assert root_span.name == "codex_conversation"
    assert root_span.span_type == SpanType.AGENT

    llm_spans = [s for s in spans if s.span_type == SpanType.LLM]
    assert len(llm_spans) == 1


def test_process_transcript_with_tool_call(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tracing_module._MODULE_LOGGER = None

    transcript = _make_rollout_transcript(with_tool_call=True)
    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text("\n".join(json.dumps(r) for r in transcript))

    trace = process_transcript(str(transcript_path), session_id="test-session")

    assert trace is not None
    spans = trace.data.spans
    # agent root + 2 LLM spans + 1 TOOL span
    assert len(spans) == 4

    root_span = next(s for s in spans if s.parent_id is None)
    assert root_span.name == "codex_conversation"
    assert root_span.span_type == SpanType.AGENT

    tool_spans = [s for s in spans if s.span_type == SpanType.TOOL]
    assert len(tool_spans) == 1
    assert tool_spans[0].name == "tool_exec_command"

    llm_spans = [s for s in spans if s.span_type == SpanType.LLM]
    assert len(llm_spans) == 2


def test_process_transcript_extracts_session_id(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tracing_module._MODULE_LOGGER = None

    transcript = _make_rollout_transcript()
    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text("\n".join(json.dumps(r) for r in transcript))

    # Don't pass session_id — should extract from session_meta
    trace = process_transcript(str(transcript_path))

    assert trace is not None


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
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "event_msg",
            "payload": {"type": "task_started"},
        },
        {
            "timestamp": "2024-01-15T10:30:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}],
            },
        },
        {
            "timestamp": "2024-01-15T10:30:02Z",
            "type": "event_msg",
            "payload": {"type": "task_complete"},
        },
    ]

    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text("\n".join(json.dumps(r) for r in transcript))

    trace = process_transcript(str(transcript_path))
    assert trace is None


def test_process_transcript_none_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tracing_module._MODULE_LOGGER = None

    trace = process_transcript(None)
    assert trace is None
