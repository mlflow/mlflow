import json
import os
from unittest.mock import MagicMock, patch

import pytest

from mlflow.gemini_cli.tracing import (
    _aggregate_token_usage,
    _extract_content_and_tools,
    extract_text_content,
    find_final_gemini_response,
    find_last_user_message_index,
    get_hook_response,
    is_tracing_enabled,
    parse_timestamp_to_ns,
    process_transcript,
    read_transcript,
)


def test_extract_text_content_from_list():
    content = [{"text": "Hello"}, {"text": " world"}]
    assert extract_text_content(content) == "Hello\n world"


def test_extract_text_content_from_string():
    assert extract_text_content("Hello world") == "Hello world"


def test_extract_text_content_empty():
    assert extract_text_content([]) == ""
    assert extract_text_content("") == ""


def test_extract_text_content_with_tool_calls():
    content = [
        {"text": "Let me check that"},
        {"functionCall": {"name": "read_file", "args": {"path": "test.py"}}},
    ]
    assert extract_text_content(content) == "Let me check that"


def test_parse_timestamp_to_ns_iso_string():
    result = parse_timestamp_to_ns("2024-01-15T10:30:00Z")
    assert result is not None
    assert isinstance(result, int)
    assert result > 0


def test_parse_timestamp_to_ns_unix_seconds():
    result = parse_timestamp_to_ns(1705312200)
    assert result is not None
    assert result == int(1705312200 * 1e9)


def test_parse_timestamp_to_ns_unix_ms():
    result = parse_timestamp_to_ns(1705312200000)
    assert result is not None
    assert result == int(1705312200000 * 1e6)


def test_parse_timestamp_to_ns_none():
    assert parse_timestamp_to_ns(None) is None
    assert parse_timestamp_to_ns("") is None


def test_find_last_user_message_index():
    transcript = [
        {"type": "session_metadata", "sessionId": "test"},
        {"type": "user", "content": [{"text": "Hello"}]},
        {"type": "gemini", "content": [{"text": "Hi there!"}]},
        {"type": "user", "content": [{"text": "How are you?"}]},
        {"type": "gemini", "content": [{"text": "I'm good!"}]},
    ]
    assert find_last_user_message_index(transcript) == 3


def test_find_last_user_message_index_no_user():
    transcript = [
        {"type": "session_metadata", "sessionId": "test"},
        {"type": "gemini", "content": [{"text": "Hi!"}]},
    ]
    assert find_last_user_message_index(transcript) is None


def test_find_last_user_message_index_empty_content():
    transcript = [
        {"type": "user", "content": []},
        {"type": "user", "content": [{"text": "Real message"}]},
    ]
    assert find_last_user_message_index(transcript) == 1


def test_extract_content_and_tools_text_only():
    content = [{"text": "Hello"}]
    text, tools = _extract_content_and_tools(content)
    assert text == "Hello"
    assert tools == []


def test_extract_content_and_tools_with_function_call():
    content = [
        {"functionCall": {"name": "read_file", "args": {"path": "test.py"}}},
    ]
    text, tools = _extract_content_and_tools(content)
    assert text == ""
    assert len(tools) == 1
    assert tools[0]["name"] == "read_file"


def test_extract_content_and_tools_mixed():
    content = [
        {"text": "Let me check"},
        {"functionCall": {"name": "run_command", "args": {"cmd": "ls"}}},
    ]
    text, tools = _extract_content_and_tools(content)
    assert text == "Let me check"
    assert len(tools) == 1


def test_aggregate_token_usage():
    transcript = [
        {"type": "user", "content": [{"text": "Hello"}]},
        {"type": "gemini", "id": "msg1", "content": [{"text": "Hi"}]},
        {"type": "message_update", "id": "msg1", "tokens": {"input": 10, "output": 5}},
        {"type": "user", "content": [{"text": "More"}]},
        {"type": "gemini", "id": "msg2", "content": [{"text": "Sure"}]},
        {"type": "message_update", "id": "msg2", "tokens": {"input": 15, "output": 8}},
    ]
    usage = _aggregate_token_usage(transcript)
    assert usage["input_tokens"] == 25
    assert usage["output_tokens"] == 13
    assert usage["total_tokens"] == 38


def test_aggregate_token_usage_no_updates():
    transcript = [
        {"type": "user", "content": [{"text": "Hello"}]},
        {"type": "gemini", "content": [{"text": "Hi"}]},
    ]
    usage = _aggregate_token_usage(transcript)
    assert usage == {}


def test_find_final_gemini_response():
    transcript = [
        {"type": "user", "content": [{"text": "Hello"}]},
        {"type": "gemini", "content": [{"text": "First response"}]},
        {"type": "gemini", "content": [{"text": "Final response"}]},
    ]
    result = find_final_gemini_response(transcript, 1)
    assert result == "Final response"


def test_find_final_gemini_response_none():
    transcript = [
        {"type": "user", "content": [{"text": "Hello"}]},
    ]
    result = find_final_gemini_response(transcript, 1)
    assert result is None


def test_get_hook_response_success():
    response = get_hook_response()
    assert response == {"continue": True}


def test_get_hook_response_error():
    response = get_hook_response(error="Something failed")
    assert response == {"continue": False, "stopReason": "Something failed"}


def test_is_tracing_enabled_true(monkeypatch):
    monkeypatch.setenv("MLFLOW_GEMINI_CLI_TRACING_ENABLED", "true")
    assert is_tracing_enabled() is True


def test_is_tracing_enabled_false(monkeypatch):
    monkeypatch.delenv("MLFLOW_GEMINI_CLI_TRACING_ENABLED", raising=False)
    assert is_tracing_enabled() is False


def test_read_transcript(tmp_path):
    transcript_file = tmp_path / "transcript.jsonl"
    entries = [
        {"type": "session_metadata", "sessionId": "test-session"},
        {"type": "user", "content": [{"text": "Hello"}]},
        {"type": "gemini", "content": [{"text": "Hi there!"}]},
    ]
    with open(transcript_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    result = read_transcript(str(transcript_file))
    assert len(result) == 3
    assert result[0]["type"] == "session_metadata"
    assert result[1]["type"] == "user"
    assert result[2]["type"] == "gemini"


def test_process_transcript_creates_trace(tmp_path):
    transcript_file = tmp_path / "transcript.jsonl"
    entries = [
        {"type": "session_metadata", "sessionId": "test-session", "startTime": "2024-01-15T10:00:00Z"},
        {"type": "user", "id": "msg1", "content": [{"text": "What is 2+2?"}], "timestamp": "2024-01-15T10:00:01Z"},
        {"type": "gemini", "id": "msg2", "content": [{"text": "2+2 equals 4."}], "timestamp": "2024-01-15T10:00:02Z"},
        {"type": "message_update", "id": "msg2", "tokens": {"input": 10, "output": 5}},
    ]
    with open(transcript_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    trace = process_transcript(str(transcript_file), "test-session")
    assert trace is not None


def test_process_transcript_empty_file(tmp_path):
    transcript_file = tmp_path / "empty.jsonl"
    transcript_file.touch()

    trace = process_transcript(str(transcript_file))
    assert trace is None


def test_process_transcript_no_user_message(tmp_path):
    transcript_file = tmp_path / "no_user.jsonl"
    entries = [
        {"type": "session_metadata", "sessionId": "test"},
        {"type": "gemini", "content": [{"text": "Hi"}]},
    ]
    with open(transcript_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    trace = process_transcript(str(transcript_file))
    assert trace is None


def test_process_transcript_with_tool_calls(tmp_path):
    transcript_file = tmp_path / "tools.jsonl"
    entries = [
        {"type": "session_metadata", "sessionId": "tool-session"},
        {
            "type": "user",
            "id": "msg1",
            "content": [{"text": "Read test.py"}],
            "timestamp": "2024-01-15T10:00:00Z",
        },
        {
            "type": "gemini",
            "id": "msg2",
            "content": [
                {"functionCall": {"name": "read_file", "args": {"path": "test.py"}}},
            ],
            "timestamp": "2024-01-15T10:00:01Z",
        },
        {
            "type": "user",
            "id": "msg3",
            "content": [
                {
                    "functionResponse": {
                        "name": "read_file",
                        "response": {"content": "print('hello')"},
                    }
                }
            ],
            "timestamp": "2024-01-15T10:00:02Z",
        },
        {
            "type": "gemini",
            "id": "msg4",
            "content": [{"text": "The file contains a simple print statement."}],
            "timestamp": "2024-01-15T10:00:03Z",
        },
        {"type": "message_update", "id": "msg2", "tokens": {"input": 10, "output": 5}},
        {"type": "message_update", "id": "msg4", "tokens": {"input": 20, "output": 15}},
    ]
    with open(transcript_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    trace = process_transcript(str(transcript_file), "tool-session")
    assert trace is not None
