"""Tests for mlflow.kiro.tracing."""

import json
from datetime import datetime, timezone
from unittest import mock

import pytest

from mlflow.kiro.tracing import (
    _build_usage_dict,
    _extract_text,
    _find_final_assistant_response,
    _find_last_user_message,
    get_hook_response,
    is_tracing_enabled,
    parse_timestamp_to_ns,
    process_session,
)
from mlflow.tracing.constant import TokenUsageKey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ISO_TS = "2024-01-15T10:00:00Z"
ISO_TS_5 = "2024-01-15T10:00:05Z"


def _iso_to_ns(s: str) -> int:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1e9)


# ---------------------------------------------------------------------------
# parse_timestamp_to_ns
# ---------------------------------------------------------------------------


def test_parse_timestamp_iso_string():
    result = parse_timestamp_to_ns("2024-01-15T10:00:00Z")
    assert isinstance(result, int)
    assert result > 0


def test_parse_timestamp_none_returns_none():
    assert parse_timestamp_to_ns(None) is None


def test_parse_timestamp_empty_string_returns_none():
    assert parse_timestamp_to_ns("") is None


def test_parse_timestamp_unix_seconds():
    ts = 1705315200  # 2024-01-15 in seconds
    result = parse_timestamp_to_ns(ts)
    assert result == ts * int(1e9)


def test_parse_timestamp_unix_ms():
    ts = 1705315200000  # 2024-01-15 in ms
    result = parse_timestamp_to_ns(ts)
    assert result == ts * int(1e6)


def test_parse_timestamp_unix_ns():
    ts = 1705315200_000_000_000  # 2024-01-15 in ns
    result = parse_timestamp_to_ns(ts)
    assert result == ts


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


def test_extract_text_plain_string():
    assert _extract_text("hello world") == "hello world"


def test_extract_text_list_of_dicts():
    content = [{"text": "part1"}, {"text": "part2"}]
    assert _extract_text(content) == "part1\npart2"


def test_extract_text_empty_list():
    assert _extract_text([]) == ""


def test_extract_text_none():
    assert _extract_text(None) == ""


def test_extract_text_mixed_list():
    content = [{"text": "a"}, "b", {"text": "c"}]
    result = _extract_text(content)
    assert "a" in result
    assert "b" in result
    assert "c" in result


# ---------------------------------------------------------------------------
# _find_last_user_message
# ---------------------------------------------------------------------------


def _conv(*roles_and_content):
    """Build a minimal conversation list: [(role, content), ...]"""
    return [{"role": r, "content": c} for r, c in roles_and_content]


def test_find_last_user_message_basic():
    conv = _conv(("user", "hello"), ("assistant", "hi"), ("user", "how are you?"))
    idx = _find_last_user_message(conv)
    assert idx == 2


def test_find_last_user_message_none_if_empty():
    assert _find_last_user_message([]) is None


def test_find_last_user_message_only_assistant():
    conv = _conv(("assistant", "hi there"))
    assert _find_last_user_message(conv) is None


def test_find_last_user_message_skips_empty_content():
    conv = _conv(("user", "first"), ("assistant", "ok"), ("user", ""))
    # Last user message has empty content; should fall back to first one
    idx = _find_last_user_message(conv)
    assert idx == 0


# ---------------------------------------------------------------------------
# _find_final_assistant_response
# ---------------------------------------------------------------------------


def test_find_final_assistant_response():
    conv = _conv(
        ("user", "hello"),
        ("assistant", "first response"),
        ("user", "follow-up"),
        ("assistant", "second response"),
    )
    result = _find_final_assistant_response(conv, start_idx=1)
    assert result == "second response"


def test_find_final_assistant_response_none_when_none():
    conv = _conv(("user", "hello"), ("user", "another user msg"))
    result = _find_final_assistant_response(conv, start_idx=1)
    assert result is None


# ---------------------------------------------------------------------------
# _build_usage_dict
# ---------------------------------------------------------------------------


def test_build_usage_dict_basic():
    usage = {"input_tokens": 100, "output_tokens": 50}
    result = _build_usage_dict(usage)
    assert result[TokenUsageKey.INPUT_TOKENS] == 100
    assert result[TokenUsageKey.OUTPUT_TOKENS] == 50
    assert result[TokenUsageKey.TOTAL_TOKENS] == 150


def test_build_usage_dict_with_cache_tokens():
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 10,
    }
    result = _build_usage_dict(usage)
    assert result[TokenUsageKey.CACHE_READ_INPUT_TOKENS] == 20
    assert result[TokenUsageKey.CACHE_CREATION_INPUT_TOKENS] == 10


def test_build_usage_dict_empty():
    result = _build_usage_dict({})
    assert result[TokenUsageKey.TOTAL_TOKENS] == 0


# ---------------------------------------------------------------------------
# get_hook_response
# ---------------------------------------------------------------------------


def test_get_hook_response_success():
    resp = get_hook_response()
    assert resp["continue"] is True
    assert "stopReason" not in resp


def test_get_hook_response_error():
    resp = get_hook_response(error="Something went wrong")
    assert resp["continue"] is False
    assert resp["stopReason"] == "Something went wrong"


# ---------------------------------------------------------------------------
# is_tracing_enabled
# ---------------------------------------------------------------------------


def test_is_tracing_enabled_false_by_default(monkeypatch):
    monkeypatch.setenv("MLFLOW_KIRO_TRACING_ENABLED", "")
    assert is_tracing_enabled() is False


def test_is_tracing_enabled_true(monkeypatch):
    monkeypatch.setenv("MLFLOW_KIRO_TRACING_ENABLED", "true")
    assert is_tracing_enabled() is True


def test_is_tracing_enabled_1(monkeypatch):
    monkeypatch.setenv("MLFLOW_KIRO_TRACING_ENABLED", "1")
    assert is_tracing_enabled() is True


# ---------------------------------------------------------------------------
# process_session
# ---------------------------------------------------------------------------


def _minimal_session_data(session_id="sess-001"):
    return {
        "session_id": session_id,
        "kiro_version": "1.2.3",
        "conversation": [
            {
                "role": "user",
                "content": "Refactor my function please",
                "timestamp": ISO_TS,
            },
            {
                "role": "assistant",
                "content": "Sure! Here is the refactored version.",
                "timestamp": ISO_TS_5,
                "model": "claude-3-5-sonnet",
                "usage": {"input_tokens": 100, "output_tokens": 200},
            },
        ],
    }


@mock.patch("mlflow.kiro.tracing.mlflow")
@mock.patch("mlflow.kiro.tracing.InMemoryTraceManager")
@mock.patch("mlflow.kiro.tracing._get_trace_exporter")
def test_process_session_basic(mock_exporter, mock_itm, mock_mlflow):
    """process_session returns a trace object on a valid conversation."""
    mock_span = mock.MagicMock()
    mock_span.trace_id = "trace-123"
    mock_mlflow.start_span_no_context.return_value = mock_span
    mock_mlflow.get_trace.return_value = mock.MagicMock()

    mock_ctx = mock.MagicMock()
    mock_itm.get_instance.return_value.get_trace.return_value.__enter__ = (
        lambda s, *a: mock_ctx
    )
    mock_itm.get_instance.return_value.get_trace.return_value.__exit__ = (
        lambda s, *a: None
    )
    mock_ctx.info.trace_metadata = {}
    mock_ctx.info.request_preview = ""
    mock_ctx.info.response_preview = ""

    result = process_session(_minimal_session_data())
    assert result is not None
    # Two calls: one for the root AGENT span, one for the LLM child span
    assert mock_mlflow.start_span_no_context.call_count >= 2


@mock.patch("mlflow.kiro.tracing.mlflow")
@mock.patch("mlflow.kiro.tracing.InMemoryTraceManager")
@mock.patch("mlflow.kiro.tracing._get_trace_exporter")
def test_process_session_empty_conversation(mock_exporter, mock_itm, mock_mlflow):
    """process_session creates minimal trace when no conversation is present."""
    mock_span = mock.MagicMock()
    mock_span.trace_id = "trace-no-conv"
    mock_mlflow.start_span_no_context.return_value = mock_span
    mock_mlflow.get_trace.return_value = mock.MagicMock()

    mock_ctx = mock.MagicMock()
    mock_itm.get_instance.return_value.get_trace.return_value.__enter__ = (
        lambda s, *a: mock_ctx
    )
    mock_itm.get_instance.return_value.get_trace.return_value.__exit__ = (
        lambda s, *a: None
    )
    mock_ctx.info.trace_metadata = {}

    result = process_session({"session_id": "sess-noconv", "conversation": []})
    assert result is not None


@mock.patch("mlflow.kiro.tracing.mlflow")
@mock.patch("mlflow.kiro.tracing.InMemoryTraceManager")
@mock.patch("mlflow.kiro.tracing._get_trace_exporter")
def test_process_session_no_user_message(mock_exporter, mock_itm, mock_mlflow):
    """process_session returns None when there is no user message."""
    result = process_session({
        "session_id": "sess-no-user",
        "conversation": [
            {"role": "assistant", "content": "I can help you!"},
        ],
    })
    assert result is None


@mock.patch("mlflow.kiro.tracing.mlflow")
@mock.patch("mlflow.kiro.tracing.InMemoryTraceManager")
@mock.patch("mlflow.kiro.tracing._get_trace_exporter")
def test_process_session_with_tool_calls(mock_exporter, mock_itm, mock_mlflow):
    """process_session handles tool calls in the conversation."""
    mock_span = mock.MagicMock()
    mock_span.trace_id = "trace-tools"
    mock_mlflow.start_span_no_context.return_value = mock_span
    mock_mlflow.get_trace.return_value = mock.MagicMock()

    mock_ctx = mock.MagicMock()
    mock_itm.get_instance.return_value.get_trace.return_value.__enter__ = (
        lambda s, *a: mock_ctx
    )
    mock_itm.get_instance.return_value.get_trace.return_value.__exit__ = (
        lambda s, *a: None
    )
    mock_ctx.info.trace_metadata = {}

    session_data = {
        "session_id": "sess-tools",
        "conversation": [
            {"role": "user", "content": "Read my file", "timestamp": ISO_TS},
            {
                "role": "assistant",
                "content": "",
                "timestamp": ISO_TS_5,
                "model": "claude-3-5-sonnet",
                "tool_calls": [
                    {
                        "id": "tool_abc",
                        "name": "read_file",
                        "input": {"path": "src/main.py"},
                        "result": "def main(): ...",
                    }
                ],
            },
        ],
    }
    result = process_session(session_data)
    # Tool spans should be created
    assert mock_mlflow.start_span_no_context.call_count >= 1
