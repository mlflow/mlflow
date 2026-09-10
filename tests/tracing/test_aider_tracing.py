"""Tests for MLflow Aider CLI tracing integration."""

import pytest
from pathlib import Path
from mlflow.aider.tracing import (
    read_aider_history,
    find_last_user_message,
    find_final_assistant_response,
    process_aider_history,
)

SAMPLE_HISTORY = """# aider chat started at 2024-01-01 10:00:00

## user

Can you help me fix the bug in my code?

## assistant

Sure! I can help you fix that bug.

## user

Here is the error: TypeError on line 42

## assistant

I see the issue. Here is the fix.
"""


def test_read_aider_history(tmp_path):
    history_file = tmp_path / ".aider.llm.history"
    history_file.write_text(SAMPLE_HISTORY)
    messages = read_aider_history(str(history_file))
    assert len(messages) > 0
    roles = [m["role"] for m in messages]
    assert "user" in roles
    assert "assistant" in roles


def test_read_aider_history_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_aider_history("/nonexistent/.aider.llm.history")


def test_find_last_user_message():
    messages = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "First response"},
        {"role": "user", "content": "Second message"},
    ]
    result = find_last_user_message(messages)
    assert result == "Second message"


def test_find_last_user_message_empty():
    assert find_last_user_message([]) is None


def test_find_final_assistant_response():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "First response"},
        {"role": "user", "content": "Follow up"},
        {"role": "assistant", "content": "Final response"},
    ]
    result = find_final_assistant_response(messages)
    assert result == "Final response"


def test_find_final_assistant_response_empty():
    assert find_final_assistant_response([]) is None


def test_process_aider_history(tmp_path):
    history_file = tmp_path / ".aider.llm.history"
    history_file.write_text(SAMPLE_HISTORY)
    import mlflow
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    trace = process_aider_history(str(history_file))
    assert trace is not None


def test_process_aider_history_missing_file():
    result = process_aider_history("/nonexistent/.aider.llm.history")
    assert result is None
