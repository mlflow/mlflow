from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlflow.server.assistant.providers.claude_code import ClaudeCodeProvider


class AsyncIterator:
    """Helper to mock async stdout iteration."""

    def __init__(self, items):
        self.items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration


@pytest.mark.parametrize(
    ("which_return", "expected"),
    [
        ("/usr/bin/claude", True),
        (None, False),
    ],
)
def test_is_available(which_return, expected):
    with patch(
        "mlflow.server.assistant.providers.claude_code.shutil.which",
        return_value=which_return,
    ):
        provider = ClaudeCodeProvider()
        assert provider.is_available() is expected


@pytest.mark.parametrize(
    ("file_content", "expected_config"),
    [
        (None, {}),  # No file
        ("not valid json", {}),  # Invalid JSON
        (
            '{"projectPath": "/my/project", "model": "opus"}',
            {"projectPath": "/my/project", "model": "opus"},
        ),
    ],
)
def test_load_config(tmp_path, file_content, expected_config, monkeypatch):
    config_file = tmp_path / "config.json"
    if file_content is not None:
        config_file.write_text(file_content)

    monkeypatch.setattr(
        "mlflow.server.assistant.providers.claude_code.CLAUDE_CONFIG_FILE", config_file
    )
    provider = ClaudeCodeProvider()
    assert provider.load_config() == expected_config


@pytest.mark.asyncio
async def test_run_yields_error_when_claude_not_found():
    with patch(
        "mlflow.server.assistant.providers.claude_code.shutil.which",
        return_value=None,
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.run("test prompt")]

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "not found" in events[0]["data"]["error"]


@pytest.mark.asyncio
async def test_run_builds_correct_command():
    mock_process = MagicMock()
    mock_process.stdout = AsyncIterator([b'{"type": "result"}\n'])
    mock_process.stderr = MagicMock()
    mock_process.stderr.read = AsyncMock(return_value=b"")
    mock_process.wait = AsyncMock()
    mock_process.returncode = 0

    with (
        patch(
            "mlflow.server.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.server.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec,
    ):
        provider = ClaudeCodeProvider()
        _ = [e async for e in provider.run("test prompt")]

    call_args = mock_exec.call_args[0]
    assert "/usr/bin/claude" in call_args
    assert "-p" in call_args
    assert "test prompt" in call_args
    assert "--output-format" in call_args
    assert "stream-json" in call_args
    assert "--verbose" in call_args
    assert "--append-system-prompt" in call_args


@pytest.mark.asyncio
async def test_run_streams_assistant_messages():
    mock_stdout = AsyncIterator(
        [
            b'{"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi!"}]}}\n',
            b'{"type": "result", "session_id": "session-123"}\n',
        ]
    )

    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = MagicMock()
    mock_process.stderr.read = AsyncMock(return_value=b"")
    mock_process.wait = AsyncMock()
    mock_process.returncode = 0

    with (
        patch(
            "mlflow.server.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.server.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ),
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.run("test prompt")]

    assert len(events) == 2
    assert events[0]["type"] == "message"
    assert events[0]["data"]["text"] == "Hi!"
    assert events[1]["type"] == "done"
    assert events[1]["data"]["session_id"] == "session-123"


@pytest.mark.asyncio
async def test_run_handles_process_error():
    mock_process = MagicMock()
    mock_process.stdout = AsyncIterator([])
    mock_process.stderr = MagicMock()
    mock_process.stderr.read = AsyncMock(return_value=b"Command failed")
    mock_process.wait = AsyncMock()
    mock_process.returncode = 1

    with (
        patch(
            "mlflow.server.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.server.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ),
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.run("test prompt")]

    assert events[-1]["type"] == "error"
    assert "Command failed" in events[-1]["data"]["error"]


@pytest.mark.asyncio
async def test_run_passes_session_id_for_resume():
    mock_process = MagicMock()
    mock_process.stdout = AsyncIterator([b'{"type": "result"}\n'])
    mock_process.stderr = MagicMock()
    mock_process.stderr.read = AsyncMock(return_value=b"")
    mock_process.wait = AsyncMock()
    mock_process.returncode = 0

    with (
        patch(
            "mlflow.server.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.server.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec,
    ):
        provider = ClaudeCodeProvider()
        _ = [e async for e in provider.run("prompt", session_id="existing-session")]

    call_args = mock_exec.call_args[0]
    assert "--resume" in call_args
    assert "existing-session" in call_args


@pytest.mark.asyncio
async def test_run_handles_non_json_output():
    mock_stdout = AsyncIterator(
        [
            b"Some plain text output\n",
            b'{"type": "result"}\n',
        ]
    )

    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = MagicMock()
    mock_process.stderr.read = AsyncMock(return_value=b"")
    mock_process.wait = AsyncMock()
    mock_process.returncode = 0

    with (
        patch(
            "mlflow.server.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.server.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ),
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.run("test prompt")]

    assert events[0]["type"] == "message"
    assert events[0]["data"]["text"] == "Some plain text output"


@pytest.mark.asyncio
async def test_run_handles_error_message_type():
    mock_stdout = AsyncIterator(
        [
            b'{"type": "error", "error": {"message": "API rate limit exceeded"}}\n',
        ]
    )

    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = MagicMock()
    mock_process.stderr.read = AsyncMock(return_value=b"")
    mock_process.wait = AsyncMock()
    mock_process.returncode = 0

    with (
        patch(
            "mlflow.server.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.server.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ),
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.run("test prompt")]

    assert events[0]["type"] == "error"
    assert "rate limit" in events[0]["data"]["error"]
