from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlflow.assistant.providers.claude_code import ClaudeCodeProvider
from mlflow.assistant.types import EventType


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


@pytest.fixture(autouse=True)
def config(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"providers": {"claude_code": {"model": "claude-opus-4"}}}')
    with patch("mlflow.assistant.config.CONFIG_PATH", config_file):
        yield config_file


@pytest.mark.parametrize(
    ("which_return", "expected"),
    [
        ("/usr/bin/claude", True),
        (None, False),
    ],
)
def test_is_available(which_return, expected):
    with patch(
        "mlflow.assistant.providers.claude_code.shutil.which",
        return_value=which_return,
    ):
        provider = ClaudeCodeProvider()
        assert provider.is_available() is expected


@pytest.mark.asyncio
async def test_astream_yields_error_when_claude_not_found():
    with patch(
        "mlflow.assistant.providers.claude_code.shutil.which",
        return_value=None,
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    assert len(events) == 1
    assert events[0].type == EventType.ERROR
    assert "not found" in events[0].data["error"]
    assert "PATH" in events[0].data["error"]


@pytest.mark.asyncio
async def test_astream_builds_correct_command(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_ENV_VAR", "test_value")

    mock_process = MagicMock()
    mock_process.stdout = AsyncIterator([b'{"type": "result"}\n'])
    mock_process.stderr = MagicMock()
    mock_process.stderr.read = AsyncMock(return_value=b"")
    mock_process.wait = AsyncMock()
    mock_process.returncode = 0

    with (
        patch(
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec,
    ):
        provider = ClaudeCodeProvider()
        _ = [
            e async for e in provider.astream("test prompt", "http://localhost:5000", cwd=tmp_path)
        ]

    call_args = mock_exec.call_args[0]
    assert "/usr/bin/claude" in call_args
    assert "-p" in call_args
    assert "test prompt" in call_args
    assert "--output-format" in call_args
    assert "stream-json" in call_args
    assert "--verbose" in call_args
    assert "--append-system-prompt" in call_args

    # Verify system prompt contains tracking URI
    system_prompt_idx = call_args.index("--append-system-prompt") + 1
    system_prompt = call_args[system_prompt_idx]
    assert "http://localhost:5000" in system_prompt

    # Verify Skill permission is granted by default
    allowed_tools = [
        call_args[i + 1] for i, arg in enumerate(call_args) if arg == "--allowed-tools"
    ]
    assert "Skill" in allowed_tools

    # Verify cwd and tracking URI env var are passed correctly
    call_kwargs = mock_exec.call_args[1]
    assert call_kwargs["cwd"] == tmp_path
    assert call_kwargs["env"]["MLFLOW_TRACKING_URI"] == "http://localhost:5000"
    assert call_kwargs["env"]["TEST_ENV_VAR"] == "test_value"


@pytest.mark.asyncio
async def test_astream_streams_assistant_messages():
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
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ),
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    assert len(events) == 2
    assert events[0].type == EventType.MESSAGE
    assert events[0].data["message"]["content"][0]["text"] == "Hi!"
    assert events[1].type == EventType.DONE
    assert events[1].data["session_id"] == "session-123"


@pytest.mark.asyncio
async def test_astream_handles_process_error():
    mock_process = MagicMock()
    mock_process.stdout = AsyncIterator([])
    mock_process.stderr = MagicMock()
    mock_process.stderr.read = AsyncMock(return_value=b"Command failed")
    mock_process.wait = AsyncMock()
    mock_process.returncode = 1

    with (
        patch(
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ),
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    assert events[-1].type == EventType.ERROR
    assert "Command failed" in events[-1].data["error"]


@pytest.mark.asyncio
async def test_astream_passes_session_id_for_resume():
    mock_process = MagicMock()
    mock_process.stdout = AsyncIterator([b'{"type": "result"}\n'])
    mock_process.stderr = MagicMock()
    mock_process.stderr.read = AsyncMock(return_value=b"")
    mock_process.wait = AsyncMock()
    mock_process.returncode = 0

    with (
        patch(
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec,
    ):
        provider = ClaudeCodeProvider()
        _ = [
            e
            async for e in provider.astream(
                "prompt", "http://localhost:5000", session_id="existing-session"
            )
        ]

    call_args = mock_exec.call_args[0]
    assert "--resume" in call_args
    assert "existing-session" in call_args


@pytest.mark.asyncio
async def test_astream_handles_non_json_output():
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
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ),
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    assert events[0].type == EventType.MESSAGE
    assert events[0].data["message"]["content"] == "Some plain text output"


@pytest.mark.asyncio
async def test_astream_handles_error_message_type():
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
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ),
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    assert events[0].type == EventType.ERROR
    assert "rate limit" in events[0].data["error"]
