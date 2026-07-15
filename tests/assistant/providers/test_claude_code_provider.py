import errno
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlflow.assistant.providers.base import NotAuthenticatedError
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


def _mock_process(stdout_lines=None, returncode=0, stderr=b"", pid=12345):
    """Create a mock process with async stdout iteration and stdin support.

    The provider pipes the user message via stdin, so mocks must support the
    async stdin write/drain/close/wait_closed sequence.
    """
    process = MagicMock()
    process.returncode = returncode
    process.pid = pid

    process.stdin = MagicMock()
    process.stdin.write = MagicMock()
    process.stdin.drain = AsyncMock()
    process.stdin.close = MagicMock()
    process.stdin.wait_closed = AsyncMock()

    process.stdout = AsyncIterator(stdout_lines or [])

    process.stderr = MagicMock()
    process.stderr.read = AsyncMock(return_value=stderr)

    process.wait = AsyncMock()
    process.kill = MagicMock()

    return process


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


def test_check_connection_runs_auth_verification_prompt():
    completed = subprocess.CompletedProcess(
        args=["claude", "-p", "hi", "--max-turns", "1", "--output-format", "json"],
        returncode=0,
        stdout='{"type": "result"}',
        stderr="",
    )

    with (
        patch(
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.subprocess.run",
            return_value=completed,
        ) as mock_run,
    ):
        provider = ClaudeCodeProvider()
        provider.check_connection()

    mock_run.assert_called_once_with(
        ["claude", "-p", "hi", "--max-turns", "1", "--output-format", "json"],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_check_connection_raises_not_authenticated_for_auth_error():
    completed = subprocess.CompletedProcess(
        args=["claude", "-p", "hi", "--max-turns", "1", "--output-format", "json"],
        returncode=1,
        stdout="",
        stderr="Login required",
    )

    with (
        patch(
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch("mlflow.assistant.providers.claude_code.subprocess.run", return_value=completed),
    ):
        provider = ClaudeCodeProvider()
        with pytest.raises(NotAuthenticatedError, match="claude login"):
            provider.check_connection()


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

    mock_process = _mock_process(stdout_lines=[b'{"type": "result"}\n'])

    # Capture the system-prompt file's contents while the subprocess is "running",
    # since the provider deletes the file after the process exits.
    captured = {}

    def _capture(*args, **kwargs):
        idx = args.index("--append-system-prompt-file")
        file_path = args[idx + 1]
        captured["argv"] = args
        captured["file_path"] = file_path
        captured["file_contents"] = Path(file_path).read_text()
        return mock_process

    with (
        patch(
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            side_effect=_capture,
        ),
    ):
        provider = ClaudeCodeProvider()
        _ = [
            e async for e in provider.astream("test prompt", "http://localhost:5000", cwd=tmp_path)
        ]

    call_args = captured["argv"]
    assert "/usr/bin/claude" in call_args
    assert "-p" in call_args
    assert "--output-format" in call_args
    assert "stream-json" in call_args
    assert "--verbose" in call_args

    # The large system prompt must NOT be passed inline as a CLI argument
    # (Windows cmd.exe caps the command line at 8191 chars). It goes in a file.
    assert "--append-system-prompt" not in call_args
    assert "--append-system-prompt-file" in call_args
    assert "http://localhost:5000" in captured["file_contents"]
    assert not any("You are an MLflow assistant" in str(a) for a in call_args)

    # The user message must NOT be an inline CLI arg either; it goes via stdin.
    assert "test prompt" not in call_args
    stdin_bytes = mock_process.stdin.write.call_args[0][0]
    assert b"test prompt" in stdin_bytes

    # Regression guard: the full command line must stay well under the 8191-char
    # Windows cmd.exe limit even though the system prompt is ~9,900 chars.
    assert len(" ".join(str(a) for a in call_args)) < 4096

    # Verify Skill permission is granted by default
    allowed_tools = [
        call_args[i + 1] for i, arg in enumerate(call_args) if arg == "--allowed-tools"
    ]
    assert "Skill" in allowed_tools


@pytest.mark.asyncio
async def test_astream_passes_cwd_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_ENV_VAR", "test_value")

    mock_process = _mock_process(stdout_lines=[b'{"type": "result"}\n'])

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

    call_kwargs = mock_exec.call_args[1]
    assert call_kwargs["cwd"] == tmp_path
    assert call_kwargs["env"]["MLFLOW_TRACKING_URI"] == "http://localhost:5000"
    assert call_kwargs["env"]["TEST_ENV_VAR"] == "test_value"


@pytest.mark.asyncio
async def test_astream_cleans_up_system_prompt_file(tmp_path):
    mock_process = _mock_process(stdout_lines=[b'{"type": "result"}\n'])

    captured = {}

    def _capture(*args, **kwargs):
        idx = args.index("--append-system-prompt-file")
        captured["file_path"] = args[idx + 1]
        return mock_process

    with (
        patch(
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            side_effect=_capture,
        ),
    ):
        provider = ClaudeCodeProvider()
        _ = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    # The temp file must be cleaned up after the subprocess completes.
    assert not Path(captured["file_path"]).exists()


@pytest.mark.asyncio
async def test_astream_temp_file_cleanup_failure_does_not_mask_result():
    # A failure while unlinking the temp file (e.g. a lingering handle on
    # Windows) must be swallowed so it doesn't escape the generator or mask
    # the real result.
    mock_process = _mock_process(stdout_lines=[b'{"type": "result", "session_id": "s1"}\n'])

    with (
        patch(
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ),
        patch(
            "mlflow.assistant.providers.claude_code.Path.unlink",
            side_effect=PermissionError("file in use"),
        ),
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    # The stream completes normally; the cleanup error is swallowed.
    assert events[-1].type == EventType.DONE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "write_error",
    [
        BrokenPipeError("broken pipe"),
        # POSIX EPIPE can surface as a bare OSError rather than BrokenPipeError.
        OSError(errno.EPIPE, "broken pipe"),
    ],
)
async def test_astream_surfaces_cli_error_when_stdin_pipe_breaks(write_error):
    # If the CLI exits before reading stdin, writing the message raises a pipe
    # error; the provider must swallow it and surface the CLI's real stderr
    # instead of a bare "Broken pipe" message.
    mock_process = _mock_process(stdout_lines=[], returncode=1, stderr=b"Invalid session id")
    mock_process.stdin.write = MagicMock(side_effect=write_error)

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
    assert "Invalid session id" in events[-1].data["error"]
    assert "broken pipe" not in events[-1].data["error"].lower()


@pytest.mark.asyncio
async def test_astream_cleans_up_temp_file_when_subprocess_launch_fails():
    # If create_subprocess_exec raises after the temp file is written, the
    # finally block must still remove it (no leaked temp files).
    created_paths = []
    real_mkstemp = tempfile.mkstemp

    def _tracking_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created_paths.append(path)
        return fd, path

    with (
        patch(
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.tempfile.mkstemp",
            side_effect=_tracking_mkstemp,
        ),
        patch(
            "mlflow.assistant.providers.claude_code.asyncio.create_subprocess_exec",
            side_effect=OSError("boom"),
        ),
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    assert events[-1].type == EventType.ERROR
    assert len(created_paths) == 1
    assert not Path(created_paths[0]).exists()


@pytest.mark.asyncio
async def test_astream_streams_assistant_messages():
    mock_process = _mock_process(
        stdout_lines=[
            b'{"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi!"}]}}\n',
            b'{"type": "result", "session_id": "session-123"}\n',
        ]
    )

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
async def test_astream_emits_usage_event_before_done():
    mock_process = _mock_process(
        stdout_lines=[
            b'{"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi!"}]}}\n',
            b'{"type": "result", "session_id": "session-123", "total_cost_usd": 0.1319, '
            b'"usage": {"input_tokens": 2, "cache_creation_input_tokens": 35155, '
            b'"cache_read_input_tokens": 100, "output_tokens": 5}}\n',
        ]
    )

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

    usage_events = [
        e
        for e in events
        if e.type == EventType.STREAM_EVENT and e.data["event"].get("type") == "usage"
    ]
    assert len(usage_events) == 1
    assert usage_events[0].data["event"]["usage"] == {
        "prompt_tokens": 35257,
        "completion_tokens": 5,
        "total_tokens": 35262,
        "total_cost_usd": 0.1319,
    }
    # The usage event must precede the DONE event, which closes the client stream.
    assert events.index(usage_events[0]) < next(
        i for i, e in enumerate(events) if e.type == EventType.DONE
    )


@pytest.mark.parametrize(
    ("usage", "cost_usd", "expected"),
    [
        (
            {"input_tokens": 10, "output_tokens": 5},
            0.25,
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "total_cost_usd": 0.25,
            },
        ),
        (
            {
                "input_tokens": 2,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 50,
                "output_tokens": 5,
            },
            None,
            {
                "prompt_tokens": 152,
                "completion_tokens": 5,
                "total_tokens": 157,
                "total_cost_usd": None,
            },
        ),
        (
            {},
            None,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": None,
            },
        ),
    ],
)
def test_build_usage_event_sums_cache_tokens(usage, cost_usd, expected):
    event = ClaudeCodeProvider._build_usage_event(usage, cost_usd)
    assert event.type == EventType.STREAM_EVENT
    assert event.data["event"]["type"] == "usage"
    assert event.data["event"]["usage"] == expected


@pytest.mark.asyncio
async def test_astream_handles_process_error():
    mock_process = _mock_process(returncode=1, stderr=b"Command failed")

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
    mock_process = _mock_process(stdout_lines=[b'{"type": "result"}\n'])

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
    mock_process = _mock_process(
        stdout_lines=[
            b"Some plain text output\n",
            b'{"type": "result"}\n',
        ]
    )

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
    mock_process = _mock_process(
        stdout_lines=[
            b'{"type": "error", "error": {"message": "API rate limit exceeded"}}\n',
        ]
    )

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


@pytest.mark.asyncio
async def test_astream_skips_rate_limit_event():
    mock_process = _mock_process(
        stdout_lines=[
            b'{"type": "rate_limit_event", "data": {"retry_after": 1.0}}\n',
            b'{"type": "assistant", "message": {"content": [{"type": "text", "text": "hello"}]}}\n',
            b'{"type": "result", "result": null, "session_id": "sess-123"}\n',
        ]
    )

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
    assert events[1].type == EventType.DONE
