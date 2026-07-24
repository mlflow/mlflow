import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlflow.assistant.providers.base import NotAuthenticatedError
from mlflow.assistant.providers.claude_code import ClaudeCodeProvider
from mlflow.assistant.types import EventType


def _mock_process(stdout_lines=None, returncode=0, stderr=b"", killed=False):
    """Create a mock SubprocessLineStream presenting the streaming surface."""
    process = MagicMock()
    process.pid = 12345
    process.returncode = returncode
    process.killed = killed

    async def _lines():
        for line in stdout_lines or []:
            yield line

    process.lines = _lines

    process.wait = AsyncMock(return_value=returncode)
    process.read_stderr = AsyncMock(return_value=stderr)
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

    with (
        patch(
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.SubprocessLineStream",
            return_value=mock_process,
        ) as mock_ctor,
    ):
        provider = ClaudeCodeProvider()
        _ = [
            e async for e in provider.astream("test prompt", "http://localhost:5000", cwd=tmp_path)
        ]

    cmd = mock_ctor.call_args[0][0]
    assert "/usr/bin/claude" in cmd
    assert "-p" in cmd
    assert "test prompt" in cmd
    assert "--output-format" in cmd
    assert "stream-json" in cmd
    assert "--verbose" in cmd
    assert "--append-system-prompt" in cmd

    # Verify system prompt contains tracking URI
    system_prompt_idx = cmd.index("--append-system-prompt") + 1
    system_prompt = cmd[system_prompt_idx]
    assert "http://localhost:5000" in system_prompt

    # Verify Skill permission is granted by default
    allowed_tools = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--allowed-tools"]
    assert "Skill" in allowed_tools

    # Verify cwd and tracking URI env var are passed correctly
    call_kwargs = mock_ctor.call_args.kwargs
    assert call_kwargs["cwd"] == tmp_path
    assert call_kwargs["env"]["MLFLOW_TRACKING_URI"] == "http://localhost:5000"
    assert call_kwargs["env"]["TEST_ENV_VAR"] == "test_value"


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
            "mlflow.assistant.providers.claude_code.SubprocessLineStream",
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
            "mlflow.assistant.providers.claude_code.SubprocessLineStream",
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
            "mlflow.assistant.providers.claude_code.SubprocessLineStream",
            return_value=mock_process,
        ),
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    assert events[-1].type == EventType.ERROR
    assert "Command failed" in events[-1].data["error"]


@pytest.mark.asyncio
async def test_astream_yields_interrupted_on_sigkill():
    mock_process = _mock_process(returncode=-9)

    with (
        patch(
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.SubprocessLineStream",
            return_value=mock_process,
        ),
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    assert len(events) == 1
    assert events[0].type == EventType.INTERRUPTED


@pytest.mark.asyncio
async def test_astream_yields_interrupted_when_killed():
    # On Windows a kill surfaces as a positive exit code, so classification
    # relies on `killed` rather than the returncode.
    mock_process = _mock_process(returncode=1, killed=True)

    with (
        patch(
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.SubprocessLineStream",
            return_value=mock_process,
        ),
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    assert len(events) == 1
    assert events[0].type == EventType.INTERRUPTED


@pytest.mark.asyncio
async def test_astream_passes_session_id_for_resume():
    mock_process = _mock_process(stdout_lines=[b'{"type": "result"}\n'])

    with (
        patch(
            "mlflow.assistant.providers.claude_code.shutil.which",
            return_value="/usr/bin/claude",
        ),
        patch(
            "mlflow.assistant.providers.claude_code.SubprocessLineStream",
            return_value=mock_process,
        ) as mock_ctor,
    ):
        provider = ClaudeCodeProvider()
        _ = [
            e
            async for e in provider.astream(
                "prompt", "http://localhost:5000", session_id="existing-session"
            )
        ]

    cmd = mock_ctor.call_args[0][0]
    assert "--resume" in cmd
    assert "existing-session" in cmd


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
            "mlflow.assistant.providers.claude_code.SubprocessLineStream",
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
            "mlflow.assistant.providers.claude_code.SubprocessLineStream",
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
            "mlflow.assistant.providers.claude_code.SubprocessLineStream",
            return_value=mock_process,
        ),
    ):
        provider = ClaudeCodeProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    assert len(events) == 2
    assert events[0].type == EventType.MESSAGE
    assert events[1].type == EventType.DONE
