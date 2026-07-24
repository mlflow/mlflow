import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlflow.assistant.providers.base import clear_config_cache
from mlflow.assistant.providers.codex import CodexProvider
from mlflow.assistant.types import EventType


def _make_stdout_lines(*dicts) -> list[bytes]:
    return [json.dumps(d).encode() + b"\n" for d in dicts]


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
    config_file.write_text('{"providers": {"codex": {"model": "default"}}}')
    clear_config_cache()
    with patch("mlflow.assistant.config.CONFIG_PATH", config_file):
        yield config_file
    clear_config_cache()


@pytest.mark.parametrize(
    ("which_return", "expected"),
    [
        ("/usr/local/bin/codex", True),
        (None, False),
    ],
)
def test_is_available(which_return, expected):
    with patch("mlflow.assistant.providers.codex.shutil.which", return_value=which_return):
        provider = CodexProvider()
        assert provider.is_available() is expected


def test_provider_name():
    assert CodexProvider().name == "codex"


def test_provider_display_name():
    assert CodexProvider().display_name == "OpenAI Codex"


@pytest.mark.asyncio
async def test_astream_yields_error_when_codex_not_found():
    with patch("mlflow.assistant.providers.codex.shutil.which", return_value=None):
        provider = CodexProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    assert len(events) == 1
    assert events[0].type == EventType.ERROR
    assert "codex" in events[0].data["error"].lower()


@pytest.mark.asyncio
async def test_astream_yields_agent_message_text():
    stdout_lines = _make_stdout_lines(
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"id": "i1", "type": "agent_message", "text": "Hello there!"},
        },
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
    )

    mock_proc = _mock_process(stdout_lines=stdout_lines)

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ),
    ):
        provider = CodexProvider()
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]

    message_events = [e for e in events if e.type == EventType.MESSAGE]
    assert len(message_events) == 1
    assert message_events[0].data["message"]["content"][0]["text"] == "Hello there!"

    done_events = [e for e in events if e.type == EventType.DONE]
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_astream_emits_usage_event_from_turn_completed():
    stdout_lines = _make_stdout_lines(
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 5},
        },
    )

    mock_proc = _mock_process(stdout_lines=stdout_lines)

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ),
        patch(
            "mlflow.assistant.providers.codex.calculate_cost_by_model_and_token_usage",
            return_value=None,
        ) as mock_cost,
    ):
        provider = CodexProvider()
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]

    usage_events = [
        e
        for e in events
        if e.type == EventType.STREAM_EVENT and e.data["event"].get("type") == "usage"
    ]
    assert len(usage_events) == 1
    usage = usage_events[0].data["event"]["usage"]
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 15
    assert usage["total_cost_usd"] is None
    mock_cost.assert_called_once()


@pytest.mark.asyncio
async def test_astream_ignores_non_agent_message_items():
    mcp_item = {"id": "i1", "type": "mcp_tool_call", "text": "ignored"}
    agent_item = {"id": "i2", "type": "agent_message", "text": "kept"}
    stdout_lines = _make_stdout_lines(
        {"type": "item.completed", "item": mcp_item},
        {"type": "item.completed", "item": agent_item},
    )

    mock_proc = _mock_process(stdout_lines=stdout_lines)

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ),
    ):
        provider = CodexProvider()
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]

    message_events = [e for e in events if e.type == EventType.MESSAGE]
    assert len(message_events) == 1
    assert message_events[0].data["message"]["content"][0]["text"] == "kept"


@pytest.mark.asyncio
async def test_astream_yields_error_on_nonzero_exit():
    mock_proc = _mock_process(returncode=1, stderr=b"OPENAI_API_KEY not set")

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ),
    ):
        provider = CodexProvider()
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]

    error_events = [e for e in events if e.type == EventType.ERROR]
    assert len(error_events) == 1
    assert "OPENAI_API_KEY" in error_events[0].data["error"]


@pytest.mark.asyncio
async def test_astream_yields_interrupted_on_sigkill():
    mock_proc = _mock_process(returncode=-9)

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ),
    ):
        provider = CodexProvider()
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]

    assert len(events) == 1
    assert events[0].type == EventType.INTERRUPTED


@pytest.mark.asyncio
async def test_astream_yields_interrupted_when_killed():
    # On Windows a kill surfaces as a positive exit code, so classification
    # relies on `killed` rather than the returncode.
    mock_proc = _mock_process(returncode=1, killed=True)

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ),
    ):
        provider = CodexProvider()
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]

    assert len(events) == 1
    assert events[0].type == EventType.INTERRUPTED


@pytest.mark.asyncio
async def test_astream_builds_correct_command():
    mock_proc = _mock_process()

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ) as mock_ctor,
    ):
        provider = CodexProvider()
        _ = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    cmd = mock_ctor.call_args[0][0]
    assert "/usr/bin/codex" in cmd
    assert "exec" in cmd
    assert "--json" in cmd
    assert "--sandbox" in cmd
    assert "danger-full-access" in cmd
    assert "--dangerously-bypass-approvals-and-sandbox" not in cmd
    assert "--ephemeral" not in cmd
    assert "--skip-git-repo-check" in cmd
    assert cmd[-1] == "-"


@pytest.mark.asyncio
async def test_astream_includes_model_flag_when_configured(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"providers": {"codex": {"model": "o4-mini"}}}')

    mock_proc = _mock_process()

    clear_config_cache()
    with (
        patch("mlflow.assistant.config.CONFIG_PATH", config_file),
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ) as mock_ctor,
    ):
        provider = CodexProvider()
        _ = [e async for e in provider.astream("prompt", "http://localhost:5000")]

    cmd = mock_ctor.call_args[0][0]
    assert "-m" in cmd
    assert "o4-mini" in cmd


@pytest.mark.asyncio
async def test_astream_skips_model_flag_when_default(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"providers": {"codex": {"model": "default"}}}')

    mock_proc = _mock_process()

    clear_config_cache()
    with (
        patch("mlflow.assistant.config.CONFIG_PATH", config_file),
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ) as mock_ctor,
    ):
        provider = CodexProvider()
        _ = [e async for e in provider.astream("prompt", "http://localhost:5000")]

    cmd = mock_ctor.call_args[0][0]
    assert "-m" not in cmd


@pytest.mark.asyncio
async def test_astream_sends_prompt_via_stdin():
    mock_proc = _mock_process()

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ) as mock_ctor,
    ):
        provider = CodexProvider()
        _ = [e async for e in provider.astream("my question", "http://localhost:5000")]

    input_bytes = mock_ctor.call_args.kwargs["input_bytes"]
    assert b"<system_instructions>" in input_bytes
    assert b"my question" in input_bytes


@pytest.mark.asyncio
async def test_astream_omits_system_instructions_on_resume():
    mock_proc = _mock_process()

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ) as mock_ctor,
    ):
        provider = CodexProvider()
        _ = [
            e
            async for e in provider.astream(
                "follow up", "http://localhost:5000", session_id="abc-123"
            )
        ]

    input_bytes = mock_ctor.call_args.kwargs["input_bytes"]
    assert b"<system_instructions>" not in input_bytes
    assert b"follow up" in input_bytes


@pytest.mark.asyncio
async def test_astream_captures_session_id_from_thread_started():
    stdout_lines = _make_stdout_lines(
        {"type": "thread.started", "thread_id": "abc-123"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}},
    )
    mock_proc = _mock_process(stdout_lines=stdout_lines)

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ),
    ):
        provider = CodexProvider()
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]

    done_events = [e for e in events if e.type == EventType.DONE]
    assert len(done_events) == 1
    assert done_events[0].data["session_id"] == "abc-123"


@pytest.mark.asyncio
async def test_astream_resumes_session_when_session_id_provided():
    mock_proc = _mock_process()

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ) as mock_ctor,
    ):
        provider = CodexProvider()
        _ = [
            e
            async for e in provider.astream(
                "follow up", "http://localhost:5000", session_id="abc-123"
            )
        ]

    cmd = mock_ctor.call_args[0][0]
    assert "resume" in cmd
    assert "abc-123" in cmd
    resume_idx = cmd.index("resume")
    assert cmd[resume_idx + 1] == "abc-123"


@pytest.mark.asyncio
async def test_astream_saves_and_clears_process_pid():
    mock_proc = _mock_process()
    mock_proc.pid = 12345

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ),
        patch("mlflow.assistant.providers.codex.save_process_pid") as mock_save,
        patch("mlflow.assistant.providers.codex.clear_process_pid") as mock_clear,
    ):
        provider = CodexProvider()
        _ = [
            e
            async for e in provider.astream(
                "hi", "http://localhost:5000", mlflow_session_id="session-xyz"
            )
        ]

    mock_save.assert_called_once_with("session-xyz", 12345)
    mock_clear.assert_called_once_with("session-xyz")


@pytest.mark.asyncio
async def test_astream_ignores_invalid_json_lines():
    valid_item = {
        "type": "item.completed",
        "item": {"type": "agent_message", "text": "valid"},
    }
    stdout_lines = [
        b"not json\n",
        json.dumps(valid_item).encode() + b"\n",
    ]

    mock_proc = _mock_process(stdout_lines=stdout_lines)

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.SubprocessLineStream",
            return_value=mock_proc,
        ),
    ):
        provider = CodexProvider()
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]

    message_events = [e for e in events if e.type == EventType.MESSAGE]
    assert len(message_events) == 1
    assert message_events[0].data["message"]["content"][0]["text"] == "valid"
