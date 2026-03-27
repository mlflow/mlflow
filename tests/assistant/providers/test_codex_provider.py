from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlflow.assistant.providers.base import clear_config_cache
from mlflow.assistant.providers.codex import CodexProvider
from mlflow.assistant.types import EventType


def _jsonl(*dicts) -> bytes:
    import json

    return b"\n".join(json.dumps(d).encode() for d in dicts)


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
    assert CodexProvider().display_name == "OpenAI Codex CLI"


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
    stdout = _jsonl(
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"id": "i1", "type": "agent_message", "text": "Hello there!"},
        },
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
    )

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(stdout, b""))

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.asyncio.create_subprocess_exec",
            return_value=mock_process,
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
async def test_astream_ignores_non_agent_message_items():
    mcp_item = {"id": "i1", "type": "mcp_tool_call", "text": "ignored"}
    agent_item = {"id": "i2", "type": "agent_message", "text": "kept"}
    stdout = _jsonl(
        {"type": "item.completed", "item": mcp_item},
        {"type": "item.completed", "item": agent_item},
    )

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(stdout, b""))

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ),
    ):
        provider = CodexProvider()
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]

    message_events = [e for e in events if e.type == EventType.MESSAGE]
    assert len(message_events) == 1
    assert message_events[0].data["message"]["content"][0]["text"] == "kept"


@pytest.mark.asyncio
async def test_astream_yields_error_on_nonzero_exit():
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.communicate = AsyncMock(return_value=(b"", b"OPENAI_API_KEY not set"))

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ),
    ):
        provider = CodexProvider()
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]

    error_events = [e for e in events if e.type == EventType.ERROR]
    assert len(error_events) == 1
    assert "OPENAI_API_KEY" in error_events[0].data["error"]


@pytest.mark.asyncio
async def test_astream_yields_error_on_timeout():
    mock_process = MagicMock()
    mock_process.returncode = None
    mock_process.communicate = AsyncMock(side_effect=TimeoutError)
    mock_process.kill = MagicMock()
    mock_process.wait = AsyncMock()

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ),
    ):
        provider = CodexProvider()
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]

    assert len(events) == 1
    assert events[0].type == EventType.ERROR
    assert "timed out" in events[0].data["error"].lower()
    mock_process.kill.assert_called()


@pytest.mark.asyncio
async def test_astream_builds_correct_command():
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"", b""))

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec,
    ):
        provider = CodexProvider()
        _ = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    args = mock_exec.call_args[0]
    assert "/usr/bin/codex" in args
    assert "exec" in args
    assert "--json" in args
    assert "--dangerously-bypass-approvals-and-sandbox" in args
    assert "--ephemeral" in args
    assert "--skip-git-repo-check" in args
    assert args[-1] == "-"


@pytest.mark.asyncio
async def test_astream_includes_model_flag_when_configured(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"providers": {"codex": {"model": "o4-mini"}}}')

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"", b""))

    clear_config_cache()
    with (
        patch("mlflow.assistant.config.CONFIG_PATH", config_file),
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec,
    ):
        provider = CodexProvider()
        _ = [e async for e in provider.astream("prompt", "http://localhost:5000")]

    args = mock_exec.call_args[0]
    assert "-m" in args
    assert "o4-mini" in args


@pytest.mark.asyncio
async def test_astream_skips_model_flag_when_default(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"providers": {"codex": {"model": "default"}}}')

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"", b""))

    clear_config_cache()
    with (
        patch("mlflow.assistant.config.CONFIG_PATH", config_file),
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec,
    ):
        provider = CodexProvider()
        _ = [e async for e in provider.astream("prompt", "http://localhost:5000")]

    args = mock_exec.call_args[0]
    assert "-m" not in args


@pytest.mark.asyncio
async def test_astream_sends_prompt_via_stdin():
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"", b""))

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ),
    ):
        provider = CodexProvider()
        _ = [e async for e in provider.astream("my question", "http://localhost:5000")]

    communicate_call = mock_process.communicate.call_args
    stdin_bytes = communicate_call[1]["input"]
    assert b"<system_instructions>" in stdin_bytes
    assert b"my question" in stdin_bytes


@pytest.mark.asyncio
async def test_astream_ignores_invalid_json_lines():
    stdout = b"not json\n" + _jsonl(
        {"type": "item.completed", "item": {"type": "agent_message", "text": "valid"}}
    )

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(stdout, b""))

    with (
        patch("mlflow.assistant.providers.codex.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "mlflow.assistant.providers.codex.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ),
    ):
        provider = CodexProvider()
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]

    message_events = [e for e in events if e.type == EventType.MESSAGE]
    assert len(message_events) == 1
    assert message_events[0].data["message"]["content"][0]["text"] == "valid"
