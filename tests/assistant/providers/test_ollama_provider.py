import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlflow.assistant.providers.base import clear_config_cache
from mlflow.assistant.providers.ollama import _MAX_SESSION_BYTES, OllamaProvider, _trim_session
from mlflow.assistant.types import EventType

# ---------------------------------------------------------------------------
# aiohttp mock helpers
# ---------------------------------------------------------------------------


class _AsyncLineIter:
    """Yields pre-encoded JSON lines, one per call to __anext__."""

    def __init__(self, lines: list[bytes]):
        self._iter = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def _make_aiohttp_session(response_lines_per_call: list[list[bytes]], status: int = 200):
    """
    Build a mock aiohttp.ClientSession whose post() returns successive
    responses, each streaming the given byte lines.
    """
    responses = []
    for lines in response_lines_per_call:
        resp = MagicMock()
        resp.status = status
        resp.content = _AsyncLineIter(lines)
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        responses.append(resp)

    call_count = 0

    def _post(*args, **kwargs):
        nonlocal call_count
        r = responses[call_count]
        call_count += 1
        return r

    session = MagicMock()
    session.post = _post
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def _line(data: dict[str, object]) -> bytes:
    return (json.dumps(data) + "\n").encode()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def config(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"providers": {"ollama": {"model": "llama3.2"}}}')
    clear_config_cache()
    with patch("mlflow.assistant.config.CONFIG_PATH", config_file):
        yield config_file
    clear_config_cache()


# ---------------------------------------------------------------------------
# Basic provider tests
# ---------------------------------------------------------------------------


def test_is_available():
    assert OllamaProvider().is_available() is True


def test_provider_name():
    assert OllamaProvider().name == "ollama"


def test_provider_display_name():
    assert OllamaProvider().display_name == "Ollama"


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


def test_list_models_returns_model_names():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": [{"model": "llama3"}, {"model": "mistral"}]}
    mock_resp.raise_for_status = MagicMock()

    with patch(
        "mlflow.assistant.providers.ollama.requests.get", return_value=mock_resp
    ) as mock_get:
        models = OllamaProvider().list_models("http://localhost:11434")

    assert models == ["llama3", "mistral"]
    mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=10)


def test_list_models_raises_on_connection_error():
    from mlflow.assistant.providers.base import ProviderNotConfiguredError

    with patch(
        "mlflow.assistant.providers.ollama.requests.get",
        side_effect=Exception("Connection refused"),
    ):
        with pytest.raises(ProviderNotConfiguredError, match="Connection refused"):
            OllamaProvider().list_models("http://localhost:11434")


# ---------------------------------------------------------------------------
# _trim_session
# ---------------------------------------------------------------------------


def test_trim_session_avoids_reserializing_full_history(monkeypatch):
    message_content = "x" * (_MAX_SESSION_BYTES // 3)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": f"old-{message_content}"},
        {"role": "assistant", "content": f"middle-{message_content}"},
        {"role": "user", "content": f"new-{message_content}"},
    ]

    original_dumps = json.dumps
    list_serialization_count = 0

    def tracked_dumps(value, *args, **kwargs):
        nonlocal list_serialization_count
        if isinstance(value, list):
            list_serialization_count += 1
            if list_serialization_count > 1:
                raise AssertionError("_trim_session reserialized the full message list")
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr("mlflow.assistant.providers.ollama.json.dumps", tracked_dumps)

    trimmed = _trim_session(messages)

    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"].startswith("new-")
    assert all(not m["content"].startswith("old-") for m in trimmed[1:])


# ---------------------------------------------------------------------------
# astream — text streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_astream_streams_text_chunks():
    lines = [
        _line({"message": {"role": "assistant", "content": "Hello"}, "done": False}),
        _line({"message": {"role": "assistant", "content": " world"}, "done": False}),
        _line({"message": {"role": "assistant", "content": ""}, "done": True}),
    ]
    session = _make_aiohttp_session([lines])

    with patch("mlflow.assistant.providers.ollama.aiohttp.ClientSession", return_value=session):
        events = [e async for e in OllamaProvider().astream("test prompt", "http://localhost:5000")]

    stream_events = [e for e in events if e.type == EventType.STREAM_EVENT]
    assert len(stream_events) == 2
    assert stream_events[0].data["event"]["delta"]["text"] == "Hello"
    assert stream_events[1].data["event"]["delta"]["text"] == " world"
    assert any(e.type == EventType.DONE for e in events)


@pytest.mark.asyncio
async def test_astream_uses_base_url_from_config(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"providers": {"ollama": {"model": "llama3.2", "base_url": "http://myhost:11434"}}}'
    )
    lines = [_line({"message": {"role": "assistant", "content": "hi"}, "done": True})]

    # Capture the URL via a wrapper
    real_session = _make_aiohttp_session([lines])
    original_post = real_session.post

    captured = {}

    def capturing_post(url, **kwargs):
        captured["url"] = url
        return original_post(url, **kwargs)

    real_session.post = capturing_post

    clear_config_cache()
    with (
        patch("mlflow.assistant.config.CONFIG_PATH", config_file),
        patch("mlflow.assistant.providers.ollama.aiohttp.ClientSession", return_value=real_session),
    ):
        _ = [e async for e in OllamaProvider().astream("prompt", "http://localhost:5000")]

    assert captured["url"] == "http://myhost:11434/api/chat"
    clear_config_cache()


@pytest.mark.asyncio
async def test_astream_yields_error_on_http_error():
    session = _make_aiohttp_session([[]], status=500)
    session.__aenter__.return_value.post = AsyncMock(
        return_value=MagicMock(
            status=500,
            text=AsyncMock(return_value="Internal Server Error"),
            __aenter__=AsyncMock(
                return_value=MagicMock(
                    status=500,
                    text=AsyncMock(return_value="Internal Server Error"),
                    content=_AsyncLineIter([]),
                )
            ),
            __aexit__=AsyncMock(return_value=False),
        )
    )

    with patch(
        "mlflow.assistant.providers.ollama.aiohttp.ClientSession",
        side_effect=Exception("Connection refused"),
    ):
        events = [e async for e in OllamaProvider().astream("test", "http://localhost:5000")]

    assert len(events) == 1
    assert events[0].type == EventType.ERROR
    assert "Connection refused" in events[0].data["error"]


# ---------------------------------------------------------------------------
# astream — tool call round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_astream_tool_call_round_trip():
    tool_call = {"function": {"name": "Bash", "arguments": {"command": "ls"}}}
    tool_msg = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
    lines_turn1 = [_line({"message": tool_msg, "done": True})]
    lines_turn2 = [
        _line({"message": {"role": "assistant", "content": "Done"}, "done": False}),
        _line({"message": {"role": "assistant", "content": ""}, "done": True}),
    ]
    session = _make_aiohttp_session([lines_turn1, lines_turn2])

    with (
        patch("mlflow.assistant.providers.ollama.aiohttp.ClientSession", return_value=session),
        patch(
            "mlflow.assistant.providers.ollama.execute_tool",
            AsyncMock(return_value=("file1.py\n", False)),
        ),
    ):
        events = [e async for e in OllamaProvider().astream("list files", "http://localhost:5000")]

    tool_use_events = [
        e
        for e in events
        if e.type == EventType.MESSAGE
        and isinstance(e.data["message"]["content"], list)
        and e.data["message"]["content"][0].get("name") == "Bash"
    ]
    assert len(tool_use_events) == 1

    stream_events = [e for e in events if e.type == EventType.STREAM_EVENT]
    assert any(ev.data["event"]["delta"]["text"] == "Done" for ev in stream_events)
