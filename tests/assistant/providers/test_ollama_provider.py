from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlflow.assistant.providers.base import clear_config_cache
from mlflow.assistant.providers.ollama import OllamaProvider
from mlflow.assistant.types import EventType


class AsyncIterator:
    def __init__(self, items):
        self.items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration


def _make_chunk(content="", tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    chunk = MagicMock()
    chunk.message = msg
    return chunk


@pytest.fixture(autouse=True)
def config(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"providers": {"ollama": {"model": "llama3.2"}}}')
    clear_config_cache()
    with patch("mlflow.assistant.config.CONFIG_PATH", config_file):
        yield config_file
    clear_config_cache()


def test_is_available_when_ollama_installed():
    with patch.dict("sys.modules", {"ollama": MagicMock()}):
        provider = OllamaProvider()
        assert provider.is_available() is True


def test_is_available_when_ollama_not_installed():
    with patch.dict("sys.modules", {"ollama": None}):
        provider = OllamaProvider()
        assert provider.is_available() is False


def test_provider_name():
    provider = OllamaProvider()
    assert provider.name == "ollama"


def test_provider_display_name():
    provider = OllamaProvider()
    assert provider.display_name == "Ollama"


@pytest.mark.asyncio
async def test_astream_yields_error_when_ollama_not_installed():
    with patch.dict("sys.modules", {"ollama": None}):
        provider = OllamaProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    assert len(events) == 1
    assert events[0].type == EventType.ERROR
    assert "ollama" in events[0].data["error"].lower()


@pytest.mark.asyncio
async def test_astream_streams_text_chunks():
    mock_ollama = MagicMock()
    chunks = [
        _make_chunk(content="Hello"),
        _make_chunk(content=" world"),
        _make_chunk(content=""),
    ]

    mock_client = MagicMock()
    mock_client.chat = AsyncMock(return_value=AsyncIterator(chunks))
    mock_ollama.AsyncClient.return_value = mock_client

    with patch.dict("sys.modules", {"ollama": mock_ollama}):
        provider = OllamaProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    stream_events = [e for e in events if e.type == EventType.STREAM_EVENT]
    assert len(stream_events) == 2
    assert stream_events[0].data["event"]["delta"]["text"] == "Hello"
    assert stream_events[1].data["event"]["delta"]["text"] == " world"

    done_events = [e for e in events if e.type == EventType.DONE]
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_astream_yields_error_on_exception():
    mock_ollama = MagicMock()
    mock_client = MagicMock()
    mock_client.chat = AsyncMock(side_effect=Exception("Connection refused"))
    mock_ollama.AsyncClient.return_value = mock_client

    with patch.dict("sys.modules", {"ollama": mock_ollama}):
        provider = OllamaProvider()
        events = [e async for e in provider.astream("test prompt", "http://localhost:5000")]

    assert len(events) == 1
    assert events[0].type == EventType.ERROR
    assert "Connection refused" in events[0].data["error"]


@pytest.mark.asyncio
async def test_astream_uses_base_url_from_config(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"providers": {"ollama": {"model": "llama3.2", "base_url": "http://myhost:11434"}}}'
    )

    mock_ollama = MagicMock()
    chunks = [_make_chunk(content="hi")]
    mock_client = MagicMock()
    mock_client.chat = AsyncMock(return_value=AsyncIterator(chunks))
    mock_ollama.AsyncClient.return_value = mock_client

    clear_config_cache()
    with (
        patch("mlflow.assistant.config.CONFIG_PATH", config_file),
        patch.dict("sys.modules", {"ollama": mock_ollama}),
    ):
        provider = OllamaProvider()
        _ = [e async for e in provider.astream("prompt", "http://localhost:5000")]

    mock_ollama.AsyncClient.assert_called_once_with(host="http://myhost:11434")


@pytest.mark.asyncio
async def test_astream_tool_call_round_trip():
    mock_ollama = MagicMock()

    tc = MagicMock()
    tc.function.name = "Bash"
    tc.function.arguments = {"command": "ls"}
    tc.model_dump.return_value = {"function": {"name": "Bash", "arguments": {"command": "ls"}}}

    chunks_turn1 = [
        _make_chunk(content="", tool_calls=[tc]),
    ]
    chunks_turn2 = [
        _make_chunk(content="Done"),
    ]

    call_count = 0

    async def fake_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AsyncIterator(chunks_turn1)
        return AsyncIterator(chunks_turn2)

    mock_client = MagicMock()
    mock_client.chat = fake_chat
    mock_ollama.AsyncClient.return_value = mock_client

    with (
        patch.dict("sys.modules", {"ollama": mock_ollama}),
        patch(
            "mlflow.assistant.providers.ollama.execute_tool",
            AsyncMock(return_value=("file1.py\n", False)),
        ),
    ):
        provider = OllamaProvider()
        events = [e async for e in provider.astream("list files", "http://localhost:5000")]

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
