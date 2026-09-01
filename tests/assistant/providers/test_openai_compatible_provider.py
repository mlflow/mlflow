import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlflow.assistant.config import PermissionsConfig
from mlflow.assistant.providers.base import clear_config_cache
from mlflow.assistant.providers.ollama import OllamaProvider
from mlflow.assistant.providers.openai_compatible import (
    _MAX_SESSION_BYTES,
    OpenAICompatibleProvider,
    _build_usage_event,
    _merge_tool_call_chunk,
    _strip_think_blocks,
    _trim_session,
)
from mlflow.assistant.providers.tool_executor import (
    RENDER_CUSTOM_VIEW_TOOL_NAME,
    static_permission_error,
)
from mlflow.assistant.types import EventType
from mlflow.tracing.constant import CostKey, TokenUsageKey

# ---------------------------------------------------------------------------
# aiohttp mock helpers
# ---------------------------------------------------------------------------


class _AsyncLineIter:
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
    responses = []
    captured_calls: list[dict[str, Any]] = []
    for lines in response_lines_per_call:
        resp = MagicMock()
        resp.status = status
        resp.content = _AsyncLineIter(lines)
        resp.text = AsyncMock(return_value="")
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        responses.append(resp)

    call_count = 0

    def _post(url, **kwargs):
        nonlocal call_count
        captured_calls.append({"url": url, **kwargs})
        r = responses[call_count]
        call_count += 1
        return r

    session = MagicMock()
    session.post = _post
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session, captured_calls


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n".encode()


def _delta(
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    role: str = "assistant",
):
    delta: dict[str, Any] = {"role": role}
    if content:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"choices": [{"delta": delta, "index": 0}]}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _list_models_stub(*_args, **_kwargs):
    return ["model-a"]


@pytest.fixture
def provider():
    return OpenAICompatibleProvider(
        name="oai_test",
        display_name="OAI Test",
        description="Test provider",
        list_models_fn=_list_models_stub,
        connection_hint="hint",
        default_base_url="http://localhost:9999",
    )


@pytest.fixture(autouse=True)
def config_file(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({
            "providers": {
                "oai_test": {"model": "model-a"},
                "ollama": {"model": "llama3.2"},
            }
        })
    )
    clear_config_cache()
    with patch("mlflow.assistant.config.CONFIG_PATH", cfg):
        yield cfg
    clear_config_cache()


def test_uses_native_client_tool_delivery(provider):
    # Schema-based providers pause on a CLIENT_TOOLS call and resume on the next
    # stream once a result is posted (see the pause/resume tests below).
    assert provider.client_tool_delivery == "tool"


def test_ollama_uses_native_client_tool_delivery():
    provider = OllamaProvider()
    assert provider.client_tool_delivery == "tool"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("buf", "in_think", "expected_emit", "expected_remaining", "expected_in_think"),
    [
        ("hello world", False, "hello world", "", False),
        ("foo<think>secret</think>bar", False, "foobar", "", False),
        ("<think>partial", False, "", "", True),
        ("rest of thought</think>after", True, "after", "", False),
        ("plain", True, "", "", True),
        # Partial-tag-at-tail: a chunk ending with a prefix of "<think>"
        # must not leak that prefix to the user — it must be held back as
        # the remainder so the next chunk can complete the tag.
        ("foo<th", False, "foo", "<th", False),
        ("foo<", False, "foo", "<", False),
        # Same for the closing tag while inside a think span.
        ("secret</th", True, "", "</th", True),
        ("secret<", True, "", "<", True),
        # Plain "<" at the end with no following partial isn't a hold case
        # outside a think span — but the prefix-match logic still treats
        # it as a potential opening "<think>" start. That's the safe
        # default: hold one char, emit it next round if it doesn't grow.
    ],
)
def test_strip_think_blocks(buf, in_think, expected_emit, expected_remaining, expected_in_think):
    emit, remaining, new_in_think = _strip_think_blocks(buf, in_think)
    assert emit == expected_emit
    assert remaining == expected_remaining
    assert new_in_think is expected_in_think


def test_strip_think_blocks_completes_partial_tag_across_chunks():
    """Reproduces the SSE-frame split that previously leaked `<think>` to
    the user. Frame 1 ends mid-opening-tag; frame 2 supplies the rest of
    the tag plus the secret content and the closing tag. The combined
    behavior must emit nothing user-visible (only "foo").
    """
    emit1, remaining1, in_think1 = _strip_think_blocks("foo<th", False)
    assert emit1 == "foo"
    assert remaining1 == "<th"
    assert in_think1 is False

    emit2, remaining2, in_think2 = _strip_think_blocks(remaining1 + "ink>secret</think>", in_think1)
    assert emit2 == ""
    assert remaining2 == ""
    assert in_think2 is False


def test_merge_tool_call_chunk_accumulates_arguments():
    acc: list[dict[str, Any]] = []
    _merge_tool_call_chunk(
        acc,
        {"index": 0, "id": "call_1", "function": {"name": "Bash", "arguments": '{"comm'}},
    )
    _merge_tool_call_chunk(acc, {"index": 0, "function": {"arguments": 'and": "ls"}'}})
    assert acc == [{"id": "call_1", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}]


def test_merge_tool_call_chunk_supports_multiple_calls():
    acc: list[dict[str, Any]] = []
    _merge_tool_call_chunk(
        acc, {"index": 0, "id": "a", "function": {"name": "X", "arguments": "{}"}}
    )
    _merge_tool_call_chunk(
        acc, {"index": 1, "id": "b", "function": {"name": "Y", "arguments": "{}"}}
    )
    assert len(acc) == 2
    assert acc[0]["id"] == "a"
    assert acc[1]["id"] == "b"


def test_trim_session_drops_oldest_keeping_system():
    big = "x" * (_MAX_SESSION_BYTES // 3)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": f"old-{big}"},
        {"role": "assistant", "content": f"middle-{big}"},
        {"role": "user", "content": f"new-{big}"},
    ]
    trimmed = _trim_session(messages)
    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"].startswith("new-")
    assert not any(m["content"].startswith("old-") for m in trimmed[1:])


def test_build_usage_event_remaps_cache_tokens_and_prices():
    usage = {
        "prompt_tokens": 35257,
        "completion_tokens": 5,
        "total_tokens": 35262,
        "prompt_tokens_details": {"cached_tokens": 100},
        "cache_creation_input_tokens": 35155,
    }
    with patch(
        "mlflow.assistant.providers.openai_compatible.calculate_cost_by_model_and_token_usage",
        return_value={CostKey.TOTAL_COST: 0.1319},
    ) as mock_cost:
        event = _build_usage_event(usage, "claude-3-5-sonnet")

    mock_cost.assert_called_once_with(
        "claude-3-5-sonnet",
        {
            TokenUsageKey.INPUT_TOKENS: 35257,
            TokenUsageKey.OUTPUT_TOKENS: 5,
            TokenUsageKey.CACHE_READ_INPUT_TOKENS: 100,
            TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: 35155,
        },
    )
    assert event.type == EventType.STREAM_EVENT
    assert event.data["event"]["usage"] == {
        "prompt_tokens": 35257,
        "completion_tokens": 5,
        "total_tokens": 35262,
        "cache_read_tokens": 100,
        "total_cost_usd": 0.1319,
    }


def test_build_usage_event_cost_none_when_model_not_priced():
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    with patch(
        "mlflow.assistant.providers.openai_compatible.calculate_cost_by_model_and_token_usage",
        return_value=None,
    ) as mock_cost:
        event = _build_usage_event(usage, "local-ollama-model")

    mock_cost.assert_called_once_with(
        "local-ollama-model",
        {TokenUsageKey.INPUT_TOKENS: 10, TokenUsageKey.OUTPUT_TOKENS: 5},
    )
    assert event.data["event"]["usage"]["total_cost_usd"] is None


# ---------------------------------------------------------------------------
# astream — basic streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_astream_emits_content_deltas(provider):
    lines = [
        _sse(_delta(content="Hello")),
        _sse(_delta(content=" world")),
        b"data: [DONE]\n",
    ]
    session, calls = _make_aiohttp_session([lines])

    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=session,
    ):
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]

    stream_events = [e for e in events if e.type == EventType.STREAM_EVENT]
    assert [e.data["event"]["delta"]["text"] for e in stream_events] == ["Hello", " world"]
    assert any(e.type == EventType.DONE for e in events)
    assert calls[0]["url"] == "http://localhost:9999/v1/chat/completions"
    assert calls[0]["headers"] == {}


@pytest.mark.asyncio
async def test_astream_omits_stream_options(provider):
    # stream_options is an OpenAI-only field. The assistant must not send it: a
    # gateway route backed by Anthropic forwards it to /v1/messages, which 400s on
    # the unknown field. The gateway self-injects it per-provider where accepted.
    lines = [_sse(_delta(content="hi")), b"data: [DONE]\n"]
    session, calls = _make_aiohttp_session([lines])

    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=session,
    ):
        _ = [e async for e in provider.astream("hi", "http://localhost:5000")]

    assert "stream_options" not in calls[0]["json"]


@pytest.mark.asyncio
async def test_astream_tolerates_done_terminator_and_blank_lines(provider):
    lines = [
        b"\n",
        _sse(_delta(content="A")),
        b":heartbeat\n",
        _sse(_delta(content="B")),
        b"data: [DONE]\n",
    ]
    session, _calls = _make_aiohttp_session([lines])
    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=session,
    ):
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]
    deltas = [e.data["event"]["delta"]["text"] for e in events if e.type == EventType.STREAM_EVENT]
    assert deltas == ["A", "B"]


@pytest.mark.asyncio
async def test_astream_strips_think_blocks_from_stream(provider):
    lines = [
        _sse(_delta(content="ans:")),
        _sse(_delta(content="<think>internal")),
        _sse(_delta(content=" reasoning</think>real")),
        _sse(_delta(content=" answer")),
        b"data: [DONE]\n",
    ]
    session, _calls = _make_aiohttp_session([lines])
    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=session,
    ):
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]
    visible = "".join(
        e.data["event"]["delta"]["text"] for e in events if e.type == EventType.STREAM_EVENT
    )
    assert "internal" not in visible
    assert "reasoning" not in visible
    assert "ans:real answer" == visible


@pytest.mark.asyncio
async def test_astream_ignores_config_stored_api_key(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({
            "providers": {
                "oai_test": {
                    "model": "model-a",
                    "base_url": "http://gateway.example",
                    "api_key": "sk-abc",
                }
            }
        })
    )
    clear_config_cache()
    provider = OpenAICompatibleProvider(
        name="oai_test",
        display_name="OAI",
        description="d",
        list_models_fn=_list_models_stub,
        connection_hint="h",
    )
    lines = [_sse(_delta(content="ok")), b"data: [DONE]\n"]
    session, calls = _make_aiohttp_session([lines])
    with (
        patch("mlflow.assistant.config.CONFIG_PATH", cfg),
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=session,
        ),
    ):
        _ = [e async for e in provider.astream("hi", "http://localhost:5000")]
    assert calls[0]["url"] == "http://gateway.example/v1/chat/completions"
    assert calls[0]["headers"] == {}
    clear_config_cache()


@pytest.mark.asyncio
async def test_astream_uses_first_model_when_unconfigured(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"providers": {}}))
    clear_config_cache()
    provider = OpenAICompatibleProvider(
        name="oai_test",
        display_name="OAI",
        description="d",
        list_models_fn=_list_models_stub,
        connection_hint="h",
        default_base_url="http://localhost:9999",
    )
    lines = [_sse(_delta(content="ok")), b"data: [DONE]\n"]
    session, calls = _make_aiohttp_session([lines])
    with (
        patch("mlflow.assistant.config.CONFIG_PATH", cfg),
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=session,
        ),
    ):
        _ = [e async for e in provider.astream("hi", "http://localhost:5000")]

    assert calls[0]["url"] == "http://localhost:9999/v1/chat/completions"
    assert calls[0]["json"]["model"] == "model-a"
    clear_config_cache()


@pytest.mark.asyncio
async def test_astream_uses_tracking_uri_via_custom_chat_url_builder(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"providers": {"gw_test": {"model": "ep-1"}}}))
    clear_config_cache()

    def chat_url_builder(_base_url, tracking_uri):
        return f"{tracking_uri.rstrip('/')}/gateway/mlflow/v1/chat/completions"

    provider = OpenAICompatibleProvider(
        name="gw_test",
        display_name="Gateway",
        description="d",
        connection_hint="h",
        chat_url_builder=chat_url_builder,
    )
    lines = [_sse(_delta(content="ok")), b"data: [DONE]\n"]
    session, calls = _make_aiohttp_session([lines])
    with (
        patch("mlflow.assistant.config.CONFIG_PATH", cfg),
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=session,
        ),
    ):
        _ = [e async for e in provider.astream("hi", "http://mlflow.server:5000")]
    assert calls[0]["url"] == "http://mlflow.server:5000/gateway/mlflow/v1/chat/completions"
    clear_config_cache()


def test_list_models_raises_not_implemented_when_no_fn():
    provider = OpenAICompatibleProvider(
        name="gw_test2",
        display_name="Gateway",
        description="d",
        connection_hint="h",
    )
    with pytest.raises(NotImplementedError, match="Model listing is not supported"):
        provider.list_models()


def test_list_models_passes_supplied_api_key():
    captured = {}

    def list_models(base_url: str, api_key: str | None = None) -> list[str]:
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return ["model-a"]

    provider = OpenAICompatibleProvider(
        name="oai_test",
        display_name="OAI",
        description="d",
        list_models_fn=list_models,
        connection_hint="h",
        default_base_url="http://localhost:9999",
    )

    assert provider.list_models(api_key="sk-test") == ["model-a"]
    assert captured == {"base_url": "http://localhost:9999", "api_key": "sk-test"}


@pytest.mark.asyncio
async def test_astream_yields_error_on_http_error(provider):
    session, _calls = _make_aiohttp_session([[b""]], status=500)
    # Wrap the failing response so .text() returns the error body.
    bad_resp = MagicMock()
    bad_resp.status = 500
    bad_resp.text = AsyncMock(return_value="boom")
    bad_resp.content = _AsyncLineIter([])
    bad_resp.__aenter__ = AsyncMock(return_value=bad_resp)
    bad_resp.__aexit__ = AsyncMock(return_value=False)
    session.post = lambda url, **kw: bad_resp

    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=session,
    ):
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]
    errors = [e for e in events if e.type == EventType.ERROR]
    assert len(errors) == 1
    assert "boom" in errors[0].data["error"]


@pytest.mark.asyncio
async def test_astream_yields_error_on_empty_truncated_stream(provider):
    # The gateway commits a 200 before proxying upstream, so an upstream failure
    # (e.g. a bad API key) truncates the body instead of returning a non-200. When
    # the failure happens before any token streams, the body is empty and ends with
    # no terminal signal: surface an error rather than a silent `done`.
    lines: list[bytes] = []
    session, _calls = _make_aiohttp_session([lines])
    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=session,
    ):
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]
    errors = [e for e in events if e.type == EventType.ERROR]
    assert len(errors) == 1
    assert "empty response" in errors[0].data["error"]
    assert not any(e.type == EventType.DONE for e in events)


@pytest.mark.asyncio
async def test_astream_surfaces_gateway_error_chunk(provider):
    # When an upstream failure happens mid-stream (after the gateway committed a
    # 200), safe_stream emits `data: {"error": {"message", "type"}}`. We must
    # surface that real message, not discard it and fall through to the generic
    # empty-response error, which would misattribute the cause (e.g. blame a bad
    # API key when the key was valid and something else failed).
    lines = [_sse({"error": {"message": "Rate limit exceeded", "type": "RateLimitError"}})]
    session, _calls = _make_aiohttp_session([lines])
    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=session,
    ):
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]
    errors = [e for e in events if e.type == EventType.ERROR]
    assert len(errors) == 1
    assert "Rate limit exceeded" in errors[0].data["error"]
    assert "empty response" not in errors[0].data["error"]
    assert not any(e.type == EventType.DONE for e in events)


@pytest.mark.asyncio
async def test_astream_error_chunk_without_message_falls_back_to_raw(provider):
    # A malformed error frame (no usable "message") must still surface something
    # concrete rather than "error: None" or the misleading generic empty-response text.
    lines = [_sse({"error": {"code": 500}})]
    session, _calls = _make_aiohttp_session([lines])
    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=session,
    ):
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]
    errors = [e for e in events if e.type == EventType.ERROR]
    assert len(errors) == 1
    assert "None" not in errors[0].data["error"]
    assert "500" in errors[0].data["error"]
    assert not any(e.type == EventType.DONE for e in events)


@pytest.mark.asyncio
async def test_astream_productive_stream_without_terminal_is_not_flagged(provider):
    # Key false-positive guard: an OpenAI-compatible server that streams content but
    # emits neither [DONE] nor a finish_reason (out of spec, but real) must NOT be
    # flagged as truncated. Any produced output is treated as a successful turn.
    lines = [_sse(_delta(content="a complete answer"))]
    session, _calls = _make_aiohttp_session([lines])
    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=session,
    ):
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]
    assert not any(e.type == EventType.ERROR for e in events)
    assert any(e.type == EventType.DONE for e in events)
    # Also assert the content actually streamed through — otherwise a regression that
    # silently dropped the delta would still pass the no-error / done checks above.
    deltas = [e.data["event"]["delta"]["text"] for e in events if e.type == EventType.STREAM_EVENT]
    assert deltas == ["a complete answer"]


@pytest.mark.asyncio
async def test_astream_empty_stream_with_terminal_is_not_flagged(provider):
    # An intentionally empty completion that still signals a normal terminal
    # ([DONE] or finish_reason) is a valid turn, not a truncation.
    lines = [_sse({"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]})]
    session, _calls = _make_aiohttp_session([lines])
    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=session,
    ):
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]
    assert not any(e.type == EventType.ERROR for e in events)
    assert any(e.type == EventType.DONE for e in events)


@pytest.mark.asyncio
async def test_astream_usage_only_stream_is_not_flagged(provider):
    # Some backends emit a trailing usage-summary chunk (choices empty) after a
    # [DONE] the gateway strips. A usage report means the server did real work,
    # so it counts as a signal and must not be flagged as truncated.
    lines = [_sse({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 5}})]
    session, _calls = _make_aiohttp_session([lines])
    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=session,
    ):
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]
    assert not any(e.type == EventType.ERROR for e in events)
    assert any(e.type == EventType.DONE for e in events)


@pytest.mark.asyncio
async def test_astream_role_only_stream_is_flagged(provider):
    # A stream whose ONLY chunk is a role preamble (no content, tool_calls,
    # finish_reason, or [DONE]) and then closes is a truncation, not a completion:
    # the server opened a message and dropped the connection before finishing it.
    # This is intentionally flagged; a role delta alone is not treated as signal.
    lines = [_sse({"choices": [{"delta": {"role": "assistant"}, "index": 0}]})]
    session, _calls = _make_aiohttp_session([lines])
    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=session,
    ):
        events = [e async for e in provider.astream("hi", "http://localhost:5000")]
    errors = [e for e in events if e.type == EventType.ERROR]
    assert len(errors) == 1
    assert "empty response" in errors[0].data["error"]
    assert not any(e.type == EventType.DONE for e in events)


# ---------------------------------------------------------------------------
# astream — tool call round trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_astream_tool_call_round_trip(provider):
    # Turn 1: streamed tool call with chunked arguments.
    lines_turn1 = [
        _sse(
            _delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "Bash", "arguments": '{"comm'},
                    }
                ]
            )
        ),
        _sse(_delta(tool_calls=[{"index": 0, "function": {"arguments": 'and": "ls"}'}}])),
        b"data: [DONE]\n",
    ]
    lines_turn2 = [_sse(_delta(content="Done")), b"data: [DONE]\n"]
    session, calls = _make_aiohttp_session([lines_turn1, lines_turn2])

    with (
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=session,
        ),
        patch(
            "mlflow.assistant.providers.openai_compatible.execute_tool",
            AsyncMock(return_value=("file1.py\n", False)),
        ) as mock_tool,
    ):
        events = [e async for e in provider.astream("ls", "http://localhost:5000")]

    mock_tool.assert_awaited_once()
    args, kwargs = mock_tool.await_args
    assert args[0] == "Bash"
    assert args[1] == {"command": "ls"}

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
    # Second request should include the tool message in history.
    second_payload = calls[1]["json"]
    assert any(m["role"] == "tool" for m in second_payload["messages"])


# ---------------------------------------------------------------------------
# astream — session-scoped permission gating
# ---------------------------------------------------------------------------

_SESSION_ID = "11111111-1111-1111-1111-111111111111"


def _tool_call_turns():
    turn1 = [
        _sse(
            _delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "Bash", "arguments": '{"command": "ls"}'},
                    }
                ]
            )
        ),
        b"data: [DONE]\n",
    ]
    turn2 = [_sse(_delta(content="Done")), b"data: [DONE]\n"]
    return [turn1, turn2]


def _client_tool_call_turns():
    turn1 = [
        _sse(
            _delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {
                            "name": RENDER_CUSTOM_VIEW_TOOL_NAME,
                            "arguments": '{"title": "Trace Summary", "messages": []}',
                        },
                    }
                ]
            )
        ),
        b"data: [DONE]\n",
    ]
    turn2 = [_sse(_delta(content="Done")), b"data: [DONE]\n"]
    return [turn1, turn2]


def _done_session_id(events) -> str:
    for e in reversed(events):
        if e.type == EventType.DONE:
            return e.data["session_id"]
    raise AssertionError("no DONE event found")


@pytest.mark.asyncio
async def test_astream_pauses_at_permission_without_executing(provider):
    # Full access off + a session: the turn must END at the prompt (no in-process
    # await), emitting PERMISSION_REQUEST then DONE, with the tool unexecuted and
    # the pending tool_call persisted in the returned history.
    session, _calls = _make_aiohttp_session([_tool_call_turns()[0]])
    with (
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=session,
        ),
        patch(
            "mlflow.assistant.providers.openai_compatible.execute_tool",
            AsyncMock(return_value=("file1.py\n", False)),
        ) as mock_tool,
    ):
        events = [
            e
            async for e in provider.astream(
                "ls", "http://localhost:5000", mlflow_session_id=_SESSION_ID
            )
        ]

    mock_tool.assert_not_awaited()
    prompts = [e for e in events if e.type == EventType.PERMISSION_REQUEST]
    assert len(prompts) == 1
    assert prompts[0].data["request_id"] == "call_1"
    assert prompts[0].data["tool_name"] == "Bash"
    assert prompts[0].data["tool_input"] == {"command": "ls"}
    assert events[-1].type == EventType.DONE

    history = json.loads(_done_session_id(events))
    assert history[-1]["role"] == "assistant"
    assert history[-1].get("tool_calls")
    assert not any(m.get("role") == "tool" for m in history)


@pytest.mark.asyncio
async def test_astream_resume_allow_executes_and_continues(provider):
    # Pause to capture the persisted history.
    s1, _ = _make_aiohttp_session([_tool_call_turns()[0]])
    with (
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=s1,
        ),
        patch(
            "mlflow.assistant.providers.openai_compatible.execute_tool",
            AsyncMock(return_value=("x", False)),
        ) as mt1,
    ):
        ev1 = [
            e
            async for e in provider.astream(
                "ls", "http://localhost:5000", mlflow_session_id=_SESSION_ID
            )
        ]
    mt1.assert_not_awaited()
    history = _done_session_id(ev1)

    # Resume with allow: the decision is delivered via context, no new user turn.
    s2, _ = _make_aiohttp_session([_tool_call_turns()[1]])
    with (
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=s2,
        ),
        patch(
            "mlflow.assistant.providers.openai_compatible.execute_tool",
            AsyncMock(return_value=("file1.py\n", False)),
        ) as mt2,
    ):
        ev2 = [
            e
            async for e in provider.astream(
                "",
                "http://localhost:5000",
                mlflow_session_id=_SESSION_ID,
                session_id=history,
                context={"tool_decisions": {"call_1": "allow"}},
            )
        ]

    mt2.assert_awaited_once()
    # An explicit allow overrides the static allowlist for this call.
    assert mt2.await_args.kwargs["permissions"].full_access is True
    assert not any(e.type == EventType.PERMISSION_REQUEST for e in ev2)
    assert any(
        e.type == EventType.STREAM_EVENT and e.data["event"]["delta"]["text"] == "Done" for e in ev2
    )


def _two_read_calls_turn():
    return [
        _sse(
            _delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {
                            "name": "Read",
                            "arguments": '{"file_path": "/etc/passwd"}',
                        },
                    },
                    {
                        "index": 1,
                        "id": "call_2",
                        "function": {
                            "name": "Read",
                            "arguments": '{"file_path": "/etc/shadow"}',
                        },
                    },
                ]
            )
        ),
        b"data: [DONE]\n",
    ]


@pytest.mark.asyncio
async def test_astream_resume_allow_does_not_leak_to_other_calls(provider):
    # Regression guard for GHSA-27c7-qx3r-x4f8 review discussion: an explicit
    # Allow for one cwd=None-denied call must not implicitly authorize a
    # different call still awaiting its own decision, even in the same
    # resumed turn. Uses Read (denied because cwd is None) rather than an
    # unrelated Bash allowlist miss, to exercise the exact class of call this
    # advisory covers.
    s1, _ = _make_aiohttp_session([_two_read_calls_turn()])
    with (
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=s1,
        ),
        patch(
            "mlflow.assistant.providers.openai_compatible.execute_tool",
            AsyncMock(return_value=("x", False)),
        ) as mt1,
    ):
        ev1 = [
            e
            async for e in provider.astream(
                "read some files", "http://localhost:5000", mlflow_session_id=_SESSION_ID
            )
        ]
    mt1.assert_not_awaited()
    prompts1 = [e for e in ev1 if e.type == EventType.PERMISSION_REQUEST]
    assert len(prompts1) == 1
    assert prompts1[0].data["request_id"] == "call_1"
    history = _done_session_id(ev1)

    # Resume with Allow for call_1 only. The pending tool_calls are already in
    # history, so no new model call happens; execute_tool is invoked directly.
    s2, _ = _make_aiohttp_session([])
    with (
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=s2,
        ),
        patch(
            "mlflow.assistant.providers.openai_compatible.execute_tool",
            AsyncMock(return_value=("passwd contents", False)),
        ) as mt2,
    ):
        ev2 = [
            e
            async for e in provider.astream(
                "",
                "http://localhost:5000",
                mlflow_session_id=_SESSION_ID,
                session_id=history,
                context={"tool_decisions": {"call_1": "allow"}},
            )
        ]

    mt2.assert_awaited_once()
    assert mt2.await_args.kwargs["permissions"].full_access is True

    # call_2 gets its own fresh prompt: the Allow for call_1 did not leak.
    prompts2 = [e for e in ev2 if e.type == EventType.PERMISSION_REQUEST]
    assert len(prompts2) == 1
    assert prompts2[0].data["request_id"] == "call_2"


@pytest.mark.asyncio
async def test_astream_resume_deny_skips_execution(provider):
    s1, _ = _make_aiohttp_session([_tool_call_turns()[0]])
    with (
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=s1,
        ),
        patch(
            "mlflow.assistant.providers.openai_compatible.execute_tool",
            AsyncMock(return_value=("x", False)),
        ),
    ):
        ev1 = [
            e
            async for e in provider.astream(
                "ls", "http://localhost:5000", mlflow_session_id=_SESSION_ID
            )
        ]
    history = _done_session_id(ev1)

    s2, _ = _make_aiohttp_session([_tool_call_turns()[1]])
    with (
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=s2,
        ),
        patch(
            "mlflow.assistant.providers.openai_compatible.execute_tool",
            AsyncMock(return_value=("file1.py\n", False)),
        ) as mt2,
    ):
        ev2 = [
            e
            async for e in provider.astream(
                "",
                "http://localhost:5000",
                mlflow_session_id=_SESSION_ID,
                session_id=history,
                context={"tool_decisions": {"call_1": "deny"}},
            )
        ]

    mt2.assert_not_awaited()
    denied = [
        e
        for e in ev2
        if e.type == EventType.MESSAGE
        and isinstance(e.data["message"]["content"], list)
        and e.data["message"]["content"][0].get("content") == "Permission denied by user."
    ]
    assert len(denied) == 1
    assert any(
        e.type == EventType.STREAM_EVENT and e.data["event"]["delta"]["text"] == "Done" for e in ev2
    )


@pytest.mark.asyncio
async def test_astream_fresh_message_after_abandoned_tool_call(provider):
    # A turn paused at a prompt, then cancelled (a no-op for this provider, so the
    # unresolved tool_call stays in history). A NEW user message must start a fresh
    # turn — NOT silently re-resume the abandoned call and drop the message.
    s1, _ = _make_aiohttp_session([_tool_call_turns()[0]])
    with (
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=s1,
        ),
        patch(
            "mlflow.assistant.providers.openai_compatible.execute_tool",
            AsyncMock(return_value=("x", False)),
        ),
    ):
        ev1 = [
            e
            async for e in provider.astream(
                "ls", "http://localhost:5000", mlflow_session_id=_SESSION_ID
            )
        ]
    history = _done_session_id(ev1)

    # New message, NO tool_decisions: the abandoned call must be closed out and the
    # new message must reach the model (turn 2 returns plain text, no tool calls).
    s2, _ = _make_aiohttp_session([_tool_call_turns()[1]])
    with (
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=s2,
        ),
        patch(
            "mlflow.assistant.providers.openai_compatible.execute_tool",
            AsyncMock(return_value=("file1.py\n", False)),
        ) as mt2,
    ):
        ev2 = [
            e
            async for e in provider.astream(
                "what is 2+2",
                "http://localhost:5000",
                mlflow_session_id=_SESSION_ID,
                session_id=history,
            )
        ]

    # No duplicate prompt for the old call, and the old call is never executed.
    assert not any(e.type == EventType.PERMISSION_REQUEST for e in ev2)
    mt2.assert_not_awaited()
    # The stream completes with the model's reply to the NEW message.
    assert any(
        e.type == EventType.STREAM_EVENT and e.data["event"]["delta"]["text"] == "Done" for e in ev2
    )
    # History: the orphaned call is closed with a cancellation result, and the new
    # user message is present.
    final = json.loads(_done_session_id(ev2))
    assert any(
        m.get("role") == "tool"
        and m.get("tool_call_id") == "call_1"
        and m.get("content") == "Tool call cancelled by user."
        for m in final
    )
    assert any(m.get("role") == "user" and m.get("content") == "what is 2+2" for m in final)


@pytest.mark.asyncio
async def test_astream_pauses_at_client_tool_call_without_prompting(provider):
    # A CLIENT_TOOLS call (e.g. render_custom_view) never goes through execute_tool or the
    # static permission gate — it always pauses for the client to execute, even though this
    # provider is NOT full-access and has no static allowlist entry for the tool name.
    session, _calls = _make_aiohttp_session([_client_tool_call_turns()[0]])
    with (
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=session,
        ),
        patch(
            "mlflow.assistant.providers.openai_compatible.execute_tool",
            AsyncMock(return_value=("should not run", False)),
        ) as mock_tool,
    ):
        events = [
            e
            async for e in provider.astream(
                "build me a view", "http://localhost:5000", mlflow_session_id=_SESSION_ID
            )
        ]

    mock_tool.assert_not_awaited()
    assert not any(e.type == EventType.PERMISSION_REQUEST for e in events)
    client_calls = [e for e in events if e.type == EventType.CLIENT_TOOL_CALL]
    assert len(client_calls) == 1
    assert client_calls[0].data["request_id"] == "call_1"
    assert client_calls[0].data["tool_name"] == RENDER_CUSTOM_VIEW_TOOL_NAME
    assert client_calls[0].data["tool_input"] == {"title": "Trace Summary", "messages": []}
    # The tool-use block is surfaced before the pause, same as a permission prompt.
    tool_use_messages = [
        e
        for e in events
        if e.type == EventType.MESSAGE
        and isinstance(e.data["message"]["content"], list)
        and e.data["message"]["content"][0].get("name") == RENDER_CUSTOM_VIEW_TOOL_NAME
    ]
    assert len(tool_use_messages) == 1
    assert events[-1].type == EventType.DONE

    history = json.loads(_done_session_id(events))
    assert history[-1]["role"] == "assistant"
    assert history[-1].get("tool_calls")
    assert not any(m.get("role") == "tool" for m in history)


@pytest.mark.asyncio
async def test_astream_resume_with_client_tool_result_continues(provider):
    # Pause to capture the persisted history.
    s1, _ = _make_aiohttp_session([_client_tool_call_turns()[0]])
    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=s1,
    ):
        ev1 = [
            e
            async for e in provider.astream(
                "build me a view", "http://localhost:5000", mlflow_session_id=_SESSION_ID
            )
        ]
    history = _done_session_id(ev1)

    # Resume with the client-reported result: the tool result is spliced in and the
    # loop continues to a normal model turn — no re-prompting, no server execution.
    s2, _ = _make_aiohttp_session([_client_tool_call_turns()[1]])
    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=s2,
    ):
        ev2 = [
            e
            async for e in provider.astream(
                "",
                "http://localhost:5000",
                mlflow_session_id=_SESSION_ID,
                session_id=history,
                context={
                    "client_tool_results": {
                        "call_1": {"content": "Applied successfully.", "is_error": False}
                    }
                },
            )
        ]

    assert not any(
        e.type in (EventType.PERMISSION_REQUEST, EventType.CLIENT_TOOL_CALL) for e in ev2
    )
    tool_results = [
        e
        for e in ev2
        if e.type == EventType.MESSAGE
        and isinstance(e.data["message"]["content"], list)
        and e.data["message"]["content"][0].get("content") == "Applied successfully."
    ]
    assert len(tool_results) == 1
    assert any(
        e.type == EventType.STREAM_EVENT and e.data["event"]["delta"]["text"] == "Done" for e in ev2
    )

    final = json.loads(_done_session_id(ev2))
    assert any(
        m.get("role") == "tool"
        and m.get("tool_call_id") == "call_1"
        and m.get("content") == "Applied successfully."
        for m in final
    )


@pytest.mark.asyncio
async def test_astream_resume_with_client_tool_error_result_continues(provider):
    s1, _ = _make_aiohttp_session([_client_tool_call_turns()[0]])
    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=s1,
    ):
        ev1 = [
            e
            async for e in provider.astream(
                "build me a view", "http://localhost:5000", mlflow_session_id=_SESSION_ID
            )
        ]
    history = _done_session_id(ev1)

    s2, _ = _make_aiohttp_session([_client_tool_call_turns()[1]])
    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=s2,
    ):
        ev2 = [
            e
            async for e in provider.astream(
                "",
                "http://localhost:5000",
                mlflow_session_id=_SESSION_ID,
                session_id=history,
                context={
                    "client_tool_results": {
                        "call_1": {"content": "Failed to render: invalid spec.", "is_error": True}
                    }
                },
            )
        ]

    error_results = [
        e
        for e in ev2
        if e.type == EventType.MESSAGE
        and isinstance(e.data["message"]["content"], list)
        and e.data["message"]["content"][0].get("content") == "Failed to render: invalid spec."
    ]
    assert len(error_results) == 1
    assert error_results[0].data["message"]["content"][0]["is_error"] is True
    final = json.loads(_done_session_id(ev2))
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "call_1" for m in final)


@pytest.mark.asyncio
async def test_astream_global_full_access_skips_prompt(tmp_path):
    # When full access is enabled in the global config, no per-call prompt fires
    # even for a session — this preserves the pre-existing "run freely" setting.
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({
            "providers": {"oai_test": {"model": "model-a", "permissions": {"full_access": True}}}
        })
    )
    clear_config_cache()
    provider = OpenAICompatibleProvider(
        name="oai_test",
        display_name="OAI Test",
        description="d",
        list_models_fn=_list_models_stub,
        connection_hint="h",
        default_base_url="http://localhost:9999",
    )
    session, _calls = _make_aiohttp_session(_tool_call_turns())
    with (
        patch("mlflow.assistant.config.CONFIG_PATH", cfg),
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=session,
        ),
        patch(
            "mlflow.assistant.providers.openai_compatible.execute_tool",
            AsyncMock(return_value=("file1.py\n", False)),
        ) as mock_tool,
    ):
        events = [
            e
            async for e in provider.astream(
                "ls", "http://localhost:5000", mlflow_session_id=_SESSION_ID
            )
        ]
    clear_config_cache()
    assert not any(e.type == EventType.PERMISSION_REQUEST for e in events)
    mock_tool.assert_awaited_once()


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "allowed"),
    [
        ("Bash", {"command": "mlflow experiments search"}, True),
        # Regression guard for GHSA-27c7-qx3r-x4f8: python/python3 must be denied
        # without a configured project directory (cwd=None), same as Read/Write/Edit.
        ("Bash", {"command": "python script.py"}, False),
        ("Bash", {"command": "rm -rf /"}, False),
        ("Bash", {"command": "ls"}, False),
    ],
)
def test_static_permission_error_bash_allowlist(tool_name, tool_input, allowed):
    err = static_permission_error(tool_name, tool_input, PermissionsConfig(full_access=False), None)
    assert (err is None) == allowed


def test_static_permission_error_bash_python_allowed_with_cwd(tmp_path):
    err = static_permission_error(
        "Bash", {"command": "python script.py"}, PermissionsConfig(full_access=False), tmp_path
    )
    assert err is None


def test_static_permission_error_full_access_allows_everything():
    assert (
        static_permission_error(
            "Bash", {"command": "rm -rf /"}, PermissionsConfig(full_access=True), None
        )
        is None
    )


@pytest.mark.asyncio
async def test_astream_allowlisted_command_runs_without_prompt(provider):
    # Regression guard for #24084: an `mlflow` CLI command is on the static allowlist, so even with
    # full access off and a session present it must run WITHOUT a per-call permission prompt.
    turn1 = [
        _sse(
            _delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {
                            "name": "Bash",
                            "arguments": '{"command": "mlflow experiments search"}',
                        },
                    }
                ]
            )
        ),
        b"data: [DONE]\n",
    ]
    turn2 = [_sse(_delta(content="Done")), b"data: [DONE]\n"]
    session, _ = _make_aiohttp_session([turn1, turn2])
    with (
        patch(
            "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
            return_value=session,
        ),
        patch(
            "mlflow.assistant.providers.openai_compatible.execute_tool",
            AsyncMock(return_value=("ok", False)),
        ) as mock_tool,
    ):
        events = [
            e
            async for e in provider.astream(
                "go", "http://localhost:5000", mlflow_session_id=_SESSION_ID
            )
        ]
    assert not any(e.type == EventType.PERMISSION_REQUEST for e in events)
    mock_tool.assert_awaited_once()
    assert events[-1].type == EventType.DONE


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_arguments", ["[]", "null", "123", '"just a string"'])
async def test_astream_non_dict_tool_arguments_does_not_abort_turn(provider, bad_arguments):
    # Regression guard: a model can emit syntactically valid JSON for a tool call's
    # "arguments" that isn't an object (e.g. "[]" or "null"). json.loads decodes this
    # without raising, so the existing "except json.JSONDecodeError: tool_input = {}"
    # never fires, and the non-dict value used to reach ToolUseBlock's pydantic
    # validation (input must be a dict) and static_permission_error's tool_input.get(...)
    # calls uncaught, aborting the turn with an ERROR event instead of a normal tool
    # result. tool_input must be normalized to a dict right after parsing.
    turn1 = [
        _sse(
            _delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "Bash", "arguments": bad_arguments},
                    }
                ]
            )
        ),
        b"data: [DONE]\n",
    ]
    turn2 = [_sse(_delta(content="Done")), b"data: [DONE]\n"]
    session, _ = _make_aiohttp_session([turn1, turn2])
    with patch(
        "mlflow.assistant.providers.openai_compatible.aiohttp.ClientSession",
        return_value=session,
    ):
        events = [
            e
            async for e in provider.astream(
                "go", "http://localhost:5000", mlflow_session_id=_SESSION_ID
            )
        ]
    assert not any(e.type == EventType.ERROR for e in events)
    assert events[-1].type == EventType.DONE
