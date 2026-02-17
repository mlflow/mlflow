import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import mlflow.anthropic
from mlflow.anthropic.autolog import patched_claude_sdk_init


def test_anthropic_autolog_without_claude_sdk():
    # Ensure claude_agent_sdk is not in sys.modules
    sys.modules.pop("claude_agent_sdk", None)

    with (
        patch.dict(
            "sys.modules",
            {
                "anthropic": MagicMock(__version__="0.35.0"),
                "anthropic.resources": MagicMock(Messages=MagicMock, AsyncMessages=MagicMock),
            },
        ),
        patch("mlflow.anthropic.safe_patch"),
    ):
        # Should not raise exception when claude_agent_sdk is not installed
        mlflow.anthropic.autolog()


def _make_mock_options(hooks=None):
    @dataclass
    class FakeOptions:
        hooks: dict[str, Any] | None = None

    return FakeOptions(hooks=hooks)


def _make_mock_hook_matcher(**kwargs):
    class FakeHookMatcher:
        def __init__(self, hooks=None):
            self.hooks = hooks or []

    return FakeHookMatcher


def _patch_sdk_init(mock_self, messages, response_messages=None):
    """Set up fake generators on mock_self and call patched_claude_sdk_init."""
    original_init = MagicMock()
    mock_self.options = _make_mock_options(hooks={"Stop": []})

    async def fake_receive_messages():
        for msg in messages:
            yield msg

    mock_self.receive_messages = fake_receive_messages

    async def fake_receive_response():
        for msg in response_messages if response_messages is not None else messages:
            yield msg

    mock_self.receive_response = fake_receive_response

    with patch.dict(
        "sys.modules",
        {
            "claude_agent_sdk": MagicMock(
                ClaudeAgentOptions=lambda: _make_mock_options(),
                HookMatcher=_make_mock_hook_matcher(),
            )
        },
    ):
        patched_claude_sdk_init(original_init, mock_self, mock_self.options)

    return original_init


def test_patched_claude_sdk_init_wraps_client_and_injects_hook():
    original_init = MagicMock()
    mock_self = MagicMock()
    mock_self.options = _make_mock_options(hooks=None)
    original_receive_messages = AsyncMock()
    original_receive_response = AsyncMock()
    mock_self.receive_messages = original_receive_messages
    mock_self.receive_response = original_receive_response

    with patch.dict(
        "sys.modules",
        {
            "claude_agent_sdk": MagicMock(
                ClaudeAgentOptions=lambda: _make_mock_options(),
                HookMatcher=_make_mock_hook_matcher(),
            )
        },
    ):
        patched_claude_sdk_init(original_init, mock_self, mock_self.options)

    original_init.assert_called_once_with(mock_self, mock_self.options)
    # Both receive_messages and receive_response should be wrapped
    assert mock_self.receive_messages is not original_receive_messages
    assert mock_self.receive_response is not original_receive_response
    assert "Stop" in mock_self.options.hooks
    assert len(mock_self.options.hooks["Stop"]) == 1


@pytest.mark.asyncio
async def test_receive_messages_wrapper_accumulates_messages():
    mock_self = MagicMock()
    messages = ["msg1", "msg2", "msg3"]
    _patch_sdk_init(mock_self, messages)

    collected = [msg async for msg in mock_self.receive_messages()]
    assert collected == messages


@pytest.mark.asyncio
async def test_stop_hook_builds_trace_when_receive_response_not_used():
    """When only receive_messages() is consumed, the stop hook builds the trace."""
    mock_self = MagicMock()
    messages = ["msg1", "msg2"]
    _patch_sdk_init(mock_self, messages)

    # Consume only via receive_messages
    [msg async for msg in mock_self.receive_messages()]

    stop_hook_fn = mock_self.options.hooks["Stop"][0].hooks[0]

    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False),
        patch(
            "mlflow.claude_code.tracing.process_sdk_messages", return_value=MagicMock()
        ) as mock_process,
    ):
        result = await stop_hook_fn({"session_id": "test-session"}, None, None)

    mock_process.assert_called_once()
    assert mock_process.call_args[0][0] == messages
    assert mock_process.call_args[0][1] == "test-session"
    assert result == {"continue": True}


@pytest.mark.asyncio
async def test_stop_hook_skips_when_autologging_disabled():
    mock_self = MagicMock()
    messages = ["msg1", "msg2"]
    _patch_sdk_init(mock_self, messages)

    [msg async for msg in mock_self.receive_messages()]

    stop_hook_fn = mock_self.options.hooks["Stop"][0].hooks[0]

    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=True),
        patch("mlflow.claude_code.tracing.process_sdk_messages") as mock_process,
    ):
        result = await stop_hook_fn({"session_id": "test"}, None, None)

    mock_process.assert_not_called()
    assert result == {"continue": True}


@pytest.mark.asyncio
async def test_receive_response_builds_trace_with_result_message():
    """When receive_response() is consumed, the trace is built after the
    generator is exhausted — at which point ResultMessage is in the buffer.
    """
    from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock, UserMessage

    mock_self = MagicMock()

    conversation_msgs = [
        UserMessage(content="Hello"),
        AssistantMessage(content=[TextBlock(text="Hi!")], model="claude-sonnet-4-20250514"),
    ]
    result_msg = ResultMessage(
        subtype="success",
        duration_ms=5000,
        duration_api_ms=4000,
        is_error=False,
        num_turns=1,
        session_id="test-session",
        usage={"input_tokens": 100, "output_tokens": 20},
    )
    response_msgs = [*conversation_msgs, result_msg]

    _patch_sdk_init(mock_self, conversation_msgs, response_messages=response_msgs)

    # Consume receive_messages (conversation messages enter buffer)
    [msg async for msg in mock_self.receive_messages()]

    # Consume receive_response — trace should be built when generator finishes
    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False),
        patch(
            "mlflow.claude_code.tracing.process_sdk_messages", return_value=MagicMock()
        ) as mock_process,
    ):
        [msg async for msg in mock_self.receive_response()]

    mock_process.assert_called_once()
    called_messages = mock_process.call_args[0][0]

    # Buffer should have conversation msgs from receive_messages + ResultMessage
    # from receive_response
    result_messages = [m for m in called_messages if isinstance(m, ResultMessage)]
    assert len(result_messages) == 1
    assert result_messages[0].usage == {"input_tokens": 100, "output_tokens": 20}
    assert mock_process.call_args[0][1] == "test-session"


@pytest.mark.asyncio
async def test_stop_hook_defers_during_receive_response():
    """When receive_response() is in progress, the stop hook should be a no-op
    (the trace will be built when the generator is exhausted instead).
    """
    from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock, UserMessage

    mock_self = MagicMock()
    conversation_msgs = [
        UserMessage(content="Hello"),
        AssistantMessage(content=[TextBlock(text="Hi!")], model="claude-sonnet-4-20250514"),
    ]
    result_msg = ResultMessage(
        subtype="success",
        duration_ms=5000,
        duration_api_ms=4000,
        is_error=False,
        num_turns=1,
        session_id="test",
    )
    # Simulate SDK behavior: conversation messages first, then ResultMessage
    response_msgs = [*conversation_msgs, result_msg]

    _patch_sdk_init(mock_self, conversation_msgs, response_messages=response_msgs)

    # Consume receive_messages so conversation msgs enter the buffer
    [msg async for msg in mock_self.receive_messages()]

    # Partially consume receive_response (just start it, don't exhaust)
    # Then call the stop hook — it should defer since we're mid-stream
    gen = mock_self.receive_response()
    first_msg = await gen.__anext__()  # Get first conversation message
    assert isinstance(first_msg, UserMessage)

    stop_hook_fn = mock_self.options.hooks["Stop"][0].hooks[0]

    with patch("mlflow.claude_code.tracing.process_sdk_messages") as mock_process:
        result = await stop_hook_fn({"session_id": "test"}, None, None)

    # Stop hook should defer — receiving_response is True
    mock_process.assert_not_called()
    assert result == {"continue": True}

    # Now exhaust the generator — trace should be built
    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False),
        patch(
            "mlflow.claude_code.tracing.process_sdk_messages", return_value=MagicMock()
        ) as mock_process,
    ):
        async for _ in gen:
            pass

    mock_process.assert_called_once()


@pytest.mark.asyncio
async def test_stop_hook_is_noop_after_receive_response():
    """After receive_response() builds the trace, the stop hook should be a no-op."""
    from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock, UserMessage

    mock_self = MagicMock()
    conversation_msgs = [
        UserMessage(content="Hello"),
        AssistantMessage(content=[TextBlock(text="Hi!")], model="claude-sonnet-4-20250514"),
    ]
    result_msg = ResultMessage(
        subtype="success",
        duration_ms=5000,
        duration_api_ms=4000,
        is_error=False,
        num_turns=1,
        session_id="test",
    )
    response_msgs = [*conversation_msgs, result_msg]

    _patch_sdk_init(mock_self, conversation_msgs, response_messages=response_msgs)

    # Consume both wrappers
    [msg async for msg in mock_self.receive_messages()]
    with patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False):
        with patch("mlflow.claude_code.tracing.process_sdk_messages", return_value=MagicMock()):
            [msg async for msg in mock_self.receive_response()]

    stop_hook_fn = mock_self.options.hooks["Stop"][0].hooks[0]

    with patch("mlflow.claude_code.tracing.process_sdk_messages") as mock_process:
        result = await stop_hook_fn({"session_id": "test"}, None, None)

    # Stop hook should NOT call process_sdk_messages again
    mock_process.assert_not_called()
    assert result == {"continue": True}


@pytest.mark.asyncio
async def test_sdk_hook_handler_when_disabled():
    from mlflow.claude_code.hooks import sdk_stop_hook_handler

    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=True),
        patch("mlflow.claude_code.hooks._process_stop_hook") as mock_process,
    ):
        result = await sdk_stop_hook_handler(
            input_data={"session_id": "test", "transcript_path": "/fake/path"},
            tool_use_id=None,
            context=None,
        )
        # Should return early without calling _process_stop_hook
        mock_process.assert_not_called()
        assert result == {"continue": True}
