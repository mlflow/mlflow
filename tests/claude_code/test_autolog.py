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


async def _setup_patched_client_and_consume(messages, response_messages=None):
    """Patch a mock SDK client, stream messages through it, and return the stop hook.

    Args:
        messages: Messages yielded by receive_messages().
        response_messages: If provided, messages yielded by receive_response() instead of
            using receive_messages(). Used to test that receive_response() wrapper captures
            ResultMessage separately from receive_messages().
    """
    original_init = MagicMock()
    mock_self = MagicMock()
    mock_self.options = _make_mock_options(hooks={"Stop": []})

    async def fake_receive_messages():
        for msg in messages:
            yield msg

    mock_self.receive_messages = fake_receive_messages

    # receive_response() normally calls receive_messages() internally,
    # then appends a ResultMessage. We simulate this with a separate generator.
    async def fake_receive_response():
        for msg in (response_messages if response_messages is not None else messages):
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

    # Consume via receive_messages to populate the buffer with conversation messages
    collected = [msg async for msg in mock_self.receive_messages()]

    # If response_messages provided, also consume receive_response (which captures ResultMessage)
    if response_messages is not None:
        [msg async for msg in mock_self.receive_response()]

    stop_hook_fn = mock_self.options.hooks["Stop"][0].hooks[0]
    return mock_self, collected, stop_hook_fn


@pytest.mark.asyncio
async def test_receive_messages_wrapper_accumulates_messages():
    messages = ["msg1", "msg2", "msg3"]
    _, collected, _ = await _setup_patched_client_and_consume(messages)
    assert collected == messages


@pytest.mark.asyncio
async def test_stop_hook_forwards_messages_to_tracing():
    messages = ["msg1", "msg2"]
    _, _, stop_hook_fn = await _setup_patched_client_and_consume(messages)

    # Hook should forward accumulated messages to process_sdk_messages
    with (
        patch(
            "mlflow.utils.autologging_utils.autologging_is_disabled",
            return_value=False,
        ),
        patch(
            "mlflow.claude_code.tracing.process_sdk_messages",
            return_value=MagicMock(),
        ) as mock_process,
    ):
        result = await stop_hook_fn({"session_id": "test-session"}, None, None)

    mock_process.assert_called_once()
    assert mock_process.call_args[0][0] == messages
    assert mock_process.call_args[0][1] == "test-session"
    assert result == {"continue": True}

    # Calling again without new messages should forward an empty list
    with (
        patch(
            "mlflow.utils.autologging_utils.autologging_is_disabled",
            return_value=False,
        ),
        patch(
            "mlflow.claude_code.tracing.process_sdk_messages",
            return_value=None,
        ) as mock_process,
    ):
        await stop_hook_fn({"session_id": "s2"}, None, None)
        assert mock_process.call_args[0][0] == []


@pytest.mark.asyncio
async def test_stop_hook_skips_when_autologging_disabled():
    messages = ["msg1", "msg2"]
    _, _, stop_hook_fn = await _setup_patched_client_and_consume(messages)

    with (
        patch(
            "mlflow.utils.autologging_utils.autologging_is_disabled",
            return_value=True,
        ),
        patch(
            "mlflow.claude_code.tracing.process_sdk_messages",
        ) as mock_process,
    ):
        result = await stop_hook_fn({"session_id": "test"}, None, None)

    mock_process.assert_not_called()
    assert result == {"continue": True}


@pytest.mark.asyncio
async def test_receive_response_wrapper_captures_result_message():
    """receive_response() yields ResultMessage after receive_messages() is done.
    The wrapper should capture ResultMessage into the buffer so token usage
    and duration are available for tracing."""
    from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock, UserMessage

    # receive_messages() only yields conversation messages
    conversation_msgs = [
        UserMessage(content="Hello"),
        AssistantMessage(content=[TextBlock(text="Hi!")], model="claude-sonnet-4-20250514"),
    ]
    # receive_response() yields conversation + ResultMessage
    result_msg = ResultMessage(
        subtype="success",
        duration_ms=5000,
        duration_api_ms=4000,
        is_error=False,
        num_turns=1,
        session_id="test",
        usage={"input_tokens": 100, "output_tokens": 20},
    )
    response_msgs = [*conversation_msgs, result_msg]

    _, _, stop_hook_fn = await _setup_patched_client_and_consume(
        conversation_msgs, response_messages=response_msgs
    )

    # The stop hook should see conversation messages + ResultMessage in the buffer
    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False),
        patch("mlflow.claude_code.tracing.process_sdk_messages", return_value=MagicMock()) as mock,
    ):
        await stop_hook_fn({"session_id": "test"}, None, None)

    # Buffer should contain: conversation msgs from receive_messages + ResultMessage
    # from receive_response. The conversation msgs appear twice (once from each wrapper),
    # but that's fine — process_sdk_messages handles duplicates.
    called_messages = mock.call_args[0][0]
    result_messages = [m for m in called_messages if isinstance(m, ResultMessage)]
    assert len(result_messages) == 1
    assert result_messages[0].usage == {"input_tokens": 100, "output_tokens": 20}


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
