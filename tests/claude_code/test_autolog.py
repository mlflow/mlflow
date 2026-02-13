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


def test_patched_claude_sdk_init_wraps_receive_messages():
    original_init = MagicMock()
    mock_self = MagicMock()
    mock_self.options = _make_mock_options(hooks={"Stop": []})
    mock_self.receive_messages = AsyncMock()

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
    # receive_messages should have been replaced with the wrapped version
    assert mock_self.receive_messages is not original_init


def test_patched_claude_sdk_init_injects_stop_hook():
    original_init = MagicMock()
    mock_self = MagicMock()
    mock_self.options = _make_mock_options(hooks={"Stop": []})
    mock_self.receive_messages = AsyncMock()

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

    assert len(mock_self.options.hooks["Stop"]) == 1


def test_patched_claude_sdk_init_creates_hooks_dict_when_none():
    original_init = MagicMock()
    mock_self = MagicMock()
    mock_self.options = _make_mock_options(hooks=None)
    mock_self.receive_messages = AsyncMock()

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

    assert "Stop" in mock_self.options.hooks
    assert len(mock_self.options.hooks["Stop"]) == 1


@pytest.mark.asyncio
async def test_receive_messages_wrapper_accumulates_messages():
    original_init = MagicMock()
    mock_self = MagicMock()
    mock_self.options = _make_mock_options(hooks={"Stop": []})

    # Create a real async generator for receive_messages
    messages = ["msg1", "msg2", "msg3"]

    async def fake_receive_messages():
        for msg in messages:
            yield msg

    mock_self.receive_messages = fake_receive_messages

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

    # Consume the wrapped async generator
    collected = [msg async for msg in mock_self.receive_messages()]

    assert collected == messages


@pytest.mark.asyncio
async def test_closure_stop_hook_calls_process_sdk_messages():
    original_init = MagicMock()
    mock_self = MagicMock()
    mock_self.options = _make_mock_options(hooks={"Stop": []})

    messages = ["msg1", "msg2"]

    async def fake_receive_messages():
        for msg in messages:
            yield msg

    mock_self.receive_messages = fake_receive_messages

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

    # First consume messages to populate the buffer
    async for _ in mock_self.receive_messages():
        pass

    # Get the injected hook from the closure
    hook_matcher = mock_self.options.hooks["Stop"][0]
    stop_hook_fn = hook_matcher.hooks[0]

    with (
        patch(
            "mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False,
        ),
        patch(
            "mlflow.claude_code.tracing.process_sdk_messages", return_value=MagicMock(),
        ) as mock_process,
    ):
        result = await stop_hook_fn(
            {"session_id": "test-session"}, None, None,
        )

    mock_process.assert_called_once()
    # Verify messages were passed to process_sdk_messages
    call_args = mock_process.call_args
    assert call_args[0][0] == messages
    assert call_args[0][1] == "test-session"
    assert result == {"continue": True}


@pytest.mark.asyncio
async def test_closure_stop_hook_clears_buffer():
    original_init = MagicMock()
    mock_self = MagicMock()
    mock_self.options = _make_mock_options(hooks={"Stop": []})

    messages = ["msg1", "msg2"]

    async def fake_receive_messages():
        for msg in messages:
            yield msg

    mock_self.receive_messages = fake_receive_messages

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

    # Consume messages to populate the buffer
    async for _ in mock_self.receive_messages():
        pass

    hook_matcher = mock_self.options.hooks["Stop"][0]
    stop_hook_fn = hook_matcher.hooks[0]

    # Verify buffer has messages before hook runs
    with (
        patch(
            "mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False,
        ),
        patch(
            "mlflow.claude_code.tracing.process_sdk_messages", return_value=MagicMock(),
        ) as mock_process,
    ):
        await stop_hook_fn({"session_id": "s1"}, None, None)

        # The hook should have received the buffered messages
        call_args = mock_process.call_args
        assert call_args[0][0] == messages

    # After hook runs, the next call should get an empty buffer
    # (verified by calling the hook again without consuming new messages)
    with (
        patch(
            "mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False,
        ),
        patch(
            "mlflow.claude_code.tracing.process_sdk_messages", return_value=None,
        ) as mock_process,
    ):
        await stop_hook_fn({"session_id": "s2"}, None, None)
        call_args = mock_process.call_args
        assert call_args[0][0] == []


@pytest.mark.asyncio
async def test_closure_stop_hook_skips_when_autologging_disabled():
    original_init = MagicMock()
    mock_self = MagicMock()
    mock_self.options = _make_mock_options(hooks={"Stop": []})
    mock_self.receive_messages = AsyncMock()

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

    hook_matcher = mock_self.options.hooks["Stop"][0]
    stop_hook_fn = hook_matcher.hooks[0]

    with (
        patch(
            "mlflow.utils.autologging_utils.autologging_is_disabled", return_value=True,
        ),
        patch(
            "mlflow.claude_code.tracing.process_sdk_messages",
        ) as mock_process,
    ):
        result = await stop_hook_fn({"session_id": "test"}, None, None)

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
