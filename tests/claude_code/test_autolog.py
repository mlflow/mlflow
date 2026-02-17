import sys
from unittest.mock import MagicMock, patch

import pytest

import mlflow.anthropic
from mlflow.anthropic.autolog import patched_claude_sdk_init


def test_anthropic_autolog_without_claude_sdk():
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
        mlflow.anthropic.autolog()


def _patch_sdk_init(mock_self, response_messages):
    """Set up a fake receive_response on mock_self and call patched_claude_sdk_init."""
    original_init = MagicMock()

    async def fake_receive_response():
        for msg in response_messages:
            yield msg

    mock_self.receive_response = fake_receive_response
    patched_claude_sdk_init(original_init, mock_self)
    return original_init


def test_patched_claude_sdk_init_wraps_receive_response():
    mock_self = MagicMock()

    async def fake_receive_response():
        yield "msg1"

    mock_self.receive_response = fake_receive_response
    original_init = MagicMock()
    patched_claude_sdk_init(original_init, mock_self)

    original_init.assert_called_once_with(mock_self, None)
    assert mock_self.receive_response is not fake_receive_response


@pytest.mark.asyncio
async def test_receive_response_builds_trace():
    from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock, UserMessage

    mock_self = MagicMock()
    messages = [
        UserMessage(content="Hello"),
        AssistantMessage(content=[TextBlock(text="Hi!")], model="claude-sonnet-4-20250514"),
        ResultMessage(
            subtype="success",
            duration_ms=5000,
            duration_api_ms=4000,
            is_error=False,
            num_turns=1,
            session_id="test-session",
            usage={"input_tokens": 100, "output_tokens": 20},
        ),
    ]
    _patch_sdk_init(mock_self, messages)

    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False),
        patch(
            "mlflow.claude_code.tracing.process_sdk_messages", return_value=MagicMock()
        ) as mock_process,
    ):
        [msg async for msg in mock_self.receive_response()]

    mock_process.assert_called_once()
    called_messages = mock_process.call_args[0][0]
    assert len(called_messages) == 3
    result_messages = [m for m in called_messages if isinstance(m, ResultMessage)]
    assert len(result_messages) == 1
    assert result_messages[0].usage == {"input_tokens": 100, "output_tokens": 20}


@pytest.mark.asyncio
async def test_receive_response_skips_when_autologging_disabled():
    mock_self = MagicMock()
    _patch_sdk_init(mock_self, ["msg1", "msg2"])

    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=True),
        patch("mlflow.claude_code.tracing.process_sdk_messages") as mock_process,
    ):
        [msg async for msg in mock_self.receive_response()]

    mock_process.assert_not_called()


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
        mock_process.assert_not_called()
        assert result == {"continue": True}
