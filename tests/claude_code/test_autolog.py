import sys
from unittest.mock import MagicMock, patch

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


def test_patched_claude_sdk_init_with_options():
    original_init = MagicMock()

    mock_options = MagicMock()
    mock_options.hooks = {"Stop": ["a"]}
    mock_self = MagicMock()

    # Mock claude_agent_sdk imports
    mock_hook_matcher = MagicMock()
    with patch.dict(
        "sys.modules",
        {
            "claude_agent_sdk": MagicMock(
                ClaudeAgentOptions=MagicMock, HookMatcher=mock_hook_matcher
            )
        },
    ):
        patched_claude_sdk_init(original_init, mock_self, mock_options)

    # Verify original_init was called
    original_init.assert_called_once_with(mock_self, mock_options)
    # Verify Stop hook was appended
    assert len(mock_options.hooks["Stop"]) == 2


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
