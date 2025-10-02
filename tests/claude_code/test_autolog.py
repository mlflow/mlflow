import sys
from unittest.mock import MagicMock, patch

import mlflow.anthropic
from mlflow.anthropic.autolog import patched_claude_sdk_init


def test_anthropic_autolog_without_claude_sdk():
    # Ensure claude_agent_sdk is not in sys.modules
    if "claude_agent_sdk" in sys.modules:
        del sys.modules["claude_agent_sdk"]

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
