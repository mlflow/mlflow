"""Tests for mlflow.claude_code.autolog functionality."""

import asyncio
from unittest.mock import MagicMock, patch

from mlflow.claude_code.autolog import patched_init
from mlflow.claude_code.hooks import sdk_stop_hook_handler


def test_autolog_function_exists():
    """Test that autolog function is exported."""
    from mlflow.claude_code import autolog

    assert callable(autolog)


def test_autolog_runs_without_sdk():
    """Test that autolog handles missing SDK gracefully."""
    from mlflow.claude_code import autolog

    # Should not raise exception when SDK is not installed
    autolog()


def test_autolog_with_options():
    """Test that autolog works when ClaudeSDKClient is created with options."""
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
        patched_init(original_init, mock_self, mock_options)

    # Verify original_init was called
    original_init.assert_called_once_with(mock_self, mock_options)
    # Verify Stop hook was appended
    assert len(mock_options.hooks["Stop"]) == 2


def test_sdk_stop_hook_handler_handles_missing_transcript():
    """Test that sdk_stop_hook_handler handles missing transcript gracefully."""

    async def test():
        input_data = {
            "session_id": "test-session-123",
            "transcript_path": "/nonexistent/path/transcript.jsonl",
        }

        result = await sdk_stop_hook_handler(input_data, None, None)
        assert result["continue"] is False
        assert "stopReason" in result

    asyncio.run(test())
