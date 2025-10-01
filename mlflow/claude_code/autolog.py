"""Autologging implementation for Claude Code SDK."""

from mlflow.claude_code.hooks import sdk_stop_hook_handler
from mlflow.claude_code.tracing import get_logger


def patched_init(original, self, options=None):
    """Patched __init__ that adds MLflow tracing hook to ClaudeSDKClient."""
    try:
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        # Create options if not provided
        if options is None:
            try:
                options = ClaudeAgentOptions()
            except Exception:
                options = {}

        if options.hooks is None:
            options.hooks = {}
        if "Stop" not in options.hooks:
            options.hooks["Stop"] = []

        options.hooks["Stop"].append(HookMatcher(hooks=[sdk_stop_hook_handler]))

        # Call original init with modified options
        return original(self, options)

    except Exception as e:
        get_logger().error("Error in patched_init: %s", e, exc_info=True)
        # Fall back to original behavior if patching fails
        return original(self, options)
