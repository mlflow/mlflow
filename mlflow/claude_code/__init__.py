"""Claude Code integration for MLflow.

This module provides automatic tracing of Claude Code conversations to MLflow.

CLI Usage:
    mlflow autolog claude [directory] [options]

After setup, use the regular 'claude' command and traces will be automatically captured.

SDK Usage:
    import mlflow.claude_code
    mlflow.claude_code.autolog()

    from claude_agent_sdk import ClaudeSDKClient
    async with ClaudeSDKClient() as client:
        await client.query("What is 2 + 2?")

Note: SDK tracing only works with the `ClaudeSDKClient`, not with `query` directly.
"""

from mlflow.claude_code.hooks import sdk_stop_hook_handler
from mlflow.claude_code.tracing import get_logger
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_TRACE_LOGGING


def autolog():
    """
    Enable automatic MLflow tracing for Claude Code SDK usage.

    This modifies the default options for ClaudeSDKClient to include a Stop hook
    that processes transcripts and creates MLflow traces.

    Usage:
        import mlflow.claude_code
        mlflow.claude_code.autolog()

        from claude_agent_sdk import ClaudeSDKClient
        async with ClaudeSDKClient() as client:
            await client.query("What is 2 + 2?")
            # Trace will be created when client exits

        trace_id = mlflow.get_last_active_trace_id()
    """
    # Enable async trace logging for SDK usage
    MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.set("true")

    try:
        import claude_agent_sdk
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        # Store original __init__
        original_init = claude_agent_sdk.ClaudeSDKClient.__init__

        def wrapped_init(self, options=None):
            # Create options if not provided
            if options is None:
                try:
                    options = ClaudeAgentOptions()
                except Exception:
                    options = {}

            # SDK options object
            if options.hooks is None:
                options.hooks = {}
            if "Stop" not in options.hooks:
                options.hooks["Stop"] = []

            options.hooks["Stop"].append(HookMatcher(hooks=[sdk_stop_hook_handler]))

            # Call original init with modified options
            original_init(self, options)

        # Replace __init__ with wrapped version
        claude_agent_sdk.ClaudeSDKClient.__init__ = wrapped_init

    except ImportError as e:
        get_logger().error("Failed to import claude_agent_sdk: %s", e)
    except Exception as e:
        get_logger().error("Failed to enable autolog: %s", e, exc_info=True)


__all__ = ["autolog"]
