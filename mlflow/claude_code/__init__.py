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

from mlflow.claude_code.autolog import patched_init
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_TRACE_LOGGING
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "claude_code"


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enable automatic MLflow tracing for Claude Code SDK usage.

    Args:
        log_traces: If ``True``, traces are logged for Claude Code SDK.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the Claude Code autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Claude Code
            autologging. If ``False``, show all events and warnings.

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

    from claude_agent_sdk import ClaudeSDKClient

    safe_patch(
        FLAVOR_NAME,
        ClaudeSDKClient,
        "__init__",
        patched_init,
    )


__all__ = ["autolog"]
