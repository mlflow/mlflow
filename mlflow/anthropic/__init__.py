import logging

from mlflow.anthropic.autolog import (
    async_patched_class_call,
    patched_class_call,
    patched_claude_sdk_init,
    patched_stream_call,
)
from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "anthropic"
_logger = logging.getLogger(__name__)


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from Anthropic to MLflow.
    Both the synchronous and the asynchronous APIs are supported, including streaming with
    ``messages.create(stream=True)`` and ``messages.stream()``.

    This also enables tracing for Claude Code SDK if available.

    Args:
        log_traces: If ``True``, traces are logged for Anthropic models.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the Anthropic autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Anthropic
            autologging. If ``False``, show all events and warnings.
    """
    from anthropic.resources import AsyncMessages, Messages

    safe_patch(
        FLAVOR_NAME,
        Messages,
        "create",
        patched_class_call,
    )

    safe_patch(
        FLAVOR_NAME,
        AsyncMessages,
        "create",
        async_patched_class_call,
    )

    # `stream()` does not go through `create()`. On both classes it is a regular function that
    # returns a manager, which the patch replaces with one that traces the stream.
    for cls in (Messages, AsyncMessages):
        safe_patch(
            FLAVOR_NAME,
            cls,
            "stream",
            patched_stream_call,
            return_patch_result=True,
        )

    # Patch Claude Code SDK if available
    try:
        from claude_agent_sdk import ClaudeSDKClient

        safe_patch(
            FLAVOR_NAME,
            ClaudeSDKClient,
            "__init__",
            patched_claude_sdk_init,
        )
    except ImportError:
        _logger.debug("Claude Agent SDK not installed, skipping Claude Code SDK patching")
    _record_event(
        AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
    )
