import importlib
import logging

from mlflow.anthropic.autolog import (
    async_patched_class_call,
    patched_class_call,
    patched_claude_sdk_init,
)
from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "anthropic"
_logger = logging.getLogger(__name__)


def _patch_messages_classes(sync_cls, async_cls):
    safe_patch(FLAVOR_NAME, sync_cls, "create", patched_class_call)
    safe_patch(FLAVOR_NAME, async_cls, "create", async_patched_class_call)
    # `parse` was added in anthropic 0.73.0 (beta) and 0.77.0.
    if hasattr(sync_cls, "parse"):
        safe_patch(FLAVOR_NAME, sync_cls, "parse", patched_class_call)
    if hasattr(async_cls, "parse"):
        safe_patch(FLAVOR_NAME, async_cls, "parse", async_patched_class_call)


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from Anthropic to MLflow.
    Only synchronous calls and asynchronous APIs are supported. Streaming is not recorded.
    ``messages.create``, ``messages.parse``, and their beta Messages API counterparts are
    traced, including the calls a non-streaming ``client.beta.messages.tool_runner`` makes.

    This also enables tracing for Claude Code SDK if available.

    Args:
        log_traces: If ``True``, traces are logged for Anthropic models.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the Anthropic autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Anthropic
            autologging. If ``False``, show all events and warnings.
    """
    from anthropic.resources import AsyncMessages, Messages

    _patch_messages_classes(Messages, AsyncMessages)

    # The beta Messages API is what `client.beta.messages.tool_runner` calls. The Bedrock and
    # Vertex AI clients copy its methods into classes of their own, so those are patched too.
    for module_name in (
        "anthropic.resources.beta.messages",
        "anthropic.lib.bedrock._beta_messages",
        "anthropic.lib.vertex._beta_messages",
    ):
        try:
            module = importlib.import_module(module_name)
            messages_classes = (module.Messages, module.AsyncMessages)
        except (ImportError, AttributeError) as e:
            _logger.debug("Skipping Anthropic beta Messages patching for %s: %s", module_name, e)
            continue
        _patch_messages_classes(*messages_classes)

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
