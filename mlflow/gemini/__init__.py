"""
The ``mlflow.gemini`` module provides an API for tracing the interaction with Gemini models.
"""

from mlflow.gemini.autolog import (
    async_patched_class_call,
    patched_class_call,
    patched_module_call,
)
from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "gemini"


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from Gemini to MLflow.
    Currently, both legacy SDK google-generativeai and new SDK google-genai are supported.
    Both synchronous and asynchronous calls are supported for the new SDK.

    Args:
        log_traces: If ``True``, traces are logged for Gemini models.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the Gemini autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Gemini
            autologging. If ``False``, show all events and warnings.
    """
    try:
        from google import generativeai

        for method in ["generate_content", "count_tokens"]:
            safe_patch(
                FLAVOR_NAME,
                generativeai.GenerativeModel,
                method,
                patched_class_call,
            )

        safe_patch(
            FLAVOR_NAME,
            generativeai.ChatSession,
            "send_message",
            patched_class_call,
        )

        safe_patch(
            FLAVOR_NAME,
            generativeai,
            "embed_content",
            patched_module_call,
        )
    except ImportError:
        pass

    try:
        from google import genai

        # Since the genai SDK calls private generation methods iteratively within their public
        # counterparts, we need to patch both methods to trace each model request.
        for method in [
            "generate_content",
            "_generate_content",
            "generate_content_stream",
            "_generate_content_stream",
            "count_tokens",
            "embed_content",
        ]:
            safe_patch(
                FLAVOR_NAME,
                genai.models.Models,
                method,
                patched_class_call,
                return_patch_result=method.endswith("_stream"),
            )
            safe_patch(
                FLAVOR_NAME,
                genai.models.AsyncModels,
                method,
                async_patched_class_call,
                return_patch_result=method.endswith("_stream"),
            )

        for method in ["send_message", "send_message_stream"]:
            safe_patch(
                FLAVOR_NAME,
                genai.chats.Chat,
                method,
                patched_class_call,
                return_patch_result=method.endswith("_stream"),
            )
            safe_patch(
                FLAVOR_NAME,
                genai.chats.AsyncChat,
                method,
                async_patched_class_call,
                return_patch_result=method.endswith("_stream"),
            )
    except ImportError:
        pass

    _record_event(
        AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
    )
