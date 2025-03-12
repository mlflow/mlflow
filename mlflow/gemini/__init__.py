"""
The ``mlflow.gemini`` module provides an API for tracing the interaction with Gemini models.
"""

from mlflow.gemini.autolog import (
    patched_class_call,
    patched_module_call,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "gemini"


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from Gemini to MLflow.
    Currently, both legacy SDK google-generativeai and new SDK google-genai are supported.
    Only synchronous calls are supported. Asynchnorous APIs and streaming are not recorded.

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

        # Since the genai SDK calls "_generate_content" iteratively within "generate_content",
        # we need to patch both "generate_content" and "_generate_content".
        for method in ["generate_content", "_generate_content", "count_tokens", "embed_content"]:
            safe_patch(
                FLAVOR_NAME,
                genai.models.Models,
                method,
                patched_class_call,
            )

        safe_patch(
            FLAVOR_NAME,
            genai.chats.Chat,
            "send_message",
            patched_class_call,
        )
    except ImportError:
        pass
