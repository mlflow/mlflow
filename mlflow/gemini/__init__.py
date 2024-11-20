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
    Only synchronous calls are supported. Asynchnorous APIs and streaming are not recorded.

    Args:
        log_traces: If ``True``, traces are logged for Gemini models.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the Gemini autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Gemini
            autologging. If ``False``, show all events and warnings.
    """
    import google.generativeai as genai

    safe_patch(
        FLAVOR_NAME,
        genai.GenerativeModel,
        "generate_content",
        patched_class_call,
    )

    safe_patch(
        FLAVOR_NAME,
        genai.GenerativeModel,
        "count_tokens",
        patched_class_call,
    )

    safe_patch(
        FLAVOR_NAME,
        genai.ChatSession,
        "send_message",
        patched_class_call,
    )

    safe_patch(
        FLAVOR_NAME,
        genai,
        "embed_content",
        patched_module_call,
    )
