"""MLflow autologging integration for Aider coding sessions."""

import functools
import logging
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

_logger = logging.getLogger(__name__)

_AIDER_PATCHED_ATTR = "_mlflow_patched"


def autolog(disable: bool = False, silent: bool = False) -> None:
    """Enable (or disable) automatic MLflow tracing for Aider coding sessions.

    When enabled, every ``coder.run()`` call is wrapped in an MLflow trace that
    captures the user prompt, model name, in-chat files, LLM response, and token
    usage.

    Args:
        disable: If True, remove the autologging patches from Aider's Coder class.
        silent: If True, suppress info-level log messages about patching status.
    """
    try:
        from aider.coders import Coder
    except ImportError as e:
        raise ImportError(
            "aider-chat is not installed. Install it with: pip install aider-chat"
        ) from e

    if disable:
        _unpatch_coder(Coder)
        if not silent:
            _logger.info("MLflow Aider autologging disabled.")
        return

    _patch_coder(Coder)
    if not silent:
        _logger.info("MLflow Aider autologging enabled.")


def _patch_coder(Coder: type) -> None:
    if getattr(Coder, _AIDER_PATCHED_ATTR, False):
        return

    original_run_one = Coder.run_one

    @functools.wraps(original_run_one)
    def patched_run_one(self: Any, user_message: str, preproc: bool = True) -> str | None:
        model_name = getattr(self.main_model, "name", "unknown")
        in_chat_files = _get_chat_files(self)

        # Snapshot token counts before the turn so we can compute per-turn usage.
        tokens_sent_before = getattr(self, "total_tokens_sent", 0) or 0
        tokens_received_before = getattr(self, "total_tokens_received", 0) or 0

        with mlflow.start_span(
            name="aider_turn",
            span_type=SpanType.AGENT,
        ) as span:
            span.set_inputs({
                "prompt": user_message,
                "model": model_name,
                "files": in_chat_files,
            })
            span.set_attribute("model", model_name)

            result = original_run_one(self, user_message, preproc)

            tokens_sent_after = getattr(self, "total_tokens_sent", 0) or 0
            tokens_received_after = getattr(self, "total_tokens_received", 0) or 0
            input_tokens = tokens_sent_after - tokens_sent_before
            output_tokens = tokens_received_after - tokens_received_before

            if input_tokens > 0 or output_tokens > 0:
                span.set_attribute(
                    SpanAttributeKey.CHAT_USAGE,
                    {
                        TokenUsageKey.INPUT_TOKENS: input_tokens,
                        TokenUsageKey.OUTPUT_TOKENS: output_tokens,
                        TokenUsageKey.TOTAL_TOKENS: input_tokens + output_tokens,
                    },
                )

            total_cost = getattr(self, "total_cost", None)
            if total_cost is not None:
                span.set_attribute("total_cost_usd", total_cost)

            span.set_outputs({"response": result or ""})

        return result

    Coder.run_one = patched_run_one
    setattr(Coder, _AIDER_PATCHED_ATTR, True)


def _unpatch_coder(Coder: type) -> None:
    if not getattr(Coder, _AIDER_PATCHED_ATTR, False):
        return

    original = getattr(Coder.run_one, "__wrapped__", None)
    if original is not None:
        Coder.run_one = original

    setattr(Coder, _AIDER_PATCHED_ATTR, False)


def _get_chat_files(coder: Any) -> list[str]:
    try:
        return list(coder.get_inchat_relative_files())
    except Exception:
        return []
