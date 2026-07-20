"""Adapter wrapper to call Inspect AI scoring tasks from MLflow.

This module provides a single entrypoint `score_with_inspect_ai` which maps
MLflow inputs/traces/sessions into a payload consumable by Inspect AI and
normalizes the response into an MLflow `Feedback` object.

The implementation is intentionally defensive: if the Inspect AI package is
not installed, callers receive a helpful error message directing them to
install the dependency. The concrete invocation of the Inspect AI library is
kept flexible to allow tests to patch/mock the underlying call site.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException

from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.inspect_ai import adapter, registry, utils

_logger = logging.getLogger(__name__)

INSPECTAI_NOT_INSTALLED_ERROR_MESSAGE = (
    "Inspect AI scorers require the 'inspectai' package. "
    "Please install it with: pip install inspectai"
)

_FRAMEWORK_NAME = "inspect_ai"

_SKIP_STATUSES = frozenset({"skip", "skipped", "not_run", "notrun", "not-run"})
_ERROR_STATUSES = frozenset({"error", "failed", "exception"})


def _normalize_inspectai_result(result: Any, name: str) -> Feedback:
    """Normalize a variety of Inspect AI return shapes into ``Feedback``.

    Terminal-state semantics (per maintainer requirements):

    - ``None`` result → ``not_run`` state (distinguishable from a scored failure).
    - Dict with ``status`` in ``_SKIP_STATUSES`` → explicit ``skip`` state.
    - Dict with ``status`` in ``_ERROR_STATUSES`` → scorer-level ``error`` state.
    - Dict with ``value``/``score``/``reason`` → normal scored result.
    - Already a ``Feedback`` → returned as-is.
    - Any other type → wrapped in metadata for inspection.

    The ``terminal_state`` metadata key is always set so downstream consumers
    can distinguish skip / not-run / error / pass / fail without inspecting
    the ``value`` field.
    """
    if isinstance(result, Feedback):
        return result

    base_metadata: Dict[str, Any] = {FRAMEWORK_METADATA_KEY: _FRAMEWORK_NAME}

    if result is None:
        return Feedback(
            name=name,
            value=None,
            rationale="Scorer returned no result (not-run or case was not evaluated).",
            metadata={**base_metadata, "terminal_state": "not_run"},
        )

    if isinstance(result, dict):
        status = (result.get("status") or "").lower().replace("-", "_")

        
        if status in _SKIP_STATUSES:
            extra = {k: v for k, v in result.items() if k not in ("status", "reason")}
            return Feedback(
                name=name,
                value=None,
                rationale=result.get("reason") or "Case was skipped by the scorer.",
                metadata={**base_metadata, "terminal_state": "skip", **extra},
            )

        
        if status in _ERROR_STATUSES:
            error_msg = (
                result.get("reason")
                or result.get("error")
                or "Scorer reported an error."
            )
            extra = {k: v for k, v in result.items() if k not in ("status", "reason", "error")}
            return Feedback(
                name=name,
                value=None,
                rationale=error_msg,
                metadata={**base_metadata, "terminal_state": "error", **extra},
            )

        value = result.get("value")
        reason = result.get("reason") or result.get("explanation")
        score = result.get("score")
        extra = {k: v for k, v in result.items() if k not in ("value", "reason", "score", "status")}
        metadata: Dict[str, Any] = {**base_metadata, "terminal_state": "scored", **extra}
        if score is not None:
            metadata["score"] = score

        return Feedback(name=name, value=value, rationale=reason, metadata=metadata)

    
    return Feedback(
        name=name,
        metadata={**base_metadata, "terminal_state": "not_run", "raw_result_type": type(result).__name__, "raw_result": result},
    )


def _ensure_inspectai_installed():
    """Import the installed Inspect AI package or raise a helpful error."""
    return adapter._import_inspectai_module()


def score_with_inspect_ai(
    *,
    metric_name: str,
    model: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    inputs: Any = None,
    outputs: Any = None,
    expectations: Optional[Dict[str, Any]] = None,
    trace: Optional[Trace] = None,
    session: Optional[list[Trace]] = None,
    scorer_kwargs: Optional[Dict[str, Any]] = None,
) -> Feedback:
    """Score a trace or session using an Inspect AI task/scorer.

    This function intentionally keeps the contract broad and defensive: it
    converts MLflow types into a small payload, delegates to the installed
    Inspect AI library (if available), and normalizes the response to an
    MLflow `Feedback`.
    """
    ia = _ensure_inspectai_installed()

    if scorer_kwargs is not None and list(scorer_kwargs.keys()) == ["scorer_kwargs"]:
        scorer_kwargs = scorer_kwargs["scorer_kwargs"]

    payload = utils.map_scorer_inputs_to_inspectai_payload(
        metric_name=metric_name,
        inputs=inputs,
        outputs=outputs,
        expectations=expectations,
        trace=trace,
        scorer_kwargs=scorer_kwargs, 
    )
    
    conversational_payload = utils.map_session_to_inspectai_conversational_payload(
        metric_name=metric_name,
        session=session,
        expectations=expectations,
        scorer_kwargs=scorer_kwargs, 
    )
    payload.update(conversational_payload)

   
    call_kwargs = {}
    if model is not None:
        from mlflow.genai.scorers.inspect_ai.models import create_inspectai_model
        call_kwargs["model"] = create_inspectai_model(model, model_kwargs)

    if scorer_kwargs is not None:
        call_kwargs["config"] = scorer_kwargs

    try:
        callable_fn = registry.get_task_callable(metric_name)
    except Exception:
        candidates = [
            getattr(ia, "score", None),
            getattr(ia, "score_task", None),
            getattr(ia, "evaluate", None),
            getattr(ia, "run_task", None),
            getattr(ia, "run", None),
        ]

        callable_fn = None
        for fn in candidates:
            if callable(fn):
                callable_fn = fn
                break

        if callable_fn is None:
            for submod_name in ("tasks", "scorers", "client"):
                sub = getattr(ia, submod_name, None)
                if sub is None:
                    continue
                for nm in ("score", "score_task", "evaluate", "run_task", "run"):
                    fn = getattr(sub, nm, None)
                    if callable(fn):
                        callable_fn = fn
                        break
                if callable_fn is not None:
                    break

    if callable_fn is None:
        raise MlflowException(
            "Unable to find a scoring entrypoint in the Inspect AI package. "
            "Please ensure the package exposes a callable named 'score', 'evaluate', or similar."
        )

    try:
        result = adapter.invoke_task_callable(
            callable_fn,
            metric_name=metric_name,
            payload=payload,
            call_kwargs=call_kwargs,
        )
        return _normalize_inspectai_result(result, name=metric_name)
    except Exception as exc:
        _logger.error("Inspect AI scoring call failed: %s", exc)
        raise