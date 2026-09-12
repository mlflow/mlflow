"""Utilities to convert MLflow traces and sessions into Inspect AI task payloads.

This module reuses existing `mlflow.genai.utils.trace_utils` helpers to extract
inputs, outputs, expectations, and conversational history, then shapes them
into small, predictable dictionaries that Inspect AI callables can consume.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException

from mlflow.genai.utils.trace_utils import (
    extract_retrieval_context_from_trace,
    parse_inputs_to_str,
    parse_outputs_to_str,
    resolve_expectations_from_session,
    resolve_expectations_from_trace,
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
    resolve_conversation_from_session,
)

INSPECTAI_NOT_INSTALLED_ERROR_MESSAGE = (
    "Inspect AI scorers require the 'inspectai' package. "
    "Please install it with: pip install inspectai"
)


def _check_inspectai_installed():
    try:
        import inspectai 

        return True
    except Exception:
        try:
            import inspect_ai  

            return True
        except Exception:
            raise MlflowException.invalid_parameter_value(INSPECTAI_NOT_INSTALLED_ERROR_MESSAGE)


def map_scorer_inputs_to_inspectai_payload(
    metric_name: str,
    inputs: Any = None,
    outputs: Any = None,
    expectations: Optional[Dict[str, Any]] = None,
    trace: Optional[Trace] = None,
    scorer_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a payload dict for single-turn Inspect AI tasks.

    The returned payload includes stringified `input`/`output` for judge
    compatibility, optional `expectations`, and helpful context such as
    `retrieval_context`, `tags`, and `additional_metadata` when available.
    """
    _check_inspectai_installed()

    inputs = resolve_inputs_from_trace(inputs, trace)
    outputs = resolve_outputs_from_trace(outputs, trace)
    expectations = resolve_expectations_from_trace(expectations, trace)

    payload: Dict[str, Any] = {
        "metric_name": metric_name,
        "input": parse_inputs_to_str(inputs) if inputs is not None else None,
        "output": parse_outputs_to_str(outputs) if outputs is not None else None,
        "expectations": expectations,
    }

    if scorer_kwargs: 
        payload["config"] = scorer_kwargs

    if trace is not None:
        try:
            retrieval_context = extract_retrieval_context_from_trace(trace)
            payload["retrieval_context"] = retrieval_context
        except Exception:
            payload["retrieval_context"] = {}

        try:
            payload["additional_metadata"] = trace.info.trace_metadata or {}
        except Exception:
            payload["additional_metadata"] = {}

        try:
            payload["tags"] = trace.info.tags or {}
        except Exception:
            payload["tags"] = {}

    return payload


def map_session_to_inspectai_conversational_payload(
    metric_name: str,
    session: Optional[List[Trace]] = None,
    expectations: Optional[Dict[str, Any]] = None,
    include_tool_calls: bool = False,
    include_timing: bool = False,
    scorer_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a payload dict for conversational Inspect AI tasks.

    The function assembles a `conversation` list of message dicts derived from
    the traces using `resolve_conversation_from_session`. It also resolves
    session-level expectations when available.
    """
    _check_inspectai_installed()

    conversation = []
    if session:
        conversation = resolve_conversation_from_session(
            session, include_tool_calls=include_tool_calls, include_timing=include_timing
        )

    expectations = resolve_expectations_from_session(expectations, session)

    payload: Dict[str, Any] = { 
        "metric_name": metric_name,
        "conversation": conversation,
        "expectations": expectations,
    }
    if scorer_kwargs: 
        payload["config"] = scorer_kwargs
    return payload