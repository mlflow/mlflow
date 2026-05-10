"""Utility functions for Google ADK integration."""

from __future__ import annotations

import json
from typing import Any

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException

GOOGLE_ADK_NOT_INSTALLED_ERROR_MESSAGE = (
    "Google ADK scorers require the `google-adk` package. Install it with: `pip install google-adk`"
)


def check_adk_installed() -> None:
    try:
        import google.adk.evaluation  # noqa: F401
    except ImportError:
        raise MlflowException.invalid_parameter_value(GOOGLE_ADK_NOT_INSTALLED_ERROR_MESSAGE)


def _to_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return str(value)


def _extract_actual_tool_calls(
    expectations: dict[str, Any] | None,
    trace: Trace | None,
) -> list[dict[str, Any]]:
    """Resolve the list of tool calls the agent actually made.

    Priority:
      1. ``expectations["actual_tool_calls"]`` when explicitly supplied
         (lets callers override trace extraction, e.g. for synthetic data).
      2. TOOL spans on ``trace`` via
         :func:`mlflow.genai.utils.trace_utils.extract_tools_called_from_trace`.
      3. Empty list.

    Returned dicts have ``name`` and ``args`` keys, matching the shape
    consumed by the ``{ "name": ..., "args": {...} }`` entries users
    already pass under ``expectations["expected_tool_calls"]``.
    """
    if expectations and (explicit := expectations.get("actual_tool_calls")):
        return list(explicit)

    if trace is None:
        return []

    from mlflow.genai.utils.trace_utils import extract_tools_called_from_trace

    return [
        {"name": call.name, "args": call.arguments or {}}
        for call in extract_tools_called_from_trace(trace)
    ]


def map_scorer_inputs_to_invocation(
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
) -> tuple[Any, Any]:
    """Convert MLflow scorer inputs to an ADK ``Invocation`` pair.

    Builds actual and expected ``google.adk.evaluation.eval_case.Invocation``
    objects from the raw scorer arguments. Expected tool calls come from
    ``expectations["expected_tool_calls"]``; actual tool calls come from the
    trace's TOOL spans, with ``expectations["actual_tool_calls"]`` honored as
    an explicit override when callers want to bypass the trace.

    Returns a tuple of ``(actual_invocation, expected_invocation)``.
    """
    from google.adk.evaluation.eval_case import IntermediateData, Invocation
    from google.genai import types as genai_types

    if trace is not None:
        from mlflow.genai.utils.trace_utils import (
            resolve_inputs_from_trace,
            resolve_outputs_from_trace,
        )

        inputs = resolve_inputs_from_trace(inputs, trace)
        outputs = resolve_outputs_from_trace(outputs, trace)

    input_text = _to_str(inputs) if inputs is not None else ""
    output_text = _to_str(outputs) if outputs is not None else ""

    user_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_text(text=input_text)],
    )

    actual_response = genai_types.Content(
        role="model",
        parts=[genai_types.Part.from_text(text=output_text)],
    )

    actual_invocation = Invocation(
        user_content=user_content,
        final_response=actual_response,
    )

    expected_invocation = Invocation(
        user_content=user_content,
    )

    # Actual tool calls: prefer the trace, fall back to an explicit override.
    if actual_tool_calls_raw := _extract_actual_tool_calls(expectations, trace):
        actual_tool_uses = [
            genai_types.FunctionCall(name=tc["name"], args=tc.get("args") or {})
            for tc in actual_tool_calls_raw
        ]
        actual_invocation.intermediate_data = IntermediateData(
            tool_uses=actual_tool_uses,
        )

    if expectations:
        if tool_calls_raw := expectations.get("expected_tool_calls"):
            expected_tool_uses = [
                genai_types.FunctionCall(name=tc["name"], args=tc.get("args") or {})
                for tc in tool_calls_raw
            ]
            expected_invocation.intermediate_data = IntermediateData(
                tool_uses=expected_tool_uses,
            )

        reference_text = (
            expectations.get("expected_response")
            or expectations.get("context")
            or expectations.get("reference")
            or expectations.get("expected_output")
        )
        if reference_text:
            expected_invocation.final_response = genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text=str(reference_text))],
            )

    return actual_invocation, expected_invocation
