"""Utility functions for Google ADK integration."""

from __future__ import annotations

import asyncio
import concurrent.futures
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


def map_scorer_inputs_to_invocation(
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
) -> tuple[Any, Any]:
    """Convert MLflow scorer inputs to an ADK ``Invocation`` pair.

    Builds actual and expected ``google.adk.evaluation.eval_case.Invocation``
    objects from the raw scorer arguments. Tool call expectations are pulled
    from ``expectations["expected_tool_calls"]`` and placed in
    ``IntermediateData.tool_uses``.

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

    if expectations:
        if tool_calls_raw := expectations.get("expected_tool_calls"):
            expected_tool_uses = [
                genai_types.FunctionCall(name=tc["name"], args=tc.get("args", {}))
                for tc in tool_calls_raw
            ]
            expected_invocation.intermediate_data = IntermediateData(
                tool_uses=expected_tool_uses,
            )

        if actual_tool_calls_raw := expectations.get("actual_tool_calls"):
            actual_tool_uses = [
                genai_types.FunctionCall(name=tc["name"], args=tc.get("args", {}))
                for tc in actual_tool_calls_raw
            ]
            actual_invocation.intermediate_data = IntermediateData(
                tool_uses=actual_tool_uses,
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


def run_async(coro):
    """Run an async coroutine from a synchronous caller.

    Falls back to a worker thread when an event loop is already running
    (e.g., when called from inside an async test or Jupyter notebook).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(thread_name_prefix="google_adk_judge") as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)
