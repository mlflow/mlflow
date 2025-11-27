"""Utility functions and constants for DeepEval integration."""

from __future__ import annotations

from typing import Any

from mlflow.entities.span import SpanAttributeKey, SpanType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.utils.trace_utils import (
    extract_retrieval_context_from_trace,
    parse_outputs_to_str,
    resolve_expectations_from_trace,
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
)

DEEPEVAL_NOT_INSTALLED_ERROR_MESSAGE = (
    "DeepEval metrics require the 'deepeval' package. Please install it with: pip install deepeval"
)


def _check_deepeval_installed():
    try:
        import deepeval  # noqa: F401
    except ImportError:
        raise MlflowException.invalid_parameter_value(DEEPEVAL_NOT_INSTALLED_ERROR_MESSAGE)


def _convert_to_deepeval_tool_calls(tool_call_dicts: list[dict[str, Any]]):
    """
    Convert tool call dicts to DeepEval ToolCall objects.

    Args:
        tool_call_dicts: List of dicts with tool call data

    Returns:
        List of DeepEval ToolCall objects
    """
    from deepeval.test_case import ToolCall as DeepEvalToolCall

    tool_calls = []
    for tc_dict in tool_call_dicts:
        tool_call = DeepEvalToolCall(
            name=tc_dict.get("name"),
            description=tc_dict.get("description"),
            reasoning=tc_dict.get("reasoning"),
            output=tc_dict.get("output"),
            input_parameters=tc_dict.get("input_parameters"),
        )
        tool_calls.append(tool_call)

    return tool_calls


def _extract_tool_calls_from_trace(trace: Trace):
    """
    Extract tool calls from trace spans with type TOOL.

    Args:
        trace: MLflow Trace object

    Returns:
        List of DeepEval ToolCall objects, or None if no tool calls found
    """
    if not trace:
        return None

    from deepeval.test_case import ToolCall as DeepEvalToolCall

    tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    if not tool_spans:
        return None

    tool_calls = []
    for span in tool_spans:
        tool_call = DeepEvalToolCall(
            name=span.name,
            input_parameters=span.attributes.get(SpanAttributeKey.INPUTS),
            output=span.attributes.get(SpanAttributeKey.OUTPUTS),
        )
        tool_calls.append(tool_call)

    return tool_calls or None


def _dict_to_kv_list(d: dict[str, Any]) -> list[str]:
    return [f"{k}: {v}" for k, v in d.items()]


def map_scorer_inputs_to_deepeval_test_case(
    metric_name: str,
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
):
    _check_deepeval_installed()
    from deepeval.test_case import LLMTestCase

    if trace:
        inputs = resolve_inputs_from_trace(inputs, trace)
        outputs = resolve_outputs_from_trace(outputs, trace)
        expectations = resolve_expectations_from_trace(expectations, trace)

    context = _dict_to_kv_list(expectations) if expectations else None
    additional_metadata = trace.info.trace_metadata if trace else {}
    tags = _dict_to_kv_list(trace.info.tags) if trace else []
    completion_time = trace.info.execution_duration * 1000 if trace else None

    expected_output = None
    expected_tools = None
    if expectations:
        if "expected_output" in expectations:
            expected_output = parse_outputs_to_str(expectations["expected_output"])

        if "expected_tool_calls" in expectations:
            expected_tool_calls = expectations["expected_tool_calls"]
            if isinstance(expected_tool_calls, list):
                expected_tools = _convert_to_deepeval_tool_calls(expected_tool_calls)

    tools_called = _extract_tool_calls_from_trace(trace) if trace else None

    span_id_to_context = extract_retrieval_context_from_trace(trace) if trace else {}
    retrieval_context = [str(context) for context in span_id_to_context.values()]

    return LLMTestCase(
        input=str(inputs),
        actual_output=str(outputs),
        expected_output=expected_output,
        context=context,
        retrieval_context=retrieval_context,
        tools_called=tools_called,
        expected_tools=expected_tools,
        additional_metadata=additional_metadata,
        tags=tags,
        completion_time=completion_time,
    )
