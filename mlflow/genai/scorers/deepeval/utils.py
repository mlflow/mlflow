"""Utility functions and constants for DeepEval integration."""

from __future__ import annotations

from typing import Any

from mlflow.entities.span import SpanAttributeKey, SpanType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.utils.trace_utils import (
    _to_dict,
    extract_retrieval_context_from_trace,
    parse_inputs_to_str,
    parse_outputs_to_str,
    resolve_expectations_from_trace,
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
)
from mlflow.tracing.utils.truncation import (
    _get_last_message,
    _get_text_content_from_message,
)

DEEPEVAL_NOT_INSTALLED_ERROR_MESSAGE = (
    "DeepEval metrics require the 'deepeval' package. Please install it with: pip install deepeval"
)

# Expectation keys for conversational test cases
EXPECTATION_KEY_SCENARIO = "scenario"
EXPECTATION_KEY_CHATBOT_ROLE = "chatbot_role"
EXPECTATION_KEY_EXPECTED_OUTCOME = "expected_outcome"
EXPECTATION_KEY_CONTEXT = "context"

try:
    from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn
    from deepeval.test_case import ToolCall as DeepEvalToolCall

    _DEEPEVAL_INSTALLED = True
except ImportError:
    _DEEPEVAL_INSTALLED = False


def _check_deepeval_installed():
    if not _DEEPEVAL_INSTALLED:
        raise MlflowException.invalid_parameter_value(DEEPEVAL_NOT_INSTALLED_ERROR_MESSAGE)


def _convert_to_deepeval_tool_calls(tool_call_dicts: list[dict[str, Any]]):
    """
    Convert tool call dicts to DeepEval ToolCall objects.

    Args:
        tool_call_dicts: List of dicts with tool call data

    Returns:
        List of DeepEval ToolCall objects
    """
    return [
        DeepEvalToolCall(
            name=tc_dict.get("name"),
            description=tc_dict.get("description"),
            reasoning=tc_dict.get("reasoning"),
            output=tc_dict.get("output"),
            input_parameters=tc_dict.get("input_parameters"),
        )
        for tc_dict in tool_call_dicts
    ]


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

    tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    if not tool_spans:
        return None

    return [
        DeepEvalToolCall(
            name=span.name,
            input_parameters=span.attributes.get(SpanAttributeKey.INPUTS),
            output=span.attributes.get(SpanAttributeKey.OUTPUTS),
        )
        for span in tool_spans
    ]


def _dict_to_kv_list(d: dict[str, Any]) -> list[str]:
    return [f"{k}: {v}" for k, v in d.items()]


def _extract_last_user_message_content(value: Any) -> str:
    """
    Extract the content of the last user message from inputs for multi-turn conversations.

    Args:
        value: Input value that may contain messages

    Returns:
        String content of the last user message
    """
    if isinstance(value, str):
        return value

    try:
        value_dict = _to_dict(value)
        messages = value_dict.get("messages")
        if messages and isinstance(messages, list) and len(messages) > 0:
            last_user_message = _get_last_message(messages, "user")
            return _get_text_content_from_message(last_user_message)
    except Exception:
        pass

    return parse_inputs_to_str(value)


def map_scorer_inputs_to_deepeval_test_case(
    metric_name: str,
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
):
    if trace:
        inputs = resolve_inputs_from_trace(inputs, trace)
        outputs = resolve_outputs_from_trace(outputs, trace)
        expectations = resolve_expectations_from_trace(expectations, trace)

    context = _dict_to_kv_list(expectations) if expectations else None
    additional_metadata = trace.info.trace_metadata if trace else {}
    tags = _dict_to_kv_list(trace.info.tags) if trace else []
    completion_time = trace.info.execution_duration / 1000 if trace else None

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
        input=parse_inputs_to_str(inputs),
        actual_output=parse_outputs_to_str(outputs),
        expected_output=expected_output,
        context=context,
        retrieval_context=retrieval_context,
        tools_called=tools_called,
        expected_tools=expected_tools,
        additional_metadata=additional_metadata,
        tags=tags,
        completion_time=completion_time,
    )


def map_session_to_deepeval_conversational_test_case(
    session: list[Trace],
    expectations: dict[str, Any] | None = None,
):
    """
    Convert list of MLflow traces (session) to DeepEval ConversationalTestCase.

    Args:
        session: List of traces in chronological order (same mlflow.trace.session ID)
        expectations: Optional conversation-level metadata. Use the EXPECTATION_KEY_* constants:
            - EXPECTATION_KEY_SCENARIO: Description of the test scenario
            - EXPECTATION_KEY_CHATBOT_ROLE: The chatbot's assigned role
            - EXPECTATION_KEY_EXPECTED_OUTCOME: The anticipated result
            - EXPECTATION_KEY_CONTEXT: Background information (str or list[str])

    Returns:
        ConversationalTestCase with turns populated from session traces
    """
    turns = []
    for trace in session:
        inputs = resolve_inputs_from_trace(None, trace)
        outputs = resolve_outputs_from_trace(None, trace)

        user_turn = Turn(
            role="user",
            content=_extract_last_user_message_content(inputs),
        )
        turns.append(user_turn)

        assistant_turn = Turn(
            role="assistant",
            content=parse_outputs_to_str(outputs),
        )
        turns.append(assistant_turn)

    kwargs = {}
    if expectations:
        if EXPECTATION_KEY_SCENARIO in expectations:
            kwargs[EXPECTATION_KEY_SCENARIO] = str(expectations[EXPECTATION_KEY_SCENARIO])
        if EXPECTATION_KEY_CHATBOT_ROLE in expectations:
            kwargs[EXPECTATION_KEY_CHATBOT_ROLE] = str(expectations[EXPECTATION_KEY_CHATBOT_ROLE])
        if EXPECTATION_KEY_EXPECTED_OUTCOME in expectations:
            kwargs[EXPECTATION_KEY_EXPECTED_OUTCOME] = str(
                expectations[EXPECTATION_KEY_EXPECTED_OUTCOME]
            )
        if EXPECTATION_KEY_CONTEXT in expectations:
            ctx = expectations[EXPECTATION_KEY_CONTEXT]
            if isinstance(ctx, list):
                kwargs[EXPECTATION_KEY_CONTEXT] = [str(c) for c in ctx]
            else:
                kwargs[EXPECTATION_KEY_CONTEXT] = [str(ctx)]

    return ConversationalTestCase(turns=turns, **kwargs)
