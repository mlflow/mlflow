from __future__ import annotations

from typing import Any

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.scorer_utils import parse_tool_call_expectations
from mlflow.genai.utils.trace_utils import (
    extract_retrieval_context_from_trace,
    extract_tools_called_from_trace,
    parse_inputs_to_str,
    parse_outputs_to_str,
    resolve_expectations_from_session,
    resolve_expectations_from_trace,
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
)

RAGAS_NOT_INSTALLED_ERROR_MESSAGE = (
    "RAGAS metrics require the 'ragas' package. Please install it with: pip install ragas"
)

try:
    from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
    from ragas.messages import AIMessage, HumanMessage, ToolCall

    _RAGAS_INSTALLED = True
except ImportError:
    _RAGAS_INSTALLED = False


def _check_ragas_installed():
    if not _RAGAS_INSTALLED:
        raise MlflowException.invalid_parameter_value(RAGAS_NOT_INSTALLED_ERROR_MESSAGE)


def map_scorer_inputs_to_ragas_sample(
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
    session: list[Trace] | None = None,
    is_agentic_or_multiturn: bool = False,
):
    """
    Convert MLflow scorer inputs to RAGAS sample format.

    For single-turn metrics, returns a SingleTurnSample.
    For agentic/multi-turn metrics, returns a MultiTurnSample.

    Args:
        inputs: The input to evaluate
        outputs: The output to evaluate
        expectations: Expected values and context for evaluation
        trace: MLflow trace for evaluation
        session: List of MLflow traces for multi-turn evaluation
        is_agentic_or_multiturn: Whether the metric is agentic or multiturn

    Returns:
        RAGAS SingleTurnSample or MultiTurnSample object
    """

    if is_agentic_or_multiturn:
        return _create_multi_turn_sample(
            expectations=expectations,
            trace=trace,
            session=session,
        )

    return _create_single_turn_sample(
        inputs=inputs,
        outputs=outputs,
        expectations=expectations,
        trace=trace,
    )


def _create_single_turn_sample(
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
):
    if trace:
        inputs = resolve_inputs_from_trace(inputs, trace)
        outputs = resolve_outputs_from_trace(outputs, trace)
        expectations = resolve_expectations_from_trace(expectations, trace)

    user_input = parse_inputs_to_str(inputs) if inputs is not None else None
    response = parse_outputs_to_str(outputs) if outputs is not None else None

    span_id_to_context = extract_retrieval_context_from_trace(trace) if trace else {}
    retrieved_contexts = [str(ctx) for contexts in span_id_to_context.values() for ctx in contexts]

    reference = None
    rubrics = None
    if expectations:
        # Extract rubrics if present (for InstanceSpecificRubrics metric)
        rubrics = expectations.get("rubrics")
        non_rubric_expectations = {
            key: value for key, value in expectations.items() if key != "rubrics"
        }
        if non_rubric_expectations:
            reference = ", ".join(str(value) for value in non_rubric_expectations.values())

    return SingleTurnSample(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts or None,
        reference=reference,
        reference_contexts=retrieved_contexts or None,
        rubrics=rubrics,
    )


def _create_multi_turn_sample(
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
    session: list[Trace] | None = None,
):
    if session:
        messages = map_session_to_ragas_messages(session, include_tool_calls=True)
        expectations = resolve_expectations_from_session(expectations, session)
    elif trace is not None:
        messages = map_session_to_ragas_messages([trace], include_tool_calls=True)
        expectations = resolve_expectations_from_trace(expectations, trace)
    else:
        messages = []

    reference_tool_calls = extract_reference_tool_calls_from_expectations(expectations)

    reference = None
    reference_topics = None
    if expectations and "expected_output" in expectations:
        reference = str(expectations["expected_output"])
    if expectations and "reference_topics" in expectations:
        reference_topics = expectations["reference_topics"]

    return MultiTurnSample(
        user_input=messages,
        reference=reference,
        reference_tool_calls=reference_tool_calls,
        reference_topics=reference_topics or [],
    )


def map_session_to_ragas_messages(
    session: list[Trace],
    *,
    include_tool_calls: bool = True,
) -> list[HumanMessage | AIMessage]:
    """
    Convert MLflow session (list of traces) to RAGAS message format.

    This converts MLflow traces into RAGAS HumanMessage, AIMessage
    objects suitable for agentic metrics evaluation.

    Args:
        session: List of traces from the same session in chronological order.
        include_tool_calls: If True, include tool call information from TOOL spans.

    Returns:
        List of RAGAS message objects (HumanMessage, AIMessage).
    """

    messages = []

    for trace in session:
        messages.extend(map_trace_to_ragas_messages(trace, include_tool_calls=include_tool_calls))

    return messages


def map_trace_to_ragas_messages(
    trace: Trace,
    *,
    include_tool_calls: bool = True,
) -> list[HumanMessage | AIMessage]:
    """
    Convert a single MLflow trace to RAGAS message format.

    This converts an MLflow trace into RAGAS HumanMessage, AIMessage
    objects suitable for agentic metrics evaluation.

    Args:
        trace: A single Trace object.
        include_tool_calls: If True, include tool call information from TOOL spans.

    Returns:
        List of RAGAS message objects (HumanMessage, AIMessage).
    """

    messages = []

    if inputs := resolve_inputs_from_trace(None, trace):
        user_content = parse_inputs_to_str(inputs)
        if user_content and user_content.strip():
            messages.append(HumanMessage(content=user_content))

    tool_calls = []
    if include_tool_calls:
        if tools_called := extract_tools_called_from_trace(trace):
            tool_calls.extend(
                ToolCall(name=tool.name, args=tool.arguments)
                for tool in tools_called
                if tool.name and tool.arguments
            )

    if outputs := resolve_outputs_from_trace(None, trace):
        assistant_content = parse_outputs_to_str(outputs)
        if assistant_content and assistant_content.strip():
            messages.append(
                AIMessage(
                    content=assistant_content,
                    tool_calls=tool_calls,
                )
            )

    return messages


def extract_reference_tool_calls_from_expectations(
    expectations: dict[str, Any] | None,
) -> list[ToolCall]:
    """
    Uses parse_tool_call_expectations to extract tool calls from expectations, then converts
    MLflow FunctionCall objects to RAGAS ToolCall format.

    Args:
        expectations: Expectations dict that may contain 'expected_tool_calls'.

    Returns:
        List of RAGAS ToolCall objects, or empty list if no tool calls are found.
    """
    function_calls = parse_tool_call_expectations(expectations)
    if not function_calls:
        return []

    return [
        ToolCall(name=fc.name, args=fc.arguments)
        for fc in function_calls
        if fc.name and fc.arguments
    ]


def create_mlflow_error_message_from_ragas_param(error_msg: str, metric_name: str) -> str:
    """
    Create an mlflow error message for missing RAGAS parameters.

    Args:
        error_msg: The error message from RAGAS
        metric_name: The name of the RAGAS metric

    Returns:
        An mlflow error message for missing RAGAS parameters
    """
    ragas_to_mlflow_param_mapping = {
        "user_input": "inputs",
        "response": "outputs",
        "reference_tool_calls": "expectations['expected_tool_calls']",
        "reference_contexts": "trace with retrieval spans",
        "reference": "expectations['expected_output']",
        "retrieved_contexts": "trace with retrieval spans",
        "rubrics": "expectations['rubrics']",
    }
    mlflow_param = error_msg
    for (
        ragas_param,
        corresponding_mlflow_param,
    ) in ragas_to_mlflow_param_mapping.items():
        if ragas_param in error_msg:
            mlflow_param = corresponding_mlflow_param
            break

    message_parts = [
        f"RAGAS metric '{metric_name}' requires '{mlflow_param}' parameter, which is missing."
    ]

    if ragas_param == "user_input":
        message_parts.append("Example: judge(inputs='What is MLflow?', outputs='...')")
    elif ragas_param == "response":
        message_parts.append("Example: judge(inputs='...', outputs='MLflow is a platform')")
    elif ragas_param == "reference":
        message_parts.append(
            "\nExample: judge(inputs='...', outputs='...', "
            "expectations={'expected_output': ...}) or log an expectation to the trace: "
            "mlflow.log_expectation(trace_id, name='expected_output', value=..., source=...)"
        )
    elif ragas_param in {"retrieved_contexts", "reference_contexts"}:
        message_parts.append(
            "\nMake sure your trace includes retrieval spans. "
            "Example: use @mlflow.trace(span_type=SpanType.RETRIEVER) decorator"
        )
    elif ragas_param == "rubrics":
        message_parts.append(
            "\nExample: judge(inputs='...', outputs='...', "
            "expectations={'rubrics': {'0': 'rubric for score 0', '1': 'rubric for score 1'}})"
        )
    elif ragas_param == "reference_tool_calls":
        message_parts.append(
            "\nExample: judge(inputs='...', outputs='...', "
            "expectations={'expected_tool_calls': ["
            "{'name': 'tool_name', 'arguments': {'arg1': 'value1'}}]})"
        )

    return " ".join(message_parts)
