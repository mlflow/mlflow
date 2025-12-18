from __future__ import annotations

from typing import Any

from mlflow.entities.trace import Trace
from mlflow.genai.utils.trace_utils import (
    extract_retrieval_context_from_trace,
    parse_inputs_to_str,
    parse_outputs_to_str,
    resolve_expectations_from_trace,
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
)


def map_scorer_inputs_to_ragas_sample(
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
):
    """
    Convert MLflow scorer inputs to RAGAS SingleTurnSample format.

    Args:
        inputs: The input to evaluate
        outputs: The output to evaluate
        expectations: Expected values and context for evaluation
        trace: MLflow trace for evaluation

    Returns:
        RAGAS SingleTurnSample object
    """
    from ragas.dataset_schema import SingleTurnSample

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
        # Extract rubrics if present (for InstanceRubrics metric)
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


def create_mlflow_error_message_from_ragas_param(ragas_param: str, metric_name: str) -> str:
    """
    Create an mlflow error message for missing RAGAS parameters.

    Args:
        ragas_param: The RAGAS parameter name that is missing
        metric_name: The name of the RAGAS metric

    Returns:
        An mlflow error message for missing RAGAS parameters
    """
    ragas_to_mlflow_param_mapping = {
        "user_input": "inputs",
        "response": "outputs",
        "reference": "expectations['expected_output']",
        "retrieved_contexts": "trace with retrieval spans",
        "reference_contexts": "trace with retrieval spans",
        "rubrics": "expectations['rubrics']",
    }
    mlflow_param = ragas_to_mlflow_param_mapping.get(ragas_param, ragas_param)

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
    elif ragas_param in ["retrieved_contexts", "reference_contexts"]:
        message_parts.append(
            "\nMake sure your trace includes retrieval spans. "
            "Example: use @mlflow.trace(span_type=SpanType.RETRIEVER) decorator"
        )
    elif ragas_param == "rubrics":
        message_parts.append(
            "\nExample: judge(inputs='...', outputs='...', "
            "expectations={'rubrics': {'0': 'rubric for score 0', '1': 'rubric for score 1'}})"
        )

    return " ".join(message_parts)
