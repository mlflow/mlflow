from __future__ import annotations

from typing import Any

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.utils.trace_utils import (
    parse_inputs_to_str,
    parse_outputs_to_str,
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
)


def check_guardrails_installed():
    try:
        import guardrails  # noqa: F401
    except ImportError:
        raise MlflowException.invalid_parameter_value(
            "Guardrails AI scorers require the `guardrails-ai` package. "
            "Install it with: `pip install guardrails-ai`"
        )


def map_scorer_inputs_to_text(
    inputs: Any = None,
    outputs: Any = None,
    trace: Trace | None = None,
) -> str:
    """
    Convert MLflow scorer inputs to text for Guardrails AI validation.

    Guardrails AI validators operate on text strings. This function extracts
    and converts the relevant text from MLflow's scorer interface.

    Args:
        inputs: The input to evaluate
        outputs: The output to evaluate (primary target for validation)
        trace: MLflow trace for evaluation

    Returns:
        Text string to validate
    """
    if trace:
        inputs = resolve_inputs_from_trace(inputs, trace)
        outputs = resolve_outputs_from_trace(outputs, trace)

    # Guardrails validators primarily validate outputs (LLM responses)
    # Fall back to inputs if outputs not provided
    if outputs is not None:
        return parse_outputs_to_str(outputs)
    elif inputs is not None:
        return parse_inputs_to_str(inputs)
    else:
        raise MlflowException.invalid_parameter_value(
            "Guardrails AI scorers require either 'outputs' or 'inputs' to validate."
        )
