"""Utility functions for DSPy-based alignment optimizers."""

import logging
from typing import TYPE_CHECKING, Any, Optional

from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.judge_trace_utils import (
    extract_request_from_trace,
    extract_response_from_trace,
)
from mlflow.metrics.genai.model_utils import _parse_model_uri

# Import dspy - raise exception if not installed
try:
    import dspy
except ImportError:
    raise MlflowException("DSPy library is required but not installed")

if TYPE_CHECKING:
    from mlflow.genai.judges.base import Judge

_logger = logging.getLogger(__name__)


def convert_mlflow_uri_to_litellm(model_uri: str) -> str:
    """
    Convert MLflow model URI format to LiteLLM format.

    MLflow uses URIs like 'openai:/gpt-4' while LiteLLM expects 'openai/gpt-4'.

    Args:
        model_uri: MLflow model URI (e.g., 'openai:/gpt-4')

    Returns:
        LiteLLM-compatible model string (e.g., 'openai/gpt-4')
    """
    try:
        scheme, path = _parse_model_uri(model_uri)
        return f"{scheme}/{path}"
    except Exception as e:
        raise MlflowException(f"Failed to convert MLflow URI to LiteLLM format: {e}")


def trace_to_dspy_example(trace: Trace, judge_name: str) -> Optional["dspy.Example"]:
    """
    Convert MLflow trace to DSPy example format.

    Extracts:
    - inputs/outputs from trace spans
    - expected result from human assessments
    - rationale from assessment feedback

    Args:
        trace: MLflow trace object
        judge_name: Name of the judge to find assessments for

    Returns:
        DSPy example object or None if conversion fails
    """
    try:
        # Extract request and response from trace
        request = extract_request_from_trace(trace)
        response = extract_response_from_trace(trace)

        if request is None or response is None:
            _logger.warning(f"Missing request or response in trace {trace.info.trace_id}")
            return None

        # Find human assessment for this judge
        expected_result = None
        sanitized_judge_name = judge_name.lower().strip()

        if trace.info.assessments:
            # Sort assessments by creation time (most recent first) then process
            sorted_assessments = sorted(
                trace.info.assessments,
                key=lambda a: (
                    a.create_time_ms if hasattr(a, "create_time_ms") and a.create_time_ms else 0
                ),
                reverse=True,
            )
            for assessment in sorted_assessments:
                if (
                    assessment.name == sanitized_judge_name
                    and assessment.source.source_type == AssessmentSourceType.HUMAN
                ):
                    expected_result = assessment
                    break

        if not expected_result:
            _logger.warning(
                f"No human assessment found for judge '{judge_name}' in trace {trace.info.trace_id}"
            )
            return None

        if not expected_result.feedback:
            _logger.warning(f"No feedback found in assessment for trace {trace.info.trace_id}")
            return None

        # Create DSPy example
        example = dspy.Example(
            inputs=request,
            outputs=response,
            result=str(expected_result.feedback.value).lower(),
            rationale=expected_result.rationale if expected_result.rationale else "",
        )

        # Set inputs (what the model should use as input)
        return example.with_inputs("inputs", "outputs")

    except Exception as e:
        _logger.error(f"Failed to create DSPy example from trace: {e}")
        return None


def create_dspy_signature(judge: "Judge") -> "dspy.Signature":
    """
    Create DSPy signature for judge evaluation.

    Args:
        judge: The judge to create signature for

    Returns:
        DSPy signature object
    """
    try:
        # Build signature fields dictionary using the judge's field definitions
        signature_fields = {}

        # Get input fields from the judge
        input_fields = judge.get_input_fields()
        for field in input_fields:
            signature_fields[field.name] = (
                str,
                dspy.InputField(desc=field.description),
            )

        # Get output fields from the judge
        output_fields = judge.get_output_fields()
        for field in output_fields:
            signature_fields[field.name] = (
                str,
                dspy.OutputField(desc=field.description),
            )

        return dspy.make_signature(signature_fields, judge.instructions)

    except Exception as e:
        raise MlflowException(f"Failed to create DSPy signature: {e}")


def agreement_metric(example: "dspy.Example", pred: Any, trace: Any | None = None):
    """Simple agreement metric for judge optimization."""
    try:
        # Extract result from example and prediction
        expected = getattr(example, "result", None)
        predicted = getattr(pred, "result", None)

        if expected is None or predicted is None:
            return False

        # Normalize both to consistent format
        expected_norm = str(expected).lower().strip()
        predicted_norm = str(predicted).lower().strip()

        _logger.debug(f"expected_norm: {expected_norm}, predicted_norm: {predicted_norm}")

        return expected_norm == predicted_norm
    except Exception as e:
        _logger.warning(f"Error in agreement_metric: {e}")
        return False
