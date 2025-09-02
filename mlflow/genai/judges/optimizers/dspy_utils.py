"""Utility functions for DSPy-based alignment optimizers."""

import logging
from typing import Any

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.judge_trace_utils import (
    extract_request_from_trace,
    extract_response_from_trace,
)

logger = logging.getLogger(__name__)


def trace_to_dspy_example(trace: Trace, judge_name: str) -> Any | None:
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
        # Import dspy here to allow graceful failure
        import dspy

        # Extract request and response from trace
        request = extract_request_from_trace(trace)
        response = extract_response_from_trace(trace)

        if not request or not response:
            logger.warning(f"Missing request or response in trace {trace.info.trace_id}")
            return None

        # Find human assessment for this judge
        expected_result = None
        sanitized_judge_name = judge_name.lower().strip()

        if trace.info.assessments:
            for assessment in trace.info.assessments:
                if (
                    assessment.name == sanitized_judge_name
                    and assessment.source.source_type == "HUMAN"
                ):
                    expected_result = assessment
                    break

        if not expected_result:
            logger.warning(
                f"No human assessment found for judge '{judge_name}' in trace {trace.info.trace_id}"
            )
            return None

        if not expected_result.feedback:
            logger.warning(f"No feedback found in assessment for trace {trace.info.trace_id}")
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

    except ImportError:
        raise MlflowException("DSPy library is required but not installed")
    except Exception as e:
        logger.error(f"Failed to create DSPy example from trace: {e}")
        return None


def create_dspy_signature(judge) -> Any:
    """
    Create DSPy signature for judge evaluation.

    Args:
        judge: The judge to create signature for

    Returns:
        DSPy signature object
    """
    try:
        import dspy

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

    except ImportError:
        raise MlflowException("DSPy library is required but not installed")


def agreement_metric(example, pred, trace=None):
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

        return expected_norm == predicted_norm
    except Exception:
        # Return 0 for any errors
        return False
