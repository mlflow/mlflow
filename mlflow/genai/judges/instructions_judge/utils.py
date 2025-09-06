"""Utility functions for InstructionsJudge."""
from typing import Any, NamedTuple

from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.trace import Trace


class EvaluationFields(NamedTuple):
    """Fields extracted from a trace for evaluation."""
    inputs: dict[str, Any] | None = None
    outputs: dict[str, Any] | None = None
    expectations: dict[str, Any] | None = None


def extract_evaluation_fields_from_trace(trace: Trace) -> EvaluationFields:
    """
    Extract inputs, outputs, and human-set expectations from a trace.

    This function extracts evaluation data from a trace in a format suitable for
    field-based evaluation (as opposed to trace-based evaluation). It differs from
    other trace extraction utilities in MLflow in that:

    1. It returns raw dict values (not converted to strings like judge_trace_utils)
    2. It filters expectations to only include human-set ones (ground truth)
    3. It wraps non-dict values in {"value": ...} for consistency

    Args:
        trace: The trace object to extract from

    Returns:
        EvaluationFields namedtuple with inputs, outputs, and expectations attributes
    """
    inputs = None
    outputs = None
    expectations = None

    # Extract inputs/outputs from root span
    # Using _get_root_span() instead of spans[0] to properly find the span with no parent
    root_span = trace.data._get_root_span()
    if root_span:
        if root_span.inputs is not None:
            # Convert to dict if it's not already
            if isinstance(root_span.inputs, dict):
                inputs = root_span.inputs
            else:
                inputs = {"value": root_span.inputs}

        if root_span.outputs is not None:
            # Convert to dict if it's not already
            if isinstance(root_span.outputs, dict):
                outputs = root_span.outputs
            else:
                outputs = {"value": root_span.outputs}

    # Extract human-set expectations from trace assessments
    # Unlike _extract_expectations_from_trace in evaluation utils, we filter
    # to only include human-set expectations as ground truth
    expectation_assessments = trace.search_assessments(type="expectation")

    # Filter for human-set expectations only (ground truth)
    human_expectations = [
        exp for exp in expectation_assessments
        if exp.source and exp.source.source_type == AssessmentSourceType.HUMAN
    ]

    if human_expectations:
        # If there's only one expectation, use its value directly
        if len(human_expectations) == 1:
            exp_value = human_expectations[0].expectation.value
            # Convert to dict if it's not already
            if exp_value is not None and not isinstance(exp_value, dict):
                expectations = {"value": exp_value}
            else:
                expectations = exp_value
        else:
            # If there are multiple expectations, create a dict with expectation names as keys
            expectations = {}
            for exp in human_expectations:
                expectations[exp.name] = exp.expectation.value

    return EvaluationFields(inputs=inputs, outputs=outputs, expectations=expectations)
