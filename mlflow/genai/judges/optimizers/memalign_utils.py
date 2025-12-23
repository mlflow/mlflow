"""Utility functions for MemAlign optimizer."""

import logging
import os
from typing import Any

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.genai.judges.base import Judge
from mlflow.genai.judges.optimizers.dspy_utils import (
    _sanitize_assessment_name,
    extract_expectations_from_trace,
    extract_request_from_trace,
    extract_response_from_trace,
)

_logger = logging.getLogger(__name__)


class Guideline(BaseModel):
    """A distilled guideline from feedback."""

    guideline_text: str
    source_ids: list[int] | None = None


class Guidelines(BaseModel):
    """Collection of guidelines."""

    guidelines: list[Guideline]


def load_memalign_template(template_name: str) -> Environment:
    """Load Jinja2 template from mlflow/genai/judges/prompts/.

    Args:
        template_name: Name of the template file to load

    Returns:
        Jinja2 Template object
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(os.path.dirname(current_dir), "prompts")

    if not os.path.exists(prompts_dir):
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    env = Environment(loader=FileSystemLoader(prompts_dir))
    return env.get_template(template_name)


def trace_to_feedback_record(trace: Trace, judge: Judge) -> dict[str, Any] | None:
    """Convert MLflow trace to MemAlign feedback record format.

    Args:
        trace: MLflow trace object
        judge: Judge instance to find assessments for

    Returns:
        Dict with both input and output fields, or None if conversion fails
    """
    try:
        judge_input_fields = judge.get_input_fields()

        judge_requires_inputs = any(field.name == "inputs" for field in judge_input_fields)
        judge_requires_outputs = any(field.name == "outputs" for field in judge_input_fields)
        judge_requires_expectations = any(
            field.name == "expectations" for field in judge_input_fields
        )

        request = extract_request_from_trace(trace)
        response = extract_response_from_trace(trace)
        expectations = extract_expectations_from_trace(trace)

        # Check for missing required fields
        if not request and judge_requires_inputs:
            _logger.warning(f"Missing required request in trace {trace.info.trace_id}")
            return None
        elif not response and judge_requires_outputs:
            _logger.warning(f"Missing required response in trace {trace.info.trace_id}")
            return None
        elif not expectations and judge_requires_expectations:
            _logger.warning(f"Missing required expectations in trace {trace.info.trace_id}")
            return None

        # Find human assessment for this judge
        expected_result = None

        if trace.info.assessments:
            sorted_assessments = sorted(
                trace.info.assessments,
                key=lambda a: (
                    a.create_time_ms if hasattr(a, "create_time_ms") and a.create_time_ms else 0
                ),
                reverse=True,
            )
            for assessment in sorted_assessments:
                sanitized_assessment_name = _sanitize_assessment_name(assessment.name)
                sanitized_judge_name = _sanitize_assessment_name(judge.name)
                if (
                    sanitized_assessment_name == sanitized_judge_name
                    and assessment.source.source_type == AssessmentSourceType.HUMAN
                ):
                    expected_result = assessment
                    break

        if not expected_result:
            _logger.warning(
                f"No human assessment found for judge '{judge.name}' in trace {trace.info.trace_id}"
            )
            return None

        if not expected_result.feedback:
            _logger.warning(f"No feedback found in assessment for trace {trace.info.trace_id}")
            return None

        # Create feedback record dict with both inputs and outputs
        feedback_record = {
            "_trace_id": trace.info.trace_id,  # Store trace ID for unalign
        }

        # Add input fields
        if judge_requires_inputs:
            feedback_record["inputs"] = request
        if judge_requires_outputs:
            feedback_record["outputs"] = response
        if judge_requires_expectations:
            feedback_record["expectations"] = expectations

        # Add output fields (result and rationale)
        feedback_record["result"] = str(expected_result.feedback.value).lower()
        feedback_record["rationale"] = expected_result.rationale or ""

        return feedback_record

    except Exception as e:
        _logger.error(f"Failed to create feedback record from trace: {e}")
        return None
