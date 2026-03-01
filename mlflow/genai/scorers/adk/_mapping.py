"""Converts between MLflow and ADK data models."""

from __future__ import annotations

import logging
from typing import Any

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY

_logger = logging.getLogger(__name__)


def _text_to_content(text: str):
    """Convert a plain text string to a genai_types.Content object."""
    from google.genai import types as genai_types

    return genai_types.Content(parts=[genai_types.Part(text=text)])


def _extract_text_from_inputs(inputs: Any) -> str:
    """Extract text from MLflow inputs (dict, str, or other)."""
    if inputs is None:
        return ""
    if isinstance(inputs, dict):
        # Try common keys for the user query
        for key in ("query", "question", "input", "prompt", "request"):
            if key in inputs:
                return str(inputs[key])
        # Fall back to first string value
        for v in inputs.values():
            if isinstance(v, str):
                return v
        return str(inputs)
    return str(inputs)


def _extract_text_from_outputs(outputs: Any) -> str:
    """Extract text from MLflow outputs (str or other)."""
    if outputs is None:
        return ""
    if isinstance(outputs, dict):
        for key in ("response", "output", "answer", "result"):
            if key in outputs:
                return str(outputs[key])
        for v in outputs.values():
            if isinstance(v, str):
                return v
        return str(outputs)
    return str(outputs)


def map_to_adk_invocations(
    inputs: Any,
    outputs: Any,
    expectations: dict[str, Any] | None = None,
    trace: Any | None = None,
):
    """Map MLflow scorer inputs to ADK Invocation objects.

    Args:
        inputs: MLflow inputs (dict or string).
        outputs: MLflow outputs (string or dict).
        expectations: Optional expected values.
        trace: Optional MLflow trace.

    Returns:
        Tuple of (actual_invocation, expected_invocation_or_None).
    """
    from google.adk.evaluation.eval_case import IntermediateData, Invocation

    # Build actual invocation
    user_text = _extract_text_from_inputs(inputs)
    response_text = _extract_text_from_outputs(outputs)

    actual_invocation = Invocation(
        user_content=_text_to_content(user_text),
        final_response=_text_to_content(response_text) if response_text else None,
        intermediate_data=_extract_intermediate_data_from_trace(trace),
    )

    # Build expected invocation from expectations
    expected_invocation = None
    if expectations:
        expected_response = expectations.get("expected_response")
        expected_tool_calls = expectations.get("expected_tool_calls")
        rubrics = expectations.get("rubrics")

        expected_intermediate_data = None
        if expected_tool_calls:
            expected_intermediate_data = _parse_expected_tool_calls(
                expected_tool_calls
            )

        expected_final_response = None
        if expected_response:
            expected_final_response = _text_to_content(str(expected_response))

        expected_rubrics = None
        if rubrics:
            expected_rubrics = _parse_rubrics(rubrics)

        expected_invocation = Invocation(
            user_content=_text_to_content(user_text),
            final_response=expected_final_response,
            intermediate_data=expected_intermediate_data,
            rubrics=expected_rubrics,
        )

    return actual_invocation, expected_invocation


def _extract_intermediate_data_from_trace(trace: Any):
    """Extract tool calls and responses from an MLflow trace."""
    if trace is None:
        return None

    from google.genai import types as genai_types

    from google.adk.evaluation.eval_case import IntermediateData

    tool_calls = []
    tool_responses = []

    try:
        # Walk spans looking for tool/function spans
        if hasattr(trace, "data") and hasattr(trace.data, "spans"):
            for span in trace.data.spans:
                if _is_tool_span(span):
                    fc, fr = _span_to_function_call_and_response(span)
                    if fc:
                        tool_calls.append(fc)
                    if fr:
                        tool_responses.append(fr)
    except Exception as e:
        _logger.debug("Failed to extract tool data from trace: %s", e)

    if tool_calls or tool_responses:
        return IntermediateData(
            tool_uses=tool_calls,
            tool_responses=tool_responses,
        )
    return None


def _is_tool_span(span) -> bool:
    """Check if a trace span represents a tool call."""
    span_type = getattr(span, "span_type", None)
    if span_type and str(span_type).upper() in ("TOOL", "FUNCTION"):
        return True
    name = getattr(span, "name", "")
    if name and any(
        kw in name.lower() for kw in ("tool", "function_call", "function")
    ):
        return True
    return False


def _span_to_function_call_and_response(span):
    """Convert a trace span to ADK FunctionCall and FunctionResponse."""
    from google.genai import types as genai_types

    fc = None
    fr = None

    name = getattr(span, "name", "unknown_tool")
    inputs = getattr(span, "inputs", None)
    outputs = getattr(span, "outputs", None)

    args = {}
    if isinstance(inputs, dict):
        args = inputs
    elif inputs is not None:
        args = {"input": str(inputs)}

    fc = genai_types.FunctionCall(name=name, args=args)

    if outputs is not None:
        response_data = {}
        if isinstance(outputs, dict):
            response_data = outputs
        else:
            response_data = {"result": str(outputs)}
        fr = genai_types.FunctionResponse(name=name, response=response_data)

    return fc, fr


def _parse_expected_tool_calls(expected_tool_calls):
    """Parse expected tool calls from expectations dict.

    Supports:
    - List of strings (tool names): [\"search\", \"book\"]
    - List of dicts: [{\"name\": \"search\", \"args\": {...}}, ...]
    """
    from google.genai import types as genai_types

    from google.adk.evaluation.eval_case import IntermediateData

    function_calls = []
    if isinstance(expected_tool_calls, list):
        for tc in expected_tool_calls:
            if isinstance(tc, str):
                function_calls.append(
                    genai_types.FunctionCall(name=tc, args={})
                )
            elif isinstance(tc, dict):
                function_calls.append(
                    genai_types.FunctionCall(
                        name=tc.get("name", ""),
                        args=tc.get("args", {}),
                    )
                )

    return IntermediateData(tool_uses=function_calls)


def _parse_rubrics(rubrics):
    """Parse rubrics from expectations."""
    from google.adk.evaluation.eval_rubrics import Rubric, RubricContent

    parsed = []
    if isinstance(rubrics, list):
        for i, rubric in enumerate(rubrics):
            if isinstance(rubric, str):
                parsed.append(
                    Rubric(
                        rubric_id=f"rubric_{i}",
                        rubric_content=RubricContent(text=rubric),
                    )
                )
            elif isinstance(rubric, dict):
                parsed.append(
                    Rubric(
                        rubric_id=rubric.get("id", f"rubric_{i}"),
                        rubric_content=RubricContent(
                            text=rubric.get("content", rubric.get("text", ""))
                        ),
                        description=rubric.get("description"),
                    )
                )
    return parsed if parsed else None


def map_adk_result_to_feedback(
    name: str,
    eval_result,
    source: AssessmentSource,
    is_deterministic: bool = False,
) -> Feedback:
    """Convert an ADK EvaluationResult to an MLflow Feedback object.

    Args:
        name: The metric/scorer name.
        eval_result: ADK EvaluationResult.
        source: AssessmentSource for the feedback.
        is_deterministic: Whether the metric is deterministic.

    Returns:
        MLflow Feedback object.
    """
    from google.adk.evaluation.eval_metrics import EvalStatus

    score = eval_result.overall_score
    status = eval_result.overall_eval_status

    # Build rationale from per-invocation results
    rationale_parts = []
    if eval_result.per_invocation_results:
        for i, inv_result in enumerate(eval_result.per_invocation_results):
            status_str = inv_result.eval_status.name if inv_result.eval_status else "UNKNOWN"
            rationale_parts.append(
                f"Invocation {i}: score={inv_result.score}, status={status_str}"
            )
            if inv_result.rubric_scores:
                for rs in inv_result.rubric_scores:
                    rationale_parts.append(
                        f"  Rubric {rs.rubric_id}: score={rs.score}"
                        + (f", rationale={rs.rationale}" if rs.rationale else "")
                    )

    rationale = "\n".join(rationale_parts) if rationale_parts else None

    metadata = {
        FRAMEWORK_METADATA_KEY: "adk",
        "eval_status": status.name if status else "UNKNOWN",
    }
    if score is not None:
        metadata["score"] = str(score)

    # Use the score as the Feedback value (float)
    # If score is None, fall back to pass/fail based on status
    if score is not None:
        value = score
    elif status == EvalStatus.PASSED:
        value = 1.0
    elif status == EvalStatus.FAILED:
        value = 0.0
    else:
        value = None

    if value is not None:
        return Feedback(
            name=name,
            value=value,
            rationale=rationale,
            source=source,
            metadata=metadata,
        )
    else:
        return Feedback(
            name=name,
            error="Evaluation did not produce a score",
            source=source,
            metadata=metadata,
        )
