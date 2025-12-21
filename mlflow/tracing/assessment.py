from typing import Any

from mlflow.entities.assessment import (
    DEFAULT_FEEDBACK_NAME,
    Assessment,
    AssessmentError,
    Expectation,
    Feedback,
    FeedbackValueType,
)
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.exceptions import MlflowException
from mlflow.tracing.client import TracingClient


def get_assessment(trace_id: str, assessment_id: str) -> Assessment:
    """
    Get an assessment entity from the backend store.

    Args:
        trace_id: The ID of the trace.
        assessment_id: The ID of the assessment to get.

    Returns:
        :py:class:`~mlflow.entities.Assessment`: The Assessment object.
    """
    return TracingClient().get_assessment(trace_id, assessment_id)


def log_assessment(trace_id: str, assessment: Assessment) -> Assessment:
    """
    Logs an assessment to a Trace. The assessment can be an expectation or a feedback.

    - Expectation: A label that represents the expected value for a particular operation.
        For example, an expected answer for a user question from a chatbot.
    - Feedback: A label that represents the feedback on the quality of the operation.
        Feedback can come from different sources, such as human judges, heuristic scorers,
        or LLM-as-a-Judge.

    The following code annotates a trace with a feedback provided by LLM-as-a-Judge.

    .. code-block:: python

        import mlflow
        from mlflow.entities import Feedback

        feedback = Feedback(
            name="faithfulness",
            value=0.9,
            rationale="The model is faithful to the input.",
            metadata={"model": "gpt-4o-mini"},
        )

        mlflow.log_assessment(trace_id="1234", assessment=feedback)

    The following code annotates a trace with human-provided ground truth with source information.
    When the source is not provided, the default source is set to "default" with type "HUMAN"

    .. code-block:: python

        import mlflow
        from mlflow.entities import AssessmentSource, AssessmentSourceType, Expectation

        # Specify the annotator information as a source.
        source = AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id="john@example.com",
        )

        expectation = Expectation(
            name="expected_answer",
            value=42,
            source=source,
        )

        mlflow.log_assessment(trace_id="1234", assessment=expectation)

    The expectation value can be any JSON-serializable value. For example, you may
     record the full LLM message as the expectation value.

    .. code-block:: python

        import mlflow
        from mlflow.entities.assessment import Expectation

        expectation = Expectation(
            name="expected_message",
            # Full LLM message including expected tool calls
            value={
                "role": "assistant",
                "content": "The answer is 42.",
                "tool_calls": [
                    {
                        "id": "1234",
                        "type": "function",
                        "function": {"name": "add", "arguments": "40 + 2"},
                    }
                ],
            },
        )
        mlflow.log_assessment(trace_id="1234", assessment=expectation)

    You can also log an error information during the feedback generation process. To do so,
    provide an instance of :py:class:`~mlflow.entities.AssessmentError` to the `error`
    parameter, and leave the `value` parameter as `None`.

    .. code-block:: python

        import mlflow
        from mlflow.entities import AssessmentError, Feedback

        error = AssessmentError(
            error_code="RATE_LIMIT_EXCEEDED",
            error_message="Rate limit for the judge exceeded.",
        )

        feedback = Feedback(
            trace_id="1234",
            name="faithfulness",
            error=error,
        )
        mlflow.log_assessment(trace_id="1234", assessment=feedback)


    """
    return TracingClient().log_assessment(trace_id, assessment)


def log_expectation(
    *,
    trace_id: str,
    name: str,
    value: Any,
    source: AssessmentSource | None = None,
    metadata: dict[str, Any] | None = None,
    span_id: str | None = None,
) -> Assessment:
    """
    Logs an expectation (e.g. ground truth label) to a Trace. This API only takes keyword arguments.

    Args:
        trace_id: The ID of the trace.
        name: The name of the expectation assessment e.g., "expected_answer
        value: The value of the expectation. It can be any JSON-serializable value.
        source: The source of the expectation assessment. Must be an instance of
                :py:class:`~mlflow.entities.AssessmentSource`. If not provided,
                default to HUMAN source type.
        metadata: Additional metadata for the expectation.
        span_id: The ID of the span associated with the expectation, if it needs be
                associated with a specific span in the trace.

    Returns:
        :py:class:`~mlflow.entities.Assessment`: The created expectation assessment.

    Examples:
        .. code-block:: python

            import mlflow
            from mlflow.entities import AssessmentSource, AssessmentSourceType

            # Log simple expected answer
            expectation = mlflow.log_expectation(
                trace_id="tr-1234567890abcdef",
                name="expected_answer",
                value="The capital of France is Paris.",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="annotator@company.com"
                ),
                metadata={"question_type": "factual", "difficulty": "easy"},
            )

            # Log expected classification label
            mlflow.log_expectation(
                trace_id="tr-1234567890abcdef",
                name="expected_category",
                value="positive",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="data_labeler_001"
                ),
                metadata={"labeling_session": "batch_01", "confidence": 0.95},
            )
    """
    assessment = Expectation(
        name=name,
        source=source,
        value=value,
        metadata=metadata,
        span_id=span_id,
    )
    return TracingClient().log_assessment(trace_id, assessment)


def update_assessment(
    trace_id: str,
    assessment_id: str,
    assessment: Assessment,
) -> Assessment:
    """
    Updates an existing expectation (ground truth) in a Trace.

    Args:
        trace_id: The ID of the trace.
        assessment_id: The ID of the expectation or feedback assessment to update.
        assessment: The updated assessment.

    Returns:
        :py:class:`~mlflow.entities.Assessment`: The updated feedback or expectation assessment.

    Example:

    The following code updates an existing expectation with a new value.
    To update other fields, provide the corresponding parameters.

    .. code-block:: python

        import mlflow
        from mlflow.entities import Expectation, ExpectationValue

        # Create an expectation with value 42.
        response = mlflow.log_assessment(
            trace_id="1234",
            assessment=Expectation(name="expected_answer", value=42),
        )
        assessment_id = response.assessment_id

        # Update the expectation with a new value 43.
        mlflow.update_assessment(
            trace_id="1234",
            assessment_id=assessment.assessment_id,
            assessment=Expectation(name="expected_answer", value=43),
        )
    """
    return TracingClient().update_assessment(
        assessment_id=assessment_id,
        trace_id=trace_id,
        assessment=assessment,
    )


def delete_assessment(trace_id: str, assessment_id: str):
    """
    Deletes an assessment associated with a trace.

    Args:
        trace_id: The ID of the trace.
        assessment_id: The ID of the assessment to delete.
    """
    return TracingClient().delete_assessment(trace_id=trace_id, assessment_id=assessment_id)


def log_feedback(
    *,
    trace_id: str,
    name: str = DEFAULT_FEEDBACK_NAME,
    value: FeedbackValueType | None = None,
    source: AssessmentSource | None = None,
    error: Exception | AssessmentError | None = None,
    rationale: str | None = None,
    metadata: dict[str, Any] | None = None,
    span_id: str | None = None,
) -> Assessment:
    """
    Logs feedback to a Trace. This API only takes keyword arguments.

    Args:
        trace_id: The ID of the trace.
        name: The name of the feedback assessment e.g., "faithfulness". Defaults to
            "feedback" if not provided.
        value: The value of the feedback. Must be one of the following types:
            - float
            - int
            - str
            - bool
            - list of values of the same types as above
            - dict with string keys and values of the same types as above
        source: The source of the feedback assessment. Must be an instance of
                :py:class:`~mlflow.entities.AssessmentSource`. If not provided, defaults to
                CODE source type
        error: An error object representing any issues encountered while computing the
            feedback, e.g., a timeout error from an LLM judge. Accepts an exception
            object, or an :py:class:`~mlflow.entities.AssessmentError` object. Either
            this or `value` must be provided.
        rationale: The rationale / justification for the feedback.
        metadata: Additional metadata for the feedback.
        span_id: The ID of the span associated with the feedback, if it needs be
                associated with a specific span in the trace.

    Returns:
        :py:class:`~mlflow.entities.Assessment`: The created feedback assessment.

    Examples:
        .. code-block:: python

            import mlflow
            from mlflow.entities import AssessmentSource, AssessmentSourceType

            # Log simple feedback score
            feedback = mlflow.log_feedback(
                trace_id="tr-1234567890abcdef",
                name="relevance",
                value=0.9,
                source=AssessmentSource(
                    source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4"
                ),
                rationale="Response directly addresses the user's question",
            )

            # Log detailed feedback with structured data
            mlflow.log_feedback(
                trace_id="tr-1234567890abcdef",
                name="quality_metrics",
                value={"accuracy": 0.95, "completeness": 0.88, "clarity": 0.92, "overall": 0.92},
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="expert_evaluator"
                ),
                rationale="High accuracy and clarity, slightly incomplete coverage",
            )
    """
    assessment = Feedback(
        name=name,
        source=source,
        value=value,
        error=error,
        rationale=rationale,
        metadata=metadata,
        span_id=span_id,
    )
    return TracingClient().log_assessment(trace_id, assessment)


def override_feedback(
    *,
    trace_id: str,
    assessment_id: str,
    value: FeedbackValueType,
    rationale: str | None = None,
    source: AssessmentSource | None = None,
    metadata: dict[str, Any] | None = None,
) -> Assessment:
    """
    Overrides an existing feedback assessment with a new assessment. This API
    logs a new assessment with the `overrides` field set to the provided assessment ID.
    The original assessment will be marked as invalid, but will otherwise be unchanged.
    This is useful when you want to correct an assessment generated by an LLM judge,
    but want to preserve the original assessment for future judge fine-tuning.

    If you want to mutate an assessment in-place, use :py:func:`update_assessment` instead.

    Args:
        trace_id: The ID of the trace.
        assessment_id: The ID of the assessment to override.
        value: The new value of the assessment.
        rationale: The rationale of the new assessment.
        source: The source of the new assessment.
        metadata: Additional metadata for the new assessment.

    Returns:
        :py:class:`~mlflow.entities.Assessment`: The created assessment.

    Examples:
        .. code-block:: python

            import mlflow
            from mlflow.entities import AssessmentSource, AssessmentSourceType

            # First, log an initial LLM-generated feedback as a simulation
            llm_feedback = mlflow.log_feedback(
                trace_id="tr-1234567890abcdef",
                name="relevance",
                value=0.6,
                source=AssessmentSource(
                    source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4"
                ),
                rationale="Response partially addresses the question",
            )

            # Later, a human reviewer disagrees and wants to override
            corrected_assessment = mlflow.override_feedback(
                trace_id="tr-1234567890abcdef",
                assessment_id=llm_feedback.assessment_id,
                value=0.9,
                rationale="Response fully addresses the question with good examples",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="expert_reviewer@company.com"
                ),
                metadata={
                    "override_reason": "LLM underestimated relevance",
                    "review_date": "2024-01-15",
                    "confidence": "high",
                },
            )
    """
    old_assessment = get_assessment(trace_id, assessment_id)
    if not isinstance(old_assessment, Feedback):
        raise MlflowException.invalid_parameter_value(
            f"The assessment with ID {assessment_id} is not a feedback assessment."
        )

    new_assessment = Feedback(
        name=old_assessment.name,
        span_id=old_assessment.span_id,
        value=value,
        rationale=rationale,
        source=source,
        metadata=metadata,
        overrides=old_assessment.assessment_id,
    )

    return TracingClient().log_assessment(trace_id, new_assessment)
