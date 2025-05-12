from typing import Any, Optional

from mlflow.entities.assessment import (
    Assessment,
    AssessmentError,
    Expectation,
    Feedback,
    FeedbackValueType,
    experimental,
)
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.exceptions import MlflowException
from mlflow.tracing.client import TracingClient


@experimental
def log_expectation(
    trace_id: str,
    name: str,
    source: AssessmentSource,
    value: Any,
    metadata: Optional[dict[str, Any]] = None,
    span_id: Optional[str] = None,
) -> Assessment:
    """
    .. important::

        This API is currently only available for `Databricks Managed MLflow <https://www.databricks.com/product/managed-mlflow>`_.

    Logs an expectation (e.g. ground truth label) to a Trace.

    Args:
        trace_id: The ID of the trace.
        name: The name of the expectation assessment e.g., "expected_answer
        source: The source of the expectation assessment. Must be an instance of
                :py:class:`~mlflow.entities.AssessmentSource`.
        value: The value of the expectation. It can be any JSON-serializable value.
        metadata: Additional metadata for the expectation.
        span_id: The ID of the span associated with the expectation, if it needs be
                associated with a specific span in the trace.

    Returns:
        :py:class:`~mlflow.entities.Assessment`: The created expectation assessment.

    Example:

    The following code annotates a trace with human-provided ground truth.

    .. code-block:: python

        import mlflow
        from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType

        # Specify the annotator information as a source.
        source = AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id="john@example.com",
        )

        mlflow.log_expectation(
            trace_id="1234",
            name="expected_answer",
            value=42,
            source=source,
        )

    The expectation value can be any JSON-serializable value. For example, you may
     record the full LLM message as the expectation value.

    .. code-block:: python

        import mlflow
        from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType

        mlflow.log_expectation(
            trace_id="1234",
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
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN,
                source_id="john@example.com",
            ),
        )
    """
    if value is None:
        raise MlflowException.invalid_parameter_value("Expectation value cannot be None.")

    if not isinstance(source, AssessmentSource):
        raise MlflowException.invalid_parameter_value(
            f"`source` must be an instance of `AssessmentSource`. Got {type(source)} instead."
        )

    return TracingClient().log_assessment(
        trace_id=trace_id,
        name=name,
        source=source,
        expectation=Expectation(value) if value is not None else None,
        metadata=metadata,
        span_id=span_id,
    )


@experimental
def update_expectation(
    trace_id: str,
    assessment_id: str,
    name: Optional[str] = None,
    value: Any = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Assessment:
    """
    .. important::

        This API is currently only available for `Databricks Managed MLflow <https://www.databricks.com/product/managed-mlflow>`_.

    Updates an existing expectation (ground truth) in a Trace.

    Args:
        trace_id: The ID of the trace.
        assessment_id: The ID of the expectation assessment to update.
        name: The updated name of the expectation. Specify only when updating the name.
        value: The updated value of the expectation. Specify only when updating the value.
        metadata: Additional metadata for the expectation. Specify only when updating the metadata.

    Returns:
        :py:class:`~mlflow.entities.Assessment`: The updated feedback assessment.

    Example:

    The following code updates an existing expectation with a new value.
    To update other fields, provide the corresponding parameters.

    .. code-block:: python

        import mlflow
        from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType

        # Create an expectation with value 42.
        assessment = mlflow.log_expectation(
            trace_id="1234",
            name="expected_answer",
            value=42,
            # Original annotator
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN,
                source_id="bob@example.com",
            ),
        )

        # Update the expectation with a new value 43.
        mlflow.update_expectation(
            trace_id="1234", assessment_id=assessment.assessment_id, value=43
        )
    """
    return TracingClient().update_assessment(
        assessment_id=assessment_id,
        trace_id=trace_id,
        name=name,
        expectation=Expectation(value) if value is not None else None,
        metadata=metadata,
    )


@experimental
def delete_expectation(trace_id: str, assessment_id: str):
    """
    .. important::

        This API is currently only available for `Databricks Managed MLflow <https://www.databricks.com/product/managed-mlflow>`_.

    Deletes an expectation associated with a trace.

    Args:
        trace_id: The ID of the trace.
        assessment_id: The ID of the expectation assessment to delete.
    """
    return TracingClient().delete_assessment(trace_id=trace_id, assessment_id=assessment_id)


@experimental
def log_feedback(
    trace_id: str,
    name: str,
    source: AssessmentSource,
    value: Optional[FeedbackValueType] = None,
    error: Optional[AssessmentError] = None,
    rationale: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    span_id: Optional[str] = None,
) -> Assessment:
    """
    .. important::

        This API is currently only available for `Databricks Managed MLflow <https://www.databricks.com/product/managed-mlflow>`_.

    Logs feedback to a Trace.

    Args:
        trace_id: The ID of the trace.
        name: The name of the feedback assessment e.g., "faithfulness"
        source: The source of the feedback assessment. Must be an instance of
                :py:class:`~mlflow.entities.AssessmentSource`.
        value: The value of the feedback. Must be one of the following types:
            - float
            - int
            - str
            - bool
            - list of values of the same types as above
            - dict with string keys and values of the same types as above
        error: An error object representing any issues encountered while computing the
            feedback, e.g., a timeout error from an LLM judge. Either this or `value`
            must be provided.
        rationale: The rationale / justification for the feedback.
        metadata: Additional metadata for the feedback.
        span_id: The ID of the span associated with the feedback, if it needs be
                associated with a specific span in the trace.

    Returns:
        :py:class:`~mlflow.entities.Assessment`: The created feedback assessment.

    Example:

    The following code annotates a trace with a feedback provided by LLM-as-a-Judge.

    .. code-block:: python

        import mlflow
        from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType

        source = AssessmentSource(
            source_type=Type.LLM_JUDGE,
            source_id="faithfulness-judge",
        )

        mlflow.log_feedback(
            trace_id="1234",
            name="faithfulness",
            source=source,
            value=0.9,
            rationale="The model is faithful to the input.",
            metadata={"model": "gpt-4o-mini"},
        )

    You can also log an error information during the feedback generation process. To do so,
    provide an instance of :py:class:`~mlflow.entities.AssessmentError` to the `error`
    parameter, and leave the `value` parameter as `None`.

    .. code-block:: python

        import mlflow
        from mlflow.entities.assessment import AssessmentError

        source = AssessmentSource(
            source_type=Type.LLM_JUDGE,
            source_id="faithfulness-judge",
        )

        error = AssessmentError(
            error_code="RATE_LIMIT_EXCEEDED",
            error_message="Rate limit for the judge exceeded.",
        )

        mlflow.log_feedback(
            trace_id="1234",
            name="faithfulness",
            source=source,
            error=error,
        )

    """
    if value is None and error is None:
        raise MlflowException.invalid_parameter_value("Either `value` or `error` must be provided.")

    if not isinstance(source, AssessmentSource):
        raise MlflowException.invalid_parameter_value(
            f"`source` must be an instance of `AssessmentSource`. Got {type(source)} instead."
        )

    return TracingClient().log_assessment(
        trace_id=trace_id,
        name=name,
        source=source,
        feedback=Feedback(value, error),
        rationale=rationale,
        metadata=metadata,
        span_id=span_id,
    )


@experimental
def update_feedback(
    trace_id: str,
    assessment_id: str,
    name: Optional[str] = None,
    value: Optional[FeedbackValueType] = None,
    rationale: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Assessment:
    """
    .. important::

        This API is currently only available for `Databricks Managed MLflow <https://www.databricks.com/product/managed-mlflow>`_.

    Updates an existing feedback in a Trace.


    Args:
        trace_id: The ID of the trace.
        assessment_id: The ID of the feedback assessment to update.
        name: The updated name of the feedback. Specify only when updating the name.
        value: The updated value of the feedback. Specify only when updating the value.
        rationale: The updated rationale of the feedback. Specify only when updating the rationale.
        metadata: Additional metadata for the feedback. Specify only when updating the metadata.

    Returns:
        :py:class:`~mlflow.entities.Assessment`: The updated feedback assessment.

    Example:

    The following code updates an existing feedback with a new value.
    To update other fields, provide the corresponding parameters.

    .. code-block:: python

        import mlflow
        from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType

        # Create a feedback with value 0.9.
        assessment = mlflow.log_feedback(
            trace_id="1234",
            name="faithfulness",
            value=0.9,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id="gpt-4o-mini",
            ),
        )

        # Update the feedback with a new value 0.95.
        mlflow.update_feedback(
            trace_id="1234",
            assessment_id=assessment.assessment_id,
            value=0.95,
        )
    """
    return TracingClient().update_assessment(
        trace_id=trace_id,
        assessment_id=assessment_id,
        name=name,
        feedback=Feedback(value) if value is not None else None,
        rationale=rationale,
        metadata=metadata,
    )


@experimental
def delete_feedback(trace_id: str, assessment_id: str):
    """
    .. important::

        This API is currently only available for `Databricks Managed MLflow <https://www.databricks.com/product/managed-mlflow>`_.

    Deletes feedback associated with a trace.

    Args:
        trace_id: The ID of the trace.
        assessment_id: The ID of the feedback assessment to delete.
    """
    return TracingClient().delete_assessment(trace_id=trace_id, assessment_id=assessment_id)
