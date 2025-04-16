from typing import Any, Optional, Union

from mlflow.entities.assessment import (
    Assessment,
    AssessmentError,
    AssessmentValueType,
    Expectation,
    Feedback,
    experimental,
)
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient


@experimental
def log_expectation(
    trace_id: str,
    name: str,
    source: Union[str, AssessmentSource],
    value: AssessmentValueType,
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
        source: The source of the expectation assessment. Must be either an instance of
                :py:class:`~mlflow.entities.AssessmentSource` or a string that
                is a valid value in the
                :py:class:`~mlflow.entities.AssessmentSourceType` enum.
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
        from mlflow.entities.assessment import AssessmentSourceType

        mlflow.log_expectation(
            trace_id="1234",
            name="expected_answer",
            value=42,
            source=AssessmentSourceType.HUMAN,
        )

    """
    if value is None:
        raise MlflowException.invalid_parameter_value("Expectation value cannot be None.")

    return MlflowClient().log_assessment(
        trace_id=trace_id,
        name=name,
        source=_parse_source(source),
        expectation=Expectation(value) if value is not None else None,
        metadata=metadata,
        span_id=span_id,
    )


@experimental
def update_expectation(
    trace_id: str,
    assessment_id: str,
    name: Optional[str] = None,
    value: Optional[AssessmentValueType] = None,
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
        from mlflow.entities.assessment import AssessmentSourceType

        # Create an expectation with value 42.
        assessment = mlflow.log_expectation(
            trace_id="1234",
            name="expected_answer",
            value=42,
            source=AssessmentSourceType.HUMAN,
        )

        # Update the expectation with a new value 43.
        mlflow.update_expectation(
            trace_id="1234",
            assessment_id=assessment.assessment_id,
            value=43,
        )
    """
    return MlflowClient().update_assessment(
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
    return MlflowClient().delete_assessment(trace_id=trace_id, assessment_id=assessment_id)


@experimental
def log_feedback(
    trace_id: str,
    name: str,
    source: Union[str, AssessmentSource],
    value: Optional[AssessmentValueType] = None,
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
        source: The source of the feedback assessment. Must be either an instance of
                :py:class:`~mlflow.entities.AssessmentSource` or a string that
                is a valid value in the
                :py:class:`~mlflow.entities.AssessmentSourceType` enum.
        value: The value of the feedback.
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
        from mlflow.entities.assessment import AssessmentSourceType

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
    return MlflowClient().log_assessment(
        trace_id=trace_id,
        name=name,
        source=_parse_source(source),
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
    value: Optional[AssessmentValueType] = None,
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
        from mlflow.entities.assessment import AssessmentSourceType

        # Create a feedback with value 0.9.
        assessment = mlflow.log_feedback(
            trace_id="1234",
            name="faithfulness",
            value=0.9,
            source=AssessmentSourceType.LLM_JUDGE,
        )

        # Update the feedback with a new value 0.95.
        mlflow.update_feedback(
            trace_id="1234",
            assessment_id=assessment.assessment_id,
            value=0.95,
        )
    """
    return MlflowClient().update_assessment(
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
    return MlflowClient().delete_assessment(trace_id=trace_id, assessment_id=assessment_id)


def _parse_source(source: Union[str, AssessmentSource]) -> AssessmentSource:
    if source is None:
        raise MlflowException.invalid_parameter_value("`source` must be provided.")

    if isinstance(source, str):
        return AssessmentSource(source_type=source)
    elif isinstance(source, AssessmentSource):
        return source

    raise MlflowException.invalid_parameter_value(
        "Invalid source type. Must be one of str, AssessmentSource, or AssessmentSourceType."
    )
