"""Utilities for session-level (multi-turn) evaluation."""

import traceback
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_error import AssessmentError
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import Scorer
from mlflow.tracing.constant import TraceMetadataKey

if TYPE_CHECKING:
    from mlflow.genai.evaluation.entities import EvalItem


def classify_scorers(scorers: list[Scorer]) -> tuple[list[Scorer], list[Scorer]]:
    """
    Separate scorers into single-turn and multi-turn categories.

    Args:
        scorers: List of scorer instances.

    Returns:
        tuple: (single_turn_scorers, multi_turn_scorers)
    """
    single_turn_scorers = []
    multi_turn_scorers = []

    for scorer in scorers:
        if scorer.is_session_level_scorer:
            multi_turn_scorers.append(scorer)
        else:
            single_turn_scorers.append(scorer)

    return single_turn_scorers, multi_turn_scorers


def group_traces_by_session(
    eval_items: list["EvalItem"],
) -> dict[str, list["EvalItem"]]:
    """
    Group evaluation items containing traces by session_id.

    Args:
        eval_items: List of EvalItem objects.

    Returns:
        dict: {session_id: [eval_item, ...]} where eval items are grouped by session.
              Only items with traces that have a session_id are included in the output.
    """
    session_groups = defaultdict(list)

    for item in eval_items:
        if not getattr(item, "trace", None):
            continue

        trace_metadata = item.trace.info.trace_metadata

        if session_id := trace_metadata.get(TraceMetadataKey.TRACE_SESSION):
            session_groups[session_id].append(item)

    return dict(session_groups)


def get_first_trace_in_session(session_items: list["EvalItem"]) -> "EvalItem":
    """
    Find the chronologically first trace in a session based on request_time.

    Args:
        session_items: List of EvalItem objects from the same session.

    Returns:
        EvalItem: The eval item with the earliest trace in chronological order.
    """
    return min(session_items, key=lambda x: x.trace.info.request_time)


def evaluate_session_level_scorers(
    multi_turn_scorers: list[Scorer],
    session_groups: dict[str, list["EvalItem"]],
) -> dict[str, list[Feedback]]:
    """
    Evaluate multi-turn scorers on grouped sessions.

    Args:
        multi_turn_scorers: List of multi-turn scorer instances
        session_groups: Dict of {session_id: [eval_item, ...]}

    Returns:
        dict: Dictionary mapping first trace ID of each session to its assessments list.
              Structure: {first_trace_id: [feedback1, feedback2, ...], ...}
    """
    # Import here to avoid circular dependency
    from mlflow.genai.evaluation.utils import (
        make_code_type_assessment_source,
        standardize_scorer_value,
    )

    multi_turn_assessments = defaultdict(list)

    for session_id, session_items in session_groups.items():
        # Get the first trace to store assessments
        first_item = get_first_trace_in_session(session_items)
        first_trace_id = first_item.trace.info.trace_id

        # Extract just the traces for scorer evaluation
        session_traces = [item.trace for item in session_items]

        # Evaluate each multi-turn scorer
        for scorer in multi_turn_scorers:
            try:
                # Call scorer with session_traces parameter
                value = scorer.run(session=session_traces)
                feedbacks = standardize_scorer_value(scorer.name, value)

                # Add session_id to metadata for each feedback
                for feedback in feedbacks:
                    if feedback.metadata is None:
                        feedback.metadata = {}
                    feedback.metadata[TraceMetadataKey.TRACE_SESSION] = session_id

                # Extend the list with all feedbacks from this scorer
                multi_turn_assessments[first_trace_id].extend(feedbacks)

            except Exception as e:
                # Use existing error handling mechanism
                error_feedback = Feedback(
                    name=scorer.name,
                    source=make_code_type_assessment_source(scorer.name),
                    error=AssessmentError(
                        error_code="SCORER_ERROR",
                        error_message=str(e),
                        stack_trace=traceback.format_exc(),
                    ),
                )

                multi_turn_assessments[first_trace_id].append(error_feedback)

    return multi_turn_assessments


def validate_session_level_evaluation_inputs(scorers: list[Scorer], predict_fn: Any) -> None:
    """
    Validate input parameters when session-level scorers are present.

    Args:
        scorers: List of scorer instances
        predict_fn: Prediction function (if provided)

    Raises:
        MlflowException: If invalid configuration is detected
    """
    session_level_scorers = [scorer for scorer in scorers if scorer.is_session_level_scorer]

    if session_level_scorers:
        if predict_fn is not None:
            scorer_names = [scorer.name for scorer in session_level_scorers]
            raise MlflowException.invalid_parameter_value(
                f"Multi-turn scorers are not yet supported with predict_fn. "
                f"The following scorers require session-level evaluation: {scorer_names}. "
                f"Please pass traces instead."
            )
