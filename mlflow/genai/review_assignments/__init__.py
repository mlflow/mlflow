"""OSS-native review assignments for SME labeling workflows.

A ``ReviewAssignment`` is one row per ``(target, reviewer)`` pair: the
piece of state that says "this trace needs review from this person."
The companion workflow lives in MLflow's review UI, where reviewers
post Feedback assessments against the assigned target and explicitly
mark the assignment ``complete`` when done. The assignment has two
states, ``pending`` and ``complete``; writing an assessment does not
change the state.
"""

from mlflow.genai.review_assignments.review_assignments import (
    BulkCreateFailure,
    BulkCreateReviewAssignmentsResult,
    ReviewAssignment,
    ReviewAssignmentState,
    ReviewTargetType,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracing.client import TracingClient


def create_review_assignment(
    experiment_id: str,
    *,
    target_id: str,
    reviewer: str,
    assigner: str,
    target_type: ReviewTargetType | str = ReviewTargetType.TRACE,
) -> ReviewAssignment:
    """Assign a single reviewer to a single target (v1: a trace).

    Identity is ``(target_id, reviewer)``: assigning the same reviewer to
    the same target twice is a no-op that returns the existing assignment.

    Args:
        experiment_id: Parent experiment ID.
        target_id: The trace ID being assigned for review.
        reviewer: Reviewer identifier (typically email). Lowercased on write
            and matched case-insensitively against assessment authorship.
        assigner: Identifier of whoever is creating the assignment.
        target_type: What kind of object is being reviewed. v1 supports
            ``"trace"`` only.

    Returns:
        The created (or pre-existing) :class:`ReviewAssignment`.
    """
    return TracingClient()._create_review_assignment(
        experiment_id,
        target_type=target_type,
        target_id=target_id,
        reviewer=reviewer,
        assigner=assigner,
    )


def bulk_create_review_assignments(
    experiment_id: str,
    *,
    target_ids: list[str],
    reviewers: list[str],
    assigner: str,
    target_type: ReviewTargetType | str = ReviewTargetType.TRACE,
) -> BulkCreateReviewAssignmentsResult:
    """Assign the cross product of (targets x reviewers) in one transaction.

    ``N`` targets times ``M`` reviewers create ``N*M`` assignments. The
    result partitions every pair into exactly one of ``created`` /
    ``existing`` / ``failed``.

    Args:
        experiment_id: Parent experiment ID.
        target_ids: Trace IDs to assign (de-duplicate before calling).
        reviewers: Reviewer identifiers to assign (de-duplicate before calling).
        assigner: Identifier of whoever is creating the assignments.
        target_type: What kind of object is being reviewed. v1 supports
            ``"trace"`` only.

    Returns:
        A :class:`BulkCreateReviewAssignmentsResult`.
    """
    return TracingClient()._bulk_create_review_assignments(
        experiment_id,
        target_type=target_type,
        target_ids=target_ids,
        reviewers=reviewers,
        assigner=assigner,
    )


def get_review_assignment(assignment_id: str) -> ReviewAssignment:
    """Get a review assignment by its server-generated ``assignment_id``."""
    return TracingClient()._get_review_assignment(assignment_id)


def list_review_assignments(
    *,
    experiment_id: str | None = None,
    reviewer: str | None = None,
    state: ReviewAssignmentState | str | None = None,
    max_results: int | None = None,
    page_token: str | None = None,
) -> PagedList[ReviewAssignment]:
    """List review assignments, paginated.

    At least one of ``experiment_id`` or ``reviewer`` must be provided;
    an unscoped listing is rejected.

    There is intentionally no ``target_type`` filter here: v1 only has
    ``TRACE``, so the filter would be a no-op. The lower store/REST layers
    carry it for forward-compatibility; it surfaces on this SDK function
    when a second target type ships.

    Args:
        experiment_id: Restrict to assignments in this experiment.
        reviewer: Restrict to assignments for this reviewer.
        state: Optional workflow-state filter.
        max_results: Page size (server default applies when unset).
        page_token: Pagination token from a prior call.

    Returns:
        A :class:`PagedList` of :class:`ReviewAssignment`.
    """
    return TracingClient()._list_review_assignments(
        experiment_id=experiment_id,
        reviewer=reviewer,
        state=state,
        max_results=max_results,
        page_token=page_token,
    )


def list_review_assignments_for_target(target_id: str) -> list[ReviewAssignment]:
    """List every reviewer assigned to a given target (trace)."""
    return TracingClient()._list_review_assignments_for_target(target_id)


def update_review_assignment(
    assignment_id: str, *, state: ReviewAssignmentState | str
) -> ReviewAssignment:
    """Update the workflow ``state`` of a review assignment.

    Only ``state`` is mutable. ``pending`` and ``complete`` are
    interchangeable via mark-complete / reopen; reopening clears the
    completion timestamp. See :class:`ReviewAssignmentState`.
    """
    return TracingClient()._update_review_assignment(assignment_id, state=state)


def delete_review_assignment(assignment_id: str) -> None:
    """Delete a review assignment. No-op when it doesn't exist."""
    TracingClient()._delete_review_assignment(assignment_id)


__all__ = [
    "BulkCreateFailure",
    "BulkCreateReviewAssignmentsResult",
    "ReviewAssignment",
    "ReviewAssignmentState",
    "ReviewTargetType",
    "create_review_assignment",
    "bulk_create_review_assignments",
    "get_review_assignment",
    "list_review_assignments",
    "list_review_assignments_for_target",
    "update_review_assignment",
    "delete_review_assignment",
]
