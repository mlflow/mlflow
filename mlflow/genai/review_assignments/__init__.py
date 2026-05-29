"""OSS-native review assignments for SME labeling workflows.

A ``ReviewAssignment`` is one row per ``(target, reviewer)`` pair: the
piece of state that says "this trace needs review from this person."
The companion workflow lives in MLflow's review UI, where reviewers
post Feedback assessments against the assigned target and explicitly
mark the assignment ``complete`` when done. The assignment has two
states, ``pending`` and ``complete``; writing an assessment does not
change the state.

The fluent SDK surface (``assign_traces``, ``list_my_assignments``,
``mark_assignment_complete``, etc.) lands in stack 3 of this PR
series; this stack ships the entity, validation, and SQL store.
"""

from mlflow.genai.review_assignments.review_assignments import (
    BulkCreateFailure,
    BulkCreateReviewAssignmentsResult,
    ReviewAssignment,
    ReviewAssignmentState,
    ReviewTargetType,
)

__all__ = [
    "BulkCreateFailure",
    "BulkCreateReviewAssignmentsResult",
    "ReviewAssignment",
    "ReviewAssignmentState",
    "ReviewTargetType",
]
