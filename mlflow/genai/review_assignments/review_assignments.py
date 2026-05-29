from dataclasses import dataclass, field
from typing import NamedTuple

from mlflow.genai.utils.enum_utils import StrEnum


class ReviewTargetType(StrEnum):
    """What kind of object is being reviewed."""

    TRACE = "trace"


class ReviewAssignmentState(StrEnum):
    """Per-assignment workflow state.

    Two states only. State changes only on explicit reviewer action —
    writing an assessment against the target does NOT advance the
    assignment.

    Transitions:
        - ``PENDING`` -> ``COMPLETE``: explicit "Mark complete" action
          (sets ``completed_time_ms``).
        - ``COMPLETE`` -> ``PENDING``: explicit reopen (clears
          ``completed_time_ms`` back to ``None``).
    """

    PENDING = "pending"
    COMPLETE = "complete"


@dataclass
class ReviewAssignment:
    """One row of the review-assignment table.

    Identity is the composite ``(target_id, reviewer)``: a single
    reviewer is at most once assigned to a single target. Repeated
    assignment is silently a no-op (bulk-create returns the existing
    row in the ``existing`` bucket).

    ``reviewer`` and ``assigner`` are free-form strings that should
    match whatever shape ``AssessmentSource.source_id`` takes in the
    caller's deployment — typically email on Databricks, username
    elsewhere. ``reviewer`` is lowercased on write so the UI can match a
    reviewer's assignments against their assessments without casing
    drift.
    """

    assignment_id: str
    experiment_id: str
    target_type: ReviewTargetType
    target_id: str
    reviewer: str
    assigner: str
    state: ReviewAssignmentState
    creation_time_ms: int
    last_update_time_ms: int
    completed_time_ms: int | None = field(default=None)


class BulkCreateFailure(NamedTuple):
    """One element of ``BulkCreateReviewAssignmentsResult.failed``.

    ``target_id`` and ``reviewer`` identify the row that didn't land;
    ``error_message`` is a human-readable explanation. Validation
    failures are the typical reason — see
    ``mlflow.genai.review_assignments.validation``.
    """

    target_id: str
    reviewer: str
    error_message: str


@dataclass
class BulkCreateReviewAssignmentsResult:
    """Outcome of :py:meth:`bulk_create_review_assignments`.

    The three buckets are disjoint: every input ``(target_id, reviewer)``
    pair lands in exactly one. Callers can size-check
    ``len(created) + len(existing) + len(failed) == N * M`` to verify
    no rows were silently dropped.
    """

    created: list[ReviewAssignment]
    """Newly-inserted assignments."""

    existing: list[str]
    """``assignment_id`` strings for rows that already existed at the
    unique key. Idempotent re-runs of the same bulk-assign land here.
    Intentionally only the id, not the full entity: re-fetch via
    ``get_review_assignment`` if the caller needs the state /
    timestamps of the existing rows. This keeps the bulk-assign
    response small for the typical "N x M with mostly new pairs" case
    where the existing bucket can be sized in the thousands."""

    failed: list[BulkCreateFailure]
    """Per-row validation failures. The whole batch is still a single
    transaction; failed rows don't roll back the rest."""
