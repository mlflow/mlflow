from dataclasses import dataclass, field
from typing import NamedTuple

from mlflow.genai.utils.enum_utils import StrEnum


class ReviewTargetType(StrEnum):
    """What kind of object is being reviewed."""

    TRACE = "trace"


class ReviewAssignmentState(StrEnum):
    """Per-assignment workflow state.

    Transitions:
        - ``PENDING`` -> ``IN_PROGRESS``: auto-flipped server-side in
          the same transaction as the first ``log_feedback`` write
          where ``source.source_id`` matches ``reviewer``
          (case-insensitively).
        - ``IN_PROGRESS`` -> ``COMPLETE``: explicit reviewer action via
          ``mark_assignment_complete``.
        - ``COMPLETE`` -> ``IN_PROGRESS``: explicit reopen via
          ``update_review_assignment(state=...)``. ``PENDING`` is a
          one-way state (post-first-assessment there's no clean
          way back).
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
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
    elsewhere. The store layer compares ``reviewer`` against
    ``source.source_id`` case-insensitively for the state-flip side
    effect; callers don't have to worry about casing drift.
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
