from dataclasses import dataclass, field
from typing import NamedTuple

from mlflow.exceptions import MlflowException
from mlflow.genai.utils.enum_utils import StrEnum
from mlflow.protos import review_assignments_pb2 as _ra_pb
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class ReviewTargetType(StrEnum):
    """What kind of object is being reviewed."""

    TRACE = "trace"

    def to_proto(self) -> int:
        # Dict-based (rather than `return _ra_pb.TRACE`) so a member added
        # later without a mapping fails loud with KeyError instead of
        # silently serializing as TRACE.
        return _TARGET_TYPE_TO_PROTO[self]

    @classmethod
    def from_proto(cls, proto: int) -> "ReviewTargetType":
        if (target_type := _TARGET_TYPE_FROM_PROTO.get(proto)) is not None:
            return target_type
        # Includes REVIEW_TARGET_TYPE_UNSPECIFIED: the proto2 optional
        # enum defaults to the zero value, which the handler must reject
        # explicitly because validate_required only checks HasField.
        raise MlflowException(
            f"`target_type` must be one of {[t.value for t in cls]}; got proto enum value {proto}.",
            error_code=INVALID_PARAMETER_VALUE,
        )


_TARGET_TYPE_TO_PROTO: dict[ReviewTargetType, int] = {ReviewTargetType.TRACE: _ra_pb.TRACE}
_TARGET_TYPE_FROM_PROTO: dict[int, ReviewTargetType] = {
    v: k for k, v in _TARGET_TYPE_TO_PROTO.items()
}


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

    def to_proto(self) -> int:
        return _STATE_TO_PROTO[self]

    @classmethod
    def from_proto(cls, proto: int) -> "ReviewAssignmentState":
        if (state := _STATE_FROM_PROTO.get(proto)) is not None:
            return state
        raise MlflowException(
            f"`state` must be one of {[s.value for s in cls]}; got proto enum value {proto}.",
            error_code=INVALID_PARAMETER_VALUE,
        )


_STATE_TO_PROTO: dict[ReviewAssignmentState, int] = {
    ReviewAssignmentState.PENDING: _ra_pb.PENDING,
    ReviewAssignmentState.COMPLETE: _ra_pb.COMPLETE,
}
_STATE_FROM_PROTO: dict[int, ReviewAssignmentState] = {v: k for k, v in _STATE_TO_PROTO.items()}


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

    def to_proto(self) -> _ra_pb.ReviewAssignment:
        proto = _ra_pb.ReviewAssignment(
            assignment_id=self.assignment_id,
            experiment_id=self.experiment_id,
            target_type=self.target_type.to_proto(),
            target_id=self.target_id,
            reviewer=self.reviewer,
            assigner=self.assigner,
            state=self.state.to_proto(),
            creation_time_ms=self.creation_time_ms,
            last_update_time_ms=self.last_update_time_ms,
        )
        if self.completed_time_ms is not None:
            proto.completed_time_ms = self.completed_time_ms
        return proto

    @classmethod
    def from_proto(cls, proto: _ra_pb.ReviewAssignment) -> "ReviewAssignment":
        return cls(
            assignment_id=proto.assignment_id,
            experiment_id=proto.experiment_id,
            target_type=ReviewTargetType.from_proto(proto.target_type),
            target_id=proto.target_id,
            reviewer=proto.reviewer,
            assigner=proto.assigner,
            state=ReviewAssignmentState.from_proto(proto.state),
            creation_time_ms=proto.creation_time_ms,
            last_update_time_ms=proto.last_update_time_ms,
            completed_time_ms=(
                proto.completed_time_ms if proto.HasField("completed_time_ms") else None
            ),
        )


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

    def to_proto(self) -> _ra_pb.BulkCreateFailure:
        return _ra_pb.BulkCreateFailure(
            target_id=self.target_id,
            reviewer=self.reviewer,
            error_message=self.error_message,
        )


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
