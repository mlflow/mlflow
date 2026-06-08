from dataclasses import dataclass, field

from mlflow.exceptions import MlflowException
from mlflow.genai.utils.enum_utils import StrEnum
from mlflow.protos import review_queues_pb2 as _rq_pb
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental


@experimental(version="3.14.0")
class ReviewTargetType(StrEnum):
    """What kind of object a queue item points at.

    v1 ships ``trace`` only; the column is kept wide enough for
    ``session`` / ``span`` to land later without a migration.
    """

    TRACE = "trace"

    def to_proto(self) -> int:
        return _rq_pb.TRACE

    @classmethod
    def from_proto(cls, proto: int) -> "ReviewTargetType":
        if proto == _rq_pb.TRACE:
            return cls.TRACE
        raise MlflowException(
            f"`target_type` must be TRACE; got proto enum value {proto}.",
            error_code=INVALID_PARAMETER_VALUE,
        )


@experimental(version="3.14.0")
class ReviewQueueType(StrEnum):
    """The flavor of a review queue.

    ``USER`` — ``name`` equals a user identifier and the queue has exactly
        one assigned user (that user). It is the reviewer's personal
        worklist and inherits *all* of the experiment's label schemas as
        its questions (no chooser, resolved live at read time), so creating
        one is just "assign these traces to this person".
    ``CUSTOM`` — an arbitrary, non-reserved ``name`` with 0..N assigned
        users and an explicitly-attached subset of label schemas. The
        analog of a Databricks ``LabelingSession``.
    """

    USER = "user"
    CUSTOM = "custom"

    def to_proto(self) -> int:
        if self is ReviewQueueType.USER:
            return _rq_pb.USER
        return _rq_pb.CUSTOM

    @classmethod
    def from_proto(cls, proto: int) -> "ReviewQueueType":
        if proto == _rq_pb.USER:
            return cls.USER
        if proto == _rq_pb.CUSTOM:
            return cls.CUSTOM
        raise MlflowException(
            f"`queue_type` must be one of USER or CUSTOM; got proto enum value {proto}.",
            error_code=INVALID_PARAMETER_VALUE,
        )


@experimental(version="3.14.0")
class ReviewStatus(StrEnum):
    """Shared-pool workflow status of a single attached trace.

    Status is per-``(queue, trace)`` — NOT per-user. The queue's assigned
    users are a *pool*: a trace is addressed when **any** assigned user acts
    on it, not when every user does. ``completed_by`` records who that was.

    Transitions are always explicit reviewer actions; writing an assessment
    against the trace does NOT advance the status:

        - ``PENDING`` -> ``COMPLETE``: any user marks it done (sets
          ``completed_by`` + ``completed_time_ms``).
        - ``PENDING`` -> ``DECLINED``: an explicit "this trace will not be
          reviewed in this queue" (out of scope / can't judge) — distinct
          from a temporary defer, and also records who declined it.
        - ``COMPLETE`` / ``DECLINED`` -> ``PENDING``: reopen, clearing the
          ``completed_by`` / ``completed_time_ms`` attribution.

    There is no ``in_progress`` state and no auto-flip.
    """

    PENDING = "pending"
    COMPLETE = "complete"
    DECLINED = "declined"

    def to_proto(self) -> int:
        return {
            ReviewStatus.PENDING: _rq_pb.PENDING,
            ReviewStatus.COMPLETE: _rq_pb.COMPLETE,
            ReviewStatus.DECLINED: _rq_pb.DECLINED,
        }[self]

    @classmethod
    def from_proto(cls, proto: int) -> "ReviewStatus":
        mapping = {
            _rq_pb.PENDING: cls.PENDING,
            _rq_pb.COMPLETE: cls.COMPLETE,
            _rq_pb.DECLINED: cls.DECLINED,
        }
        if proto not in mapping:
            raise MlflowException(
                f"`status` must be one of PENDING, COMPLETE, or DECLINED; "
                f"got proto enum value {proto}.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return mapping[proto]


@experimental(version="3.14.0")
@dataclass
class ReviewQueueItem:
    """One trace attached to a queue plus its shared-pool workflow status.

    A row of the ``review_queue_traces`` table. The same trace attached to
    two different queues has an independent :class:`ReviewQueueItem` (and
    therefore an independent status) in each — intentional, since the
    review contexts differ.

    ``completed_by`` / ``completed_time_ms`` are populated only in the
    ``COMPLETE`` and ``DECLINED`` terminal states and cleared on reopen.
    """

    queue_id: str
    target_type: ReviewTargetType
    target_id: str
    status: ReviewStatus
    creation_time_ms: int
    last_update_time_ms: int
    completed_by: str | None = None
    completed_time_ms: int | None = None

    def to_proto(self) -> "_rq_pb.ReviewQueueItem":
        proto = _rq_pb.ReviewQueueItem(
            queue_id=self.queue_id,
            target_type=self.target_type.to_proto(),
            target_id=self.target_id,
            status=self.status.to_proto(),
            creation_time_ms=self.creation_time_ms,
            last_update_time_ms=self.last_update_time_ms,
        )
        if self.completed_by is not None:
            proto.completed_by = self.completed_by
        if self.completed_time_ms is not None:
            proto.completed_time_ms = self.completed_time_ms
        return proto

    @classmethod
    def from_proto(cls, proto: "_rq_pb.ReviewQueueItem") -> "ReviewQueueItem":
        return cls(
            queue_id=proto.queue_id,
            target_type=ReviewTargetType.from_proto(proto.target_type),
            target_id=proto.target_id,
            status=ReviewStatus.from_proto(proto.status),
            creation_time_ms=proto.creation_time_ms,
            last_update_time_ms=proto.last_update_time_ms,
            completed_by=proto.completed_by if proto.HasField("completed_by") else None,
            completed_time_ms=(
                proto.completed_time_ms if proto.HasField("completed_time_ms") else None
            ),
        )


@experimental(version="3.14.0")
@dataclass
class ReviewQueue:
    """A named bundle of attached traces, questions, and assigned users.

    Scoped to an experiment and keyed on ``(experiment_id, name)``. The
    attached traces are paged separately (they can be many) via
    ``list_review_queue_traces``; the small association sets — assigned
    ``users`` and attached label-schema ids — are hydrated inline here.

    ``schema_ids`` reflects the literal ``review_queue_label_schemas`` rows:
    it is the chosen subset for a ``CUSTOM`` queue and **empty** for a
    ``USER`` queue (which resolves to all of the experiment's schemas live
    at read time, a concern of the read/handler layer, not storage).
    """

    queue_id: str
    experiment_id: str
    name: str
    queue_type: ReviewQueueType
    created_by: str | None
    creation_time_ms: int
    last_update_time_ms: int
    users: list[str] = field(default_factory=list)
    schema_ids: list[str] = field(default_factory=list)
    # The experiment's single default queue: a CUSTOM queue that inherits all of
    # the experiment's label schemas (questions resolved live at read time, like
    # a USER queue), whose questions cannot be edited and which cannot be deleted.
    is_default: bool = False

    def to_proto(self) -> "_rq_pb.ReviewQueue":
        proto = _rq_pb.ReviewQueue(
            queue_id=self.queue_id,
            experiment_id=self.experiment_id,
            name=self.name,
            queue_type=self.queue_type.to_proto(),
            creation_time_ms=self.creation_time_ms,
            last_update_time_ms=self.last_update_time_ms,
            users=self.users,
            schema_ids=self.schema_ids,
            is_default=self.is_default,
        )
        if self.created_by is not None:
            proto.created_by = self.created_by
        return proto

    @classmethod
    def from_proto(cls, proto: "_rq_pb.ReviewQueue") -> "ReviewQueue":
        return cls(
            queue_id=proto.queue_id,
            experiment_id=proto.experiment_id,
            name=proto.name,
            queue_type=ReviewQueueType.from_proto(proto.queue_type),
            created_by=proto.created_by if proto.HasField("created_by") else None,
            creation_time_ms=proto.creation_time_ms,
            last_update_time_ms=proto.last_update_time_ms,
            users=list(proto.users),
            schema_ids=list(proto.schema_ids),
            is_default=proto.is_default,
        )
