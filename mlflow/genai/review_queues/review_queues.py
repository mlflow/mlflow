from dataclasses import dataclass, field

from mlflow.genai.utils.enum_utils import StrEnum


class ReviewTargetType(StrEnum):
    """What kind of object a queue item points at.

    v1 ships ``trace`` only; the column is kept wide enough for
    ``session`` / ``span`` to land later without a migration.
    """

    TRACE = "trace"


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
