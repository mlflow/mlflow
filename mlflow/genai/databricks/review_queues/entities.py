from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mlflow.genai.utils.enum_utils import StrEnum
from mlflow.utils.annotations import experimental


def _parse_time_ms(value: str | None) -> int | None:
    if not value:
        return None
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def _last_segment(resource_name: str) -> str:
    return resource_name.rsplit("/", 1)[-1]


def _experiment_id(resource_name: str) -> str:
    # Resource names are rooted at `experiments/{experiment_id}/...`.
    return resource_name.split("/")[1]


@experimental(version="3.17.1")
class ReviewQueueType(StrEnum):
    """The flavor of a Databricks review queue."""

    USER = "USER"
    CUSTOM = "CUSTOM"


@experimental(version="3.17.1")
class ItemKind(StrEnum):
    """What a review queue item points at."""

    V4_TRACE = "V4_TRACE"
    DATASET_RECORD = "DATASET_RECORD"


@experimental(version="3.17.1")
class ReviewStatus(StrEnum):
    """Per-item review status."""

    PENDING = "PENDING"
    COMPLETE = "COMPLETE"
    DECLINED = "DECLINED"


@experimental(version="3.17.1")
class QuestionType(StrEnum):
    """The kind of answer a review question collects."""

    FEEDBACK = "FEEDBACK"
    EXPECTATION = "EXPECTATION"


@experimental(version="3.17.1")
@dataclass
class InputPassFail:
    """A pass/fail toggle."""

    positive_label: str | None = None
    negative_label: str | None = None

    _FIELD = "pass_fail"

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in {
                "positive_label": self.positive_label,
                "negative_label": self.negative_label,
            }.items()
            if v is not None
        }


@experimental(version="3.17.1")
@dataclass
class InputCategorical:
    """A single- or multi-select from a fixed set of options."""

    options: list[str]
    multi_select: bool = False

    _FIELD = "categorical"

    def to_dict(self) -> dict[str, Any]:
        return {"options": list(self.options), "multi_select": self.multi_select}


@experimental(version="3.17.1")
@dataclass
class InputNumeric:
    """A numeric input, optionally bounded."""

    min_value: float | None = None
    max_value: float | None = None

    _FIELD = "numeric"

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in {"min_value": self.min_value, "max_value": self.max_value}.items()
            if v is not None
        }


@experimental(version="3.17.1")
@dataclass
class InputText:
    """A free-form text input."""

    max_length: int | None = None

    _FIELD = "text"

    def to_dict(self) -> dict[str, Any]:
        return {} if self.max_length is None else {"max_length": self.max_length}


QuestionInput = InputPassFail | InputCategorical | InputNumeric | InputText

_INPUT_TYPES: dict[str, type] = {
    cls._FIELD: cls for cls in (InputPassFail, InputCategorical, InputNumeric, InputText)
}


def _input_from_dict(d: dict[str, Any]) -> QuestionInput | None:
    for name, cls in _INPUT_TYPES.items():
        if name in d:
            return cls(**d[name])
    return None


@experimental(version="3.17.1")
@dataclass
class ReviewQuestion:
    """An experiment-scoped question that reviewers answer."""

    question_id: str
    experiment_id: str
    title: str | None
    type: QuestionType | None
    input: QuestionInput | None
    instruction: str | None = None
    enable_comment: bool = False
    is_default: bool = False
    position: int | None = None
    created_by: str | None = None
    create_time_ms: int | None = None
    last_updated_by: str | None = None
    update_time_ms: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReviewQuestion":
        return cls(
            question_id=_last_segment(d["name"]),
            experiment_id=_experiment_id(d["name"]),
            title=d.get("title"),
            type=QuestionType(d["type"]) if d.get("type") else None,
            input=_input_from_dict(d),
            instruction=d.get("instruction"),
            enable_comment=d.get("enable_comment", False),
            is_default=d.get("is_default", False),
            position=int(d["position"]) if "position" in d else None,
            created_by=d.get("created_by"),
            create_time_ms=_parse_time_ms(d.get("create_time")),
            last_updated_by=d.get("last_updated_by"),
            update_time_ms=_parse_time_ms(d.get("update_time")),
        )


@experimental(version="3.17.1")
@dataclass
class ReviewQueue:
    """An experiment-scoped worklist of items for reviewers.

    The questions a queue asks are not included; use
    :py:func:`mlflow.genai.databricks.review_queues.resolve_effective_review_questions`
    to get them.
    """

    queue_id: str
    experiment_id: str
    display_name: str
    queue_type: ReviewQueueType
    owner: str | None = None
    total_item_count: int | None = None
    pending_item_count: int | None = None
    created_by: str | None = None
    create_time_ms: int | None = None
    last_updated_by: str | None = None
    update_time_ms: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReviewQueue":
        return cls(
            queue_id=_last_segment(d["name"]),
            experiment_id=_experiment_id(d["name"]),
            display_name=d.get("display_name", ""),
            queue_type=ReviewQueueType(d["queue_type"]),
            owner=d.get("owner"),
            total_item_count=int(d["total_item_count"]) if "total_item_count" in d else None,
            pending_item_count=(
                int(d["pending_item_count"]) if "pending_item_count" in d else None
            ),
            created_by=d.get("created_by"),
            create_time_ms=_parse_time_ms(d.get("create_time")),
            last_updated_by=d.get("last_updated_by"),
            update_time_ms=_parse_time_ms(d.get("update_time")),
        )


@experimental(version="3.17.1")
@dataclass
class ReviewQueueMember:
    """A reviewer assigned to a queue."""

    user: str
    added_by: str | None = None
    add_time_ms: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReviewQueueMember":
        return cls(
            user=d["user"],
            added_by=d.get("added_by"),
            add_time_ms=_parse_time_ms(d.get("add_time")),
        )


@experimental(version="3.17.1")
@dataclass
class ReviewQueueItem:
    """One item attached to a queue, plus its review status.

    Exactly one address is populated, depending on ``kind``: ``trace_id`` for
    ``V4_TRACE`` items, or ``dataset_id`` + ``dataset_record_id`` for
    ``DATASET_RECORD`` items.
    """

    item_id: str
    queue_id: str
    experiment_id: str
    kind: ItemKind
    status: ReviewStatus
    trace_id: str | None = None
    dataset_id: str | None = None
    dataset_record_id: str | None = None
    completed_by: str | None = None
    complete_time_ms: int | None = None
    added_by: str | None = None
    create_time_ms: int | None = None
    update_time_ms: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReviewQueueItem":
        # Format: experiments/{experiment}/reviewQueues/{queue}/items/{item}
        parts = d["name"].split("/")
        trace = d.get("v4_trace", {})
        record = d.get("dataset_record", {})
        return cls(
            item_id=parts[5],
            queue_id=parts[3],
            experiment_id=parts[1],
            kind=ItemKind(d["kind"]),
            status=ReviewStatus(d["status"]),
            trace_id=trace.get("trace_id"),
            dataset_id=record.get("dataset_id"),
            dataset_record_id=record.get("dataset_record_id"),
            # The server sends empty values while an item is PENDING.
            completed_by=d.get("completed_by") or None,
            complete_time_ms=_parse_time_ms(d.get("complete_time")) or None,
            added_by=d.get("added_by"),
            create_time_ms=_parse_time_ms(d.get("create_time")),
            update_time_ms=_parse_time_ms(d.get("update_time")),
        )


@experimental(version="3.19.0")
class LabelingSessionMigrationIssueSeverity(StrEnum):
    """Whether a labeling session migration issue stops the migration."""

    WARNING = "WARNING"
    BLOCKING = "BLOCKING"


@experimental(version="3.19.0")
@dataclass
class LabelingSessionMigrationIssue:
    """Something a labeling session migration changes or leaves out, or a reason it can't run.

    ``code`` names the kind of issue, e.g. ``"TEXT_LIST_QUESTION_DROPPED"`` or
    ``"TOO_MANY_ITEMS"``, and ``message`` describes it.
    """

    code: str
    severity: LabelingSessionMigrationIssueSeverity
    count: int
    message: str
    limit: int | None = None
    subjects: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabelingSessionMigrationIssue":
        return cls(
            code=d.get("code", "CODE_UNSPECIFIED"),
            # Anything not marked WARNING blocks, so a severity a newer server adds never
            # makes a blocked migration look ready.
            severity=(
                LabelingSessionMigrationIssueSeverity.WARNING
                if d.get("severity") == LabelingSessionMigrationIssueSeverity.WARNING
                else LabelingSessionMigrationIssueSeverity.BLOCKING
            ),
            count=int(d.get("count", 0)),
            message=d.get("message", ""),
            limit=int(d["limit"]) if "limit" in d else None,
            subjects=list(d.get("subjects", [])),
        )


@experimental(version="3.19.0")
@dataclass
class LabelingSessionMigrationPlan:
    """What migrating a labeling session creates, or would create on a dry run."""

    review_queue_id: str
    experiment_id: str
    review_queue_display_name: str
    questions: list[ReviewQuestion]
    items_to_add: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabelingSessionMigrationPlan":
        return cls(
            review_queue_id=_last_segment(d["review_queue"]),
            experiment_id=_experiment_id(d["review_queue"]),
            review_queue_display_name=d.get("review_queue_display_name", ""),
            questions=[ReviewQuestion.from_dict(q) for q in d.get("questions", [])],
            items_to_add=int(d.get("items_to_add", 0)),
        )


@experimental(version="3.19.0")
@dataclass
class LabelingSessionMigrationResult:
    """The result of migrating a labeling session, or of a dry run.

    A dry run returns ``plan``, ``issues`` and, unless an issue is blocking, ``plan_token``.
    A migration returns ``plan``, the warnings, and the ``review_queue`` it migrated to.
    """

    plan: LabelingSessionMigrationPlan | None
    issues: list[LabelingSessionMigrationIssue]
    plan_token: str | None = None
    review_queue: ReviewQueue | None = None

    @property
    def warnings(self) -> list[LabelingSessionMigrationIssue]:
        return [
            i for i in self.issues if i.severity == LabelingSessionMigrationIssueSeverity.WARNING
        ]

    @property
    def blocking_issues(self) -> list[LabelingSessionMigrationIssue]:
        return [
            i for i in self.issues if i.severity == LabelingSessionMigrationIssueSeverity.BLOCKING
        ]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabelingSessionMigrationResult":
        return cls(
            plan=LabelingSessionMigrationPlan.from_dict(d["plan"]) if "plan" in d else None,
            issues=[LabelingSessionMigrationIssue.from_dict(i) for i in d.get("issues", [])],
            plan_token=d.get("plan_token") or None,
            review_queue=ReviewQueue.from_dict(d["review_queue"]) if "review_queue" in d else None,
        )
