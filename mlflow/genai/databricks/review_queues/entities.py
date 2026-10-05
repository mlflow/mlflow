from dataclasses import dataclass
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


@experimental(version="3.18.0")
class ReviewQueueType(StrEnum):
    """The flavor of a Databricks review queue."""

    USER = "USER"
    CUSTOM = "CUSTOM"
    DATASET = "DATASET"


@experimental(version="3.18.0")
class ItemKind(StrEnum):
    """What a review queue item points at."""

    V4_TRACE = "V4_TRACE"
    DATASET_RECORD = "DATASET_RECORD"


@experimental(version="3.18.0")
class ReviewStatus(StrEnum):
    """Per-item review status."""

    PENDING = "PENDING"
    COMPLETE = "COMPLETE"
    DECLINED = "DECLINED"


@experimental(version="3.18.0")
class QuestionType(StrEnum):
    """The kind of answer a review question collects."""

    FEEDBACK = "FEEDBACK"
    EXPECTATION = "EXPECTATION"


@experimental(version="3.18.0")
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


@experimental(version="3.18.0")
@dataclass
class InputCategorical:
    """A single- or multi-select from a fixed set of options."""

    options: list[str]
    multi_select: bool = False

    _FIELD = "categorical"

    def to_dict(self) -> dict[str, Any]:
        return {"options": list(self.options), "multi_select": self.multi_select}


@experimental(version="3.18.0")
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


@experimental(version="3.18.0")
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


@experimental(version="3.18.0")
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


@experimental(version="3.18.0")
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
    dataset_id: str | None = None
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
            dataset_id=d.get("dataset_id") or None,
            total_item_count=int(d["total_item_count"]) if "total_item_count" in d else None,
            pending_item_count=(
                int(d["pending_item_count"]) if "pending_item_count" in d else None
            ),
            created_by=d.get("created_by"),
            create_time_ms=_parse_time_ms(d.get("create_time")),
            last_updated_by=d.get("last_updated_by"),
            update_time_ms=_parse_time_ms(d.get("update_time")),
        )


@experimental(version="3.18.0")
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


@experimental(version="3.18.0")
@dataclass
class ReviewQueueItem:
    """One item attached to a queue, plus its review status.

    Exactly one address is populated, depending on ``kind``: ``trace_id`` (and
    ``uc_location`` for traces stored in Unity Catalog) for ``V4_TRACE`` items, or
    ``dataset_id`` + ``dataset_record_id`` for ``DATASET_RECORD`` items.
    """

    item_id: str
    queue_id: str
    experiment_id: str
    kind: ItemKind
    status: ReviewStatus
    trace_id: str | None = None
    uc_location: str | None = None
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
            uc_location=trace.get("uc_location") or None,
            dataset_id=record.get("dataset_id"),
            dataset_record_id=record.get("dataset_record_id"),
            # The server sends empty values while an item is PENDING.
            completed_by=d.get("completed_by") or None,
            complete_time_ms=_parse_time_ms(d.get("complete_time")) or None,
            added_by=d.get("added_by"),
            create_time_ms=_parse_time_ms(d.get("create_time")),
            update_time_ms=_parse_time_ms(d.get("update_time")),
        )
