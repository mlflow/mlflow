import pytest

from mlflow.genai.databricks.review_queues import (
    InputCategorical,
    InputNumeric,
    InputPassFail,
    InputText,
    ItemKind,
    QuestionType,
    ReviewQuestion,
    ReviewQueue,
    ReviewQueueItem,
    ReviewQueueMember,
    ReviewQueueType,
    ReviewStatus,
)

_TIME = "2026-01-02T03:04:05Z"
_TIME_MS = 1767323045000


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (InputPassFail(), {}),
        (
            InputPassFail(positive_label="good", negative_label="bad"),
            {"positive_label": "good", "negative_label": "bad"},
        ),
        (InputCategorical(options=["a", "b"]), {"options": ["a", "b"], "multi_select": False}),
        (
            InputCategorical(options=["a"], multi_select=True),
            {"options": ["a"], "multi_select": True},
        ),
        (InputNumeric(), {}),
        (InputNumeric(min_value=0, max_value=5), {"min_value": 0, "max_value": 5}),
        (InputText(), {}),
        (InputText(max_length=100), {"max_length": 100}),
    ],
)
def test_input_to_dict(input, expected):
    assert input.to_dict() == expected


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("pass_fail", {"positive_label": "y"}, InputPassFail(positive_label="y")),
        ("categorical", {"options": ["a"]}, InputCategorical(options=["a"])),
        ("numeric", {"min_value": 1}, InputNumeric(min_value=1)),
        ("text", {"max_length": 3}, InputText(max_length=3)),
    ],
)
def test_review_question_from_dict_parses_input(field, value, expected):
    question = ReviewQuestion.from_dict({
        "name": "experiments/1/reviewQuestions/q1",
        "type": "FEEDBACK",
        field: value,
    })
    assert question.input == expected


def test_review_question_from_dict():
    question = ReviewQuestion.from_dict({
        "name": "experiments/123/reviewQuestions/q1",
        "title": "Correct?",
        "type": "EXPECTATION",
        "categorical": {"options": ["yes", "no"], "multi_select": False},
        "instruction": "Check the answer",
        "enable_comment": True,
        "is_default": True,
        "position": "2",
        "created_by": "alice@example.com",
        "create_time": _TIME,
        "last_updated_by": "bob@example.com",
        "update_time": _TIME,
    })
    assert question == ReviewQuestion(
        question_id="q1",
        experiment_id="123",
        title="Correct?",
        type=QuestionType.EXPECTATION,
        input=InputCategorical(options=["yes", "no"]),
        instruction="Check the answer",
        enable_comment=True,
        is_default=True,
        position=2,
        created_by="alice@example.com",
        create_time_ms=_TIME_MS,
        last_updated_by="bob@example.com",
        update_time_ms=_TIME_MS,
    )


def test_review_question_from_dict_minimal():
    question = ReviewQuestion.from_dict({"name": "experiments/1/reviewQuestions/q1"})
    assert question == ReviewQuestion(
        question_id="q1", experiment_id="1", title=None, type=None, input=None
    )


def test_review_queue_from_dict():
    queue = ReviewQueue.from_dict({
        "name": "experiments/123/reviewQueues/rq1",
        "display_name": "My queue",
        "queue_type": "DATASET",
        "owner": "alice@example.com",
        "dataset_id": "ds1",
        "total_item_count": "5",
        "pending_item_count": "3",
        "created_by": "alice@example.com",
        "create_time": _TIME,
        "last_updated_by": "bob@example.com",
        "update_time": _TIME,
    })
    assert queue == ReviewQueue(
        queue_id="rq1",
        experiment_id="123",
        display_name="My queue",
        queue_type=ReviewQueueType.DATASET,
        owner="alice@example.com",
        dataset_id="ds1",
        total_item_count=5,
        pending_item_count=3,
        created_by="alice@example.com",
        create_time_ms=_TIME_MS,
        last_updated_by="bob@example.com",
        update_time_ms=_TIME_MS,
    )


def test_review_queue_from_dict_minimal():
    queue = ReviewQueue.from_dict({
        "name": "experiments/1/reviewQueues/rq1",
        "queue_type": "USER",
        "dataset_id": "",
    })
    assert queue == ReviewQueue(
        queue_id="rq1", experiment_id="1", display_name="", queue_type=ReviewQueueType.USER
    )


def test_review_queue_member_from_dict():
    member = ReviewQueueMember.from_dict({
        "user": "alice@example.com",
        "added_by": "bob@example.com",
        "add_time": _TIME,
    })
    assert member == ReviewQueueMember(
        user="alice@example.com", added_by="bob@example.com", add_time_ms=_TIME_MS
    )


def test_review_queue_item_from_dict_trace():
    item = ReviewQueueItem.from_dict({
        "name": "experiments/1/reviewQueues/rq1/items/i1",
        "kind": "V4_TRACE",
        "status": "COMPLETE",
        "v4_trace": {"trace_id": "tr-1", "uc_location": "catalog.schema"},
        "completed_by": "alice@example.com",
        "complete_time": _TIME,
        "added_by": "bob@example.com",
        "create_time": _TIME,
        "update_time": _TIME,
    })
    assert item == ReviewQueueItem(
        item_id="i1",
        queue_id="rq1",
        experiment_id="1",
        kind=ItemKind.V4_TRACE,
        status=ReviewStatus.COMPLETE,
        trace_id="tr-1",
        uc_location="catalog.schema",
        completed_by="alice@example.com",
        complete_time_ms=_TIME_MS,
        added_by="bob@example.com",
        create_time_ms=_TIME_MS,
        update_time_ms=_TIME_MS,
    )


def test_review_queue_item_from_dict_pending_dataset_record():
    item = ReviewQueueItem.from_dict({
        "name": "experiments/1/reviewQueues/rq1/items/i2",
        "kind": "DATASET_RECORD",
        "status": "PENDING",
        "dataset_record": {"dataset_id": "ds1", "dataset_record_id": "r1"},
        "completed_by": "",
        "complete_time": "",
    })
    assert item == ReviewQueueItem(
        item_id="i2",
        queue_id="rq1",
        experiment_id="1",
        kind=ItemKind.DATASET_RECORD,
        status=ReviewStatus.PENDING,
        dataset_id="ds1",
        dataset_record_id="r1",
    )
