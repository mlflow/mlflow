from unittest import mock

import pytest

from mlflow.genai.databricks.review_queues import (
    InputCategorical,
    InputNumeric,
    InputPassFail,
    InputText,
    ItemKind,
    LabelingSessionMigrationIssue,
    LabelingSessionMigrationIssueSeverity,
    LabelingSessionMigrationPlan,
    LabelingSessionMigrationResult,
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
        "queue_type": "CUSTOM",
        "owner": "alice@example.com",
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
        queue_type=ReviewQueueType.CUSTOM,
        owner="alice@example.com",
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
    })
    assert queue == ReviewQueue(
        queue_id="rq1", experiment_id="1", display_name="", queue_type=ReviewQueueType.USER
    )


_ENTITIES = "mlflow.genai.databricks.review_queues.entities"


def _queue(queue_id="rq1"):
    return ReviewQueue(
        queue_id=queue_id, experiment_id="123", display_name="", queue_type=ReviewQueueType.USER
    )


def test_review_queue_url_in_a_notebook_names_the_workspace():
    with (
        mock.patch(
            f"{_ENTITIES}.get_workspace_url", return_value="https://example.cloud.databricks.com/"
        ),
        mock.patch(f"{_ENTITIES}.get_workspace_id", return_value="6051"),
    ):
        url = _queue().url
    assert url == (
        "https://example.cloud.databricks.com/ml/experiments/123/review-queue"
        "?o=6051&selectedQueueId=rq1"
    )


def test_review_queue_url_outside_a_notebook_uses_the_tracking_host():
    creds = mock.Mock(host="https://example.cloud.databricks.com/")
    with (
        mock.patch(f"{_ENTITIES}.get_workspace_url", return_value=None),
        mock.patch(f"{_ENTITIES}.get_databricks_host_creds", return_value=creds),
    ):
        url = _queue("my queue/1").url
    assert url == (
        "https://example.cloud.databricks.com/ml/experiments/123/review-queue"
        "?selectedQueueId=my+queue%2F1"
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


def test_labeling_session_migration_result_from_dict_dry_run():
    result = LabelingSessionMigrationResult.from_dict({
        "plan": {
            "review_queue": "experiments/1/reviewQueues/rq-ls1",
            "review_queue_display_name": "Support triage",
            "questions": [
                {"name": "experiments/1/reviewQuestions/q1", "title": "quality", "type": "FEEDBACK"}
            ],
            "items_to_add": "6",
        },
        "issues": [
            {
                "code": "TOO_MANY_ITEMS",
                "severity": "BLOCKING",
                "count": "300",
                "limit": "150",
                "message": "The session has 300 items, more than a review queue holds.",
            },
            {
                "code": "TEXT_LIST_QUESTION_DROPPED",
                "severity": "WARNING",
                "count": 1,
                "subjects": ["expected_facts"],
                "message": "1 text-list question(s) are not migrated.",
            },
        ],
        "plan_token": "v1.tok",
    })

    assert result.plan == LabelingSessionMigrationPlan(
        review_queue_id="rq-ls1",
        experiment_id="1",
        review_queue_display_name="Support triage",
        questions=[
            ReviewQuestion(
                question_id="q1",
                experiment_id="1",
                title="quality",
                type=QuestionType.FEEDBACK,
                input=None,
            )
        ],
        items_to_add=6,
    )
    too_many = LabelingSessionMigrationIssue(
        code="TOO_MANY_ITEMS",
        severity=LabelingSessionMigrationIssueSeverity.BLOCKING,
        count=300,
        message="The session has 300 items, more than a review queue holds.",
        limit=150,
    )
    text_list = LabelingSessionMigrationIssue(
        code="TEXT_LIST_QUESTION_DROPPED",
        severity=LabelingSessionMigrationIssueSeverity.WARNING,
        count=1,
        message="1 text-list question(s) are not migrated.",
        subjects=["expected_facts"],
    )
    assert result.issues == [too_many, text_list]
    assert result.blocking_issues == [too_many]
    assert result.warnings == [text_list]
    assert result.plan_token == "v1.tok"
    assert result.review_queue is None


def test_labeling_session_migration_result_from_dict_migration():
    result = LabelingSessionMigrationResult.from_dict({
        "plan": {"review_queue": "experiments/1/reviewQueues/rq-ls1", "items_to_add": 2},
        "review_queue": {
            "name": "experiments/1/reviewQueues/rq-ls1",
            "display_name": "Support triage",
            "queue_type": "CUSTOM",
        },
    })

    assert result.plan.questions == []
    assert result.issues == []
    assert result.plan_token is None
    assert result.review_queue.queue_id == "rq-ls1"
    assert result.review_queue.queue_type == ReviewQueueType.CUSTOM


def test_labeling_session_migration_result_from_dict_empty():
    assert LabelingSessionMigrationResult.from_dict({}) == LabelingSessionMigrationResult(
        plan=None, issues=[]
    )


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        ("WARNING", LabelingSessionMigrationIssueSeverity.WARNING),
        ("BLOCKING", LabelingSessionMigrationIssueSeverity.BLOCKING),
        ("SEVERITY_UNSPECIFIED", LabelingSessionMigrationIssueSeverity.BLOCKING),
        ("A_NEWER_SEVERITY", LabelingSessionMigrationIssueSeverity.BLOCKING),
        (None, LabelingSessionMigrationIssueSeverity.BLOCKING),
    ],
)
def test_labeling_session_migration_issue_treats_unknown_severity_as_blocking(severity, expected):
    d = {"code": "TOO_MANY_MEMBERS", "count": 20}
    if severity is not None:
        d["severity"] = severity
    assert LabelingSessionMigrationIssue.from_dict(d).severity == expected


def test_labeling_session_migration_issue_from_dict_defaults():
    assert LabelingSessionMigrationIssue.from_dict({
        "code": "ALREADY_MIGRATED",
        "severity": "WARNING",
    }) == LabelingSessionMigrationIssue(
        code="ALREADY_MIGRATED",
        severity=LabelingSessionMigrationIssueSeverity.WARNING,
        count=0,
        message="",
    )
