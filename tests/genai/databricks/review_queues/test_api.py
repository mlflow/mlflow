from unittest import mock

import pytest
import requests

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.databricks.review_queues import (
    InputCategorical,
    InputText,
    ItemKind,
    QuestionType,
    ReviewQueueType,
    ReviewStatus,
    _client,
    add_review_queue_items,
    add_review_queue_member,
    create_review_question,
    create_review_queue,
    delete_review_question,
    delete_review_queue,
    get_review_question,
    get_review_queue,
    list_review_questions,
    list_review_queue_items,
    list_review_queue_members,
    list_review_queues,
    migrate_labeling_session,
    remove_review_queue_items,
    remove_review_queue_member,
    resolve_effective_review_questions,
    set_review_queue_item_status,
    update_review_question,
    update_review_queue,
)

_CALL = "mlflow.genai.databricks.review_queues._client.call"

_QUESTION = {
    "name": "experiments/1/reviewQuestions/q1",
    "title": "Correct?",
    "type": "FEEDBACK",
    "categorical": {"options": ["yes", "no"], "multi_select": False},
}
_QUEUE = {
    "name": "experiments/1/reviewQueues/rq1",
    "display_name": "My queue",
    "queue_type": "CUSTOM",
}
_ITEM = {
    "name": "experiments/1/reviewQueues/rq1/items/i1",
    "kind": "V4_TRACE",
    "status": "PENDING",
    "v4_trace": {"trace_id": "tr-1"},
}


def test_create_review_question():
    with mock.patch(_CALL, return_value=_QUESTION) as mock_call:
        question = create_review_question(
            "1",
            title="Correct?",
            type="FEEDBACK",
            input=InputCategorical(options=["yes", "no"]),
            instruction="Check it",
            enable_comment=True,
        )

    mock_call.assert_called_once_with(
        "POST",
        "experiments/1/reviewQuestions",
        json={
            "title": "Correct?",
            "type": "FEEDBACK",
            "categorical": {"options": ["yes", "no"], "multi_select": False},
            "instruction": "Check it",
            "enable_comment": True,
        },
    )
    assert question.question_id == "q1"
    assert question.type == QuestionType.FEEDBACK


def test_create_review_question_rejects_invalid_type():
    with mock.patch(_CALL) as mock_call:
        with pytest.raises(ValueError, match="INVALID"):
            create_review_question("1", title="t", type="INVALID", input=InputText())
    mock_call.assert_not_called()


def test_get_review_question():
    with mock.patch(_CALL, return_value=_QUESTION) as mock_call:
        question = get_review_question("1", "q1")
    mock_call.assert_called_once_with("GET", "experiments/1/reviewQuestions/q1")
    assert question.question_id == "q1"


def test_list_review_questions():
    resp = {"review_questions": [_QUESTION], "next_page_token": "tok2"}
    with mock.patch(_CALL, return_value=resp) as mock_call:
        questions = list_review_questions("1", max_results=10, page_token="tok1")
    mock_call.assert_called_once_with(
        "GET",
        "experiments/1/reviewQuestions",
        params={"page_size": 10, "page_token": "tok1"},
    )
    assert [q.question_id for q in questions] == ["q1"]
    assert questions.token == "tok2"


def test_list_review_questions_empty():
    with mock.patch(_CALL, return_value={}) as mock_call:
        questions = list_review_questions("1")
    mock_call.assert_called_once()
    assert list(questions) == []
    assert questions.token is None


def test_update_review_question():
    with mock.patch(_CALL, return_value=_QUESTION) as mock_call:
        update_review_question("1", "q1", title="New", input=InputText(max_length=5))
    mock_call.assert_called_once_with(
        "PATCH",
        "experiments/1/reviewQuestions/q1",
        json={
            "name": "experiments/1/reviewQuestions/q1",
            "title": "New",
            "text": {"max_length": 5},
        },
        params={"update_mask": "title,text"},
    )


def test_update_review_question_requires_a_field():
    with mock.patch(_CALL) as mock_call:
        with pytest.raises(MlflowException, match="at least one field"):
            update_review_question("1", "q1")
    mock_call.assert_not_called()


def test_delete_review_question():
    with mock.patch(_CALL, return_value={}) as mock_call:
        delete_review_question("1", "q1")
    mock_call.assert_called_once_with("DELETE", "experiments/1/reviewQuestions/q1")


def test_create_review_queue():
    with mock.patch(_CALL, return_value=_QUEUE) as mock_call:
        queue = create_review_queue("1", "My queue", question_ids=["q1", "q2"])
    mock_call.assert_called_once_with(
        "POST",
        "experiments/1/reviewQueues",
        json={
            "display_name": "My queue",
            "queue_type": "CUSTOM",
            "question_names": [
                "experiments/1/reviewQuestions/q1",
                "experiments/1/reviewQuestions/q2",
            ],
        },
    )
    assert queue.queue_id == "rq1"
    assert queue.queue_type == ReviewQueueType.CUSTOM


def test_create_review_queue_user_sets_owner():
    with mock.patch(_CALL, return_value={**_QUEUE, "queue_type": "USER"}) as mock_call:
        queue = create_review_queue("1", "alice@example.com", queue_type="USER")
    mock_call.assert_called_once_with(
        "POST",
        "experiments/1/reviewQueues",
        json={
            "display_name": "alice@example.com",
            "queue_type": "USER",
            "owner": "alice@example.com",
        },
    )
    assert queue.queue_type == ReviewQueueType.USER


def test_get_review_queue():
    with mock.patch(_CALL, return_value=_QUEUE) as mock_call:
        queue = get_review_queue("1", "rq1")
    mock_call.assert_called_once_with("GET", "experiments/1/reviewQueues/rq1")
    assert queue.display_name == "My queue"


def test_list_review_queues():
    resp = {"review_queues": [_QUEUE], "next_page_token": "tok"}
    with mock.patch(_CALL, return_value=resp) as mock_call:
        queues = list_review_queues("1", filter_string="queue_type=USER", max_results=5)
    mock_call.assert_called_once_with(
        "GET",
        "experiments/1/reviewQueues",
        params={"filter": "queue_type=USER", "page_size": 5, "page_token": None},
    )
    assert [q.queue_id for q in queues] == ["rq1"]
    assert queues.token == "tok"


def test_update_review_queue():
    with mock.patch(_CALL, return_value=_QUEUE) as mock_call:
        update_review_queue("1", "rq1", owner="alice@example.com", question_ids=[])
    mock_call.assert_called_once_with(
        "PATCH",
        "experiments/1/reviewQueues/rq1",
        json={
            "name": "experiments/1/reviewQueues/rq1",
            "owner": "alice@example.com",
            "question_names": [],
        },
        params={"update_mask": "owner,question_names"},
    )


def test_update_review_queue_requires_a_field():
    with mock.patch(_CALL) as mock_call:
        with pytest.raises(MlflowException, match="at least one field"):
            update_review_queue("1", "rq1")
    mock_call.assert_not_called()


def test_delete_review_queue():
    with mock.patch(_CALL, return_value={}) as mock_call:
        delete_review_queue("1", "rq1")
    mock_call.assert_called_once_with("DELETE", "experiments/1/reviewQueues/rq1")


def test_add_review_queue_member():
    with mock.patch(_CALL, return_value={"user": "alice@example.com"}) as mock_call:
        member = add_review_queue_member("1", "rq1", "alice@example.com")
    mock_call.assert_called_once_with(
        "POST", "experiments/1/reviewQueues/rq1/members", json={"user": "alice@example.com"}
    )
    assert member.user == "alice@example.com"


def test_remove_review_queue_member_quotes_user():
    with mock.patch(_CALL, return_value={}) as mock_call:
        remove_review_queue_member("1", "rq1", "a+b/c@example.com")
    mock_call.assert_called_once_with(
        "DELETE",
        "experiments/1/reviewQueues/rq1/members/a%2Bb%2Fc@example.com",
        params={"parent": "experiments/1/reviewQueues/rq1"},
    )


def test_list_review_queue_members():
    resp = {"review_queue_members": [{"user": "alice@example.com"}]}
    with mock.patch(_CALL, return_value=resp) as mock_call:
        members = list_review_queue_members("1", "rq1", page_token="tok")
    mock_call.assert_called_once_with(
        "GET",
        "experiments/1/reviewQueues/rq1/members",
        params={"page_size": None, "page_token": "tok"},
    )
    assert [m.user for m in members] == ["alice@example.com"]
    assert members.token is None


def test_add_review_queue_items():
    parent = "experiments/1/reviewQueues/rq1"
    with mock.patch(_CALL, return_value={"review_queue_items": [_ITEM]}) as mock_call:
        items = add_review_queue_items(
            "1",
            "rq1",
            trace_ids=["tr-1", "trace:/catalog.schema/tr-2"],
            dataset_id="ds1",
            dataset_record_ids=["r1"],
        )
    mock_call.assert_called_once_with(
        "POST",
        f"{parent}/items:batchCreate",
        json={
            "requests": [
                {
                    "parent": parent,
                    "review_queue_item": {"kind": "V4_TRACE", "v4_trace": {"trace_id": "tr-1"}},
                },
                {
                    "parent": parent,
                    "review_queue_item": {
                        "kind": "V4_TRACE",
                        "v4_trace": {"trace_id": "tr-2"},
                    },
                },
                {
                    "parent": parent,
                    "review_queue_item": {
                        "kind": "DATASET_RECORD",
                        "dataset_record": {"dataset_id": "ds1", "dataset_record_id": "r1"},
                    },
                },
            ]
        },
    )
    assert [i.item_id for i in items] == ["i1"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "Specify `trace_ids` and/or `dataset_record_ids`"),
        ({"trace_ids": []}, "Specify `trace_ids` and/or `dataset_record_ids`"),
        ({"dataset_record_ids": ["r1"]}, "`dataset_id` is required"),
    ],
)
def test_add_review_queue_items_validation(kwargs, match):
    with mock.patch(_CALL) as mock_call:
        with pytest.raises(MlflowException, match=match):
            add_review_queue_items("1", "rq1", **kwargs)
    mock_call.assert_not_called()


def test_remove_review_queue_items():
    with mock.patch(_CALL, return_value={}) as mock_call:
        remove_review_queue_items("1", "rq1", ["i1", "i2"])
    mock_call.assert_called_once_with(
        "POST",
        "experiments/1/reviewQueues/rq1/items:batchDelete",
        json={
            "names": [
                "experiments/1/reviewQueues/rq1/items/i1",
                "experiments/1/reviewQueues/rq1/items/i2",
            ]
        },
    )


@pytest.mark.parametrize(
    ("status", "expected_filter"),
    [
        (None, None),
        ("COMPLETE", "status=COMPLETE"),
        (ReviewStatus.PENDING, "status=PENDING"),
    ],
)
def test_list_review_queue_items(status, expected_filter):
    resp = {"review_queue_items": [_ITEM], "next_page_token": "tok"}
    with mock.patch(_CALL, return_value=resp) as mock_call:
        items = list_review_queue_items("1", "rq1", status=status, max_results=20)
    mock_call.assert_called_once_with(
        "GET",
        "experiments/1/reviewQueues/rq1/items",
        params={"filter": expected_filter, "page_size": 20, "page_token": None},
    )
    assert [i.item_id for i in items] == ["i1"]
    assert items.token == "tok"


def test_set_review_queue_item_status():
    resp = {**_ITEM, "status": "COMPLETE", "completed_by": "alice@example.com"}
    with mock.patch(_CALL, return_value=resp) as mock_call:
        item = set_review_queue_item_status("1", "rq1", "i1", "COMPLETE")
    name = "experiments/1/reviewQueues/rq1/items/i1"
    mock_call.assert_called_once_with(
        "PATCH",
        name,
        json={"name": name, "status": "COMPLETE"},
        params={"update_mask": "status"},
    )
    assert item.status == ReviewStatus.COMPLETE
    assert item.completed_by == "alice@example.com"


@pytest.mark.parametrize(
    ("item_kind", "expected"),
    [(None, None), ("V4_TRACE", "V4_TRACE"), (ItemKind.DATASET_RECORD, "DATASET_RECORD")],
)
def test_resolve_effective_review_questions(item_kind, expected):
    with mock.patch(_CALL, return_value={"review_questions": [_QUESTION]}) as mock_call:
        questions = resolve_effective_review_questions("1", "rq1", item_kind=item_kind)
    mock_call.assert_called_once_with(
        "GET",
        "experiments/1/reviewQueues/rq1/effectiveQuestions:resolve",
        params={"item_kind": expected},
    )
    assert [q.question_id for q in questions] == ["q1"]


_MIGRATE_PATH = "experiments/1/reviewQueues:migrateLabelingSession"
_MIGRATION_PLAN = {"review_queue": "experiments/1/reviewQueues/rq-ls1", "items_to_add": 2}
_WARNING = {
    "code": "TEXT_LIST_QUESTION_DROPPED",
    "severity": "WARNING",
    "count": 1,
    "message": "1 text-list question(s) are not migrated.",
}
_READY_DRY_RUN = {"plan": _MIGRATION_PLAN, "issues": [_WARNING], "plan_token": "v1.tok"}
_BLOCKED_DRY_RUN = {
    "plan": _MIGRATION_PLAN,
    "issues": [
        _WARNING,
        {
            "code": "TOO_MANY_MEMBERS",
            "severity": "BLOCKING",
            "count": 18,
            "limit": 16,
            "message": "The review queue would have 18 members, more than a review queue allows.",
        },
    ],
}
_MIGRATED = {"plan": _MIGRATION_PLAN, "issues": [_WARNING], "review_queue": _QUEUE}


def test_migrate_labeling_session_dry_run():
    with mock.patch(_CALL, return_value=_READY_DRY_RUN) as mock_call:
        result = migrate_labeling_session("1", "ls1", dry_run=True)

    mock_call.assert_called_once_with(
        "POST", _MIGRATE_PATH, json={"labeling_session_id": "ls1", "dry_run": True}
    )
    assert result.plan.review_queue_id == "rq-ls1"
    assert [w.code for w in result.warnings] == ["TEXT_LIST_QUESTION_DROPPED"]
    assert result.plan_token == "v1.tok"
    assert result.review_queue is None


def test_migrate_labeling_session_with_plan_token():
    with mock.patch(_CALL, return_value=_MIGRATED) as mock_call:
        result = migrate_labeling_session("1", "ls1", plan_token="v1.tok")

    mock_call.assert_called_once_with(
        "POST",
        _MIGRATE_PATH,
        json={"labeling_session_id": "ls1", "dry_run": False, "plan_token": "v1.tok"},
    )
    assert result.review_queue.queue_id == "rq1"


def test_migrate_labeling_session_previews_then_migrates():
    with mock.patch(_CALL, side_effect=[_READY_DRY_RUN, _MIGRATED]) as mock_call:
        result = migrate_labeling_session("1", "ls1")

    assert mock_call.call_args_list == [
        mock.call("POST", _MIGRATE_PATH, json={"labeling_session_id": "ls1", "dry_run": True}),
        mock.call(
            "POST",
            _MIGRATE_PATH,
            json={"labeling_session_id": "ls1", "dry_run": False, "plan_token": "v1.tok"},
        ),
    ]
    assert result.review_queue.queue_id == "rq1"
    assert [w.code for w in result.warnings] == ["TEXT_LIST_QUESTION_DROPPED"]


def test_migrate_labeling_session_raises_when_blocked():
    with mock.patch(_CALL, return_value=_BLOCKED_DRY_RUN) as mock_call:
        with pytest.raises(
            MlflowException, match="18 members, more than a review queue allows"
        ) as e:
            migrate_labeling_session("1", "ls1")

    mock_call.assert_called_once_with(
        "POST", _MIGRATE_PATH, json={"labeling_session_id": "ls1", "dry_run": True}
    )
    assert e.value.error_code == "INVALID_STATE"
    # Only blocking issues explain the failure.
    assert "text-list" not in e.value.message


def test_migrate_labeling_session_rejects_dry_run_with_plan_token():
    with mock.patch(_CALL) as mock_call:
        with pytest.raises(MlflowException, match="can't be combined with `dry_run=True`"):
            migrate_labeling_session("1", "ls1", dry_run=True, plan_token="v1.tok")
    mock_call.assert_not_called()


@pytest.mark.parametrize(
    ("status_code", "error_code"),
    [(403, "FEATURE_DISABLED"), (409, "ABORTED"), (400, "INVALID_STATE")],
)
def test_migrate_labeling_session_surfaces_server_error_codes(status_code, error_code):
    mlflow.set_tracking_uri("databricks")
    response = requests.Response()
    response.status_code = status_code
    response._content = f'{{"error_code": "{error_code}", "message": "Rejected."}}'.encode()
    with (
        mock.patch.object(_client, "get_databricks_host_creds") as mock_creds,
        mock.patch.object(_client, "http_request", return_value=response) as mock_request,
    ):
        with pytest.raises(MlflowException, match="Rejected.") as exc_info:
            migrate_labeling_session("1", "ls1", plan_token="v1.tok")

    mock_creds.assert_called_once_with("databricks")
    mock_request.assert_called_once()
    assert exc_info.value.error_code == error_code
