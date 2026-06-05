from unittest.mock import patch

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.review_queues import (
    ReviewQueue,
    ReviewQueueItem,
    ReviewQueueType,
    ReviewStatus,
    ReviewTargetType,
    add_traces_to_review_queue,
    create_review_queue,
    delete_review_queue,
    get_or_create_user_queue,
    get_review_queue,
    list_review_queue_traces,
    list_review_queues,
    remove_traces_from_review_queue,
    set_review_queue_trace_status,
    update_review_queue,
)
from mlflow.store.entities.paged_list import PagedList

_BASE = "mlflow.genai.review_queues.TracingClient"


def _queue(queue_id="rq-1", name="alice", queue_type=ReviewQueueType.USER):
    return ReviewQueue(
        queue_id=queue_id,
        experiment_id="1",
        name=name,
        queue_type=queue_type,
        created_by=None,
        creation_time_ms=1,
        last_update_time_ms=1,
        users=["alice"],
        schema_ids=[],
    )


def _item(target_id="tr-1", status=ReviewStatus.PENDING):
    return ReviewQueueItem(
        queue_id="rq-1",
        target_type=ReviewTargetType.TRACE,
        target_id=target_id,
        status=status,
        creation_time_ms=1,
        last_update_time_ms=1,
    )


# --------------------------------------------------------------------------
# Delegation: each fluent function forwards to the right TracingClient method.
# --------------------------------------------------------------------------


def test_create_review_queue_delegates():
    with patch(f"{_BASE}._create_review_queue", return_value=_queue()) as m:
        result = create_review_queue(
            "Q3",
            queue_type="custom",
            users=["bob"],
            schema_ids=["ls-1"],
            created_by="kris",
            experiment_id="1",
        )
    assert result.queue_id == "rq-1"
    args, kwargs = m.call_args
    assert args == ("1",)
    assert kwargs == {
        "name": "Q3",
        "queue_type": "custom",
        "users": ["bob"],
        "schema_ids": ["ls-1"],
        "created_by": "kris",
    }


def test_create_review_queue_defaults_experiment_id_to_current():
    with (
        patch(f"{_BASE}._create_review_queue", return_value=_queue()) as m,
        patch("mlflow.tracking.fluent._get_experiment_id", return_value="42"),
    ):
        create_review_queue("alice", queue_type="user")
    assert m.call_args[0] == ("42",)


def test_get_or_create_user_queue_delegates():
    with patch(f"{_BASE}._get_or_create_user_queue", return_value=_queue()) as m:
        get_or_create_user_queue("alice", created_by="kris", experiment_id="1")
    assert m.call_args[0] == ("1",)
    assert m.call_args[1] == {"user": "alice", "created_by": "kris"}


def test_get_review_queue_by_id_delegates():
    with patch(f"{_BASE}._get_review_queue", return_value=_queue()) as m:
        get_review_queue("rq-1")
    m.assert_called_once_with("rq-1")


def test_get_review_queue_by_name_delegates():
    with patch(f"{_BASE}._get_review_queue_by_name", return_value=_queue()) as m:
        get_review_queue(name="alice", experiment_id="1")
    m.assert_called_once_with("1", "alice")


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # neither queue_id nor name
        {"queue_id": "rq-1", "name": "alice"},  # both
    ],
)
def test_get_review_queue_requires_exactly_one_selector(kwargs):
    with pytest.raises(MlflowException, match="exactly one of `queue_id` or `name`"):
        get_review_queue(**kwargs)


def test_list_review_queues_delegates():
    with patch(f"{_BASE}._list_review_queues", return_value=PagedList([_queue()], None)) as m:
        list_review_queues(user="bob", experiment_id="1", max_results=5)
    assert m.call_args[0] == ("1",)
    assert m.call_args[1] == {"user": "bob", "max_results": 5, "page_token": None}


def test_update_review_queue_delegates():
    with patch(f"{_BASE}._update_review_queue", return_value=_queue()) as m:
        update_review_queue("rq-1", users=["dave"])
    m.assert_called_once_with("rq-1", users=["dave"], schema_ids=None)


def test_delete_review_queue_delegates():
    with patch(f"{_BASE}._delete_review_queue", return_value=None) as m:
        delete_review_queue("rq-1")
    m.assert_called_once_with("rq-1")


def test_add_traces_maps_trace_ids_to_target_ids():
    with patch(f"{_BASE}._add_traces_to_review_queue", return_value=[_item()]) as m:
        add_traces_to_review_queue("rq-1", trace_ids=["tr-1", "tr-2"])
    m.assert_called_once_with("rq-1", target_ids=["tr-1", "tr-2"])


def test_remove_traces_maps_trace_ids_to_target_ids():
    with patch(f"{_BASE}._remove_traces_from_review_queue", return_value=None) as m:
        remove_traces_from_review_queue("rq-1", trace_ids=["tr-2"])
    m.assert_called_once_with("rq-1", target_ids=["tr-2"])


def test_list_review_queue_traces_delegates():
    with patch(f"{_BASE}._list_review_queue_traces", return_value=PagedList([_item()], None)) as m:
        list_review_queue_traces("rq-1", status="pending", max_results=3)
    m.assert_called_once_with("rq-1", status="pending", max_results=3, page_token=None)


def test_set_status_maps_trace_id_to_target_id():
    with patch(f"{_BASE}._set_review_queue_trace_status", return_value=_item()) as m:
        set_review_queue_trace_status(
            "rq-1", trace_id="tr-1", status="complete", completed_by="bob"
        )
    m.assert_called_once_with("rq-1", target_id="tr-1", status="complete", completed_by="bob")


# --------------------------------------------------------------------------
# End-to-end against a real sqlite tracking store (fluent -> client -> store).
# --------------------------------------------------------------------------


@pytest.fixture
def sqlite_tracking(tmp_path):
    original = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield
    mlflow.set_tracking_uri(original)


def test_sdk_end_to_end(sqlite_tracking):
    exp_id = mlflow.create_experiment("rq-sdk-e2e")

    custom = create_review_queue(
        "Q3 review", queue_type="custom", users=["bob"], experiment_id=exp_id
    )
    assert custom.queue_type == ReviewQueueType.CUSTOM

    user_queue = get_or_create_user_queue("alice", experiment_id=exp_id)
    assert get_or_create_user_queue("alice", experiment_id=exp_id).queue_id == user_queue.queue_id

    items = add_traces_to_review_queue(custom.queue_id, trace_ids=["tr-1", "tr-2"])
    # The returned items cover every requested trace_id, in request order.
    assert [i.target_id for i in items] == ["tr-1", "tr-2"]

    set_review_queue_trace_status(
        custom.queue_id, trace_id="tr-1", status="complete", completed_by="Bob"
    )
    pending = list_review_queue_traces(custom.queue_id, status="pending")
    assert {i.target_id for i in pending} == {"tr-2"}

    assert {q.name for q in list_review_queues(user="alice", experiment_id=exp_id)} == {"alice"}

    update_review_queue(custom.queue_id, users=["dave"])
    assert get_review_queue(custom.queue_id).users == ["dave"]
    assert get_review_queue(name="Q3 review", experiment_id=exp_id).queue_id == custom.queue_id

    remove_traces_from_review_queue(custom.queue_id, trace_ids=["tr-2"])
    delete_review_queue(custom.queue_id)
    with pytest.raises(MlflowException, match="not found"):
        get_review_queue(custom.queue_id)
