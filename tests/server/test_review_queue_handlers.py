import json
from types import SimpleNamespace
from unittest import mock

import pytest

from mlflow.genai.review_queues import (
    ReviewItemType,
    ReviewQueue,
    ReviewQueueItem,
    ReviewQueueType,
    ReviewStatus,
)
from mlflow.protos.review_queues_pb2 import (
    COMPLETE,
    CUSTOM,
    PENDING,
    REVIEW_QUEUE_TYPE_UNSPECIFIED,
    REVIEW_STATUS_UNSPECIFIED,
    TRACE,
    USER,
    AddItemsToReviewQueue,
    CreateReviewQueue,
    DeleteReviewQueue,
    GetOrCreateUserQueue,
    GetReviewQueue,
    GetReviewQueueByName,
    ListReviewQueueItems,
    ListReviewQueues,
    RemoveItemsFromReviewQueue,
    SetReviewQueueItemStatus,
    UpdateReviewQueue,
)
from mlflow.server.handlers import (
    _add_items_to_review_queue,
    _create_review_queue,
    _delete_review_queue,
    _get_or_create_user_queue,
    _get_review_queue,
    _get_review_queue_by_name,
    _list_review_queue_items,
    _list_review_queues,
    _remove_items_from_review_queue,
    _set_review_queue_item_status,
    _update_review_queue,
)
from mlflow.store.entities.paged_list import PagedList

_BASE_PATCH = "mlflow.server.handlers"


def _queue_entity(
    queue_id="rq-1",
    experiment_id="1",
    name="alice",
    queue_type=ReviewQueueType.USER,
    created_by="alice@example.com",
    users=None,
    schema_ids=None,
):
    return ReviewQueue(
        queue_id=queue_id,
        experiment_id=experiment_id,
        name=name,
        queue_type=queue_type,
        created_by=created_by,
        creation_time_ms=1000,
        last_update_time_ms=1000,
        users=users if users is not None else ["alice"],
        schema_ids=schema_ids if schema_ids is not None else [],
    )


def _item_entity(item_id="tr-1", status=ReviewStatus.PENDING, completed_by=None):
    return ReviewQueueItem(
        queue_id="rq-1",
        item_type=ReviewItemType.TRACE,
        item_id=item_id,
        status=status,
        creation_time_ms=1000,
        last_update_time_ms=1000,
        completed_by=completed_by,
        completed_time_ms=2000 if completed_by else None,
    )


def _run_handler(handler, request_message, store_attr, return_value):
    with (
        mock.patch(f"{_BASE_PATCH}._get_tracking_store") as mock_store,
        mock.patch(f"{_BASE_PATCH}._get_request_message", return_value=request_message),
        # `_create_review_queue` stamps the owner from the authenticated user
        # (read off `flask.g`); these are routing tests, so default to no auth.
        # Owner-stamping behavior itself is covered in `test_handlers.py`.
        mock.patch(f"{_BASE_PATCH}._get_request_username", return_value=None),
    ):
        getattr(mock_store.return_value, store_attr).return_value = return_value
        response = handler()
        return mock_store.return_value, response


def test_create_review_queue_routes_custom_with_users_and_schemas():
    request_message = CreateReviewQueue(
        experiment_id="1",
        name="Q3",
        queue_type=CUSTOM,
        users=["bob", "carol"],
        schema_ids=["ls-1"],
    )
    entity = _queue_entity(
        queue_id="rq-2",
        name="Q3",
        queue_type=ReviewQueueType.CUSTOM,
        users=["bob", "carol"],
        schema_ids=["ls-1"],
    )
    store, response = _run_handler(
        _create_review_queue, request_message, "create_review_queue", entity
    )
    kwargs = store.create_review_queue.call_args[1]
    assert kwargs["experiment_id"] == "1"
    assert kwargs["name"] == "Q3"
    assert kwargs["queue_type"] == ReviewQueueType.CUSTOM
    assert kwargs["users"] == ["bob", "carol"]
    assert kwargs["schema_ids"] == ["ls-1"]

    body = json.loads(response.get_data())
    assert body["review_queue"]["queue_id"] == "rq-2"
    assert body["review_queue"]["queue_type"] == "CUSTOM"
    assert body["review_queue"]["users"] == ["bob", "carol"]


def test_create_review_queue_omits_created_by_when_absent():
    request_message = CreateReviewQueue(experiment_id="1", name="alice", queue_type=USER)
    store, _ = _run_handler(
        _create_review_queue, request_message, "create_review_queue", _queue_entity()
    )
    assert "created_by" not in store.create_review_queue.call_args[1]


def test_create_review_queue_rejects_unspecified_type():
    request_message = CreateReviewQueue(
        experiment_id="1", name="x", queue_type=REVIEW_QUEUE_TYPE_UNSPECIFIED
    )
    with (
        mock.patch(f"{_BASE_PATCH}._get_tracking_store") as mock_store,
        mock.patch(f"{_BASE_PATCH}._get_request_message", return_value=request_message),
    ):
        response = _create_review_queue()
        assert response.status_code == 400
        assert json.loads(response.get_data())["error_code"] == "INVALID_PARAMETER_VALUE"
        mock_store.return_value.create_review_queue.assert_not_called()


def test_get_or_create_user_queue_routes():
    request_message = GetOrCreateUserQueue(experiment_id="1", user="Alice")
    store, response = _run_handler(
        _get_or_create_user_queue,
        request_message,
        "get_or_create_user_queue",
        _queue_entity(),
    )
    args, kwargs = store.get_or_create_user_queue.call_args
    assert args == ("1",)
    assert kwargs == {"user": "Alice"}
    assert json.loads(response.get_data())["review_queue"]["queue_id"] == "rq-1"


def test_get_review_queue_routes():
    request_message = GetReviewQueue(queue_id="rq-1")
    store, response = _run_handler(
        _get_review_queue, request_message, "get_review_queue", _queue_entity()
    )
    store.get_review_queue.assert_called_once_with("rq-1")
    assert json.loads(response.get_data())["review_queue"]["name"] == "alice"


def test_get_review_queue_by_name_routes():
    request_message = GetReviewQueueByName(experiment_id="1", name="alice")
    store, response = _run_handler(
        _get_review_queue_by_name,
        request_message,
        "get_review_queue_by_name",
        _queue_entity(),
    )
    store.get_review_queue_by_name.assert_called_once_with("1", name="alice")
    assert json.loads(response.get_data())["review_queue"]["queue_id"] == "rq-1"


def test_list_review_queues_passes_user_filter_and_paginates():
    request_message = ListReviewQueues(experiment_id="1", user="bob", max_results=10)
    paged = PagedList([_queue_entity()], token="next-tok")
    store, response = _run_handler(
        _list_review_queues, request_message, "list_review_queues", paged
    )
    kwargs = store.list_review_queues.call_args[1]
    assert kwargs["user"] == "bob"
    assert kwargs["max_results"] == 10
    assert kwargs["page_token"] is None
    body = json.loads(response.get_data())
    assert body["next_page_token"] == "next-tok"
    assert len(body["review_queues"]) == 1


def test_list_review_queues_defaults_user_to_none():
    request_message = ListReviewQueues(experiment_id="1")
    store, _ = _run_handler(
        _list_review_queues, request_message, "list_review_queues", PagedList([], token=None)
    )
    assert store.list_review_queues.call_args[1]["user"] is None


@pytest.mark.parametrize(
    ("update_users", "users", "update_schema_ids", "schema_ids", "exp_users", "exp_schema_ids"),
    [
        # flag off -> None (untouched); flag on -> replace with the (possibly empty) list.
        (True, ["dave"], False, [], ["dave"], None),
        (True, [], False, [], [], None),
        (False, [], True, ["ls-1"], None, ["ls-1"]),
        (False, [], True, [], None, []),
        (True, ["dave"], True, ["ls-1"], ["dave"], ["ls-1"]),
        (False, [], False, [], None, None),
    ],
)
def test_update_review_queue_flag_gated_replacement(
    update_users, users, update_schema_ids, schema_ids, exp_users, exp_schema_ids
):
    request_message = UpdateReviewQueue(
        queue_id="rq-2",
        update_users=update_users,
        users=users,
        update_schema_ids=update_schema_ids,
        schema_ids=schema_ids,
    )
    entity = _queue_entity(queue_id="rq-2", queue_type=ReviewQueueType.CUSTOM)
    store, _ = _run_handler(_update_review_queue, request_message, "update_review_queue", entity)
    kwargs = store.update_review_queue.call_args[1]
    assert kwargs["users"] == exp_users
    assert kwargs["schema_ids"] == exp_schema_ids


def test_delete_review_queue_routes():
    request_message = DeleteReviewQueue(queue_id="rq-1")
    store, response = _run_handler(
        _delete_review_queue, request_message, "delete_review_queue", None
    )
    store.delete_review_queue.assert_called_once_with("rq-1")
    assert response.status_code == 200


def _trace_info(trace_id, experiment_id="1"):
    # The add-items handler only reads .trace_id / .experiment_id off the
    # batch_get_trace_infos result, so a lightweight stand-in suffices.
    return SimpleNamespace(trace_id=trace_id, experiment_id=experiment_id)


def _run_add_items(request_message, *, queue_experiment_id="1", trace_infos, add_return):
    with (
        mock.patch(f"{_BASE_PATCH}._get_tracking_store") as mock_store,
        mock.patch(f"{_BASE_PATCH}._get_request_message", return_value=request_message),
    ):
        store = mock_store.return_value
        store.get_review_queue.return_value = _queue_entity(experiment_id=queue_experiment_id)
        store.batch_get_trace_infos.return_value = trace_infos
        store.add_items_to_review_queue.return_value = add_return
        return store, _add_items_to_review_queue()


def test_add_items_defaults_item_type_to_trace():
    request_message = AddItemsToReviewQueue(queue_id="rq-1", item_ids=["tr-1", "tr-2"])
    store, response = _run_add_items(
        request_message,
        trace_infos=[_trace_info("tr-1"), _trace_info("tr-2")],
        add_return=[_item_entity("tr-1"), _item_entity("tr-2")],
    )
    kwargs = store.add_items_to_review_queue.call_args[1]
    assert kwargs["item_ids"] == ["tr-1", "tr-2"]
    # Unset item_type is not forwarded; the store applies its TRACE default.
    assert "item_type" not in kwargs
    # The existence check ran against exactly the requested ids.
    store.batch_get_trace_infos.assert_called_once_with(["tr-1", "tr-2"])
    body = json.loads(response.get_data())
    assert [i["item_id"] for i in body["items"]] == ["tr-1", "tr-2"]


def test_add_items_forwards_explicit_item_type():
    request_message = AddItemsToReviewQueue(queue_id="rq-1", item_type=TRACE, item_ids=["tr-1"])
    store, _ = _run_add_items(
        request_message,
        trace_infos=[_trace_info("tr-1")],
        add_return=[_item_entity("tr-1")],
    )
    assert store.add_items_to_review_queue.call_args[1]["item_type"] == ReviewItemType.TRACE
    store.batch_get_trace_infos.assert_called_once_with(["tr-1"])


def test_add_items_rejects_traces_not_in_queue_experiment():
    # tr-2 doesn't exist anywhere; tr-3 exists but in a different experiment.
    request_message = AddItemsToReviewQueue(queue_id="rq-1", item_ids=["tr-1", "tr-2", "tr-3"])
    store, response = _run_add_items(
        request_message,
        queue_experiment_id="1",
        trace_infos=[_trace_info("tr-1", "1"), _trace_info("tr-3", "2")],
        add_return=[],
    )
    assert response.status_code == 404
    body = json.loads(response.get_data())
    assert body["error_code"] == "RESOURCE_DOES_NOT_EXIST"
    assert "tr-2" in body["message"]
    assert "tr-3" in body["message"]
    store.add_items_to_review_queue.assert_not_called()


def test_remove_items_routes():
    request_message = RemoveItemsFromReviewQueue(queue_id="rq-1", item_ids=["tr-2"])
    store, response = _run_handler(
        _remove_items_from_review_queue,
        request_message,
        "remove_items_from_review_queue",
        None,
    )
    store.remove_items_from_review_queue.assert_called_once_with("rq-1", item_ids=["tr-2"])
    assert response.status_code == 200


def test_list_review_queue_items_filters_status():
    request_message = ListReviewQueueItems(queue_id="rq-1", status=COMPLETE)
    paged = PagedList([_item_entity("tr-1", ReviewStatus.COMPLETE, "bob")], token=None)
    store, response = _run_handler(
        _list_review_queue_items, request_message, "list_review_queue_items", paged
    )
    assert store.list_review_queue_items.call_args[1]["status"] == ReviewStatus.COMPLETE
    body = json.loads(response.get_data())
    assert body["items"][0]["status"] == "COMPLETE"
    assert body["items"][0]["completed_by"] == "bob"


def test_list_review_queue_items_no_status_filter():
    request_message = ListReviewQueueItems(queue_id="rq-1")
    store, _ = _run_handler(
        _list_review_queue_items,
        request_message,
        "list_review_queue_items",
        PagedList([], token=None),
    )
    assert store.list_review_queue_items.call_args[1]["status"] is None


def test_set_status_forwards_completed_by():
    request_message = SetReviewQueueItemStatus(
        queue_id="rq-1", item_id="tr-1", status=COMPLETE, completed_by="bob"
    )
    entity = _item_entity("tr-1", ReviewStatus.COMPLETE, "bob")
    store, response = _run_handler(
        _set_review_queue_item_status,
        request_message,
        "set_review_queue_item_status",
        entity,
    )
    kwargs = store.set_review_queue_item_status.call_args[1]
    assert kwargs["item_id"] == "tr-1"
    assert kwargs["status"] == ReviewStatus.COMPLETE
    assert kwargs["completed_by"] == "bob"
    body = json.loads(response.get_data())
    assert body["item"]["status"] == "COMPLETE"


def test_set_status_reopen_omits_completed_by():
    request_message = SetReviewQueueItemStatus(queue_id="rq-1", item_id="tr-1", status=PENDING)
    store, _ = _run_handler(
        _set_review_queue_item_status,
        request_message,
        "set_review_queue_item_status",
        _item_entity("tr-1", ReviewStatus.PENDING),
    )
    assert store.set_review_queue_item_status.call_args[1]["completed_by"] is None


def test_set_status_rejects_unspecified_status():
    # `status` is required in the proto but is intentionally absent from the
    # handler's input schema; rejection of the proto2 zero-value is delegated
    # to ReviewStatus.from_proto, so an absent/UNSPECIFIED status must 400.
    request_message = SetReviewQueueItemStatus(
        queue_id="rq-1", item_id="tr-1", status=REVIEW_STATUS_UNSPECIFIED
    )
    with (
        mock.patch(f"{_BASE_PATCH}._get_tracking_store") as mock_store,
        mock.patch(f"{_BASE_PATCH}._get_request_message", return_value=request_message),
    ):
        response = _set_review_queue_item_status()
        assert response.status_code == 400
        assert json.loads(response.get_data())["error_code"] == "INVALID_PARAMETER_VALUE"
        mock_store.return_value.set_review_queue_item_status.assert_not_called()
