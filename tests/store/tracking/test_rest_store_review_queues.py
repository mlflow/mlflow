import json
from unittest import mock

from mlflow.protos import review_queues_pb2 as pb
from mlflow.store.tracking.rest_store import RestStore
from mlflow.utils.rest_utils import MlflowHostCreds

_CALL = "mlflow.store.tracking.rest_store.call_endpoint"


def _store():
    return RestStore(lambda: MlflowHostCreds("https://hello"))


def _capture(response_proto):
    """Patch call_endpoint, return (mock, captured) where captured holds the
    endpoint + parsed request proto from the single call.
    """
    captured = {}

    def _side_effect(host_creds, endpoint, method, json_body, resp, **kwargs):
        captured["endpoint"] = endpoint
        captured["method"] = method
        captured["body"] = json.loads(json_body)
        resp.CopyFrom(response_proto)
        return resp

    return mock.patch(_CALL, side_effect=_side_effect), captured


def test_create_review_queue_builds_request_and_parses_response():
    resp = pb.CreateReviewQueue.Response(
        review_queue=pb.ReviewQueue(
            queue_id="rq-1",
            experiment_id="1",
            name="Q3",
            queue_type=pb.CUSTOM,
            users=["bob"],
            schema_ids=["ls-1"],
            creation_time_ms=1,
            last_update_time_ms=1,
        )
    )
    patcher, captured = _capture(resp)
    with patcher:
        result = _store().create_review_queue(
            "1", name="Q3", queue_type="custom", users=["bob"], schema_ids=["ls-1"], created_by="k"
        )
    assert captured["endpoint"] == "/api/3.0/mlflow/review-queues/create"
    assert captured["body"]["queue_type"] == "CUSTOM"
    assert captured["body"]["users"] == ["bob"]
    assert captured["body"]["schema_ids"] == ["ls-1"]
    assert captured["body"]["created_by"] == "k"
    assert result.queue_id == "rq-1"
    assert result.users == ["bob"]


def test_get_or_create_user_queue_endpoint():
    resp = pb.GetOrCreateUserQueue.Response(
        review_queue=pb.ReviewQueue(queue_id="rq-1", name="alice", queue_type=pb.USER)
    )
    patcher, captured = _capture(resp)
    with patcher:
        _store().get_or_create_user_queue("1", user="alice")
    assert captured["endpoint"] == "/api/3.0/mlflow/review-queues/get-or-create-user"
    assert captured["body"]["user"] == "alice"


def test_update_review_queue_sets_flags():
    resp = pb.UpdateReviewQueue.Response(
        review_queue=pb.ReviewQueue(queue_id="rq-1", queue_type=pb.CUSTOM)
    )
    patcher, captured = _capture(resp)
    with patcher:
        _store().update_review_queue("rq-1", users=["dave"])
    assert captured["endpoint"] == "/api/3.0/mlflow/review-queues/update"
    assert captured["body"]["update_users"] is True
    assert captured["body"]["users"] == ["dave"]
    # schema_ids left untouched -> flag absent
    assert "update_schema_ids" not in captured["body"]


def test_update_review_queue_empty_list_still_flags():
    resp = pb.UpdateReviewQueue.Response(
        review_queue=pb.ReviewQueue(queue_id="rq-1", queue_type=pb.CUSTOM)
    )
    patcher, captured = _capture(resp)
    with patcher:
        _store().update_review_queue("rq-1", schema_ids=[])
    assert captured["body"]["update_schema_ids"] is True
    # empty repeated field serializes as absent in JSON, but the flag carries intent
    assert captured["body"].get("schema_ids", []) == []


def test_add_items_sends_item_type_and_ids():
    resp = pb.AddItemsToReviewQueue.Response(
        items=[
            pb.ReviewQueueItem(
                queue_id="rq-1", item_id="tr-1", item_type=pb.TRACE, status=pb.PENDING
            )
        ]
    )
    patcher, captured = _capture(resp)
    with patcher:
        items = _store().add_items_to_review_queue("rq-1", item_ids=["tr-1"])
    assert captured["endpoint"] == "/api/3.0/mlflow/review-queues/items/add"
    assert captured["body"]["item_type"] == "TRACE"
    assert captured["body"]["item_ids"] == ["tr-1"]
    assert items[0].item_id == "tr-1"


def test_list_review_queue_items_status_filter_and_pagination():
    resp = pb.ListReviewQueueItems.Response(
        items=[
            pb.ReviewQueueItem(
                queue_id="rq-1", item_id="tr-1", item_type=pb.TRACE, status=pb.COMPLETE
            )
        ],
        next_page_token="tok",
    )
    patcher, captured = _capture(resp)
    with patcher:
        page = _store().list_review_queue_items("rq-1", status="complete", max_results=5)
    assert captured["endpoint"] == "/api/3.0/mlflow/review-queues/items/list"
    assert captured["body"]["status"] == "COMPLETE"
    assert captured["body"]["max_results"] == 5
    assert page.token == "tok"
    assert page[0].item_id == "tr-1"


def test_set_status_sends_completed_by():
    resp = pb.SetReviewQueueItemStatus.Response(
        item=pb.ReviewQueueItem(
            queue_id="rq-1",
            item_id="tr-1",
            item_type=pb.TRACE,
            status=pb.COMPLETE,
            completed_by="bob",
        )
    )
    patcher, captured = _capture(resp)
    with patcher:
        item = _store().set_review_queue_item_status(
            "rq-1", item_id="tr-1", status="complete", completed_by="bob"
        )
    assert captured["endpoint"] == "/api/3.0/mlflow/review-queues/items/set-status"
    assert captured["body"]["status"] == "COMPLETE"
    assert captured["body"]["completed_by"] == "bob"
    assert item.completed_by == "bob"


def test_list_review_queues_parses_paged_response():
    resp = pb.ListReviewQueues.Response(
        review_queues=[pb.ReviewQueue(queue_id="rq-1", name="alice", queue_type=pb.USER)],
        next_page_token="",
    )
    patcher, captured = _capture(resp)
    with patcher:
        page = _store().list_review_queues("1", user="alice")
    assert captured["endpoint"] == "/api/3.0/mlflow/review-queues/list"
    assert captured["body"]["user"] == "alice"
    # empty next_page_token -> None
    assert page.token is None
    assert page[0].queue_id == "rq-1"


def test_list_review_queues_sends_item_filter():
    resp = pb.ListReviewQueues.Response(
        review_queues=[pb.ReviewQueue(queue_id="rq-1", name="q", queue_type=pb.CUSTOM)],
        next_page_token="",
    )
    patcher, captured = _capture(resp)
    with patcher:
        page = _store().list_review_queues("1", item_id="tr-1")
    assert captured["endpoint"] == "/api/3.0/mlflow/review-queues/list"
    assert captured["body"]["item_id"] == "tr-1"
    assert page[0].queue_id == "rq-1"


def test_delete_review_queue_endpoint():
    patcher, captured = _capture(pb.DeleteReviewQueue.Response())
    with patcher:
        _store().delete_review_queue("rq-1")
    assert captured["endpoint"] == "/api/3.0/mlflow/review-queues/delete"
    assert captured["body"]["queue_id"] == "rq-1"


def test_get_review_queue_endpoint():
    resp = pb.GetReviewQueue.Response(
        review_queue=pb.ReviewQueue(queue_id="rq-1", name="alice", queue_type=pb.USER)
    )
    patcher, captured = _capture(resp)
    with patcher:
        result = _store().get_review_queue("rq-1")
    assert captured["endpoint"] == "/api/3.0/mlflow/review-queues/get"
    assert captured["body"]["queue_id"] == "rq-1"
    assert result.queue_id == "rq-1"


def test_get_review_queue_by_name_endpoint():
    resp = pb.GetReviewQueueByName.Response(
        review_queue=pb.ReviewQueue(queue_id="rq-1", name="alice", queue_type=pb.USER)
    )
    patcher, captured = _capture(resp)
    with patcher:
        result = _store().get_review_queue_by_name("1", name="alice")
    assert captured["endpoint"] == "/api/3.0/mlflow/review-queues/get-by-name"
    assert captured["body"]["experiment_id"] == "1"
    assert captured["body"]["name"] == "alice"
    assert result.queue_id == "rq-1"


def test_remove_items_endpoint():
    patcher, captured = _capture(pb.RemoveItemsFromReviewQueue.Response())
    with patcher:
        _store().remove_items_from_review_queue("rq-1", item_ids=["tr-1", "tr-2"])
    assert captured["endpoint"] == "/api/3.0/mlflow/review-queues/items/remove"
    assert captured["body"]["queue_id"] == "rq-1"
    assert captured["body"]["item_ids"] == ["tr-1", "tr-2"]
