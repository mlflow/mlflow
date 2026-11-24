import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.review_queues import (
    ReviewItemType,
    ReviewQueue,
    ReviewQueueItem,
    ReviewQueueType,
    ReviewStatus,
)
from mlflow.protos import review_queues_pb2 as pb


@pytest.mark.parametrize(
    ("enum_cls", "member", "proto_value"),
    [
        (ReviewItemType, ReviewItemType.TRACE, pb.TRACE),
        (ReviewQueueType, ReviewQueueType.USER, pb.USER),
        (ReviewQueueType, ReviewQueueType.CUSTOM, pb.CUSTOM),
        (ReviewStatus, ReviewStatus.PENDING, pb.PENDING),
        (ReviewStatus, ReviewStatus.COMPLETE, pb.COMPLETE),
        (ReviewStatus, ReviewStatus.DECLINED, pb.DECLINED),
    ],
)
def test_enum_proto_round_trip(enum_cls, member, proto_value):
    assert member.to_proto() == proto_value
    assert enum_cls.from_proto(proto_value) is member


@pytest.mark.parametrize(
    ("enum_cls", "bad_proto"),
    [
        (ReviewItemType, pb.REVIEW_ITEM_TYPE_UNSPECIFIED),
        (ReviewQueueType, pb.REVIEW_QUEUE_TYPE_UNSPECIFIED),
        (ReviewStatus, pb.REVIEW_STATUS_UNSPECIFIED),
        (ReviewStatus, 999),
    ],
)
def test_enum_from_proto_rejects_invalid(enum_cls, bad_proto):
    with pytest.raises(MlflowException, match="must be"):
        enum_cls.from_proto(bad_proto)


@pytest.mark.parametrize(
    "queue",
    [
        ReviewQueue(
            queue_id="rq-1",
            experiment_id="1",
            name="alice",
            queue_type=ReviewQueueType.USER,
            created_by=None,
            creation_time_ms=10,
            last_update_time_ms=10,
            users=["alice"],
            schema_ids=[],
        ),
        ReviewQueue(
            queue_id="rq-2",
            experiment_id="7",
            name="Q3 review",
            queue_type=ReviewQueueType.CUSTOM,
            created_by="kris",
            creation_time_ms=10,
            last_update_time_ms=20,
            users=["bob", "carol"],
            schema_ids=["ls-1", "ls-2"],
        ),
        ReviewQueue(
            queue_id="rq-3",
            experiment_id="7",
            name="custom-empty",
            queue_type=ReviewQueueType.CUSTOM,
            created_by=None,
            creation_time_ms=10,
            last_update_time_ms=10,
            users=[],
            schema_ids=[],
        ),
    ],
)
def test_review_queue_proto_round_trip(queue):
    assert ReviewQueue.from_proto(queue.to_proto()) == queue


def test_review_queue_created_by_none_absent_on_wire():
    queue = ReviewQueue(
        queue_id="rq-1",
        experiment_id="1",
        name="alice",
        queue_type=ReviewQueueType.USER,
        created_by=None,
        creation_time_ms=10,
        last_update_time_ms=10,
        users=["alice"],
        schema_ids=[],
    )
    assert queue.to_proto().HasField("created_by") is False


@pytest.mark.parametrize(
    "item",
    [
        ReviewQueueItem(
            queue_id="rq-1",
            item_type=ReviewItemType.TRACE,
            item_id="tr-1",
            status=ReviewStatus.PENDING,
            creation_time_ms=1,
            last_update_time_ms=1,
        ),
        ReviewQueueItem(
            queue_id="rq-1",
            item_type=ReviewItemType.TRACE,
            item_id="tr-2",
            status=ReviewStatus.COMPLETE,
            creation_time_ms=1,
            last_update_time_ms=2,
            completed_by="bob",
            completed_time_ms=2,
        ),
    ],
)
def test_review_queue_item_proto_round_trip(item):
    assert ReviewQueueItem.from_proto(item.to_proto()) == item


def test_review_queue_item_pending_attribution_absent_on_wire():
    item = ReviewQueueItem(
        queue_id="rq-1",
        item_type=ReviewItemType.TRACE,
        item_id="tr-1",
        status=ReviewStatus.PENDING,
        creation_time_ms=1,
        last_update_time_ms=1,
    )
    proto = item.to_proto()
    assert proto.HasField("completed_by") is False
    assert proto.HasField("completed_time_ms") is False
