import time

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.label_schemas.label_schemas import InputPassFail
from mlflow.genai.review_queues import ReviewItemType, ReviewQueueType, ReviewStatus
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.tracking.dbmodels.models import SqlReviewQueue

from tests.store.tracking.sqlalchemy_store.conftest import _create_experiments, _create_trace

pytestmark = pytest.mark.notrackingurimock


def _pass_fail(store, experiment_id, name):
    return store.create_label_schema(
        experiment_id=experiment_id,
        name=name,
        type="feedback",
        input=InputPassFail(positive_label="Yes", negative_label="No"),
    )


def _assert_error_code(exc_info, error_code):
    assert exc_info.value.error_code == ErrorCode.Name(error_code)


# --------------------------------------------------------------------------
# Create
# --------------------------------------------------------------------------


def test_create_user_queue_defaults_to_all_schemas_and_single_user(store):
    exp_id = _create_experiments(store, "user_queue")
    queue = store.create_review_queue(exp_id, name="Alice", queue_type="user")

    assert queue.queue_id.startswith(SqlReviewQueue.QUEUE_ID_PREFIX)
    assert queue.experiment_id == exp_id
    # The user identifier is normalized (lowercased) and becomes the name.
    assert queue.name == "alice"
    assert queue.queue_type == ReviewQueueType.USER
    assert queue.users == ["alice"]
    # User queues attach no schemas; they resolve to all at read time.
    assert queue.schema_ids == []
    assert queue.creation_time_ms > 0
    assert queue.last_update_time_ms == queue.creation_time_ms


def test_create_custom_queue_with_users_and_schema_subset(store):
    exp_id = _create_experiments(store, "custom_queue")
    ls1 = _pass_fail(store, exp_id, "quality")
    ls2 = _pass_fail(store, exp_id, "safety")

    queue = store.create_review_queue(
        exp_id,
        name="Q3 hallucinations",
        queue_type="custom",
        created_by="kris",
        users=["Bob", "Carol"],
        schema_ids=[ls1.schema_id],
    )

    assert queue.queue_type == ReviewQueueType.CUSTOM
    assert queue.name == "Q3 hallucinations"
    assert queue.created_by == "kris"
    assert queue.users == ["bob", "carol"]
    assert queue.schema_ids == [ls1.schema_id]
    assert ls2.schema_id not in queue.schema_ids


def test_create_custom_queue_with_no_users(store):
    exp_id = _create_experiments(store, "open_queue")
    queue = store.create_review_queue(exp_id, name="open", queue_type="custom")
    assert queue.users == []
    assert queue.schema_ids == []


def test_create_user_queue_rejects_schema_ids(store):
    exp_id = _create_experiments(store, "user_queue_schema")
    ls = _pass_fail(store, exp_id, "quality")
    with pytest.raises(MlflowException, match="cannot have explicitly-attached schemas") as exc:
        store.create_review_queue(
            exp_id, name="alice", queue_type="user", schema_ids=[ls.schema_id]
        )
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


def test_create_user_queue_rejects_mismatched_users(store):
    exp_id = _create_experiments(store, "user_queue_users")
    with pytest.raises(MlflowException, match="exactly one assigned user equal to its name") as exc:
        store.create_review_queue(exp_id, name="alice", queue_type="user", users=["bob"])
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


def test_create_custom_queue_rejects_reserved_name(store):
    exp_id = _create_experiments(store, "reserved")
    with pytest.raises(MlflowException, match="reserved queue name") as exc:
        store.create_review_queue(exp_id, name="default", queue_type="custom")
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


def test_create_user_queue_allows_reserved_default_name(store):
    exp_id = _create_experiments(store, "default_user_queue")
    queue = store.create_review_queue(exp_id, name="default", queue_type="user")
    assert queue.name == "default"
    assert queue.users == ["default"]


def test_create_rejects_unknown_queue_type(store):
    exp_id = _create_experiments(store, "bad_type")
    with pytest.raises(MlflowException, match="`queue_type` must be one of") as exc:
        store.create_review_queue(exp_id, name="x", queue_type="nonsense")
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


def test_create_rejects_schema_from_other_experiment(store):
    exp_a = _create_experiments(store, "exp_a")
    exp_b = _create_experiments(store, "exp_b")
    ls_b = _pass_fail(store, exp_b, "quality")
    with pytest.raises(MlflowException, match="not found for experiment") as exc:
        store.create_review_queue(exp_a, name="q", queue_type="custom", schema_ids=[ls_b.schema_id])
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


def test_create_duplicate_name_raises(store):
    exp_id = _create_experiments(store, "dup")
    store.create_review_queue(exp_id, name="dupe", queue_type="custom")
    with pytest.raises(MlflowException, match="already exists") as exc:
        store.create_review_queue(exp_id, name="dupe", queue_type="custom")
    _assert_error_code(exc, RESOURCE_ALREADY_EXISTS)


def test_create_against_missing_experiment_raises(store):
    with pytest.raises(MlflowException, match="No Experiment with id") as exc:
        store.create_review_queue("999999", name="q", queue_type="custom")
    _assert_error_code(exc, RESOURCE_DOES_NOT_EXIST)


def test_same_name_different_experiments_coexist(store):
    exp_a = _create_experiments(store, "coexist_a")
    exp_b = _create_experiments(store, "coexist_b")
    qa = store.create_review_queue(exp_a, name="shared", queue_type="custom")
    qb = store.create_review_queue(exp_b, name="shared", queue_type="custom")
    assert qa.queue_id != qb.queue_id


def test_create_custom_queue_accepts_users_at_cap(store):
    exp_id = _create_experiments(store, "cap_at_limit")
    queue = store.create_review_queue(
        exp_id, name="full", queue_type="custom", users=["a", "b", "c", "d"]
    )
    assert queue.users == ["a", "b", "c", "d"]


def test_create_custom_queue_rejects_users_over_cap(store):
    exp_id = _create_experiments(store, "cap_over")
    with pytest.raises(MlflowException, match="at most 4 assigned users") as exc:
        store.create_review_queue(
            exp_id, name="too_many", queue_type="custom", users=["a", "b", "c", "d", "e"]
        )
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


def test_create_custom_queue_caps_after_dedup(store):
    exp_id = _create_experiments(store, "cap_dedup")
    # Five raw users that de-duplicate to four are under the cap (the cap is
    # checked on the de-duplicated set, not the raw input).
    queue = store.create_review_queue(
        exp_id, name="dupes", queue_type="custom", users=["a", "b", "c", "d", "a"]
    )
    assert queue.users == ["a", "b", "c", "d"]


def test_update_custom_queue_rejects_users_over_cap(store):
    exp_id = _create_experiments(store, "cap_update")
    queue = store.create_review_queue(exp_id, name="growing", queue_type="custom", users=["a"])
    with pytest.raises(MlflowException, match="at most 4 assigned users") as exc:
        store.update_review_queue(queue.queue_id, users=["a", "b", "c", "d", "e"])
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


# --------------------------------------------------------------------------
# get_or_create_user_queue
# --------------------------------------------------------------------------


def test_get_or_create_user_queue_is_idempotent(store):
    exp_id = _create_experiments(store, "goc")
    first = store.get_or_create_user_queue(exp_id, user="Alice")
    second = store.get_or_create_user_queue(exp_id, user="alice")
    assert first.queue_id == second.queue_id
    assert first.queue_type == ReviewQueueType.USER
    assert first.users == ["alice"]


def test_get_or_create_user_queue_finds_existing_created_queue(store):
    exp_id = _create_experiments(store, "goc_existing")
    created = store.create_review_queue(exp_id, name="bob", queue_type="user")
    fetched = store.get_or_create_user_queue(exp_id, user="bob")
    assert fetched.queue_id == created.queue_id


def test_get_or_create_user_queue_rejects_custom_name_collision(store):
    exp_id = _create_experiments(store, "goc_collision")
    # A custom queue squatting on the user's normalized name must not be
    # returned as if it were the user's personal queue.
    store.create_review_queue(exp_id, name="alice", queue_type="custom")
    with pytest.raises(MlflowException, match="non-user queue named") as exc:
        store.get_or_create_user_queue(exp_id, user="Alice")
    _assert_error_code(exc, RESOURCE_ALREADY_EXISTS)


# --------------------------------------------------------------------------
# Get / get-by-name / list
# --------------------------------------------------------------------------


def test_get_review_queue(store):
    exp_id = _create_experiments(store, "get")
    created = store.create_review_queue(exp_id, name="q", queue_type="custom", users=["bob"])
    fetched = store.get_review_queue(created.queue_id)
    assert fetched.queue_id == created.queue_id
    assert fetched.users == ["bob"]


def test_get_review_queue_missing_raises(store):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.get_review_queue("rq-does-not-exist")
    _assert_error_code(exc, RESOURCE_DOES_NOT_EXIST)


def test_get_review_queue_by_name(store):
    exp_id = _create_experiments(store, "by_name")
    created = store.create_review_queue(exp_id, name="My Queue", queue_type="custom")
    fetched = store.get_review_queue_by_name(exp_id, name="My Queue")
    assert fetched.queue_id == created.queue_id


def test_get_review_queue_by_name_missing_raises(store):
    exp_id = _create_experiments(store, "by_name_missing")
    with pytest.raises(MlflowException, match="not found") as exc:
        store.get_review_queue_by_name(exp_id, name="nope")
    _assert_error_code(exc, RESOURCE_DOES_NOT_EXIST)


def test_list_review_queues_newest_first(store):
    exp_id = _create_experiments(store, "list_order")
    q1 = store.create_review_queue(exp_id, name="first", queue_type="custom")
    # Sleep so creation_time_ms differs: SQLite's millisecond resolution can
    # collide on back-to-back creates, leaving the queue_id tiebreaker to decide
    # order and making the assertion flaky.
    time.sleep(0.005)
    q2 = store.create_review_queue(exp_id, name="second", queue_type="custom")
    listed = store.list_review_queues(exp_id)
    assert [q.queue_id for q in listed] == [q2.queue_id, q1.queue_id]


def test_list_review_queues_filtered_by_user(store):
    exp_id = _create_experiments(store, "list_user")
    store.create_review_queue(exp_id, name="alice", queue_type="user")
    custom = store.create_review_queue(
        exp_id, name="team", queue_type="custom", users=["alice", "bob"]
    )
    store.create_review_queue(exp_id, name="solo", queue_type="custom", users=["carol"])

    alice_queues = {q.name for q in store.list_review_queues(exp_id, user="Alice")}
    assert alice_queues == {"alice", "team"}

    bob_queues = {q.name for q in store.list_review_queues(exp_id, user="bob")}
    assert bob_queues == {"team"}

    assert custom.users == ["alice", "bob"]


def test_list_review_queues_scopes_to_assigned_user(store):
    exp_id = _create_experiments(store, "list_scoped")
    store.get_or_create_user_queue(exp_id, user="default")
    store.create_review_queue(exp_id, name="solo", queue_type="custom", users=["carol"])

    # A user's scoped list contains only the queues they are assigned to; the
    # no-auth 'default' user queue belongs to 'default', not bob, so it is not
    # included for bob (neither is the 'solo' queue, which bob is not on).
    bob_queues = {q.name for q in store.list_review_queues(exp_id, user="bob")}
    assert bob_queues == set()


def test_list_review_queues_pagination(store):
    exp_id = _create_experiments(store, "list_paginate")
    for i in range(5):
        store.create_review_queue(exp_id, name=f"q{i}", queue_type="custom")
    page1 = store.list_review_queues(exp_id, max_results=2)
    assert len(page1) == 2
    assert page1.token is not None
    page2 = store.list_review_queues(exp_id, max_results=2, page_token=page1.token)
    assert len(page2) == 2
    page3 = store.list_review_queues(exp_id, max_results=2, page_token=page2.token)
    assert len(page3) == 1
    assert page3.token is None
    all_ids = {q.queue_id for q in [*page1, *page2, *page3]}
    assert len(all_ids) == 5


# --------------------------------------------------------------------------
# Update
# --------------------------------------------------------------------------


def test_update_custom_queue_replaces_users_and_schemas(store):
    exp_id = _create_experiments(store, "update")
    ls1 = _pass_fail(store, exp_id, "quality")
    ls2 = _pass_fail(store, exp_id, "safety")
    queue = store.create_review_queue(
        exp_id, name="q", queue_type="custom", users=["bob"], schema_ids=[ls1.schema_id]
    )

    updated = store.update_review_queue(
        queue.queue_id, users=["dave", "erin"], schema_ids=[ls1.schema_id, ls2.schema_id]
    )
    assert updated.users == ["dave", "erin"]
    assert sorted(updated.schema_ids) == sorted([ls1.schema_id, ls2.schema_id])
    assert updated.last_update_time_ms >= queue.creation_time_ms


def test_update_with_empty_lists_clears_associations(store):
    exp_id = _create_experiments(store, "update_clear")
    ls = _pass_fail(store, exp_id, "quality")
    queue = store.create_review_queue(
        exp_id, name="q", queue_type="custom", users=["bob"], schema_ids=[ls.schema_id]
    )
    updated = store.update_review_queue(queue.queue_id, users=[], schema_ids=[])
    assert updated.users == []
    assert updated.schema_ids == []


def test_update_only_users_leaves_schemas_untouched(store):
    exp_id = _create_experiments(store, "update_partial")
    ls = _pass_fail(store, exp_id, "quality")
    queue = store.create_review_queue(
        exp_id, name="q", queue_type="custom", users=["bob"], schema_ids=[ls.schema_id]
    )
    updated = store.update_review_queue(queue.queue_id, users=["dave"])
    assert updated.users == ["dave"]
    assert updated.schema_ids == [ls.schema_id]


def test_update_user_queue_raises(store):
    exp_id = _create_experiments(store, "update_user_queue")
    queue = store.create_review_queue(exp_id, name="alice", queue_type="user")
    with pytest.raises(MlflowException, match="fixed and cannot be updated") as exc:
        store.update_review_queue(queue.queue_id, users=["bob"])
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


def test_update_missing_queue_raises(store):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.update_review_queue("rq-missing", users=["bob"])
    _assert_error_code(exc, RESOURCE_DOES_NOT_EXIST)


def test_update_rejects_schema_from_other_experiment(store):
    exp_a = _create_experiments(store, "update_xexp_a")
    exp_b = _create_experiments(store, "update_xexp_b")
    ls_b = _pass_fail(store, exp_b, "quality")
    queue = store.create_review_queue(exp_a, name="q", queue_type="custom")
    with pytest.raises(MlflowException, match="not found for experiment") as exc:
        store.update_review_queue(queue.queue_id, schema_ids=[ls_b.schema_id])
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


def test_update_questions_locked_once_traces_assigned(store):
    exp_id = _create_experiments(store, "update_locked")
    ls1 = _pass_fail(store, exp_id, "quality")
    ls2 = _pass_fail(store, exp_id, "safety")
    queue = store.create_review_queue(
        exp_id, name="q", queue_type="custom", schema_ids=[ls1.schema_id]
    )
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1"])
    with pytest.raises(MlflowException, match="locked once items are assigned") as exc:
        store.update_review_queue(queue.queue_id, schema_ids=[ls1.schema_id, ls2.schema_id])
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


def test_update_users_allowed_after_traces_assigned(store):
    exp_id = _create_experiments(store, "update_users_with_traces")
    ls = _pass_fail(store, exp_id, "quality")
    queue = store.create_review_queue(
        exp_id, name="q", queue_type="custom", users=["bob"], schema_ids=[ls.schema_id]
    )
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1"])
    # Questions are locked, but assigned users stay editable.
    updated = store.update_review_queue(queue.queue_id, users=["dave"])
    assert updated.users == ["dave"]
    assert updated.schema_ids == [ls.schema_id]


def test_update_questions_allowed_after_traces_detached(store):
    exp_id = _create_experiments(store, "update_after_detach")
    ls1 = _pass_fail(store, exp_id, "quality")
    ls2 = _pass_fail(store, exp_id, "safety")
    queue = store.create_review_queue(
        exp_id, name="q", queue_type="custom", schema_ids=[ls1.schema_id]
    )
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1"])
    store.remove_items_from_review_queue(queue.queue_id, item_ids=["tr-1"])
    # With the queue empty again, questions can be edited.
    updated = store.update_review_queue(queue.queue_id, schema_ids=[ls1.schema_id, ls2.schema_id])
    assert sorted(updated.schema_ids) == sorted([ls1.schema_id, ls2.schema_id])


def test_get_or_create_user_queue_owner_is_the_user(store):
    exp_id = _create_experiments(store, "user_queue_owner")
    # A user queue is owned by its user — there is no caller-supplied owner.
    queue = store.get_or_create_user_queue(exp_id, user="Alice")
    assert queue.created_by == "alice"


# --------------------------------------------------------------------------
# Delete
# --------------------------------------------------------------------------


def test_delete_review_queue_removes_queue_and_children(store):
    exp_id = _create_experiments(store, "delete")
    ls = _pass_fail(store, exp_id, "quality")
    queue = store.create_review_queue(
        exp_id, name="q", queue_type="custom", users=["bob"], schema_ids=[ls.schema_id]
    )
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1"])

    store.delete_review_queue(queue.queue_id)
    with pytest.raises(MlflowException, match="not found"):
        store.get_review_queue(queue.queue_id)

    # The label schema itself survives; only the association is removed.
    assert store.get_label_schema(ls.schema_id).schema_id == ls.schema_id


def test_delete_missing_queue_is_noop(store):
    store.delete_review_queue("rq-missing")  # no raise


# --------------------------------------------------------------------------
# Attach / detach items
# --------------------------------------------------------------------------


def test_add_traces_returns_items_in_order(store):
    exp_id = _create_experiments(store, "attach")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    items = store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-3", "tr-1", "tr-2"])
    assert [i.item_id for i in items] == ["tr-3", "tr-1", "tr-2"]
    assert all(i.status == ReviewStatus.PENDING for i in items)
    assert all(i.item_type == ReviewItemType.TRACE for i in items)
    assert all(i.completed_by is None for i in items)


def test_add_traces_dedups_within_call(store):
    exp_id = _create_experiments(store, "attach_dedup")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    items = store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1", "tr-1"])
    assert len(items) == 1


def test_add_traces_is_idempotent_and_preserves_status(store):
    exp_id = _create_experiments(store, "attach_idempotent")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1"])
    store.set_review_queue_item_status(
        queue.queue_id, item_id="tr-1", status="complete", completed_by="bob"
    )
    items = store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1", "tr-2"])
    by_id = {i.item_id: i for i in items}
    assert by_id["tr-1"].status == ReviewStatus.COMPLETE
    assert by_id["tr-1"].completed_by == "bob"
    assert by_id["tr-2"].status == ReviewStatus.PENDING


def test_add_traces_to_missing_queue_raises(store):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.add_items_to_review_queue("rq-missing", item_ids=["tr-1"])
    _assert_error_code(exc, RESOURCE_DOES_NOT_EXIST)


def test_add_traces_rejects_empty_list(store):
    exp_id = _create_experiments(store, "attach_empty")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    with pytest.raises(MlflowException, match="non-empty list") as exc:
        store.add_items_to_review_queue(queue.queue_id, item_ids=[])
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


def test_remove_traces(store):
    exp_id = _create_experiments(store, "detach")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1", "tr-2", "tr-3"])
    store.remove_items_from_review_queue(queue.queue_id, item_ids=["tr-2"])
    remaining = {i.item_id for i in store.list_review_queue_items(queue.queue_id)}
    assert remaining == {"tr-1", "tr-3"}


def test_remove_unattached_trace_is_noop(store):
    exp_id = _create_experiments(store, "detach_noop")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1"])
    store.remove_items_from_review_queue(queue.queue_id, item_ids=["tr-9"])
    assert len(store.list_review_queue_items(queue.queue_id)) == 1


def test_same_trace_in_two_queues_has_independent_status(store):
    exp_id = _create_experiments(store, "two_queues")
    q1 = store.create_review_queue(exp_id, name="q1", queue_type="custom")
    q2 = store.create_review_queue(exp_id, name="q2", queue_type="custom")
    store.add_items_to_review_queue(q1.queue_id, item_ids=["tr-1"])
    store.add_items_to_review_queue(q2.queue_id, item_ids=["tr-1"])
    store.set_review_queue_item_status(
        q1.queue_id, item_id="tr-1", status="complete", completed_by="bob"
    )
    item_q1 = store.list_review_queue_items(q1.queue_id)[0]
    item_q2 = store.list_review_queue_items(q2.queue_id)[0]
    assert item_q1.status == ReviewStatus.COMPLETE
    assert item_q2.status == ReviewStatus.PENDING


# --------------------------------------------------------------------------
# List items
# --------------------------------------------------------------------------


def test_list_traces_filtered_by_status(store):
    exp_id = _create_experiments(store, "list_traces_status")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1", "tr-2", "tr-3"])
    store.set_review_queue_item_status(
        queue.queue_id, item_id="tr-1", status="complete", completed_by="bob"
    )
    store.set_review_queue_item_status(
        queue.queue_id, item_id="tr-2", status="declined", completed_by="carol"
    )

    pending = {i.item_id for i in store.list_review_queue_items(queue.queue_id, status="pending")}
    assert pending == {"tr-3"}
    complete = {i.item_id for i in store.list_review_queue_items(queue.queue_id, status="complete")}
    assert complete == {"tr-1"}
    declined = {i.item_id for i in store.list_review_queue_items(queue.queue_id, status="declined")}
    assert declined == {"tr-2"}


def test_list_traces_pagination(store):
    exp_id = _create_experiments(store, "list_traces_paginate")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    store.add_items_to_review_queue(queue.queue_id, item_ids=[f"tr-{i}" for i in range(5)])
    page1 = store.list_review_queue_items(queue.queue_id, max_results=2)
    assert len(page1) == 2
    assert page1.token is not None
    page2 = store.list_review_queue_items(queue.queue_id, max_results=2, page_token=page1.token)
    page3 = store.list_review_queue_items(queue.queue_id, max_results=2, page_token=page2.token)
    all_ids = {i.item_id for i in [*page1, *page2, *page3]}
    assert all_ids == {f"tr-{i}" for i in range(5)}


def test_list_traces_missing_queue_raises(store):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.list_review_queue_items("rq-missing")
    _assert_error_code(exc, RESOURCE_DOES_NOT_EXIST)


# --------------------------------------------------------------------------
# Status transitions (shared-pool semantics)
# --------------------------------------------------------------------------


def test_complete_records_attribution(store):
    exp_id = _create_experiments(store, "complete")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1"])
    item = store.set_review_queue_item_status(
        queue.queue_id, item_id="tr-1", status="complete", completed_by="Bob"
    )
    assert item.status == ReviewStatus.COMPLETE
    assert item.completed_by == "bob"  # normalized
    assert item.completed_time_ms is not None


def test_decline_records_attribution(store):
    exp_id = _create_experiments(store, "decline")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1"])
    item = store.set_review_queue_item_status(
        queue.queue_id, item_id="tr-1", status="declined", completed_by="carol"
    )
    assert item.status == ReviewStatus.DECLINED
    assert item.completed_by == "carol"
    assert item.completed_time_ms is not None


def test_reopen_clears_attribution(store):
    exp_id = _create_experiments(store, "reopen")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1"])
    store.set_review_queue_item_status(
        queue.queue_id, item_id="tr-1", status="complete", completed_by="bob"
    )
    reopened = store.set_review_queue_item_status(queue.queue_id, item_id="tr-1", status="pending")
    assert reopened.status == ReviewStatus.PENDING
    assert reopened.completed_by is None
    assert reopened.completed_time_ms is None


def test_repeat_same_status_preserves_attribution(store):
    exp_id = _create_experiments(store, "repeat_status")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1"])
    first = store.set_review_queue_item_status(
        queue.queue_id, item_id="tr-1", status="complete", completed_by="bob"
    )
    # Re-completing with the same status (a repeat or a concurrent second
    # reviewer) is a no-op and must not overwrite the original completer.
    again = store.set_review_queue_item_status(
        queue.queue_id, item_id="tr-1", status="complete", completed_by="carol"
    )
    assert again.completed_by == "bob"
    assert again.completed_time_ms == first.completed_time_ms


def test_terminal_status_change_reattributes(store):
    exp_id = _create_experiments(store, "status_change")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1"])
    store.set_review_queue_item_status(
        queue.queue_id, item_id="tr-1", status="complete", completed_by="bob"
    )
    # A genuine status change (complete -> declined) is a real action and
    # re-records the reviewer who made it.
    changed = store.set_review_queue_item_status(
        queue.queue_id, item_id="tr-1", status="declined", completed_by="carol"
    )
    assert changed.status == ReviewStatus.DECLINED
    assert changed.completed_by == "carol"


def test_complete_requires_completed_by(store):
    exp_id = _create_experiments(store, "complete_requires")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1"])
    with pytest.raises(MlflowException, match="`completed_by` is required") as exc:
        store.set_review_queue_item_status(queue.queue_id, item_id="tr-1", status="complete")
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


def test_reopen_rejects_completed_by(store):
    exp_id = _create_experiments(store, "reopen_rejects")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-1"])
    with pytest.raises(MlflowException, match="must not be set when reopening") as exc:
        store.set_review_queue_item_status(
            queue.queue_id, item_id="tr-1", status="pending", completed_by="bob"
        )
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


def test_set_status_on_unattached_trace_raises(store):
    exp_id = _create_experiments(store, "status_unattached")
    queue = store.create_review_queue(exp_id, name="q", queue_type="custom")
    with pytest.raises(MlflowException, match="not attached") as exc:
        store.set_review_queue_item_status(
            queue.queue_id, item_id="tr-x", status="complete", completed_by="bob"
        )
    _assert_error_code(exc, RESOURCE_DOES_NOT_EXIST)


def test_set_status_missing_queue_raises(store):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.set_review_queue_item_status(
            "rq-missing", item_id="tr-1", status="complete", completed_by="bob"
        )
    _assert_error_code(exc, RESOURCE_DOES_NOT_EXIST)


# --------------------------------------------------------------------------
# Reserved queue name
# --------------------------------------------------------------------------


def test_create_custom_queue_rejects_default_name(store):
    exp_id = _create_experiments(store, "default_reserved")
    with pytest.raises(MlflowException, match="reserved queue name") as exc:
        store.create_review_queue(exp_id, name="Default", queue_type="custom")
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize("name", ["DEFAULT", "DeFault", "default", "  Default  "])
def test_create_custom_queue_rejects_default_name_case_insensitive(store, name):
    exp_id = _create_experiments(store, "default_reserved_ci")
    with pytest.raises(MlflowException, match="reserved queue name") as exc:
        store.create_review_queue(exp_id, name=name, queue_type="custom")
    _assert_error_code(exc, INVALID_PARAMETER_VALUE)


# --------------------------------------------------------------------------
# Trace deletion cleans up attached items
# --------------------------------------------------------------------------


def test_delete_trace_removes_its_review_queue_items(store):
    exp_id = _create_experiments(store, "delete_trace_items")
    _create_trace(store, "tr-keep", experiment_id=exp_id)
    _create_trace(store, "tr-del", experiment_id=exp_id)
    queue = store.create_review_queue(exp_id, name="Q", queue_type="custom")
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-keep", "tr-del"])

    store.delete_traces(exp_id, trace_ids=["tr-del"])

    remaining = {i.item_id for i in store.list_review_queue_items(queue.queue_id)}
    assert remaining == {"tr-keep"}


def test_delete_trace_removes_items_from_every_queue(store):
    exp_id = _create_experiments(store, "delete_trace_items_multi")
    _create_trace(store, "tr-del", experiment_id=exp_id)
    q1 = store.create_review_queue(exp_id, name="Q1", queue_type="custom")
    q2 = store.create_review_queue(exp_id, name="Q2", queue_type="custom")
    store.add_items_to_review_queue(q1.queue_id, item_ids=["tr-del"])
    store.add_items_to_review_queue(q2.queue_id, item_ids=["tr-del"])

    store.delete_traces(exp_id, trace_ids=["tr-del"])

    assert list(store.list_review_queue_items(q1.queue_id)) == []
    assert list(store.list_review_queue_items(q2.queue_id)) == []
