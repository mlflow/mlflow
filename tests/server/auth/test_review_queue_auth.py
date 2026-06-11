# Unit tests for the review-queue / label-schema authorization validators: they
# exercise the gate decisions in ``mlflow.server.auth`` directly by mocking the
# request / store / experiment-permission boundary, so no live auth server is
# needed. End-to-end routing is covered by the broader auth client tests.
import json
from types import SimpleNamespace

import pytest

from mlflow.genai.review_queues import ReviewQueueType
from mlflow.protos.review_queues_pb2 import ListReviewQueues
from mlflow.utils.proto_json_utils import message_to_json, parse_dict

# flask-wtf (the auth extra) is required to import the auth app.
pytest.importorskip("flask_wtf")

from mlflow.server import auth
from mlflow.server.auth.permissions import get_permission


def _setup(
    monkeypatch,
    *,
    permission,
    queue_users=None,
    username="alice",
    created_by=None,
    queue_type=ReviewQueueType.CUSTOM,
    update_body=None,
):
    """Patch the request / store / permission boundary the validators read."""
    perm = get_permission(permission)
    queue = SimpleNamespace(
        experiment_id="123",
        users=list(queue_users or []),
        created_by=created_by,
        queue_type=queue_type,
    )
    schema = SimpleNamespace(experiment_id="123")
    store = SimpleNamespace(
        get_review_queue=lambda _qid: queue,
        get_review_queue_by_name=lambda _exp, name: queue,
        get_label_schema=lambda _sid: schema,
    )
    monkeypatch.setattr(auth, "authenticate_request", lambda: SimpleNamespace(username=username))
    monkeypatch.setattr(
        auth,
        "_get_request_param",
        lambda p: {"queue_id": "q1", "schema_id": "s1", "experiment_id": "123", "name": "Q"}[p],
    )
    monkeypatch.setattr(auth, "_get_experiment_permission", lambda _exp, _user: perm)
    monkeypatch.setattr(auth, "_get_tracking_store", lambda: store)
    # The owner-reassignment gate parses the live request body; stub it so the
    # real `_update_review_queue_reassigns_owner` detection runs against it.
    monkeypatch.setattr(
        auth, "request", SimpleNamespace(get_json=lambda silent=False: dict(update_body or {}))
    )


def test_review_queue_has_member_is_case_insensitive():
    queue = SimpleNamespace(users=["Alice", " Bob "])
    assert auth._review_queue_has_member(queue, "alice")
    assert auth._review_queue_has_member(queue, "BOB")
    assert not auth._review_queue_has_member(queue, "carol")
    assert not auth._review_queue_has_member(SimpleNamespace(users=[]), "alice")


def test_is_review_queue_owner_is_case_insensitive():
    assert auth._is_review_queue_owner(SimpleNamespace(created_by="Alice"), "alice")
    assert auth._is_review_queue_owner(SimpleNamespace(created_by=" alice "), "ALICE")
    assert not auth._is_review_queue_owner(SimpleNamespace(created_by="bob"), "alice")
    # No owner (no-auth-era rows) is never an owner match.
    assert not auth._is_review_queue_owner(SimpleNamespace(created_by=None), "alice")
    assert not auth._is_review_queue_owner(SimpleNamespace(created_by=""), "alice")


@pytest.mark.parametrize(
    "validator",
    ["validate_can_create_label_schema", "validate_can_manage_label_schema"],
)
@pytest.mark.parametrize(
    ("permission", "expected"), [("READ", False), ("EDIT", False), ("MANAGE", True)]
)
def test_label_schema_management_requires_manage(monkeypatch, validator, permission, expected):
    _setup(monkeypatch, permission=permission)
    assert getattr(auth, validator)() is expected


@pytest.mark.parametrize(
    ("permission", "expected"), [("READ", False), ("EDIT", True), ("MANAGE", True)]
)
def test_create_add_items_and_user_queue_require_edit(monkeypatch, permission, expected):
    # Creating (and owning) a queue, flagging items, and routing to a user queue
    # are all experiment-EDIT operations now.
    _setup(monkeypatch, permission=permission)
    assert auth.validate_can_create_review_queue() is expected
    assert auth.validate_can_add_items_to_review_queue() is expected
    assert auth.validate_can_get_or_create_user_queue() is expected


def test_update_and_remove_items_allow_owner_or_manager(monkeypatch):
    # Manager edits any queue.
    _setup(monkeypatch, permission="MANAGE", created_by="bob", username="alice")
    assert auth.validate_can_update_review_queue() is True
    assert auth.validate_can_remove_items_from_review_queue() is True

    # Owning EDIT user edits their own queue.
    _setup(monkeypatch, permission="EDIT", created_by="alice", username="alice")
    assert auth.validate_can_update_review_queue() is True
    assert auth.validate_can_remove_items_from_review_queue() is True

    # EDIT non-owner is denied.
    _setup(monkeypatch, permission="EDIT", created_by="bob", username="alice")
    assert auth.validate_can_update_review_queue() is False
    assert auth.validate_can_remove_items_from_review_queue() is False

    # READ owner is denied (ownership amplifies EDIT, never substitutes).
    _setup(monkeypatch, permission="READ", created_by="alice", username="alice")
    assert auth.validate_can_update_review_queue() is False


def test_owner_reassignment_requires_manage(monkeypatch):
    # An owning EDIT user may edit shape but NOT reassign the owner.
    _setup(
        monkeypatch,
        permission="EDIT",
        created_by="alice",
        username="alice",
        update_body={"queue_id": "q1", "new_owner": "victim"},
    )
    assert auth.validate_can_update_review_queue() is False

    # A manager may reassign the owner.
    _setup(
        monkeypatch,
        permission="MANAGE",
        created_by="bob",
        username="alice",
        update_body={"queue_id": "q1", "new_owner": "victim"},
    )
    assert auth.validate_can_update_review_queue() is True


def test_owner_reassignment_via_camelcase_still_requires_manage(monkeypatch):
    # Regression: protobuf JSON also accepts the camelCase `newOwner`. The gate
    # must detect it by parsing the proto (not scanning raw JSON keys), so an
    # owning EDIT user can't dodge the MANAGE-only owner reassignment.
    _setup(
        monkeypatch,
        permission="EDIT",
        created_by="alice",
        username="alice",
        update_body={"queue_id": "q1", "newOwner": "victim"},
    )
    assert auth.validate_can_update_review_queue() is False


def test_delete_owner_can_delete_own_custom_but_not_user(monkeypatch):
    # Manager deletes any queue.
    _setup(monkeypatch, permission="MANAGE", created_by="bob", username="alice")
    assert auth.validate_can_delete_review_queue() is True

    # Owning EDIT user deletes their own CUSTOM queue.
    _setup(monkeypatch, permission="EDIT", created_by="alice", username="alice")
    assert auth.validate_can_delete_review_queue() is True

    # ...but not a USER queue (those are MANAGE-only to delete).
    _setup(
        monkeypatch,
        permission="EDIT",
        created_by="alice",
        username="alice",
        queue_type=ReviewQueueType.USER,
    )
    assert auth.validate_can_delete_review_queue() is False

    # EDIT non-owner is denied.
    _setup(monkeypatch, permission="EDIT", created_by="bob", username="alice")
    assert auth.validate_can_delete_review_queue() is False


@pytest.mark.parametrize(
    ("permission", "expected"), [("READ", True), ("EDIT", True), ("MANAGE", True)]
)
def test_read_label_schema_requires_read(monkeypatch, permission, expected):
    _setup(monkeypatch, permission=permission)
    assert auth.validate_can_read_label_schema() is expected


def test_review_queue_item_requires_edit_and_membership(monkeypatch):
    # EDIT + assigned -> allowed.
    _setup(monkeypatch, permission="EDIT", queue_users=["alice"], username="alice")
    assert auth.validate_can_review_queue_item() is True

    # EDIT but not assigned -> denied (even a manager must self-assign first).
    _setup(monkeypatch, permission="MANAGE", queue_users=["bob"], username="alice")
    assert auth.validate_can_review_queue_item() is False

    # Assigned but only READ -> denied (reviewing needs EDIT).
    _setup(monkeypatch, permission="READ", queue_users=["alice"], username="alice")
    assert auth.validate_can_review_queue_item() is False


def test_view_review_queue_manage_owner_or_membership(monkeypatch):
    # Manager sees any queue, assigned or not.
    _setup(monkeypatch, permission="MANAGE", queue_users=["bob"], username="alice")
    assert auth.validate_can_view_review_queue() is True

    # READ + assigned -> visible.
    _setup(monkeypatch, permission="READ", queue_users=["alice"], username="alice")
    assert auth.validate_can_view_review_queue() is True

    # READ + not assigned -> hidden.
    _setup(monkeypatch, permission="READ", queue_users=["bob"], username="alice")
    assert auth.validate_can_view_review_queue() is False

    # EDIT owner (not assigned) -> visible.
    _setup(
        monkeypatch, permission="EDIT", queue_users=["bob"], created_by="alice", username="alice"
    )
    assert auth.validate_can_view_review_queue() is True

    # READ owner -> hidden (ownership amplifies EDIT, never substitutes).
    _setup(
        monkeypatch, permission="READ", queue_users=["bob"], created_by="alice", username="alice"
    )
    assert auth.validate_can_view_review_queue() is False


def test_view_review_queue_by_name_manage_or_membership(monkeypatch):
    # Manager sees any queue by name.
    _setup(monkeypatch, permission="MANAGE", queue_users=["bob"], username="alice")
    assert auth.validate_can_view_review_queue_by_name() is True

    # READ + assigned -> visible; READ + not assigned -> hidden.
    _setup(monkeypatch, permission="READ", queue_users=["alice"], username="alice")
    assert auth.validate_can_view_review_queue_by_name() is True
    _setup(monkeypatch, permission="READ", queue_users=["bob"], username="alice")
    assert auth.validate_can_view_review_queue_by_name() is False

    # EDIT owner -> visible; READ owner -> hidden.
    _setup(
        monkeypatch, permission="EDIT", queue_users=["bob"], created_by="alice", username="alice"
    )
    assert auth.validate_can_view_review_queue_by_name() is True
    _setup(
        monkeypatch, permission="READ", queue_users=["bob"], created_by="alice", username="alice"
    )
    assert auth.validate_can_view_review_queue_by_name() is False


def _filter_response(monkeypatch, *, permission, username, queues):
    _setup(monkeypatch, permission=permission, username=username)
    monkeypatch.setattr(auth, "sender_is_admin", lambda: False)
    msg = ListReviewQueues.Response()
    parse_dict({"review_queues": queues}, msg)
    resp = SimpleNamespace(json=json.loads(message_to_json(msg)), data=None)
    auth.filter_list_review_queues(resp)
    return resp


def test_filter_list_review_queues_read_only_sees_only_assigned(monkeypatch):
    resp = _filter_response(
        monkeypatch,
        permission="READ",
        username="alice",
        queues=[{"queue_id": "q1", "users": ["alice"]}, {"queue_id": "q2", "users": ["bob"]}],
    )
    out = ListReviewQueues.Response()
    parse_dict(json.loads(resp.data), out)
    assert [q.queue_id for q in out.review_queues] == ["q1"]


@pytest.mark.parametrize("permission", ["EDIT", "MANAGE"])
def test_filter_list_review_queues_editor_and_manager_see_all(monkeypatch, permission):
    resp = _filter_response(
        monkeypatch,
        permission=permission,
        username="alice",
        queues=[{"queue_id": "q1", "users": ["bob"]}, {"queue_id": "q2", "users": ["carol"]}],
    )
    # EDIT and MANAGE both short-circuit before rewriting the response (all rows).
    assert resp.data is None
