# Unit tests for the review-queue / label-schema authorization validators: they
# exercise the gate decisions in ``mlflow.server.auth`` directly by mocking the
# request / store / experiment-permission boundary, so no live auth server is
# needed. End-to-end routing is covered by the broader auth client tests.
import json
from types import SimpleNamespace

import pytest

from mlflow.protos.review_queues_pb2 import ListReviewQueues
from mlflow.utils.proto_json_utils import message_to_json, parse_dict

# flask-wtf (the auth extra) is required to import the auth app.
pytest.importorskip("flask_wtf")

from mlflow.server import auth
from mlflow.server.auth.permissions import get_permission


def _setup(monkeypatch, *, permission, queue_users=None, username="alice"):
    """Patch the request / store / permission boundary the validators read."""
    perm = get_permission(permission)
    queue = SimpleNamespace(experiment_id="123", users=list(queue_users or []))
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


def test_review_queue_has_member_is_case_insensitive():
    queue = SimpleNamespace(users=["Alice", " Bob "])
    assert auth._review_queue_has_member(queue, "alice")
    assert auth._review_queue_has_member(queue, "BOB")
    assert not auth._review_queue_has_member(queue, "carol")
    assert not auth._review_queue_has_member(SimpleNamespace(users=[]), "alice")


@pytest.mark.parametrize(
    "validator",
    [
        "validate_can_create_review_queue",
        "validate_can_manage_review_queue",
        "validate_can_create_label_schema",
        "validate_can_manage_label_schema",
    ],
)
@pytest.mark.parametrize(
    ("permission", "expected"), [("READ", False), ("EDIT", False), ("MANAGE", True)]
)
def test_management_validators_require_manage(monkeypatch, validator, permission, expected):
    _setup(monkeypatch, permission=permission)
    assert getattr(auth, validator)() is expected


@pytest.mark.parametrize(
    ("permission", "expected"), [("READ", False), ("EDIT", True), ("MANAGE", True)]
)
def test_add_items_and_user_queue_require_edit(monkeypatch, permission, expected):
    _setup(monkeypatch, permission=permission)
    assert auth.validate_can_add_items_to_review_queue() is expected
    assert auth.validate_can_get_or_create_user_queue() is expected


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


def test_view_review_queue_manage_or_membership(monkeypatch):
    # Manager sees any queue, assigned or not.
    _setup(monkeypatch, permission="MANAGE", queue_users=["bob"], username="alice")
    assert auth.validate_can_view_review_queue() is True

    # READ + assigned -> visible.
    _setup(monkeypatch, permission="READ", queue_users=["alice"], username="alice")
    assert auth.validate_can_view_review_queue() is True

    # READ + not assigned -> hidden.
    _setup(monkeypatch, permission="READ", queue_users=["bob"], username="alice")
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


def _filter_response(monkeypatch, *, permission, username, queues):
    _setup(monkeypatch, permission=permission, username=username)
    monkeypatch.setattr(auth, "sender_is_admin", lambda: False)
    msg = ListReviewQueues.Response()
    parse_dict({"review_queues": queues}, msg)
    resp = SimpleNamespace(json=json.loads(message_to_json(msg)), data=None)
    auth.filter_list_review_queues(resp)
    return resp


def test_filter_list_review_queues_non_manager_sees_only_assigned(monkeypatch):
    resp = _filter_response(
        monkeypatch,
        permission="EDIT",
        username="alice",
        queues=[{"queue_id": "q1", "users": ["alice"]}, {"queue_id": "q2", "users": ["bob"]}],
    )
    out = ListReviewQueues.Response()
    parse_dict(json.loads(resp.data), out)
    assert [q.queue_id for q in out.review_queues] == ["q1"]


def test_filter_list_review_queues_manager_sees_all(monkeypatch):
    resp = _filter_response(
        monkeypatch,
        permission="MANAGE",
        username="alice",
        queues=[{"queue_id": "q1", "users": ["bob"]}, {"queue_id": "q2", "users": ["carol"]}],
    )
    # Manager short-circuits before rewriting the response.
    assert resp.data is None
