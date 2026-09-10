# Unit tests for the issue authorization validators' request-body handling.

import json
from types import SimpleNamespace

import flask

import mlflow.server.auth as a


def _run(monkeypatch, validator, body, *, can_update=False, can_read=False):
    monkeypatch.setattr(a, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        a,
        "_get_experiment_permission",
        lambda eid, user: SimpleNamespace(can_update=can_update, can_read=can_read),
    )
    app = flask.Flask(__name__)
    with app.test_request_context(data=body, content_type="application/json"):
        return validator()


def test_create_issue_handles_double_encoded_body(monkeypatch):
    # Older clients post JSON double-encoded as a string; the validator must decode it like
    # the handler does instead of raising AttributeError (a 500) on the str.
    double_encoded = json.dumps(json.dumps({"experiment_id": "e1"}))
    assert _run(monkeypatch, a.validate_can_create_issue, double_encoded, can_update=True) is True
    assert _run(monkeypatch, a.validate_can_create_issue, double_encoded, can_update=False) is False


def test_search_issues_handles_double_encoded_body(monkeypatch):
    double_encoded = json.dumps(json.dumps({"experiment_id": "e1"}))
    assert _run(monkeypatch, a.validate_can_search_issues, double_encoded, can_read=True) is True
    assert _run(monkeypatch, a.validate_can_search_issues, double_encoded, can_read=False) is False


def test_issue_validators_deny_empty_body(monkeypatch):
    # No experiment_id to scope against -> fail closed rather than crash.
    assert _run(monkeypatch, a.validate_can_create_issue, "", can_update=True) is False
    assert _run(monkeypatch, a.validate_can_search_issues, "", can_read=True) is False
