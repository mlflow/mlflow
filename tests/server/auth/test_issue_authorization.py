# Unit tests for the issue authorization validators' request-body handling.

import json
from types import SimpleNamespace

import flask

import mlflow.server.auth as a
from mlflow.exceptions import MlflowException
from mlflow.genai.discovery.schedule import DetectionCadence, DetectionSchedule, ScheduleModel
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.protos.issues_pb2 import Issue as ProtoIssue
from mlflow.utils.mlflow_tags import MLFLOW_ISSUE_DETECTION_SCHEDULE


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


def test_issue_run_provenance_is_redacted(monkeypatch):
    monkeypatch.setattr(a, "_denied_sibling_tiers", lambda *_: {a.RESOURCE_TYPE_RUN})
    issues = [ProtoIssue(source_run_id="run-1"), ProtoIssue()]
    assert a._withhold_denied_issue_source_runs(issues, "u") is True
    assert not issues[0].HasField("source_run_id")
    assert not issues[1].HasField("source_run_id")


def _run_detect_issues(
    monkeypatch,
    body,
    *,
    can_create=True,
    can_search=True,
    schedule_json=None,
    can_use=False,
    searches=None,
):
    secret_checks = []
    monkeypatch.setattr(a, "_authorize_create_in_experiment", lambda eid, created: can_create)

    def authorize_trace_search(experiment_ids, *filter_strings):
        if searches is not None:
            searches.append((experiment_ids, filter_strings))
        return can_search

    monkeypatch.setattr(a, "_authorize_trace_search", authorize_trace_search)

    def get_experiment(experiment_id):
        if experiment_id == "missing":
            raise MlflowException("no such experiment", error_code=RESOURCE_DOES_NOT_EXIST)
        tags = {MLFLOW_ISSUE_DETECTION_SCHEDULE: schedule_json} if schedule_json else {}
        return SimpleNamespace(experiment_id=experiment_id, tags=tags)

    monkeypatch.setattr(
        a, "_get_tracking_store", lambda: SimpleNamespace(get_experiment=get_experiment)
    )

    def secret_permission(secret_id):
        secret_checks.append(secret_id)
        return SimpleNamespace(can_use=can_use)

    monkeypatch.setattr(a, "_get_gateway_secret_permission", secret_permission)
    app = flask.Flask(__name__)
    with app.test_request_context(data=json.dumps(body), content_type="application/json"):
        return a.validate_can_detect_issues(), secret_checks


def _schedule_json(model):
    return DetectionSchedule(cadence=DetectionCadence.DAILY, model=model).to_json()


def test_detect_issues_requires_creating_a_run_in_the_experiment(monkeypatch):
    body = {"experiment_id": "e1", "model": "gateway:/e"}
    assert _run_detect_issues(monkeypatch, body, can_create=True) == (True, [])
    assert _run_detect_issues(monkeypatch, body, can_create=False) == (False, [])
    assert _run_detect_issues(monkeypatch, {}, can_create=True) == (False, [])


def test_detect_issues_requires_searching_the_experiments_traces(monkeypatch):
    # The job searches and reads the traces straight from the store, so the request is held to
    # what a trace search with the same filter requires.
    body = {"experiment_id": "e1", "model": "gateway:/e", "filter_string": "tags.env = 'prod'"}
    searches = []
    assert _run_detect_issues(monkeypatch, body, searches=searches) == (True, [])
    assert searches == [(["e1"], ("tags.env = 'prod'",))]
    assert _run_detect_issues(monkeypatch, body, can_search=False) == (False, [])


def test_detect_issues_requires_use_on_the_schedule_secret(monkeypatch):
    # Without `model` the run uses the schedule's saved model, and a direct provider selection
    # runs on the stored secret it names, which requires USE on that secret.
    schedule = _schedule_json(ScheduleModel(provider="openai", model="gpt-5", secret_id="s-1"))
    body = {"experiment_id": "e1"}
    assert _run_detect_issues(monkeypatch, body, schedule_json=schedule, can_use=False) == (
        False,
        ["s-1"],
    )
    assert _run_detect_issues(monkeypatch, body, schedule_json=schedule, can_use=True) == (
        True,
        ["s-1"],
    )


def test_detect_issues_skips_the_secret_check_when_none_is_used(monkeypatch):
    endpoint_schedule = _schedule_json(ScheduleModel(endpoint_name="saved-endpoint"))
    secret_schedule = _schedule_json(
        ScheduleModel(provider="openai", model="gpt-5", secret_id="s-1")
    )
    # A gateway endpoint carries its own credential, and no schedule leaves the handler to reject.
    assert _run_detect_issues(
        monkeypatch, {"experiment_id": "e1"}, schedule_json=endpoint_schedule
    ) == (True, [])
    assert _run_detect_issues(monkeypatch, {"experiment_id": "e1"}) == (True, [])
    # An explicit model replaces the schedule's, so its secret is never used.
    assert _run_detect_issues(
        monkeypatch,
        {"experiment_id": "e1", "model": "openai:/gpt-5"},
        schedule_json=secret_schedule,
    ) == (True, [])


def test_detect_issues_denies_a_missing_experiment(monkeypatch):
    assert _run_detect_issues(monkeypatch, {"experiment_id": "missing"}) == (False, [])
