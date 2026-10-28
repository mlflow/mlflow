# Unit tests for the evaluation-dataset authorization validators and route routing.

from types import SimpleNamespace

import flask

import mlflow.server.auth as a


class _Req:
    def __init__(self, path, method):
        self.path = path
        self.method = method


def _perm(read=False, update=False, delete=False):
    return SimpleNamespace(can_read=read, can_update=update, can_delete=delete)


def _validator_name(path, method):
    v = a._find_validator(_Req(path, method))
    return getattr(v, "__name__", v)


def test_literal_routes_not_shadowed_by_dataset_id_wildcard():
    # /datasets/search and /datasets/create must resolve to their own validators, not be
    # captured by the {dataset_id} regex (which matches a single non-slash segment).
    assert _validator_name("/api/3.0/mlflow/datasets/search", "GET") == (
        "validate_can_search_evaluation_datasets"
    )
    assert _validator_name("/api/3.0/mlflow/datasets/search", "POST") == (
        "validate_can_search_evaluation_datasets"
    )
    assert (
        _validator_name("/api/3.0/mlflow/datasets/create", "POST") == "validate_can_create_dataset"
    )
    assert _validator_name("/api/3.0/mlflow/datasets/d-abc", "GET") == "validate_can_read_dataset"


def _mock_dataset(monkeypatch, exp_ids, perms):
    monkeypatch.setattr(a, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(a, "_get_request_param", lambda p: "d-1")
    monkeypatch.setattr(
        a,
        "_get_tracking_store",
        lambda: SimpleNamespace(get_dataset_experiment_ids=lambda did: exp_ids),
    )
    monkeypatch.setattr(a, "_get_experiment_permission", lambda eid, user: perms[eid])


def test_read_gated_on_experiment_permission(monkeypatch):
    _mock_dataset(monkeypatch, ["e1"], {"e1": _perm(read=True)})
    assert a.validate_can_read_dataset() is True
    _mock_dataset(monkeypatch, ["e1"], {"e1": _perm(read=False)})
    assert a.validate_can_read_dataset() is False


def test_update_and_delete_gated(monkeypatch):
    _mock_dataset(monkeypatch, ["e1"], {"e1": _perm(update=True, delete=False)})
    assert a.validate_can_update_dataset() is True
    assert a.validate_can_delete_dataset() is False


def test_multi_experiment_requires_permission_on_all(monkeypatch):
    _mock_dataset(monkeypatch, ["e1", "e2"], {"e1": _perm(read=True), "e2": _perm(read=False)})
    assert a.validate_can_read_dataset() is False


def test_zero_experiment_dataset_denied_for_non_admin(monkeypatch):
    _mock_dataset(monkeypatch, [], {})
    assert a.validate_can_read_dataset() is False
    assert a.validate_can_update_dataset() is False
    assert a.validate_can_delete_dataset() is False


def _run_add(monkeypatch, current, added, perms):
    monkeypatch.setattr(a, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(a, "_get_request_param", lambda p: "d-1")
    monkeypatch.setattr(
        a,
        "_get_tracking_store",
        lambda: SimpleNamespace(get_dataset_experiment_ids=lambda did: current),
    )
    monkeypatch.setattr(a, "_get_experiment_permission", lambda eid, user: perms[eid])
    with flask.Flask(__name__).test_request_context(json={"experiment_ids": added}):
        return a.validate_can_add_dataset_to_experiments()


def test_add_to_experiments_requires_update_on_current_and_added(monkeypatch):
    both = {"A": _perm(update=True), "B": _perm(update=True)}
    assert _run_add(monkeypatch, ["A"], ["B"], both) is True
    # can't attach an experiment you don't control
    assert _run_add(monkeypatch, ["A"], ["B"], {"A": _perm(update=True), "B": _perm()}) is False
    # can't modify a dataset whose current experiment you don't control
    assert _run_add(monkeypatch, ["A"], ["B"], {"A": _perm(), "B": _perm(update=True)}) is False
