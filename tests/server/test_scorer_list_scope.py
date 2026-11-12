from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from flask import Flask, request

from mlflow.exceptions import MlflowNotImplementedException
from mlflow.server import handlers
from mlflow.server._request_keys import SCOPED_SCORER_LIST, SCORER_LIST_SCOPE_REQUIRED
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.databricks_rest_store import DatabricksTracingRestStore
from mlflow.store.tracking.rest_store import RestStore
from mlflow.store.tracking.scorer_list_scope import ScorerListScope
from mlflow.utils.rest_utils import MlflowHostCreds


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("state", ["missing", None, "bad", ScorerListScope()])
@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_handoff_before_store_acquisition(monkeypatch, required, state, method):
    app = Flask(__name__)
    app.config[SCORER_LIST_SCOPE_REQUIRED] = required
    store = SimpleNamespace(supports_scorer_list_scope=True, list_scorers=Mock(return_value=[]))
    acquire = Mock(return_value=store)
    monkeypatch.setattr(handlers, "_get_tracking_store", acquire)
    with app.test_request_context("/?experiment_id=1", method=method):
        if state != "missing":
            setattr(request, SCOPED_SCORER_LIST, state)
        resp = handlers._list_scorers()
    if state == "bad" or (required and state == "missing"):
        assert resp.status_code == 500
        assert resp.json["error_code"] == "INTERNAL_ERROR"
        acquire.assert_not_called()
    else:
        assert resp.status_code == 200
        if state is None or state == "missing":
            store.list_scorers.assert_called_once_with("1")
        else:
            store.list_scorers.assert_called_once_with("1", scorer_scope=state)


@pytest.mark.parametrize("scope", [ScorerListScope(), ScorerListScope(experiment_ids=None)])
@pytest.mark.parametrize("selector", ["?experiment_id=1", "?experiment_ids=1", ""])
def test_handler_unsupported_501_no_scorer_reads(monkeypatch, scope, selector):
    store = SimpleNamespace(
        supports_scorer_list_scope=False,
        filter_active_experiment_ids=lambda ids: ids,
        list_scorers=Mock(),
        list_scorers_across_experiments=Mock(),
        search_experiments=Mock(),
    )
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: store)
    with Flask(__name__).test_request_context("/" + selector):
        setattr(request, SCOPED_SCORER_LIST, scope)
        resp = handlers._list_scorers()
    assert resp.status_code == 501
    assert resp.json["error_code"] == "NOT_IMPLEMENTED"
    assert resp.json["message"] == (
        "This tracking backend cannot enforce an internal scorer-list authorization scope. "
        "Use a SQL tracking backend."
    )
    store.list_scorers.assert_not_called()
    store.list_scorers_across_experiments.assert_not_called()
    store.search_experiments.assert_not_called()


@pytest.mark.parametrize("store_type", [RestStore, DatabricksTracingRestStore])
@pytest.mark.parametrize("scope", [ScorerListScope(), ScorerListScope(experiment_ids=None)])
def test_direct_remote_and_abstract_reject_before_io(store_type, scope):
    store = store_type(lambda: MlflowHostCreds("https://unused"))
    store._call_endpoint = Mock(side_effect=AssertionError("outbound scorer I/O"))
    for call in (
        lambda: store.list_scorers("1", scorer_scope=scope),
        lambda: store.list_scorers_across_experiments(["1"], scorer_scope=scope),
        lambda: AbstractStore.list_scorers(store, "1", scorer_scope=scope),
    ):
        with pytest.raises(MlflowNotImplementedException, match="cannot enforce") as exc:
            call()
        assert exc.value.error_code == "NOT_IMPLEMENTED"
    store._call_endpoint.assert_not_called()


def test_finite_experiment_plugin_open_ended_and_request_isolation(monkeypatch):
    app = Flask(__name__)
    app.config[SCORER_LIST_SCOPE_REQUIRED] = True
    scope = ScorerListScope(experiment_ids={"1", "2"})
    store = SimpleNamespace(
        supports_scorer_list_scope=True,
        _filter_valid_scorer_experiment_ids=lambda ids: sorted(ids),
        filter_active_experiment_ids=Mock(return_value=["1"]),
        list_scorers_across_experiments=Mock(return_value=[]),
        search_experiments=Mock(),
    )
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: store)
    with app.app_context():
        with app.test_request_context("/"):
            setattr(request, SCOPED_SCORER_LIST, scope)
            assert handlers._list_scorers().status_code == 200
        with app.test_request_context("/"):
            assert handlers._list_scorers().status_code == 500
    store.filter_active_experiment_ids.assert_called_once_with(["1", "2"])
    store.list_scorers_across_experiments.assert_called_once_with(["1"], scorer_scope=scope)
    store.search_experiments.assert_not_called()


def test_legacy_experiment_plugin_actually_constrains_and_omits_keyword(monkeypatch):
    seen = []

    def old_batch(experiment_ids):
        seen.extend(experiment_ids)
        return []

    store = SimpleNamespace(
        filter_active_experiment_ids=lambda ids: [e for e in ids if e == "1"],
        list_scorers_across_experiments=old_batch,
    )
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: store)
    with Flask(__name__).test_request_context("/?experiment_ids=1&experiment_ids=2"):
        assert handlers._list_scorers().status_code == 200
    assert seen == ["1"]


def test_databricks_plural_precedence(monkeypatch):
    store = DatabricksTracingRestStore(lambda: MlflowHostCreds("https://unused"))
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: store)
    with Flask(__name__).test_request_context("/?experiment_ids=1"):
        setattr(request, SCOPED_SCORER_LIST, ScorerListScope())
        resp = handlers._list_scorers()
    assert resp.status_code == 400
    assert resp.json["error_code"] == "INVALID_PARAMETER_VALUE"


def test_public_input_cannot_forge_required_handoff(monkeypatch):
    app = Flask(__name__)
    app.config[SCORER_LIST_SCOPE_REQUIRED] = True
    acquire = Mock()
    monkeypatch.setattr(handlers, "_get_tracking_store", acquire)
    with app.test_request_context(
        "/",
        json={
            "experiment_id": "1",
            SCOPED_SCORER_LIST: None,
            SCORER_LIST_SCOPE_REQUIRED: False,
            "scorer_scope": {"experiment_ids": None},
        },
    ):
        response = handlers._list_scorers()
    assert response.status_code == 500
    assert response.json["error_code"] == "INTERNAL_ERROR"
    acquire.assert_not_called()
