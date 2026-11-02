from unittest import mock

import pytest

import mlflow
import mlflow.server as server_module
from mlflow.entities import IssueReference
from mlflow.exceptions import MlflowException
from mlflow.gateway.tracing_utils import maybe_traced_gateway_call
from mlflow.server import app, handlers
from mlflow.store.tracking.abstract_trace_backend import AbstractTraceBackend
from mlflow.store.tracking.dbmodels.models import SqlIssue
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.trace_backend_registry import TraceBackendRegistry
from mlflow.tracking._tracking_service import utils

from tests.store.tracking.sqlalchemy_store.test_trace_backend_associations import (
    MemoryTraceBackend,
    create_trace,
)


@pytest.fixture
def selected_backend(monkeypatch):
    registry = TraceBackendRegistry()
    builder = mock.Mock(side_effect=lambda config: MemoryTraceBackend())
    registry.register("test", builder)
    monkeypatch.setattr(
        "mlflow.store.tracking.trace_backend_registry._trace_backend_registry", registry
    )
    monkeypatch.setenv("MLFLOW_TRACE_BACKEND", "test")
    return builder


def test_server_constructs_backend_aware_host_once(monkeypatch, selected_backend, db_uri, tmp_path):
    monkeypatch.setattr(handlers, "_tracking_store", None)
    store = handlers._get_tracking_store(db_uri, tmp_path.as_uri())
    assert handlers._get_tracking_store() is store
    assert isinstance(store, SqlAlchemyStore)
    assert store.trace_backend._tracking_store is store
    selected_backend.assert_called_once_with({})


@pytest.mark.parametrize("job_id", [None, "job"])
def test_only_server_owned_clients_select_backend(monkeypatch, job_id):
    monkeypatch.delenv("_MLFLOW_SERVER_BOOT_ID", raising=False)
    monkeypatch.setattr(server_module, "is_running_as_server", False)
    monkeypatch.setenv("MLFLOW_BACKEND_STORE_URI", "sqlite:///primary.db")
    monkeypatch.setenv("MLFLOW_TRACE_BACKEND", "test")
    if job_id:
        monkeypatch.setenv("_MLFLOW_SERVER_JOB_ID", job_id)
    else:
        monkeypatch.delenv("_MLFLOW_SERVER_JOB_ID", raising=False)
    with mock.patch.object(
        utils._tracking_store_registry, "get_store", return_value=mock.sentinel.store
    ) as get_store:
        assert utils._get_store("sqlite:///primary.db") is mock.sentinel.store
    if job_id:
        get_store.assert_called_once_with("sqlite:///primary.db", None, trace_backend_kind="test")
    else:
        get_store.assert_called_once_with("sqlite:///primary.db", None)


def test_invalid_backend_fails_server_construction(monkeypatch, db_uri, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACE_BACKEND", "not-registered")
    monkeypatch.setattr(handlers, "_tracking_store", None)
    with pytest.raises(MlflowException, match="Unknown trace backend"):
        handlers._get_tracking_store(db_uri, tmp_path.as_uri())
    assert handlers._tracking_store is None


@pytest.mark.parametrize("async_export", [False, True])
def test_gateway_generated_traces_use_host_backend(monkeypatch, db_uri, tmp_path, async_export):
    backend = MemoryTraceBackend()
    store = SqlAlchemyStore(db_uri, tmp_path.as_uri(), trace_backend=backend)
    experiment_id = store.create_experiment("gateway-backend")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", str(async_export).lower())
    try:
        with mock.patch.object(
            utils._tracking_store_registry, "get_store", return_value=store
        ) as get_store:
            call = maybe_traced_gateway_call(
                lambda payload: {"result": payload},
                GatewayEndpointConfig(
                    endpoint_id="endpoint",
                    endpoint_name="test",
                    experiment_id=experiment_id,
                    usage_tracking=True,
                    models=[],
                ),
            )
            assert call("hello") == {"result": "hello"}
            mlflow.flush_trace_async_logging()
            get_store.assert_called()
        assert len(backend.traces) == 1
        with store.disable_trace_backend():
            assert store.search_traces([experiment_id])[0] == []
    finally:
        store._dispose_engine()


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/api/2.0/mlflow/traces", {"experiment_id": "0", "timestamp_ms": 1}),
        ("patch", "/api/2.0/mlflow/traces/tr-a", {"timestamp_ms": 2, "status": "OK"}),
    ],
)
def test_legacy_writes_cannot_fall_through_to_sql(
    monkeypatch, db_uri, tmp_path, method, path, payload
):
    backend = MemoryTraceBackend()
    store = SqlAlchemyStore(db_uri, tmp_path.as_uri(), trace_backend=backend)
    monkeypatch.setattr(handlers, "_tracking_store", store)
    try:
        with app.test_client() as client:
            response = getattr(client, method)(
                path, json={**payload, "request_metadata": [], "tags": []}
            )
        assert response.status_code == 501
        assert response.get_json()["error_code"] == "NOT_IMPLEMENTED"
        assert backend.traces == {}
    finally:
        store._dispose_engine()


def test_unsupported_trace_mutation_returns_structured_501(monkeypatch, db_uri, tmp_path):
    store = SqlAlchemyStore(db_uri, tmp_path.as_uri(), trace_backend=AbstractTraceBackend())
    monkeypatch.setattr(handlers, "_tracking_store", store)
    try:
        with app.test_client() as client:
            response = client.patch(
                "/api/3.0/mlflow/traces/tr-a/tags", json={"key": "custom", "value": "value"}
            )
        assert response.status_code == 501
        assert response.get_json()["error_code"] == "NOT_IMPLEMENTED"
    finally:
        store._dispose_engine()


def test_issue_http_crud_and_counted_search_use_backend_storage(monkeypatch, db_uri, tmp_path):
    backend = MemoryTraceBackend()
    store = SqlAlchemyStore(db_uri, tmp_path.as_uri(), trace_backend=backend)
    monkeypatch.setattr(handlers, "_tracking_store", store)
    try:
        exp_id = store.create_experiment("backend-http-issues")
        create_trace(store, exp_id, "tr-http")
        with app.test_client() as client:
            response = client.post(
                "/api/3.0/mlflow/issues",
                json={
                    "experiment_id": exp_id,
                    "name": "Issue",
                    "description": "Details",
                },
            )
            assert response.status_code == 200
            issue_id = response.get_json()["issue"]["issue_id"]
            store.create_assessment(
                IssueReference(issue_id=issue_id, issue_name="Issue", trace_id="tr-http")
            )
            second = store.create_issue(exp_id, "Issue without assessments", "")
            response = client.post(
                "/ajax-api/3.0/mlflow/issues/search",
                json={
                    "experiment_id": exp_id,
                    "include_trace_count": True,
                    "include_total_count": True,
                    "view": "basic",
                    "max_results": 1,
                },
            )
            assert response.status_code == 200
            assert response.get_json()["issues"][0]["trace_count"] == 1
            assert response.get_json()["total_count"] == 2
            assert response.get_json()["next_page_token"] == "memory-issues:1"
            response = client.post(
                "/ajax-api/3.0/mlflow/issues/search",
                json={
                    "experiment_id": exp_id,
                    "include_trace_count": True,
                    "page_token": "memory-issues:1",
                    "max_results": 1,
                },
            )
            assert response.status_code == 200
            assert response.get_json()["issues"][0]["issue_id"] == second.issue_id
            assert response.get_json()["issues"][0]["trace_count"] == 0
            response = client.patch(
                f"/api/3.0/mlflow/issues/{issue_id}",
                json={"issue_id": issue_id, "status": "resolved"},
            )
            assert response.status_code == 200
            response = client.get(f"/api/3.0/mlflow/issues/{issue_id}")
            assert response.status_code == 200
            assert response.get_json()["issue"]["status"] == "resolved"
        with store.ManagedSessionMaker() as session:
            assert session.query(SqlIssue).count() == 0
    finally:
        store._dispose_engine()
