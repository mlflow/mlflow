# Cross-dialect (Docker matrix) tests for MLFLOW_SQLALCHEMYSTORE_SEARCH_TRACES_TIMEOUT_SECONDS.
# PostgreSQL and MySQL cancel a search that exceeds the limit; other dialects ignore it.

import time
import uuid
from pathlib import Path
from unittest import mock

import pytest
from sqlalchemy import sql

from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.environment_variables import (
    MLFLOW_SQLALCHEMYSTORE_SEARCH_TRACES_TIMEOUT_SECONDS,
    MLFLOW_TRACKING_URI,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import DEADLINE_EXCEEDED, ErrorCode
from mlflow.store.db.db_types import MYSQL, POSTGRES
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

pytestmark = pytest.mark.notrackingurimock

_SLEEP_CONDITIONS = {
    POSTGRES: "(SELECT 1 FROM pg_sleep(2)) = 1",
    MYSQL: "SLEEP(2) = 0",
}


@pytest.fixture
def store(tmp_path: Path):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    store = SqlAlchemyStore(MLFLOW_TRACKING_URI.get(), artifact_uri.as_uri())
    try:
        yield store
    finally:
        store._dispose_engine()


@pytest.fixture
def experiment_id(store: SqlAlchemyStore) -> str:
    experiment_id = store.create_experiment(f"search-traces-timeout-{uuid.uuid4().hex[:12]}")
    store.start_trace(
        TraceInfo(
            trace_id=f"tr-{uuid.uuid4().hex}",
            trace_location=TraceLocation.from_experiment_id(experiment_id),
            request_time=int(time.time() * 1000),
            execution_duration=0,
            state=TraceState.OK,
        )
    )
    return experiment_id


def _search_with_sleep(store: SqlAlchemyStore, experiment_id: str):
    apply_filters = store._apply_trace_filter_clauses

    def apply_filters_with_sleep(stmt, *args):
        return apply_filters(stmt, *args).filter(sql.text(_SLEEP_CONDITIONS[store.db_type]))

    with mock.patch.object(
        store, "_apply_trace_filter_clauses", side_effect=apply_filters_with_sleep
    ) as mock_apply_filters:
        try:
            return store.search_traces(locations=[experiment_id])
        finally:
            mock_apply_filters.assert_called_once()


def test_search_traces_timeout_cancels_slow_search(store, experiment_id, monkeypatch):
    if store.db_type not in (POSTGRES, MYSQL):
        pytest.skip("Only PostgreSQL and MySQL support the search timeout")

    monkeypatch.setenv(MLFLOW_SQLALCHEMYSTORE_SEARCH_TRACES_TIMEOUT_SECONDS.name, "0.5")
    start = time.monotonic()
    with pytest.raises(
        MlflowException, match=MLFLOW_SQLALCHEMYSTORE_SEARCH_TRACES_TIMEOUT_SECONDS.name
    ) as exc_info:
        _search_with_sleep(store, experiment_id)
    assert time.monotonic() - start < 2
    assert exc_info.value.error_code == ErrorCode.Name(DEADLINE_EXCEEDED)

    # The limit is scoped to the search, so the pooled connection is not left with it.
    traces, _ = store.search_traces(locations=[experiment_id])
    assert len(traces) == 1
    if store.db_type == POSTGRES:
        with store.ManagedSessionMaker() as session:
            assert session.execute(sql.text("SHOW statement_timeout")).scalar() == "0"


def test_search_traces_without_timeout_runs_to_completion(store, experiment_id):
    if store.db_type not in (POSTGRES, MYSQL):
        pytest.skip("Only PostgreSQL and MySQL support the search timeout")

    traces, _ = _search_with_sleep(store, experiment_id)
    assert len(traces) == 1


def test_search_traces_timeout_is_ignored_on_other_databases(store, experiment_id, monkeypatch):
    if store.db_type in (POSTGRES, MYSQL):
        pytest.skip("PostgreSQL and MySQL apply the search timeout")

    monkeypatch.setenv(MLFLOW_SQLALCHEMYSTORE_SEARCH_TRACES_TIMEOUT_SECONDS.name, "0.001")
    traces, _ = store.search_traces(locations=[experiment_id])
    assert len(traces) == 1
