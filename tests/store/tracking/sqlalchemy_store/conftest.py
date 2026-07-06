import json
import time
import uuid
from pathlib import Path
from unittest import mock

import pytest
import sqlalchemy
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource as _OTelResource
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities import ViewType, trace_location
from mlflow.entities.span import Span, create_mlflow_span
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_state import TraceState
from mlflow.entities.workspace import Workspace
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES, MLFLOW_TRACKING_URI
from mlflow.store.db.db_types import MSSQL, MYSQL, POSTGRES, SQLITE
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.dbmodels.models import (
    SqlDataset,
    SqlEntityAssociation,
    SqlEvaluationDataset,
    SqlEvaluationDatasetRecord,
    SqlExperiment,
    SqlExperimentTag,
    SqlGatewaySecret,
    SqlInput,
    SqlInputTag,
    SqlLatestMetric,
    SqlLoggedModel,
    SqlLoggedModelMetric,
    SqlLoggedModelParam,
    SqlLoggedModelTag,
    SqlMetric,
    SqlOnlineScoringConfig,
    SqlParam,
    SqlRun,
    SqlScorer,
    SqlScorerVersion,
    SqlTag,
    SqlTraceInfo,
    SqlTraceMetadata,
    SqlTraceTag,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils import mlflow_tags
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

DB_URI = "sqlite:///"
ARTIFACT_URI = "artifact_folder"

pytestmark = pytest.mark.notrackingurimock

IS_MSSQL = MLFLOW_TRACKING_URI.get() and MLFLOW_TRACKING_URI.get().startswith("mssql+pyodbc")


@pytest.fixture(params=[False, True], ids=["workspace-disabled", "workspace-enabled"])
def workspaces_enabled(request, monkeypatch, disable_workspace_mode_by_default):
    """
    Run every test in this module with workspaces disabled and enabled to cover both code paths.
    """
    enabled = request.param
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true" if enabled else "false")
    if enabled:
        with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
            yield enabled
    else:
        yield enabled


@pytest.fixture
def store(tmp_path: Path, db_uri: str, workspaces_enabled: bool) -> SqlAlchemyStore:
    store_cls = WorkspaceAwareSqlAlchemyStore if workspaces_enabled else SqlAlchemyStore
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    if db_uri_env := MLFLOW_TRACKING_URI.get():
        s = store_cls(db_uri_env, artifact_uri.as_uri())
        yield s
        _cleanup_database(s)
    else:
        s = store_cls(db_uri, artifact_uri.as_uri())
        yield s


@pytest.fixture
def store_and_trace_info(store):
    exp_id = store.create_experiment("test")
    timestamp_ms = get_current_time_millis()
    return store, store.start_trace(
        TraceInfo(
            trace_id=f"tr-{uuid.uuid4()}",
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms,
            execution_duration=0,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"tr-{uuid.uuid4()}",
            request_preview=None,
            response_preview=None,
        ),
    )


def _get_store(tmp_path: Path):
    db_uri = MLFLOW_TRACKING_URI.get() or f"{DB_URI}{tmp_path / 'temp.db'}"
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


def _get_query_to_reset_experiment_id(store: SqlAlchemyStore):
    dialect = store._get_dialect()
    if dialect == POSTGRES:
        return "ALTER SEQUENCE experiments_experiment_id_seq RESTART WITH 1"
    elif dialect == MYSQL:
        return "ALTER TABLE experiments AUTO_INCREMENT = 1"
    elif dialect == MSSQL:
        return "DBCC CHECKIDENT (experiments, RESEED, 0)"
    elif dialect == SQLITE:
        # In SQLite, deleting all experiments resets experiment_id
        return None
    raise ValueError(f"Invalid dialect: {dialect}")


def _cleanup_database(store: SqlAlchemyStore):
    with store.ManagedSessionMaker() as session:
        # Delete all rows in all tables
        for model in (
            SqlLoggedModel,
            SqlLoggedModelMetric,
            SqlLoggedModelParam,
            SqlLoggedModelTag,
            SqlParam,
            SqlMetric,
            SqlLatestMetric,
            SqlTag,
            SqlInputTag,
            SqlInput,
            SqlDataset,
            SqlRun,
            SqlTraceTag,
            SqlTraceMetadata,
            SqlTraceInfo,
            SqlEvaluationDatasetRecord,
            SqlEntityAssociation,
            SqlEvaluationDataset,
            SqlExperimentTag,
            SqlOnlineScoringConfig,
            SqlScorerVersion,
            SqlScorer,
            SqlGatewaySecret,
            SqlExperiment,
        ):
            session.query(model).delete()

        # Reset experiment_id to start at 1
        if reset_experiment_id := _get_query_to_reset_experiment_id(store):
            session.execute(sqlalchemy.sql.text(reset_experiment_id))

        # Recreate the default experiment (id=0) so that tests using the global registry
        # cache (e.g., mlflow.start_run()) can still find it after cleanup.
        store._create_default_experiment(session)

    if isinstance(store, WorkspaceAwareSqlAlchemyStore):
        provider = store._get_workspace_provider_instance()

        # Reset workspace-level overrides when tests share a cached workspace store
        # against a long-lived backend store URI.
        for workspace in provider.list_workspaces():
            if workspace.name != DEFAULT_WORKSPACE_NAME:
                provider.delete_workspace(workspace.name)

        provider.update_workspace(
            Workspace(
                name=DEFAULT_WORKSPACE_NAME,
                default_artifact_root="",
                trace_archival_location="",
                trace_archival_retention="",
            )
        )

        with provider._artifact_root_cache_lock:
            provider._artifact_root_cache.clear()
        with provider._trace_archival_config_cache_lock:
            provider._trace_archival_config_cache.clear()


def _create_experiments(store: SqlAlchemyStore, names) -> str | list[str]:
    if isinstance(names, (list, tuple)):
        ids = []
        for name in names:
            # Sleep to ensure each experiment has a unique creation_time for
            # deterministic experiment search results
            time.sleep(0.001)
            ids.append(store.create_experiment(name=name))
        return ids

    time.sleep(0.001)
    return store.create_experiment(name=names)


def _get_run_configs(experiment_id=None, tags=None, start_time=None):
    return {
        "experiment_id": experiment_id,
        "user_id": "Anderson",
        "start_time": get_current_time_millis() if start_time is None else start_time,
        "tags": tags,
        "run_name": "name",
    }


def _run_factory(store: SqlAlchemyStore, config=None):
    if not config:
        config = _get_run_configs()
    if not config.get("experiment_id", None):
        config["experiment_id"] = _create_experiments(store, "test exp")

    return store.create_run(**config)


def _clear_in_memory_engine():
    engine = SqlAlchemyStore._engine_map.pop("sqlite:///:memory:", None)
    if engine is not None:
        engine.dispose()


def _search_runs(
    store: SqlAlchemyStore,
    experiment_id,
    filter_string=None,
    run_view_type=ViewType.ALL,
    max_results=SEARCH_MAX_RESULTS_DEFAULT,
):
    exps = [experiment_id] if isinstance(experiment_id, str) else experiment_id
    return [
        r.info.run_id for r in store.search_runs(exps, filter_string, run_view_type, max_results)
    ]


def _get_ordered_runs(store: SqlAlchemyStore, order_clauses, experiment_id):
    return [
        r.data.tags[mlflow_tags.MLFLOW_RUN_NAME]
        for r in store.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            run_view_type=ViewType.ALL,
            order_by=order_clauses,
        )
    ]


def _verify_logged(store, run_id, metrics, params, tags):
    run = store.get_run(run_id)
    all_metrics = sum((store.get_metric_history(run_id, key) for key in run.data.metrics), [])
    assert len(all_metrics) == len(metrics)
    logged_metrics = [(m.key, m.value, m.timestamp, m.step) for m in all_metrics]
    assert set(logged_metrics) == {(m.key, m.value, m.timestamp, m.step) for m in metrics}
    logged_tags = set(run.data.tags.items())
    assert {(tag.key, tag.value) for tag in tags} <= logged_tags
    assert len(run.data.params) == len(params)
    assert set(run.data.params.items()) == {(param.key, param.value) for param in params}


def create_mock_span_context(trace_id_num=12345, span_id_num=111) -> trace_api.SpanContext:
    context = mock.Mock()
    context.trace_id = trace_id_num
    context.span_id = span_id_num
    context.is_remote = False
    context.trace_flags = trace_api.TraceFlags(1)
    context.trace_state = trace_api.TraceState()
    return context


def create_test_span(
    trace_id,
    name="test_span",
    span_id=111,
    parent_id=None,
    status=trace_api.StatusCode.UNSET,
    status_desc=None,
    start_ns=1000000000,
    end_ns=2000000000,
    span_type="LLM",
    trace_num=12345,
    attributes=None,
    links=None,
) -> Span:
    context = create_mock_span_context(trace_num, span_id)
    parent_context = create_mock_span_context(trace_num, parent_id) if parent_id else None

    attributes = attributes or {}
    otel_span = OTelReadableSpan(
        name=name,
        context=context,
        parent=parent_context,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id),
            "mlflow.spanType": json.dumps(span_type, cls=TraceJSONEncoder),
            **{k: json.dumps(v, cls=TraceJSONEncoder) for k, v in attributes.items()},
        },
        start_time=start_ns,
        end_time=end_ns,
        status=trace_api.Status(status, status_desc),
        resource=_OTelResource.get_empty(),
    )
    span = create_mlflow_span(otel_span, trace_id, span_type)
    if links:
        span._links = list(links)
    return span


def create_test_otel_span(
    trace_id,
    name="test_span",
    parent=None,
    status_code=trace_api.StatusCode.UNSET,
    status_description=None,
    start_time=1000000000,
    end_time=2000000000,
    span_type="LLM",
    trace_id_num=12345,
    span_id_num=111,
) -> OTelReadableSpan:
    context = create_mock_span_context(trace_id_num, span_id_num)

    return OTelReadableSpan(
        name=name,
        context=context,
        parent=parent,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id),
            "mlflow.spanType": json.dumps(span_type, cls=TraceJSONEncoder),
        },
        start_time=start_time,
        end_time=end_time,
        status=trace_api.Status(status_code, status_description),
        resource=_OTelResource.get_empty(),
    )


def _create_trace(
    store: SqlAlchemyStore,
    trace_id: str,
    experiment_id=0,
    request_time=0,
    execution_duration=0,
    state=TraceState.OK,
    trace_metadata=None,
    tags=None,
    client_request_id=None,
) -> TraceInfo:
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=request_time,
        execution_duration=execution_duration,
        state=state,
        tags=tags or {},
        trace_metadata=trace_metadata or {},
        client_request_id=client_request_id,
    )
    return store.start_trace(trace_info)
