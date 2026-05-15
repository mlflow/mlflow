import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from unittest import mock

import pytest
import sqlalchemy
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource as _OTelResource
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow.db
from mlflow import entities
from mlflow.entities import (
    AssessmentSource,
    AssessmentSourceType,
    Feedback,
    ViewType,
    trace_location,
)
from mlflow.entities.assessment import FeedbackValue
from mlflow.entities.gateway_endpoint import GatewayEndpoint
from mlflow.entities.span import Span, create_mlflow_span
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_state import TraceState
from mlflow.environment_variables import (
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_TRACKING_URI,
)
from mlflow.exceptions import MlflowException
from mlflow.store.db.db_types import MSSQL, MYSQL, POSTGRES, SQLITE
from mlflow.store.db.utils import (
    _get_latest_schema_revision,
    _get_schema_version,
)
from mlflow.store.tracking import (
    SEARCH_MAX_RESULTS_DEFAULT,
)
from mlflow.store.tracking.dbmodels import models
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
from mlflow.store.tracking.sqlalchemy_store import (
    SqlAlchemyStore,
    _get_orderby_clauses,
)
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils import mlflow_tags
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.integration.utils import invoke_cli_runner

DB_URI = "sqlite:///"
ARTIFACT_URI = "artifact_folder"

pytestmark = pytest.mark.notrackingurimock

IS_MSSQL = MLFLOW_TRACKING_URI.get() and MLFLOW_TRACKING_URI.get().startswith("mssql+pyodbc")


@pytest.fixture(autouse=True, params=[False, True], ids=["workspace-disabled", "workspace-enabled"])
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


# Helper functions for span tests
def create_mock_span_context(trace_id_num=12345, span_id_num=111) -> trace_api.SpanContext:
    """Create a mock span context for testing."""
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
    """
    Create an MLflow span for testing with minimal boilerplate.

    Args:
        trace_id: The trace ID string
        name: Span name
        span_id: Span ID number (default: 111)
        parent_id: Parent span ID number, or None for root span
        status: StatusCode enum value (default: UNSET)
        status_desc: Status description string
        start_ns: Start time in nanoseconds
        end_ns: End time in nanoseconds
        span_type: Span type (default: "LLM")
        trace_num: Trace ID number for context (default: 12345)
        attributes: Attributes dictionary
        links: List of Link objects to attach to the span

    Returns:
        MLflow Span object ready for use in tests
    """
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


# Keep the old function for backward compatibility but delegate to new one
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
    """Create an OTelReadableSpan for testing with common defaults."""
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


def db_types_and_drivers():
    d = {
        "sqlite": [
            "pysqlite",
            "pysqlcipher",
        ],
        "postgresql": [
            "psycopg2",
            "pg8000",
            "psycopg2cffi",
            "pypostgresql",
            "pygresql",
            "zxjdbc",
        ],
        "mysql": [
            "mysqldb",
            "pymysql",
            "mysqlconnector",
            "cymysql",
            "oursql",
            "gaerdbms",
            "pyodbc",
            "zxjdbc",
        ],
        "mssql": [
            "pyodbc",
            "mxodbc",
            "pymssql",
            "zxjdbc",
            "adodbapi",
        ],
    }
    for db_type, drivers in d.items():
        for driver in drivers:
            yield db_type, driver


@pytest.mark.parametrize(("db_type", "driver"), db_types_and_drivers())
def test_correct_db_type_from_uri(db_type, driver):
    assert extract_db_type_from_uri(f"{db_type}+{driver}://...") == db_type
    # try the driver-less version, which will revert SQLAlchemy to the default driver
    assert extract_db_type_from_uri(f"{db_type}://...") == db_type


@pytest.mark.parametrize(
    "db_uri",
    [
        "oracle://...",
        "oracle+cx_oracle://...",
        "snowflake://...",
        "://...",
        "abcdefg",
    ],
)
def test_fail_on_unsupported_db_type(db_uri):
    with pytest.raises(MlflowException, match=r"Invalid database engine"):
        extract_db_type_from_uri(db_uri)


def test_fail_on_multiple_drivers():
    with pytest.raises(MlflowException, match=r"Invalid database URI"):
        extract_db_type_from_uri("mysql+pymsql+pyodbc://...")


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


# Tests for Search API
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


def test_upgrade_cli_idempotence(store: SqlAlchemyStore):
    # Repeatedly run `mlflow db upgrade` against our database, verifying that the command
    # succeeds and that the DB has the latest schema
    engine = sqlalchemy.create_engine(store.db_uri)
    assert _get_schema_version(engine) == _get_latest_schema_revision()
    for _ in range(3):
        invoke_cli_runner(mlflow.db.commands, ["upgrade", store.db_uri])
        assert _get_schema_version(engine) == _get_latest_schema_revision()
    engine.dispose()


def test_metrics_materialization_upgrade_succeeds_and_produces_expected_latest_metric_values(
    store: SqlAlchemyStore, tmp_path
):
    """
    Tests the ``89d4b8295536_create_latest_metrics_table`` migration by migrating and querying
    the MLflow Tracking SQLite database located at
    /mlflow/tests/resources/db/db_version_7ac759974ad8_with_metrics.sql. This database contains
    metric entries populated by the following metrics generation script:
    https://gist.github.com/dbczumar/343173c6b8982a0cc9735ff19b5571d9.

    First, the database is upgraded from its HEAD revision of
    ``7ac755974ad8_update_run_tags_with_larger_limit`` to the latest revision via
    ``mlflow db upgrade``.

    Then, the test confirms that the metric entries returned by calls
    to ``SqlAlchemyStore.get_run()`` are consistent between the latest revision and the
    ``7ac755974ad8_update_run_tags_with_larger_limit`` revision. This is confirmed by
    invoking ``SqlAlchemyStore.get_run()`` for each run id that is present in the upgraded
    database and comparing the resulting runs' metric entries to a JSON dump taken from the
    SQLite database prior to the upgrade (located at
    mlflow/tests/resources/db/db_version_7ac759974ad8_with_metrics_expected_values.json).
    This JSON dump can be replicated by installing MLflow version 1.2.0 and executing the
    following code from the directory containing this test suite:

    .. code-block:: python

        import json
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(
            "sqlite:///../../resources/db/db_version_7ac759974ad8_with_metrics.sql"
        )
        client = MlflowClient()
        summary_metrics = {
            run.info.run_id: run.data.metrics for run in client.search_runs(experiment_ids="0")
        }
        with open("dump.json", "w") as dump_file:
            json.dump(summary_metrics, dump_file, indent=4)

    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_resources_path = os.path.normpath(
        os.path.join(current_dir, os.pardir, os.pardir, os.pardir, "resources", "db")
    )
    expected_metric_values_path = os.path.join(
        db_resources_path, "db_version_7ac759974ad8_with_metrics_expected_values.json"
    )
    db_path = tmp_path / "tmp_db.sql"
    db_url = "sqlite:///" + str(db_path)
    shutil.copy2(
        src=os.path.join(db_resources_path, "db_version_7ac759974ad8_with_metrics.sql"),
        dst=db_path,
    )

    invoke_cli_runner(mlflow.db.commands, ["upgrade", db_url])
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    store = SqlAlchemyStore(db_url, artifact_uri.as_uri())
    with open(expected_metric_values_path) as f:
        expected_metric_values = json.load(f)

    for run_id, expected_metrics in expected_metric_values.items():
        fetched_run = store.get_run(run_id=run_id)
        assert fetched_run.data.metrics == expected_metrics


def test_sqlalchemy_store_behaves_as_expected_with_inmemory_sqlite_db(
    monkeypatch, workspaces_enabled
):
    monkeypatch.setenv("MLFLOW_SQLALCHEMYSTORE_POOLCLASS", "SingletonThreadPool")
    _clear_in_memory_engine()
    store_cls = WorkspaceAwareSqlAlchemyStore if workspaces_enabled else SqlAlchemyStore
    store = store_cls("sqlite:///:memory:", ARTIFACT_URI)
    experiment_id = store.create_experiment(name="exp1")
    run = store.create_run(
        experiment_id=experiment_id,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )
    run_id = run.info.run_id
    metric = entities.Metric("mymetric", 1, 0, 0)
    store.log_metric(run_id=run_id, metric=metric)
    param = entities.Param("myparam", "A")
    store.log_param(run_id=run_id, param=param)
    fetched_run = store.get_run(run_id=run_id)
    assert fetched_run.info.run_id == run_id
    assert metric.key in fetched_run.data.metrics
    assert param.key in fetched_run.data.params
    store._dispose_engine()
    _clear_in_memory_engine()


def test_sqlalchemy_store_can_be_initialized_when_default_experiment_has_been_deleted(
    tmp_sqlite_uri,
):
    store = SqlAlchemyStore(tmp_sqlite_uri, ARTIFACT_URI)
    store.delete_experiment("0")
    assert store.get_experiment("0").lifecycle_stage == entities.LifecycleStage.DELETED
    SqlAlchemyStore(tmp_sqlite_uri, ARTIFACT_URI)


def test_sqlalchemy_store_does_not_create_artifact_root_directory_on_init(tmp_path, db_uri):
    """
    Verify that SqlAlchemyStore does NOT create the artifact root directory during initialization.

    The directory should only be created lazily when the first artifact is logged. This allows
    MLflow servers to run in read-only environments (e.g., K8s containers) when artifacts are
    stored remotely and the local artifact root is never actually used.

    See: https://github.com/mlflow/mlflow/issues/19658
    """
    artifact_root = tmp_path / "artifacts"

    store = SqlAlchemyStore(db_uri, str(artifact_root))

    assert not artifact_root.exists()

    store._dispose_engine()


def test_sqlalchemy_store_creates_artifact_directory_on_log_artifact(tmp_path, db_uri):
    from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
    from mlflow.utils.file_utils import path_to_local_file_uri

    artifact_root = tmp_path / "artifacts"

    store = SqlAlchemyStore(db_uri, path_to_local_file_uri(str(artifact_root)))
    exp_id = store.create_experiment("test")
    run = store.create_run(exp_id, user_id="user", start_time=0, tags=[], run_name="run")

    assert not artifact_root.exists()

    src_file = tmp_path / "test.txt"
    src_file.write_text("hello")

    artifact_repo = get_artifact_repository(run.info.artifact_uri)
    artifact_repo.log_artifact(str(src_file))

    assert artifact_root.exists()

    store._dispose_engine()


class TextClauseMatcher:
    def __init__(self, text):
        self.text = text

    def __eq__(self, other):
        return self.text == other.text


def test_set_zero_value_insertion_for_autoincrement_column_MYSQL():
    mock_store = mock.Mock(SqlAlchemyStore)
    mock_store.db_type = MYSQL
    with mock.patch("sqlalchemy.orm.session.Session", spec=True) as mock_session:
        SqlAlchemyStore._set_zero_value_insertion_for_autoincrement_column(mock_store, mock_session)
        mock_session.execute.assert_called_with(
            TextClauseMatcher("SET @@SESSION.sql_mode='NO_AUTO_VALUE_ON_ZERO';")
        )


def test_set_zero_value_insertion_for_autoincrement_column_MSSQL():
    mock_store = mock.Mock(SqlAlchemyStore)
    mock_store.db_type = MSSQL
    with mock.patch("sqlalchemy.orm.session.Session", spec=True) as mock_session:
        SqlAlchemyStore._set_zero_value_insertion_for_autoincrement_column(mock_store, mock_session)
        mock_session.execute.assert_called_with(
            TextClauseMatcher("SET IDENTITY_INSERT experiments ON;")
        )


def test_unset_zero_value_insertion_for_autoincrement_column_MYSQL():
    mock_store = mock.Mock(SqlAlchemyStore)
    mock_store.db_type = MYSQL
    with mock.patch("sqlalchemy.orm.session.Session", spec=True) as mock_session:
        SqlAlchemyStore._unset_zero_value_insertion_for_autoincrement_column(
            mock_store, mock_session
        )
        mock_session.execute.assert_called_with(TextClauseMatcher("SET @@SESSION.sql_mode='';"))


def test_unset_zero_value_insertion_for_autoincrement_column_MSSQL():
    mock_store = mock.Mock(SqlAlchemyStore)
    mock_store.db_type = MSSQL
    with mock.patch("sqlalchemy.orm.session.Session", spec=True) as mock_session:
        SqlAlchemyStore._unset_zero_value_insertion_for_autoincrement_column(
            mock_store, mock_session
        )
        mock_session.execute.assert_called_with(
            TextClauseMatcher("SET IDENTITY_INSERT experiments OFF;")
        )


def test_get_attribute_name():
    assert models.SqlRun.get_attribute_name("artifact_uri") == "artifact_uri"
    assert models.SqlRun.get_attribute_name("status") == "status"
    assert models.SqlRun.get_attribute_name("start_time") == "start_time"
    assert models.SqlRun.get_attribute_name("end_time") == "end_time"
    assert models.SqlRun.get_attribute_name("deleted_time") == "deleted_time"
    assert models.SqlRun.get_attribute_name("run_name") == "name"
    assert models.SqlRun.get_attribute_name("run_id") == "run_uuid"

    # we want this to break if a searchable or orderable attribute has been added
    # and not referred to in this test
    # searchable attributes are also orderable
    assert len(entities.RunInfo.get_orderable_attributes()) == 7


def test_get_orderby_clauses(tmp_sqlite_uri):
    store = SqlAlchemyStore(tmp_sqlite_uri, ARTIFACT_URI)
    with store.ManagedSessionMaker() as session:
        # test that ['runs.start_time DESC', 'SqlRun.run_uuid'] is returned by default
        parsed = [str(x) for x in _get_orderby_clauses([], session)[1]]
        assert parsed == ["runs.start_time DESC", "SqlRun.run_uuid"]

        # test that the given 'start_time' replaces the default one ('runs.start_time DESC')
        parsed = [str(x) for x in _get_orderby_clauses(["attribute.start_time ASC"], session)[1]]
        assert "SqlRun.start_time" in parsed
        assert "SqlRun.start_time DESC" not in parsed

        # test that an exception is raised when 'order_by' contains duplicates
        match = "`order_by` contains duplicate fields"
        with pytest.raises(MlflowException, match=match):
            _get_orderby_clauses(["attribute.start_time", "attribute.start_time"], session)

        with pytest.raises(MlflowException, match=match):
            _get_orderby_clauses(["param.p", "param.p"], session)

        with pytest.raises(MlflowException, match=match):
            _get_orderby_clauses(["metric.m", "metric.m"], session)

        with pytest.raises(MlflowException, match=match):
            _get_orderby_clauses(["tag.t", "tag.t"], session)

        # test that an exception is NOT raised when key types are different
        _get_orderby_clauses(["param.a", "metric.a", "tag.a"], session)

        select_clause, parsed, _ = _get_orderby_clauses(["metric.a"], session)
        select_clause = [str(x) for x in select_clause]
        parsed = [str(x) for x in parsed]
        # test that "=" is used rather than "is" when comparing to True
        assert "is_nan = true" in select_clause[0]
        assert "value IS NULL" in select_clause[0]
        # test that clause name is in parsed
        assert "clause_1" in parsed[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_default_trace_status_in_progress(store: SqlAlchemyStore, is_async: bool):
    experiment_id = store.create_experiment("test_default_in_progress")
    # Generate a proper MLflow trace ID in the format "tr-<32-char-hex>"
    trace_id = f"tr-{uuid.uuid4().hex}"

    # Create a child span (has parent, not a root span)
    child_context = mock.Mock()
    child_context.trace_id = 56789
    child_context.span_id = 777
    child_context.is_remote = False
    child_context.trace_flags = trace_api.TraceFlags(1)
    child_context.trace_state = trace_api.TraceState()

    parent_context = mock.Mock()
    parent_context.trace_id = 56789
    parent_context.span_id = 888  # Parent span not included in log
    parent_context.is_remote = False
    parent_context.trace_flags = trace_api.TraceFlags(1)
    parent_context.trace_state = trace_api.TraceState()

    child_otel_span = OTelReadableSpan(
        name="child_span_only",
        context=child_context,
        parent=parent_context,  # Has parent, not a root span
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id),
            "mlflow.spanType": json.dumps("LLM", cls=TraceJSONEncoder),
        },
        start_time=2000000000,
        end_time=3000000000,
        status=trace_api.Status(trace_api.StatusCode.OK),
        resource=_OTelResource.get_empty(),
    )
    child_span = create_mlflow_span(child_otel_span, trace_id, "LLM")

    # Log only the child span (no root span)
    if is_async:
        await store.log_spans_async(experiment_id, [child_span])
    else:
        store.log_spans(experiment_id, [child_span])

    # Check trace was created with IN_PROGRESS status (default when no root span)
    traces, _ = store.search_traces([experiment_id])
    trace = next(t for t in traces if t.request_id == trace_id)
    assert trace.state.value == "IN_PROGRESS"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    ("span_status_code", "expected_trace_status"),
    [
        (trace_api.StatusCode.OK, "OK"),
        (trace_api.StatusCode.ERROR, "ERROR"),
    ],
)
async def test_log_spans_sets_trace_status_from_root_span(
    store: SqlAlchemyStore,
    is_async: bool,
    span_status_code: trace_api.StatusCode,
    expected_trace_status: str,
):
    experiment_id = store.create_experiment("test_trace_status_from_root")
    # Generate a proper MLflow trace ID in the format "tr-<32-char-hex>"
    trace_id = f"tr-{uuid.uuid4().hex}"

    # Create root span with specified status
    description = (
        f"Root span {span_status_code.name}"
        if span_status_code == trace_api.StatusCode.ERROR
        else None
    )
    root_otel_span = create_test_otel_span(
        trace_id=trace_id,
        name=f"root_span_{span_status_code.name}",
        status_code=span_status_code,
        status_description=description,
        trace_id_num=12345 + span_status_code.value,
        span_id_num=111 + span_status_code.value,
    )
    root_span = create_mlflow_span(root_otel_span, trace_id, "LLM")

    # Log the span
    if is_async:
        await store.log_spans_async(experiment_id, [root_span])
    else:
        store.log_spans(experiment_id, [root_span])

    # Verify trace has expected status from root span
    traces, _ = store.search_traces([experiment_id])
    trace = next(t for t in traces if t.request_id == trace_id)
    assert trace.state.value == expected_trace_status


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_unset_root_span_status_defaults_to_ok(
    store: SqlAlchemyStore, is_async: bool
):
    experiment_id = store.create_experiment("test_unset_root_span")
    # Generate a proper MLflow trace ID in the format "tr-<32-char-hex>"
    trace_id = f"tr-{uuid.uuid4().hex}"

    # Create root span with UNSET status (this is unexpected in practice)
    root_unset_span = create_test_otel_span(
        trace_id=trace_id,
        name="root_span_unset",
        status_code=trace_api.StatusCode.UNSET,  # Unexpected in practice
        start_time=3000000000,
        end_time=4000000000,
        trace_id_num=23456,
        span_id_num=333,
    )
    root_span = create_mlflow_span(root_unset_span, trace_id, "LLM")

    if is_async:
        await store.log_spans_async(experiment_id, [root_span])
    else:
        store.log_spans(experiment_id, [root_span])

    # Verify trace defaults to OK status when root span has UNSET status
    traces, _ = store.search_traces([experiment_id])
    trace = next(t for t in traces if t.request_id == trace_id)
    assert trace.state.value == "OK"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_updates_in_progress_trace_status_from_root_span(
    store: SqlAlchemyStore, is_async: bool
):
    experiment_id = store.create_experiment("test_trace_status_update")
    # Generate a proper MLflow trace ID in the format "tr-<32-char-hex>"
    trace_id = f"tr-{uuid.uuid4().hex}"

    # First, log a non-root span which will create trace with default IN_PROGRESS status
    parent_context = create_mock_span_context(45678, 555)  # Will be root span later

    child_otel_span = create_test_otel_span(
        trace_id=trace_id,
        name="child_span",
        parent=parent_context,  # Has parent, not a root span
        status_code=trace_api.StatusCode.OK,
        start_time=1100000000,
        end_time=1900000000,
        trace_id_num=45678,
        span_id_num=666,
    )
    child_span = create_mlflow_span(child_otel_span, trace_id, "LLM")

    if is_async:
        await store.log_spans_async(experiment_id, [child_span])
    else:
        store.log_spans(experiment_id, [child_span])

    # Verify trace was created with IN_PROGRESS status (default when no root span)
    traces, _ = store.search_traces([experiment_id])
    trace = next(t for t in traces if t.request_id == trace_id)
    assert trace.state.value == "IN_PROGRESS"

    # Now log root span with ERROR status
    root_otel_span = create_test_otel_span(
        trace_id=trace_id,
        name="root_span",
        parent=None,  # Root span
        status_code=trace_api.StatusCode.ERROR,
        status_description="Root span error",
        trace_id_num=45678,
        span_id_num=555,
    )
    root_span = create_mlflow_span(root_otel_span, trace_id, "LLM")

    if is_async:
        await store.log_spans_async(experiment_id, [root_span])
    else:
        store.log_spans(experiment_id, [root_span])

    # Check trace status was updated to ERROR from root span
    traces, _ = store.search_traces([experiment_id])
    trace = next(t for t in traces if t.request_id == trace_id)
    assert trace.state.value == "ERROR"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_updates_state_unspecified_trace_status_from_root_span(
    store: SqlAlchemyStore, is_async: bool
):
    experiment_id = store.create_experiment("test_unspecified_update")
    # Generate a proper MLflow trace ID in the format "tr-<32-char-hex>"
    trace_id = f"tr-{uuid.uuid4().hex}"

    # First, create a trace with OK status by logging a root span with OK status
    initial_span = create_test_span(
        trace_id=trace_id,
        name="initial_unset_span",
        span_id=999,
        status=trace_api.StatusCode.OK,
        trace_num=67890,
    )

    if is_async:
        await store.log_spans_async(experiment_id, [initial_span])
    else:
        store.log_spans(experiment_id, [initial_span])

    # Verify trace was created with OK status
    trace = store.get_trace_info(trace_id)
    assert trace.state.value == "OK"

    # Now log a new root span with OK status (earlier start time makes it the new root)
    new_root_span = create_test_span(
        trace_id=trace_id,
        name="new_root_span",
        span_id=1000,
        status=trace_api.StatusCode.OK,
        start_ns=500000000,  # Earlier than initial span
        end_ns=2500000000,
        trace_num=67890,
    )

    if is_async:
        await store.log_spans_async(experiment_id, [new_root_span])
    else:
        store.log_spans(experiment_id, [new_root_span])

    # Check trace status was updated to OK from root span
    traces, _ = store.search_traces([experiment_id])
    trace = next(t for t in traces if t.request_id == trace_id)
    assert trace.state.value == "OK"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_does_not_update_finalized_trace_status(
    store: SqlAlchemyStore, is_async: bool
):
    experiment_id = store.create_experiment("test_no_update_finalized")

    # Test that OK status is not updated
    # Generate a proper MLflow trace ID in the format "tr-<32-char-hex>"
    trace_id_ok = f"tr-{uuid.uuid4().hex}"

    # Create initial root span with OK status
    ok_span = create_test_span(
        trace_id=trace_id_ok,
        name="ok_root_span",
        span_id=1111,
        status=trace_api.StatusCode.OK,
        trace_num=78901,
    )

    if is_async:
        await store.log_spans_async(experiment_id, [ok_span])
    else:
        store.log_spans(experiment_id, [ok_span])

    # Verify trace has OK status
    traces, _ = store.search_traces([experiment_id])
    trace_ok = next(t for t in traces if t.request_id == trace_id_ok)
    assert trace_ok.state.value == "OK"

    # Now log a new root span with ERROR status
    error_span = create_test_span(
        trace_id=trace_id_ok,
        name="error_root_span",
        span_id=2222,
        status=trace_api.StatusCode.ERROR,
        status_desc="New error",
        start_ns=500000000,
        end_ns=2500000000,
        trace_num=78901,
    )

    if is_async:
        await store.log_spans_async(experiment_id, [error_span])
    else:
        store.log_spans(experiment_id, [error_span])

    # Verify trace status is still OK (not updated to ERROR)
    traces, _ = store.search_traces([experiment_id])
    trace_ok = next(t for t in traces if t.request_id == trace_id_ok)
    assert trace_ok.state.value == "OK"


def test_scorer_operations(store: SqlAlchemyStore):
    """
    Test the scorer operations: register_scorer, list_scorers, get_scorer, and delete_scorer.

    This test covers:
    1. Registering multiple scorers with different names
    2. Registering multiple versions of the same scorer
    3. Listing scorers (should return latest version for each name)
    4. Getting specific scorer versions
    5. Getting latest scorer version when version is not specified
    6. Deleting scorers and verifying they are deleted
    """
    # Create an experiment for testing
    experiment_id = store.create_experiment("test_scorer_experiment")

    store.register_scorer(experiment_id, "accuracy_scorer", '{"data": "accuracy_scorer1"}')
    store.register_scorer(experiment_id, "accuracy_scorer", '{"data": "accuracy_scorer2"}')
    store.register_scorer(experiment_id, "accuracy_scorer", '{"data": "accuracy_scorer3"}')

    store.register_scorer(experiment_id, "safety_scorer", '{"data": "safety_scorer1"}')
    store.register_scorer(experiment_id, "safety_scorer", '{"data": "safety_scorer2"}')

    store.register_scorer(experiment_id, "relevance_scorer", '{"data": "relevance_scorer1"}')

    # Step 2: Test list_scorers - should return latest version for each scorer name
    scorers = store.list_scorers(experiment_id)

    # Should return 3 scorers (one for each unique name)
    assert len(scorers) == 3, f"Expected 3 scorers, got {len(scorers)}"

    scorer_names = [scorer.scorer_name for scorer in scorers]
    # Verify the order is sorted by scorer_name
    assert scorer_names == ["accuracy_scorer", "relevance_scorer", "safety_scorer"], (
        f"Expected sorted order, got {scorer_names}"
    )

    # Verify versions are the latest and check serialized_scorer content
    for scorer in scorers:
        if scorer.scorer_name == "accuracy_scorer":
            assert scorer.scorer_version == 3, (
                f"Expected version 3 for accuracy_scorer, got {scorer.scorer_version}"
            )
            assert scorer._serialized_scorer == '{"data": "accuracy_scorer3"}'
        elif scorer.scorer_name == "safety_scorer":
            assert scorer.scorer_version == 2, (
                f"Expected version 2 for safety_scorer, got {scorer.scorer_version}"
            )
            assert scorer._serialized_scorer == '{"data": "safety_scorer2"}'
        elif scorer.scorer_name == "relevance_scorer":
            assert scorer.scorer_version == 1, (
                f"Expected version 1 for relevance_scorer, got {scorer.scorer_version}"
            )
            assert scorer._serialized_scorer == '{"data": "relevance_scorer1"}'

    # Test list_scorer_versions
    accuracy_scorer_versions = store.list_scorer_versions(experiment_id, "accuracy_scorer")
    assert len(accuracy_scorer_versions) == 3, (
        f"Expected 3 versions, got {len(accuracy_scorer_versions)}"
    )

    # Verify versions are ordered by version number
    assert accuracy_scorer_versions[0].scorer_version == 1
    assert accuracy_scorer_versions[0]._serialized_scorer == '{"data": "accuracy_scorer1"}'
    assert accuracy_scorer_versions[1].scorer_version == 2
    assert accuracy_scorer_versions[1]._serialized_scorer == '{"data": "accuracy_scorer2"}'
    assert accuracy_scorer_versions[2].scorer_version == 3
    assert accuracy_scorer_versions[2]._serialized_scorer == '{"data": "accuracy_scorer3"}'

    # Step 3: Test get_scorer with specific versions
    # Get accuracy_scorer version 1
    accuracy_v1 = store.get_scorer(experiment_id, "accuracy_scorer", version=1)
    assert accuracy_v1._serialized_scorer == '{"data": "accuracy_scorer1"}'
    assert accuracy_v1.scorer_version == 1

    # Get accuracy_scorer version 2
    accuracy_v2 = store.get_scorer(experiment_id, "accuracy_scorer", version=2)
    assert accuracy_v2._serialized_scorer == '{"data": "accuracy_scorer2"}'
    assert accuracy_v2.scorer_version == 2

    # Get accuracy_scorer version 3 (latest)
    accuracy_v3 = store.get_scorer(experiment_id, "accuracy_scorer", version=3)
    assert accuracy_v3._serialized_scorer == '{"data": "accuracy_scorer3"}'
    assert accuracy_v3.scorer_version == 3

    # Step 4: Test get_scorer without version (should return latest)
    accuracy_latest = store.get_scorer(experiment_id, "accuracy_scorer")
    assert accuracy_latest._serialized_scorer == '{"data": "accuracy_scorer3"}'
    assert accuracy_latest.scorer_version == 3

    safety_latest = store.get_scorer(experiment_id, "safety_scorer")
    assert safety_latest._serialized_scorer == '{"data": "safety_scorer2"}'
    assert safety_latest.scorer_version == 2

    relevance_latest = store.get_scorer(experiment_id, "relevance_scorer")
    assert relevance_latest._serialized_scorer == '{"data": "relevance_scorer1"}'
    assert relevance_latest.scorer_version == 1

    # Step 5: Test error cases for get_scorer
    # Try to get non-existent scorer
    with pytest.raises(MlflowException, match="Scorer with name 'non_existent' not found"):
        store.get_scorer(experiment_id, "non_existent")

    # Try to get non-existent version
    with pytest.raises(
        MlflowException, match="Scorer with name 'accuracy_scorer' and version 999 not found"
    ):
        store.get_scorer(experiment_id, "accuracy_scorer", version=999)

    # Step 6: Test delete_scorer - delete specific version of accuracy_scorer
    # Delete version 1 of accuracy_scorer
    store.delete_scorer(experiment_id, "accuracy_scorer", version=1)

    # Verify version 1 is deleted but other versions still exist
    with pytest.raises(
        MlflowException, match="Scorer with name 'accuracy_scorer' and version 1 not found"
    ):
        store.get_scorer(experiment_id, "accuracy_scorer", version=1)

    # Verify versions 2 and 3 still exist
    accuracy_v2 = store.get_scorer(experiment_id, "accuracy_scorer", version=2)
    assert accuracy_v2._serialized_scorer == '{"data": "accuracy_scorer2"}'
    assert accuracy_v2.scorer_version == 2

    accuracy_v3 = store.get_scorer(experiment_id, "accuracy_scorer", version=3)
    assert accuracy_v3._serialized_scorer == '{"data": "accuracy_scorer3"}'
    assert accuracy_v3.scorer_version == 3

    # Verify latest version still works
    accuracy_latest_after_partial_delete = store.get_scorer(experiment_id, "accuracy_scorer")
    assert accuracy_latest_after_partial_delete._serialized_scorer == '{"data": "accuracy_scorer3"}'
    assert accuracy_latest_after_partial_delete.scorer_version == 3

    # Step 7: Test delete_scorer - delete all versions of accuracy_scorer
    store.delete_scorer(experiment_id, "accuracy_scorer")

    # Verify accuracy_scorer is completely deleted
    with pytest.raises(MlflowException, match="Scorer with name 'accuracy_scorer' not found"):
        store.get_scorer(experiment_id, "accuracy_scorer")

    # Verify other scorers still exist
    safety_latest_after_delete = store.get_scorer(experiment_id, "safety_scorer")
    assert safety_latest_after_delete._serialized_scorer == '{"data": "safety_scorer2"}'
    assert safety_latest_after_delete.scorer_version == 2

    relevance_latest_after_delete = store.get_scorer(experiment_id, "relevance_scorer")
    assert relevance_latest_after_delete._serialized_scorer == '{"data": "relevance_scorer1"}'
    assert relevance_latest_after_delete.scorer_version == 1

    # Step 8: Test list_scorers after deletion
    scorers_after_delete = store.list_scorers(experiment_id)
    assert len(scorers_after_delete) == 2, (
        f"Expected 2 scorers after deletion, got {len(scorers_after_delete)}"
    )

    scorer_names_after_delete = [scorer.scorer_name for scorer in scorers_after_delete]
    assert "accuracy_scorer" not in scorer_names_after_delete
    assert "safety_scorer" in scorer_names_after_delete
    assert "relevance_scorer" in scorer_names_after_delete

    # Step 9: Test delete_scorer for non-existent scorer
    with pytest.raises(MlflowException, match="Scorer with name 'non_existent' not found"):
        store.delete_scorer(experiment_id, "non_existent")

    # Step 10: Test delete_scorer for non-existent version
    with pytest.raises(
        MlflowException, match="Scorer with name 'safety_scorer' and version 999 not found"
    ):
        store.delete_scorer(experiment_id, "safety_scorer", version=999)

    # Step 11: Test delete_scorer for remaining scorers
    store.delete_scorer(experiment_id, "safety_scorer")
    store.delete_scorer(experiment_id, "relevance_scorer")

    # Verify all scorers are deleted
    final_scorers = store.list_scorers(experiment_id)
    assert len(final_scorers) == 0, (
        f"Expected 0 scorers after all deletions, got {len(final_scorers)}"
    )

    # Step 12: Test list_scorer_versions
    store.register_scorer(experiment_id, "accuracy_scorer", '{"data": "accuracy_scorer1"}')
    store.register_scorer(experiment_id, "accuracy_scorer", '{"data": "accuracy_scorer2"}')
    store.register_scorer(experiment_id, "accuracy_scorer", '{"data": "accuracy_scorer3"}')

    # Test list_scorer_versions for non-existent scorer
    with pytest.raises(MlflowException, match="Scorer with name 'non_existent_scorer' not found"):
        store.list_scorer_versions(experiment_id, "non_existent_scorer")


@pytest.mark.parametrize(
    ("name", "error_match"),
    [
        (None, "cannot be None"),
        (123, "must be a string"),
        ("", "cannot be empty"),
        ("   ", "cannot be empty"),
    ],
)
def test_register_scorer_validates_name(store: SqlAlchemyStore, name, error_match):
    experiment_id = store.create_experiment("test_scorer_name_validation")
    with pytest.raises(MlflowException, match=error_match):
        store.register_scorer(experiment_id, name, '{"data": "test"}')


@pytest.mark.parametrize(
    ("model", "error_match"),
    [
        ("", "cannot be empty"),
        ("   ", "cannot be empty"),
    ],
)
def test_register_scorer_validates_model(store: SqlAlchemyStore, model, error_match):
    experiment_id = store.create_experiment("test_scorer_model_validation")
    scorer_json = json.dumps({"instructions_judge_pydantic_data": {"model": model}})
    with pytest.raises(MlflowException, match=error_match):
        store.register_scorer(experiment_id, "test_scorer", scorer_json)


def _gateway_model_scorer_json():
    return json.dumps({"instructions_judge_pydantic_data": {"model": "gateway:/my-endpoint"}})


def _non_gateway_model_scorer_json():
    return json.dumps({"instructions_judge_pydantic_data": {"model": "openai:/gpt-4"}})


def _mock_gateway_endpoint():
    return GatewayEndpoint(
        endpoint_id="test-endpoint-id",
        name="my-endpoint",
        created_at=0,
        last_updated_at=0,
    )


def test_get_online_scoring_configs_batch(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_batch_configs")
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "scorer1", _gateway_model_scorer_json())
        store.register_scorer(experiment_id, "scorer2", _gateway_model_scorer_json())
        store.register_scorer(experiment_id, "scorer3", _gateway_model_scorer_json())

    config1 = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer1",
        sample_rate=0.1,
        filter_string="status = 'OK'",
    )
    config2 = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer2",
        sample_rate=0.5,
    )

    scorer_ids = [config1.scorer_id, config2.scorer_id]
    configs = store.get_online_scoring_configs(scorer_ids)

    assert len(configs) == 2
    configs_by_id = {c.scorer_id: c for c in configs}
    assert configs_by_id[config1.scorer_id].sample_rate == 0.1
    assert configs_by_id[config1.scorer_id].filter_string == "status = 'OK'"
    assert configs_by_id[config2.scorer_id].sample_rate == 0.5
    assert configs_by_id[config2.scorer_id].filter_string is None


def test_get_online_scoring_configs_empty_list(store: SqlAlchemyStore):
    configs = store.get_online_scoring_configs([])
    assert configs == []


def test_get_online_scoring_configs_nonexistent_ids(store: SqlAlchemyStore):
    configs = store.get_online_scoring_configs(["nonexistent_id_1", "nonexistent_id_2"])
    assert configs == []


def test_upsert_online_scoring_config_creates_config(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_online_config_create")
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "scorer", _gateway_model_scorer_json())

    config = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer",
        sample_rate=0.1,
        filter_string="status = 'OK'",
    )

    assert config.sample_rate == 0.1
    assert config.filter_string == "status = 'OK'"
    assert config.online_scoring_config_id is not None


def test_upsert_online_scoring_config_overwrites(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_online_config_overwrite")
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "scorer", _gateway_model_scorer_json())

    store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer",
        sample_rate=0.1,
    )

    new_config = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer",
        sample_rate=0.5,
    )

    assert new_config.sample_rate == 0.5

    # Verify the config is persisted by fetching via get_online_scoring_configs
    configs = store.get_online_scoring_configs([new_config.scorer_id])
    assert len(configs) == 1
    assert configs[0].scorer_id == new_config.scorer_id
    assert configs[0].sample_rate == 0.5


def test_upsert_online_scoring_config_rejects_non_gateway_model(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_online_config_non_gateway")
    non_gateway_scorer = json.dumps({
        "instructions_judge_pydantic_data": {"model": "openai:/gpt-4"}
    })
    store.register_scorer(experiment_id, "scorer", non_gateway_scorer)

    with pytest.raises(MlflowException, match="does not use a gateway model"):
        store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name="scorer",
            sample_rate=0.1,
        )


def test_upsert_online_scoring_config_rejects_scorer_requiring_expectations(
    store: SqlAlchemyStore,
):
    experiment_id = store.create_experiment("test_online_config_expectations")

    # Complete serialized scorer with {{ expectations }} template variable
    expectations_scorer = json.dumps({
        "name": "expectations_scorer",
        "description": None,
        "aggregations": [],
        "is_session_level_scorer": False,
        "mlflow_version": "3.0.0",
        "serialization_version": 1,
        "instructions_judge_pydantic_data": {
            "model": "gateway:/my-endpoint",
            "instructions": "Compare {{ outputs }} against {{ expectations }}",
        },
        "builtin_scorer_class": None,
        "builtin_scorer_pydantic_data": None,
        "call_source": None,
        "call_signature": None,
        "original_func_name": None,
    })

    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "expectations_scorer", expectations_scorer)

    # Mock LiteLLM availability to allow scorer deserialization during validation
    with mock.patch("mlflow.genai.judges.utils._is_litellm_available", return_value=True):
        with pytest.raises(MlflowException, match="requires expectations.*not currently supported"):
            store.upsert_online_scoring_config(
                experiment_id=experiment_id,
                scorer_name="expectations_scorer",
                sample_rate=0.1,
            )

        # Setting sample_rate to 0 should work (disables automatic evaluation)
        config = store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name="expectations_scorer",
            sample_rate=0.0,
        )
        assert config.sample_rate == 0.0


def test_upsert_online_scoring_config_nonexistent_scorer(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_online_config_error")

    with pytest.raises(MlflowException, match="not found"):
        store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name="nonexistent",
            sample_rate=0.1,
        )


def test_upsert_online_scoring_config_validates_filter_string(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_filter_validation")
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "test_scorer", _gateway_model_scorer_json())

    config = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="test_scorer",
        sample_rate=0.5,
        filter_string="status = 'OK'",
    )
    assert config.filter_string == "status = 'OK'"

    with pytest.raises(MlflowException, match="Invalid clause"):
        store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name="test_scorer",
            sample_rate=0.5,
            filter_string="this is not a valid filter !!@@##",
        )


def test_upsert_online_scoring_config_validates_sample_rate(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_sample_rate_validation")
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "test_scorer", _gateway_model_scorer_json())

    # Valid sample rates should work
    config = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="test_scorer",
        sample_rate=0.0,
    )
    assert config.sample_rate == 0.0

    config = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="test_scorer",
        sample_rate=1.0,
    )
    assert config.sample_rate == 1.0

    # Invalid sample rates should raise
    with pytest.raises(MlflowException, match="sample_rate must be between 0.0 and 1.0"):
        store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name="test_scorer",
            sample_rate=-0.1,
        )

    # Non-numeric sample_rate should raise
    with pytest.raises(MlflowException, match="sample_rate must be a number"):
        store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name="test_scorer",
            sample_rate="0.5",
        )

    # Non-string filter_string should raise
    with pytest.raises(MlflowException, match="filter_string must be a string"):
        store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name="test_scorer",
            sample_rate=0.5,
            filter_string=123,
        )


def test_get_active_online_scorers_filters_by_sample_rate(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_active_configs")
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "active", _gateway_model_scorer_json())
        store.register_scorer(experiment_id, "inactive", _gateway_model_scorer_json())

    store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="active",
        sample_rate=0.1,
    )
    store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="inactive",
        sample_rate=0.0,
    )

    active_scorers = store.get_active_online_scorers()
    # Filter to only scorers we created in this test using name and experiment_id
    test_scorers = [
        s
        for s in active_scorers
        if s.name == "active" and s.online_config.experiment_id == experiment_id
    ]

    assert len(test_scorers) == 1
    assert test_scorers[0].online_config.sample_rate == 0.1


def test_get_active_online_scorers_returns_scorer_fields(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_active_configs_info")
    scorer_json = _gateway_model_scorer_json()
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "scorer", scorer_json)

    store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer",
        sample_rate=0.5,
        filter_string="status = 'OK'",
    )

    active_scorers = store.get_active_online_scorers()
    active_scorer = next(
        s
        for s in active_scorers
        if s.name == "scorer" and s.online_config.experiment_id == experiment_id
    )

    assert active_scorer.name == "scorer"
    assert active_scorer.online_config.experiment_id == experiment_id
    assert active_scorer.online_config.sample_rate == 0.5
    assert active_scorer.online_config.filter_string == "status = 'OK'"


def test_get_active_online_scorers_filters_non_gateway_model(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_filter_non_gateway")

    # Register scorer with gateway model (version 1)
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "scorer", _gateway_model_scorer_json())

    # Set up online scoring config (validation passes for version 1)
    store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer",
        sample_rate=0.5,
    )

    # Verify scorer is returned initially (max version uses gateway model)
    active_scorers = store.get_active_online_scorers()
    test_scorers = [
        s
        for s in active_scorers
        if s.name == "scorer" and s.online_config.experiment_id == experiment_id
    ]
    assert len(test_scorers) == 1

    # Register same scorer with non-gateway model (version 2)
    store.register_scorer(experiment_id, "scorer", _non_gateway_model_scorer_json())

    # Verify scorer is NOT returned now (max version uses non-gateway model)
    active_scorers = store.get_active_online_scorers()
    test_scorers = [
        s
        for s in active_scorers
        if s.name == "scorer" and s.online_config.experiment_id == experiment_id
    ]
    assert len(test_scorers) == 0


def test_scorer_deletion_cascades_to_online_configs(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_cascade_delete")
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "scorer", _gateway_model_scorer_json())

    config = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer",
        sample_rate=0.5,
    )
    config_id = config.online_scoring_config_id

    with store.ManagedSessionMaker() as session:
        assert (
            session
            .query(SqlOnlineScoringConfig)
            .filter_by(online_scoring_config_id=config_id)
            .count()
            == 1
        )

    store.delete_scorer(experiment_id, "scorer")

    with store.ManagedSessionMaker() as session:
        assert (
            session
            .query(SqlOnlineScoringConfig)
            .filter_by(online_scoring_config_id=config_id)
            .count()
            == 0
        )


def test_dataset_experiment_associations(store):
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store):
        exp_ids = _create_experiments(
            store, ["exp_assoc_1", "exp_assoc_2", "exp_assoc_3", "exp_assoc_4"]
        )
        exp1, exp2, exp3, exp4 = exp_ids

        dataset = store.create_dataset(
            name="test_dataset_associations", experiment_ids=[exp1], tags={"test": "associations"}
        )

        assert dataset.experiment_ids == [exp1]

        updated = store.add_dataset_to_experiments(
            dataset_id=dataset.dataset_id, experiment_ids=[exp2, exp3]
        )
        assert set(updated.experiment_ids) == {exp1, exp2, exp3}

        result = store.add_dataset_to_experiments(
            dataset_id=dataset.dataset_id, experiment_ids=[exp2, exp4]
        )
        assert set(result.experiment_ids) == {exp1, exp2, exp3, exp4}

        removed = store.remove_dataset_from_experiments(
            dataset_id=dataset.dataset_id, experiment_ids=[exp2, exp3]
        )
        assert set(removed.experiment_ids) == {exp1, exp4}

        with mock.patch("mlflow.store.tracking.sqlalchemy_store._logger.warning") as mock_warning:
            idempotent = store.remove_dataset_from_experiments(
                dataset_id=dataset.dataset_id, experiment_ids=[exp2, exp3]
            )
            assert mock_warning.call_count == 2
            assert "was not associated" in mock_warning.call_args_list[0][0][0]

        assert set(idempotent.experiment_ids) == {exp1, exp4}

        with pytest.raises(MlflowException, match="not found"):
            store.add_dataset_to_experiments(dataset_id="d-nonexistent", experiment_ids=[exp1])

        with pytest.raises(MlflowException, match=r"No Experiment with id="):
            store.add_dataset_to_experiments(
                dataset_id=dataset.dataset_id, experiment_ids=["999999"]
            )

        with pytest.raises(MlflowException, match="not found"):
            store.remove_dataset_from_experiments(dataset_id="d-nonexistent", experiment_ids=[exp1])


def _create_simple_trace(store, experiment_id, tags=None):
    trace_id = f"tr-{uuid.uuid4()}"
    timestamp_ms = time.time_ns() // 1_000_000

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=timestamp_ms,
        execution_duration=100,
        state=TraceState.OK,
        tags=tags or {},
    )

    return store.start_trace(trace_info)


def _create_trace_for_correlation(store, experiment_id, spans=None, assessments=None, tags=None):
    trace_id = f"tr-{uuid.uuid4()}"
    timestamp_ms = time.time_ns() // 1_000_000

    trace_tags = tags or {}

    if spans:
        span_types = [span.get("type", "LLM") for span in spans]
        span_statuses = [span.get("status", "OK") for span in spans]

        if "TOOL" in span_types:
            trace_tags["primary_span_type"] = "TOOL"
        elif "LLM" in span_types:
            trace_tags["primary_span_type"] = "LLM"

        if "LLM" in span_types:
            trace_tags["has_llm"] = "true"
        if "TOOL" in span_types:
            trace_tags["has_tool"] = "true"

        trace_tags["has_error"] = "true" if "ERROR" in span_statuses else "false"

        tool_count = sum(1 for t in span_types if t == "TOOL")
        if tool_count > 0:
            trace_tags["tool_count"] = str(tool_count)

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=timestamp_ms,
        execution_duration=100,
        state=TraceState.OK,
        tags=trace_tags,
    )
    store.start_trace(trace_info)

    if assessments:
        for assessment_data in assessments:
            assessment = Feedback(
                assessment_id=assessment_data.get("assessment_id", f"fb-{uuid.uuid4()}"),
                trace_id=trace_id,
                name=assessment_data.get("name", "quality"),
                assessment_type=assessment_data.get("assessment_type", "feedback"),
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN,
                    source_id=assessment_data.get("source_id", "user123"),
                ),
                value=FeedbackValue(assessment_data.get("value", 0.8)),
                created_timestamp=timestamp_ms,
                last_updated_timestamp=timestamp_ms,
            )
            store.log_assessments([assessment])

    return trace_id


def _create_trace_with_spans_for_correlation(store, experiment_id, span_configs):
    return _create_trace_for_correlation(store, experiment_id, spans=span_configs)


def test_calculate_trace_filter_correlation_basic(store):
    exp_id = _create_experiments(store, "correlation_test")

    for i in range(10):
        _create_trace_with_spans_for_correlation(
            store,
            exp_id,
            span_configs=[{"name": "tool_operation", "type": "TOOL", "status": "ERROR"}],
        )

    for i in range(5):
        _create_trace_with_spans_for_correlation(
            store,
            exp_id,
            span_configs=[{"name": "llm_call", "type": "LLM", "status": "OK"}],
        )

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.primary_span_type = "TOOL"',
        filter_string2='tags.has_error = "true"',
    )

    assert result.npmi == pytest.approx(1.0)
    assert result.filter1_count == 10
    assert result.filter2_count == 10
    assert result.joint_count == 10
    assert result.total_count == 15


def test_calculate_trace_filter_correlation_perfect(store):
    exp_id = _create_experiments(store, "correlation_test")

    for i in range(8):
        _create_trace_with_spans_for_correlation(
            store,
            exp_id,
            span_configs=[{"name": "operation", "type": "TOOL", "status": "ERROR"}],
        )

    for i in range(7):
        _create_trace_with_spans_for_correlation(
            store,
            exp_id,
            span_configs=[{"name": "operation", "type": "LLM", "status": "OK"}],
        )

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.primary_span_type = "TOOL"',
        filter_string2='tags.has_error = "true"',
    )

    assert result.npmi == pytest.approx(1.0)
    assert result.npmi_smoothed > 0.8
    assert result.filter1_count == 8
    assert result.filter2_count == 8
    assert result.joint_count == 8
    assert result.total_count == 15


def test_calculate_trace_filter_correlation_count_expressions(store):
    exp_id = _create_experiments(store, "correlation_test")

    for i in range(15):
        num_tool_calls = 5 if i < 10 else 2
        spans = [{"type": "TOOL", "name": f"tool_{j}"} for j in range(num_tool_calls)]
        spans.append({"type": "LLM", "name": "llm_call"})
        _create_trace_with_spans_for_correlation(store, exp_id, span_configs=spans)

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.tool_count = "5"',
        filter_string2='tags.has_llm = "true"',
    )

    assert result.filter1_count == 10
    assert result.filter2_count == 15
    assert result.joint_count == 10
    assert result.total_count == 15


def test_calculate_trace_filter_correlation_negative_correlation(store):
    exp_id = _create_experiments(store, "negative_correlation_test")

    for i in range(10):
        _create_trace_for_correlation(
            store, exp_id, spans=[{"type": "LLM", "status": "ERROR"}], tags={"version": "v1"}
        )

    for i in range(10):
        _create_trace_for_correlation(
            store, exp_id, spans=[{"type": "LLM", "status": "OK"}], tags={"version": "v2"}
        )

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.version = "v1"',
        filter_string2='tags.has_error = "false"',
    )

    assert result.total_count == 20
    assert result.filter1_count == 10
    assert result.filter2_count == 10
    assert result.joint_count == 0
    assert result.npmi == pytest.approx(-1.0)


def test_calculate_trace_filter_correlation_zero_counts(store):
    exp_id = _create_experiments(store, "zero_counts_test")

    for i in range(5):
        _create_trace_for_correlation(store, exp_id, spans=[{"type": "LLM", "status": "OK"}])

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.has_error = "true"',
        filter_string2='tags.has_llm = "true"',
    )

    assert result.total_count == 5
    assert result.filter1_count == 0
    assert result.filter2_count == 5
    assert result.joint_count == 0
    assert math.isnan(result.npmi)


def test_calculate_trace_filter_correlation_multiple_experiments(store):
    exp_id1 = _create_experiments(store, "multi_exp_1")
    exp_id2 = _create_experiments(store, "multi_exp_2")

    for i in range(4):
        _create_trace_for_correlation(
            store, exp_id1, spans=[{"type": "TOOL", "status": "OK"}], tags={"env": "prod"}
        )

    _create_trace_for_correlation(
        store, exp_id1, spans=[{"type": "LLM", "status": "OK"}], tags={"env": "prod"}
    )

    _create_trace_for_correlation(
        store, exp_id2, spans=[{"type": "TOOL", "status": "OK"}], tags={"env": "dev"}
    )

    for i in range(4):
        _create_trace_for_correlation(
            store, exp_id2, spans=[{"type": "LLM", "status": "OK"}], tags={"env": "dev"}
        )

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id1, exp_id2],
        filter_string1='tags.env = "prod"',
        filter_string2='tags.primary_span_type = "TOOL"',
    )

    assert result.total_count == 10
    assert result.filter1_count == 5
    assert result.filter2_count == 5
    assert result.joint_count == 4
    assert result.npmi > 0.4


def test_calculate_trace_filter_correlation_independent_events(store):
    exp_id = _create_experiments(store, "independent_test")

    configurations = [
        *[{"spans": [{"type": "TOOL", "status": "ERROR"}]} for _ in range(5)],
        *[{"spans": [{"type": "TOOL", "status": "OK"}]} for _ in range(5)],
        *[{"spans": [{"type": "LLM", "status": "ERROR"}]} for _ in range(5)],
        *[{"spans": [{"type": "LLM", "status": "OK"}]} for _ in range(5)],
    ]

    for config in configurations:
        _create_trace_for_correlation(store, exp_id, **config)

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.primary_span_type = "TOOL"',
        filter_string2='tags.has_error = "true"',
    )

    assert result.total_count == 20
    assert result.filter1_count == 10
    assert result.filter2_count == 10
    assert result.joint_count == 5

    # Independent events should have NPMI close to 0
    # P(TOOL) = 10/20 = 0.5, P(ERROR) = 10/20 = 0.5
    # P(TOOL & ERROR) = 5/20 = 0.25
    # Expected joint = 0.5 * 0.5 * 20 = 5, so no correlation
    assert abs(result.npmi) < 0.1


def test_calculate_trace_filter_correlation_simplified_example(store):
    exp_id = _create_experiments(store, "simple_correlation_test")

    for _ in range(5):
        _create_simple_trace(store, exp_id, {"category": "A", "status": "success"})

    for _ in range(3):
        _create_simple_trace(store, exp_id, {"category": "A", "status": "failure"})

    for _ in range(7):
        _create_simple_trace(store, exp_id, {"category": "B", "status": "success"})

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.category = "A"',
        filter_string2='tags.status = "success"',
    )

    assert result.filter1_count == 8
    assert result.filter2_count == 12
    assert result.joint_count == 5
    assert result.total_count == 15


def test_calculate_trace_filter_correlation_empty_experiment_list(store):
    result = store.calculate_trace_filter_correlation(
        experiment_ids=[],
        filter_string1='tags.has_error = "true"',
        filter_string2='tags.primary_span_type = "TOOL"',
    )

    assert result.total_count == 0
    assert result.filter1_count == 0
    assert result.filter2_count == 0
    assert result.joint_count == 0
    assert math.isnan(result.npmi)


def test_calculate_trace_filter_correlation_with_base_filter(store):
    exp_id = _create_experiments(store, "base_filter_test")

    early_time = 1000000000000
    for i in range(5):
        trace_info = TraceInfo(
            trace_id=f"tr-early-{i}",
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=early_time + i,
            execution_duration=100,
            state=TraceState.OK,
            tags={
                "has_error": "true" if i < 3 else "false",
                "has_tool": "true" if i % 2 == 0 else "false",
            },
        )
        store.start_trace(trace_info)

    later_time = 2000000000000
    # Create traces in the later period:
    # - 10 total traces in the time window
    # - 6 with has_error=true
    # - 4 with has_tool=true
    # - 3 with both has_error=true AND has_tool=true
    for i in range(10):
        tags = {}
        if i < 6:
            tags["has_error"] = "true"
        if i < 3 or i == 6:
            tags["has_tool"] = "true"

        trace_info = TraceInfo(
            trace_id=f"tr-later-{i}",
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=later_time + i,
            execution_duration=100,
            state=TraceState.OK,
            tags=tags,
        )
        store.start_trace(trace_info)

    base_filter = f"timestamp_ms >= {later_time} and timestamp_ms < {later_time + 100}"
    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.has_error = "true"',
        filter_string2='tags.has_tool = "true"',
        base_filter=base_filter,
    )

    assert result.total_count == 10
    assert result.filter1_count == 6
    assert result.filter2_count == 4
    assert result.joint_count == 3

    # Calculate expected NPMI
    # P(error) = 6/10 = 0.6
    # P(tool) = 4/10 = 0.4
    # P(error AND tool) = 3/10 = 0.3
    # PMI = log(P(error AND tool) / (P(error) * P(tool))) = log(0.3 / (0.6 * 0.4)) = log(1.25)
    # NPMI = PMI / -log(P(error AND tool)) = log(1.25) / -log(0.3)

    p_error = 6 / 10
    p_tool = 4 / 10
    p_joint = 3 / 10

    if p_joint > 0:
        pmi = math.log(p_joint / (p_error * p_tool))
        npmi = pmi / -math.log(p_joint)
        assert abs(result.npmi - npmi) < 0.001

    result_no_base = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.has_error = "true"',
        filter_string2='tags.has_tool = "true"',
    )

    assert result_no_base.total_count == 15
    assert result_no_base.filter1_count == 9
    assert result_no_base.filter2_count == 7
    assert result_no_base.joint_count == 5


def test_get_decrypted_secret_integration_simple(store):
    secret_info = store.create_gateway_secret(
        secret_name="test-simple-secret",
        secret_value={"api_key": "sk-test-123456"},
        provider="openai",
    )

    decrypted = store._get_decrypted_secret(secret_info.secret_id)

    assert decrypted == {"api_key": "sk-test-123456"}


def test_get_decrypted_secret_integration_compound(store):
    secret_info = store.create_gateway_secret(
        secret_name="test-compound-secret",
        secret_value={
            "aws_access_key_id": "AKIA1234567890",
            "aws_secret_access_key": "secret-key-value",
        },
        provider="bedrock",
    )

    decrypted = store._get_decrypted_secret(secret_info.secret_id)

    assert decrypted == {
        "aws_access_key_id": "AKIA1234567890",
        "aws_secret_access_key": "secret-key-value",
    }


def test_get_decrypted_secret_integration_with_auth_config(store):
    secret_info = store.create_gateway_secret(
        secret_name="test-auth-config-secret",
        secret_value={"api_key": "aws-secret"},
        provider="bedrock",
        auth_config={"region": "us-east-1", "profile": "default"},
    )

    decrypted = store._get_decrypted_secret(secret_info.secret_id)

    assert decrypted == {"api_key": "aws-secret"}


def test_get_decrypted_secret_integration_not_found(store):
    with pytest.raises(MlflowException, match="not found"):
        store._get_decrypted_secret("nonexistent-secret-id")


def test_get_decrypted_secret_integration_multiple_secrets(store):
    secret1 = store.create_gateway_secret(
        secret_name="secret-1",
        secret_value={"api_key": "key-1"},
        provider="openai",
    )
    secret2 = store.create_gateway_secret(
        secret_name="secret-2",
        secret_value={"api_key": "key-2"},
        provider="anthropic",
    )

    decrypted1 = store._get_decrypted_secret(secret1.secret_id)
    decrypted2 = store._get_decrypted_secret(secret2.secret_id)

    assert decrypted1 == {"api_key": "key-1"}
    assert decrypted2 == {"api_key": "key-2"}
