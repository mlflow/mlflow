import json
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
    ViewType,
    trace_location,
)
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
