import json
import math
import os
import random
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pytest
import sqlalchemy
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource as _OTelResource
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from sqlalchemy.exc import IntegrityError

import mlflow
import mlflow.db
from mlflow import entities
from mlflow.entities import (
    AssessmentSource,
    AssessmentSourceType,
    Expectation,
    Feedback,
    Link,
    Metric,
    ViewType,
    trace_location,
)
from mlflow.entities.assessment import ExpectationValue, FeedbackValue
from mlflow.entities.dataset_record import DatasetRecord
from mlflow.entities.gateway_endpoint import GatewayEndpoint
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.span import Span, create_mlflow_span
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_state import TraceState
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import (
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_TRACKING_URI,
)
from mlflow.exceptions import MlflowException, MlflowTracingException
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
    SqlSpan,
    SqlSpanMetrics,
    SqlTag,
    SqlTraceInfo,
    SqlTraceMetadata,
    SqlTraceMetrics,
    SqlTraceTag,
)
from mlflow.store.tracking.sqlalchemy_store import (
    SqlAlchemyStore,
    _get_orderby_clauses,
)
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.tracing.constant import (
    MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE,
    CostKey,
    SpanAttributeKey,
    SpansLocation,
    TraceMetadataKey,
    TraceSizeStatsKey,
    TraceTagKey,
)
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils import mlflow_tags
from mlflow.utils.mlflow_tags import (
    MLFLOW_ARTIFACT_LOCATION,
)
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


def test_legacy_start_and_end_trace_v2(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_experiment")
    trace_info = store.deprecated_start_trace_v2(
        experiment_id=experiment_id,
        timestamp_ms=1234,
        request_metadata={"rq1": "foo", "rq2": "bar"},
        tags={"tag1": "apple", "tag2": "orange"},
    )
    request_id = trace_info.request_id

    assert trace_info.request_id is not None
    assert trace_info.experiment_id == experiment_id
    assert trace_info.timestamp_ms == 1234
    assert trace_info.execution_time_ms is None
    assert trace_info.status == TraceStatus.IN_PROGRESS
    assert trace_info.request_metadata == {
        "rq1": "foo",
        "rq2": "bar",
    }
    artifact_location = trace_info.tags[MLFLOW_ARTIFACT_LOCATION]
    assert artifact_location.endswith(f"/{experiment_id}/traces/{request_id}/artifacts")
    assert trace_info.tags == {
        "tag1": "apple",
        "tag2": "orange",
        MLFLOW_ARTIFACT_LOCATION: artifact_location,
    }
    assert trace_info.to_v3() == store.get_trace_info(request_id)

    trace_info = store.deprecated_end_trace_v2(
        request_id=request_id,
        timestamp_ms=2345,
        status=TraceStatus.OK,
        # Update one key and add a new key
        request_metadata={
            "rq1": "updated",
            "rq3": "baz",
        },
        tags={"tag1": "updated", "tag3": "grape"},
    )
    assert trace_info.request_id == request_id
    assert trace_info.experiment_id == experiment_id
    assert trace_info.timestamp_ms == 1234
    assert trace_info.execution_time_ms == 2345 - 1234
    assert trace_info.status == TraceStatus.OK
    assert trace_info.request_metadata == {
        "rq1": "updated",
        "rq2": "bar",
        "rq3": "baz",
    }
    assert trace_info.tags == {
        "tag1": "updated",
        "tag2": "orange",
        "tag3": "grape",
        MLFLOW_ARTIFACT_LOCATION: artifact_location,
    }
    assert trace_info.to_v3() == store.get_trace_info(request_id)


def test_start_trace(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_experiment")
    trace_info = TraceInfo(
        trace_id="tr-123",
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1234,
        execution_duration=100,
        state=TraceState.OK,
        tags={"tag1": "apple", "tag2": "orange"},
        trace_metadata={"rq1": "foo", "rq2": "bar"},
    )
    trace_info = store.start_trace(trace_info)
    trace_id = trace_info.trace_id

    assert trace_info.trace_id is not None
    assert trace_info.experiment_id == experiment_id
    assert trace_info.request_time == 1234
    assert trace_info.execution_duration == 100
    assert trace_info.state == TraceState.OK
    assert {"rq1": "foo", "rq2": "bar"}.items() <= trace_info.trace_metadata.items()
    assert trace_info.trace_metadata.get(TraceMetadataKey.TRACE_INFO_FINALIZED) == "true"
    artifact_location = trace_info.tags[MLFLOW_ARTIFACT_LOCATION]
    assert artifact_location.endswith(f"/{experiment_id}/traces/{trace_id}/artifacts")
    assert trace_info.tags == {
        "tag1": "apple",
        "tag2": "orange",
        MLFLOW_ARTIFACT_LOCATION: artifact_location,
    }
    assert trace_info == store.get_trace_info(trace_id)


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
    """Helper function to create a test trace in the database."""
    if not store.get_experiment(experiment_id):
        store.create_experiment(store, experiment_id)

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


@pytest.fixture
def store_with_traces(store):
    exp1 = store.create_experiment("exp1")
    exp2 = store.create_experiment("exp2")

    _create_trace(
        store,
        "tr-0",
        exp2,
        request_time=0,
        execution_duration=6,
        state=TraceState.OK,
        tags={"mlflow.traceName": "ddd"},
        trace_metadata={TraceMetadataKey.SOURCE_RUN: "run0"},
    )
    _create_trace(
        store,
        "tr-1",
        exp2,
        request_time=1,
        execution_duration=2,
        state=TraceState.ERROR,
        tags={"mlflow.traceName": "aaa", "fruit": "apple", "color": "red"},
        trace_metadata={TraceMetadataKey.SOURCE_RUN: "run1"},
    )
    _create_trace(
        store,
        "tr-2",
        exp1,
        request_time=2,
        execution_duration=4,
        state=TraceState.STATE_UNSPECIFIED,
        tags={"mlflow.traceName": "bbb", "fruit": "apple", "color": "green"},
    )
    _create_trace(
        store,
        "tr-3",
        exp1,
        request_time=3,
        execution_duration=10,
        state=TraceState.OK,
        tags={"mlflow.traceName": "ccc", "fruit": "orange"},
    )
    _create_trace(
        store,
        "tr-4",
        exp1,
        request_time=4,
        execution_duration=10,
        state=TraceState.OK,
        tags={"mlflow.traceName": "ddd", "color": "blue"},
    )

    return store


@pytest.mark.parametrize(
    ("order_by", "expected_ids"),
    [
        # Default order: descending by start time
        ([], ["tr-4", "tr-3", "tr-2", "tr-1", "tr-0"]),
        # Order by start time
        (["timestamp"], ["tr-0", "tr-1", "tr-2", "tr-3", "tr-4"]),
        (["timestamp DESC"], ["tr-4", "tr-3", "tr-2", "tr-1", "tr-0"]),
        # Order by execution_time and timestamp
        (
            ["execution_time DESC", "timestamp ASC"],
            ["tr-3", "tr-4", "tr-0", "tr-2", "tr-1"],
        ),
        # Order by experiment ID
        (["experiment_id"], ["tr-4", "tr-3", "tr-2", "tr-1", "tr-0"]),
        # Order by status
        (["status"], ["tr-1", "tr-4", "tr-3", "tr-0", "tr-2"]),
        # Order by name
        (["name"], ["tr-1", "tr-2", "tr-3", "tr-4", "tr-0"]),
        # Order by tag (null comes last)
        (["tag.fruit"], ["tr-2", "tr-1", "tr-3", "tr-4", "tr-0"]),
        # Order by multiple tags
        (["tag.fruit", "tag.color"], ["tr-2", "tr-1", "tr-3", "tr-4", "tr-0"]),
        # Order by non-existent tag (should be ordered by default order)
        (["tag.nonexistent"], ["tr-4", "tr-3", "tr-2", "tr-1", "tr-0"]),
        # Order by run Id
        (["run_id"], ["tr-0", "tr-1", "tr-4", "tr-3", "tr-2"]),
    ],
)
def test_search_traces_order_by(store_with_traces, order_by, expected_ids):
    exp1 = store_with_traces.get_experiment_by_name("exp1").experiment_id
    exp2 = store_with_traces.get_experiment_by_name("exp2").experiment_id
    trace_infos, _ = store_with_traces.search_traces(
        locations=[exp1, exp2],
        filter_string=None,
        max_results=5,
        order_by=order_by,
    )
    actual_ids = [trace_info.trace_id for trace_info in trace_infos]
    assert actual_ids == expected_ids


@pytest.mark.parametrize(
    ("filter_string", "expected_ids"),
    [
        # Search by name
        ("name = 'aaa'", ["tr-1"]),
        ("name != 'aaa'", ["tr-4", "tr-3", "tr-2", "tr-0"]),
        # Search by status
        ("status = 'OK'", ["tr-4", "tr-3", "tr-0"]),
        ("status != 'OK'", ["tr-2", "tr-1"]),
        ("attributes.status = 'OK'", ["tr-4", "tr-3", "tr-0"]),
        ("attributes.name != 'aaa'", ["tr-4", "tr-3", "tr-2", "tr-0"]),
        ("trace.status = 'OK'", ["tr-4", "tr-3", "tr-0"]),
        ("trace.name != 'aaa'", ["tr-4", "tr-3", "tr-2", "tr-0"]),
        # Search by timestamp
        ("`timestamp` >= 1 AND execution_time < 10", ["tr-2", "tr-1"]),
        # Search by tag
        ("tag.fruit = 'apple'", ["tr-2", "tr-1"]),
        # tags is an alias for tag
        ("tags.fruit = 'apple' and tags.color != 'red'", ["tr-2"]),
        # Search by request metadata
        ("run_id = 'run0'", ["tr-0"]),
        (f"request_metadata.{TraceMetadataKey.SOURCE_RUN} = 'run0'", ["tr-0"]),
        (f"request_metadata.{TraceMetadataKey.SOURCE_RUN} = 'run1'", ["tr-1"]),
        (f"request_metadata.`{TraceMetadataKey.SOURCE_RUN}` = 'run0'", ["tr-0"]),
        (f"metadata.{TraceMetadataKey.SOURCE_RUN} = 'run0'", ["tr-0"]),
        (f"metadata.{TraceMetadataKey.SOURCE_RUN} != 'run0'", ["tr-1"]),
    ],
)
def test_search_traces_with_filter(store_with_traces, filter_string, expected_ids):
    exp1 = store_with_traces.get_experiment_by_name("exp1").experiment_id
    exp2 = store_with_traces.get_experiment_by_name("exp2").experiment_id

    trace_infos, _ = store_with_traces.search_traces(
        locations=[exp1, exp2],
        filter_string=filter_string,
        max_results=5,
        order_by=[],
    )
    actual_ids = [trace_info.trace_id for trace_info in trace_infos]
    assert actual_ids == expected_ids


@pytest.mark.parametrize(
    ("filter_string", "error"),
    [
        ("invalid", r"Invalid clause\(s\) in filter string"),
        ("name = 'foo' AND invalid", r"Invalid clause\(s\) in filter string"),
        ("foo.bar = 'baz'", r"Invalid entity type 'foo'"),
        ("invalid = 'foo'", r"Invalid attribute key 'invalid'"),
        ("trace.tags.foo = 'bar'", r"Invalid attribute key 'tags\.foo'"),
        ("trace.status < 'OK'", r"Invalid comparator '<'"),
        ("name IN ('foo', 'bar')", r"Invalid comparator 'IN'"),
    ],
)
def test_search_traces_with_invalid_filter(store_with_traces, filter_string, error):
    exp1 = store_with_traces.get_experiment_by_name("exp1").experiment_id
    exp2 = store_with_traces.get_experiment_by_name("exp2").experiment_id

    with pytest.raises(MlflowException, match=error):
        store_with_traces.search_traces(
            locations=[exp1, exp2],
            filter_string=filter_string,
        )


def test_search_traces_raise_if_max_results_arg_is_invalid(store):
    with pytest.raises(
        MlflowException,
        match="Invalid value 50001 for parameter 'max_results' supplied.",
    ):
        store.search_traces(locations=[], max_results=50001)

    with pytest.raises(
        MlflowException, match="Invalid value -1 for parameter 'max_results' supplied."
    ):
        store.search_traces(locations=[], max_results=-1)


def test_search_traces_pagination(store_with_traces):
    exps = [
        store_with_traces.get_experiment_by_name("exp1").experiment_id,
        store_with_traces.get_experiment_by_name("exp2").experiment_id,
    ]

    traces, token = store_with_traces.search_traces(exps, max_results=2)
    assert [t.trace_id for t in traces] == ["tr-4", "tr-3"]

    traces, token = store_with_traces.search_traces(exps, max_results=2, page_token=token)
    assert [t.trace_id for t in traces] == ["tr-2", "tr-1"]

    traces, token = store_with_traces.search_traces(exps, max_results=2, page_token=token)
    assert [t.trace_id for t in traces] == ["tr-0"]
    assert token is None


def test_search_traces_pagination_tie_breaker(store):
    # This test is for ensuring the tie breaker for ordering traces with the same timestamp
    # works correctly.
    exp1 = store.create_experiment("exp1")

    trace_ids = [f"tr-{i}" for i in range(5)]
    random.shuffle(trace_ids)
    # Insert traces with random order
    for rid in trace_ids:
        _create_trace(store, rid, exp1, request_time=0)

    # Insert 5 more traces with newer timestamp
    trace_ids = [f"tr-{i + 5}" for i in range(5)]
    random.shuffle(trace_ids)
    for rid in trace_ids:
        _create_trace(store, rid, exp1, request_time=1)

    traces, token = store.search_traces([exp1], max_results=3)
    assert [t.trace_id for t in traces] == ["tr-5", "tr-6", "tr-7"]
    traces, token = store.search_traces([exp1], max_results=3, page_token=token)
    assert [t.trace_id for t in traces] == ["tr-8", "tr-9", "tr-0"]
    traces, token = store.search_traces([exp1], max_results=3, page_token=token)
    assert [t.trace_id for t in traces] == ["tr-1", "tr-2", "tr-3"]
    traces, token = store.search_traces([exp1], max_results=3, page_token=token)
    assert [t.trace_id for t in traces] == ["tr-4"]


def test_search_traces_with_run_id_filter(store: SqlAlchemyStore):
    # Create experiment and run
    exp_id = store.create_experiment("test_run_filter")
    run = store.create_run(exp_id, user_id="user", start_time=0, tags=[], run_name="test_run")
    run_id = run.info.run_id

    # Create traces with different relationships to the run
    # Trace 1: Has run_id in metadata (direct association)
    trace1_id = "tr-direct"
    _create_trace(store, trace1_id, exp_id, trace_metadata={"mlflow.sourceRun": run_id})

    # Trace 2: Linked via entity association
    trace2_id = "tr-linked"
    _create_trace(store, trace2_id, exp_id)
    store.link_traces_to_run([trace2_id], run_id)

    # Trace 3: Both metadata and entity association
    trace3_id = "tr-both"
    _create_trace(store, trace3_id, exp_id, trace_metadata={"mlflow.sourceRun": run_id})
    store.link_traces_to_run([trace3_id], run_id)

    # Trace 4: No association with the run
    trace4_id = "tr-unrelated"
    _create_trace(store, trace4_id, exp_id)

    # Search for traces with run_id filter
    traces, _ = store.search_traces([exp_id], filter_string=f'attributes.run_id = "{run_id}"')
    trace_ids = {t.trace_id for t in traces}

    # Should return traces 1, 2, and 3 but not 4
    assert trace_ids == {trace1_id, trace2_id, trace3_id}

    # Test with another run to ensure isolation
    run2 = store.create_run(exp_id, user_id="user", start_time=0, tags=[], run_name="test_run2")
    run2_id = run2.info.run_id

    # Create a trace linked to run2
    trace5_id = "tr-run2"
    _create_trace(store, trace5_id, exp_id)
    store.link_traces_to_run([trace5_id], run2_id)

    # Search for traces with run2_id filter
    traces, _ = store.search_traces([exp_id], filter_string=f'attributes.run_id = "{run2_id}"')
    trace_ids = {t.trace_id for t in traces}

    # Should only return trace5
    assert trace_ids == {trace5_id}

    # Original run_id search should still return the same traces
    traces, _ = store.search_traces([exp_id], filter_string=f'attributes.run_id = "{run_id}"')
    trace_ids = {t.trace_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}


def test_search_traces_with_run_id_and_other_filters(store: SqlAlchemyStore):
    # Create experiment and run
    exp_id = store.create_experiment("test_combined_filters")
    run = store.create_run(exp_id, user_id="user", start_time=0, tags=[], run_name="test_run")
    run_id = run.info.run_id

    # Create traces with different tags and run associations
    trace1_id = "tr-tag1-linked"
    _create_trace(store, trace1_id, exp_id, tags={"type": "training"})
    store.link_traces_to_run([trace1_id], run_id)

    trace2_id = "tr-tag2-linked"
    _create_trace(store, trace2_id, exp_id, tags={"type": "inference"})
    store.link_traces_to_run([trace2_id], run_id)

    trace3_id = "tr-tag1-notlinked"
    _create_trace(store, trace3_id, exp_id, tags={"type": "training"})

    # Search with run_id and tag filter
    traces, _ = store.search_traces(
        [exp_id], filter_string=f'run_id = "{run_id}" AND tags.type = "training"'
    )
    trace_ids = {t.trace_id for t in traces}

    # Should only return trace1 (linked to run AND has training tag)
    assert trace_ids == {trace1_id}

    # Search with run_id only
    traces, _ = store.search_traces([exp_id], filter_string=f'run_id = "{run_id}"')
    trace_ids = {t.trace_id for t in traces}

    # Should return both linked traces
    assert trace_ids == {trace1_id, trace2_id}


def test_search_traces_with_span_name_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_search")

    # Create traces with spans that have different names
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create spans with different names
    span1 = create_test_span(trace1_id, name="database_query", span_id=111, span_type="FUNCTION")
    span2 = create_test_span(trace2_id, name="api_call", span_id=222, span_type="FUNCTION")
    span3 = create_test_span(trace3_id, name="database_update", span_id=333, span_type="FUNCTION")

    # Add spans to store
    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])

    # Test exact match
    traces, _ = store.search_traces([exp_id], filter_string='span.name = "database_query"')
    assert len(traces) == 1
    assert traces[0].trace_id == trace1_id

    # Test LIKE pattern matching
    traces, _ = store.search_traces([exp_id], filter_string='span.name LIKE "database%"')
    trace_ids = {t.trace_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test match trace2 specifically
    traces, _ = store.search_traces([exp_id], filter_string='span.name = "api_call"')
    assert len(traces) == 1
    assert traces[0].trace_id == trace2_id

    # Test NOT EQUAL
    traces, _ = store.search_traces([exp_id], filter_string='span.name != "api_call"')
    trace_ids = {t.trace_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test no matches
    traces, _ = store.search_traces([exp_id], filter_string='span.name = "nonexistent"')
    assert len(traces) == 0


def test_search_traces_with_full_text_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_plain_text_search")

    # Create traces with spans that have different content
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create spans with different content
    span1 = create_test_span(
        trace1_id,
        name="database_query",
        span_id=111,
        span_type="FUNCTION",
        attributes={"llm.inputs": "what's MLflow?"},
    )
    span2 = create_test_span(
        trace2_id,
        name="api_request",
        span_id=222,
        span_type="TOOL",
        attributes={"response.token.usage": "123"},
    )
    span3 = create_test_span(
        trace3_id,
        name="computation",
        span_id=333,
        span_type="FUNCTION",
        attributes={"llm.outputs": 'MLflow is a tool for " testing " ...'},
    )
    span4 = create_test_span(
        trace3_id,
        name="result",
        span_id=444,
        parent_id=333,
        span_type="WORKFLOW",
        attributes={"test": '"the number increased 90%"'},
    )

    # Add spans to store
    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3, span4])

    # Test full text search using trace.text LIKE
    # match span name
    traces, _ = store.search_traces([exp_id], filter_string='trace.text LIKE "%database_query%"')
    assert len(traces) == 1
    assert traces[0].trace_id == trace1_id

    # match span type
    traces, _ = store.search_traces([exp_id], filter_string='trace.text LIKE "%FUNCTION%"')
    trace_ids = {t.trace_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # match span content / attributes
    traces, _ = store.search_traces([exp_id], filter_string='trace.text LIKE "%what\'s MLflow?%"')
    assert len(traces) == 1
    assert traces[0].trace_id == trace1_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='trace.text LIKE "%MLflow is a tool for%"'
    )
    assert len(traces) == 1
    assert traces[0].trace_id == trace3_id

    traces, _ = store.search_traces([exp_id], filter_string='trace.text LIKE "%llm.%"')
    trace_ids = {t.trace_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    traces, _ = store.search_traces([exp_id], filter_string='trace.text LIKE "%90%%"')
    assert len(traces) == 1
    assert traces[0].trace_id == trace3_id


def test_search_traces_with_invalid_span_attribute(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_error")

    # Test invalid span attribute should raise error
    with pytest.raises(
        MlflowException,
        match=(
            "Invalid span attribute 'duration'. Supported attributes: name, status, "
            "type, attributes.<attribute_name>."
        ),
    ):
        store.search_traces([exp_id], filter_string='span.duration = "1000"')

    with pytest.raises(
        MlflowException,
        match=(
            "Invalid span attribute 'parent_id'. Supported attributes: name, status, "
            "type, attributes.<attribute_name>."
        ),
    ):
        store.search_traces([exp_id], filter_string='span.parent_id = "123"')

    with pytest.raises(
        MlflowException,
        match="span.content comparator '=' not one of ",
    ):
        store.search_traces([exp_id], filter_string='span.content = "test"')


def test_search_traces_with_span_type_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_type_search")

    # Create traces with spans that have different types
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create spans with different types
    span1 = create_test_span(trace1_id, name="llm_call", span_id=111, span_type="LLM")
    span2 = create_test_span(trace2_id, name="retriever_call", span_id=222, span_type="RETRIEVER")
    span3 = create_test_span(trace3_id, name="chain_call", span_id=333, span_type="CHAIN")

    # Add spans to store
    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])

    # Test exact match
    traces, _ = store.search_traces([exp_id], filter_string='span.type = "LLM"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test IN operator
    traces, _ = store.search_traces([exp_id], filter_string='span.type IN ("LLM", "RETRIEVER")')
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace1_id, trace2_id}

    # Test NOT IN operator
    traces, _ = store.search_traces([exp_id], filter_string='span.type NOT IN ("LLM", "RETRIEVER")')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test != operator
    traces, _ = store.search_traces([exp_id], filter_string='span.type != "LLM"')
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace2_id, trace3_id}

    # Test LIKE operator
    traces, _ = store.search_traces([exp_id], filter_string='span.type LIKE "LLM"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test ILIKE operator
    traces, _ = store.search_traces([exp_id], filter_string='span.type ILIKE "llm"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id


def test_search_traces_with_span_status_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_status_search")

    # Create traces with spans that have different statuses
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create spans with different statuses
    span1 = create_test_span(
        trace1_id, name="success_span", span_id=111, status=trace_api.StatusCode.OK
    )
    span2 = create_test_span(
        trace2_id, name="error_span", span_id=222, status=trace_api.StatusCode.ERROR
    )
    span3 = create_test_span(
        trace3_id, name="unset_span", span_id=333, status=trace_api.StatusCode.UNSET
    )

    # Add spans to store
    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])

    # Test exact match with OK status
    traces, _ = store.search_traces([exp_id], filter_string='span.status = "OK"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test exact match with ERROR status
    traces, _ = store.search_traces([exp_id], filter_string='span.status = "ERROR"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test IN operator
    traces, _ = store.search_traces([exp_id], filter_string='span.status IN ("OK", "ERROR")')
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace1_id, trace2_id}

    # Test != operator
    traces, _ = store.search_traces([exp_id], filter_string='span.status != "ERROR"')
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace1_id, trace3_id}


def create_test_span_with_content(
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
    custom_attributes=None,
    inputs=None,
    outputs=None,
) -> Span:
    context = create_mock_span_context(trace_num, span_id)
    parent_context = create_mock_span_context(trace_num, parent_id) if parent_id else None

    attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id),
        "mlflow.spanType": json.dumps(span_type, cls=TraceJSONEncoder),
    }

    # Add custom attributes
    if custom_attributes:
        for key, value in custom_attributes.items():
            attributes[key] = json.dumps(value, cls=TraceJSONEncoder)

    # Add inputs and outputs
    if inputs:
        attributes["mlflow.spanInputs"] = json.dumps(inputs, cls=TraceJSONEncoder)
    if outputs:
        attributes["mlflow.spanOutputs"] = json.dumps(outputs, cls=TraceJSONEncoder)

    otel_span = OTelReadableSpan(
        name=name,
        context=context,
        parent=parent_context,
        attributes=attributes,
        start_time=start_ns,
        end_time=end_ns,
        status=trace_api.Status(status, status_desc),
        resource=_OTelResource.get_empty(),
    )
    return create_mlflow_span(otel_span, trace_id, span_type)


def test_search_traces_with_span_content_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_content_search")

    # Create traces
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create spans with different content
    span1 = create_test_span_with_content(
        trace1_id,
        name="gpt_span",
        span_id=111,
        span_type="LLM",
        custom_attributes={"model": "gpt-4", "temperature": 0.7},
        inputs={"prompt": "Tell me about machine learning"},
        outputs={"response": "Machine learning is a subset of AI"},
    )

    span2 = create_test_span_with_content(
        trace2_id,
        name="claude_span",
        span_id=222,
        span_type="LLM",
        custom_attributes={"model": "claude-3", "max_tokens": 1000},
        inputs={"query": "What is neural network?"},
        outputs={"response": "A neural network is..."},
    )

    span3 = create_test_span_with_content(
        trace3_id,
        name="vector_span",
        span_id=333,
        span_type="RETRIEVER",
        custom_attributes={"database": "vector_store"},
        inputs={"search": "embeddings"},
        outputs={"documents": ["doc1", "doc2"]},
    )

    # Add spans to store
    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])

    # Test LIKE operator for model in content
    traces, _ = store.search_traces([exp_id], filter_string='span.content LIKE "%gpt-4%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test LIKE operator for input text
    traces, _ = store.search_traces([exp_id], filter_string='span.content LIKE "%neural network%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test LIKE operator for attribute
    traces, _ = store.search_traces([exp_id], filter_string='span.content LIKE "%temperature%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test ILIKE operator (case-insensitive)
    traces, _ = store.search_traces(
        [exp_id], filter_string='span.content ILIKE "%MACHINE LEARNING%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test LIKE with wildcard patterns
    traces, _ = store.search_traces([exp_id], filter_string='span.content LIKE "%model%"')
    assert len(traces) == 2  # Both LLM spans have "model" in their attributes
    assert {t.request_id for t in traces} == {trace1_id, trace2_id}

    # Test searching for array content
    traces, _ = store.search_traces([exp_id], filter_string='span.content LIKE "%doc1%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id


def test_search_traces_with_combined_span_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_combined_span_search")

    # Create traces
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)
    _create_trace(store, trace4_id, exp_id)

    # Create spans with various combinations
    span1 = create_test_span_with_content(
        trace1_id,
        name="llm_success",
        span_id=111,
        span_type="LLM",
        status=trace_api.StatusCode.OK,
        custom_attributes={"model": "gpt-4"},
    )

    span2 = create_test_span_with_content(
        trace2_id,
        name="llm_error",
        span_id=222,
        span_type="LLM",
        status=trace_api.StatusCode.ERROR,
        custom_attributes={"model": "gpt-3.5"},
    )

    span3 = create_test_span_with_content(
        trace3_id,
        name="retriever_success",
        span_id=333,
        span_type="RETRIEVER",
        status=trace_api.StatusCode.OK,
        custom_attributes={"database": "pinecone"},
    )

    span4 = create_test_span_with_content(
        trace4_id,
        name="llm_success_claude",
        span_id=444,
        span_type="LLM",
        status=trace_api.StatusCode.OK,
        custom_attributes={"model": "claude-3"},
    )

    # Add spans to store (must log spans for each trace separately)
    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])
    store.log_spans(exp_id, [span4])

    # Test: type = LLM AND status = OK
    traces, _ = store.search_traces(
        [exp_id], filter_string='span.type = "LLM" AND span.status = "OK"'
    )
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace1_id, trace4_id}

    # Test: type = LLM AND content contains gpt
    traces, _ = store.search_traces(
        [exp_id], filter_string='span.type = "LLM" AND span.content LIKE "%gpt%"'
    )
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace1_id, trace2_id}

    # Test: name LIKE pattern AND status = OK
    traces, _ = store.search_traces(
        [exp_id], filter_string='span.name LIKE "%success%" AND span.status = "OK"'
    )
    assert len(traces) == 3
    assert {t.request_id for t in traces} == {trace1_id, trace3_id, trace4_id}

    # Test: Complex combination - (type = LLM AND status = OK) AND content LIKE gpt
    traces, _ = store.search_traces(
        [exp_id],
        filter_string='span.type = "LLM" AND span.status = "OK" AND span.content LIKE "%gpt-4%"',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id


def test_search_traces_combined_span_filters_match_same_span(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_same_span_filter")

    trace1_id = "trace1"
    _create_trace(store, trace1_id, exp_id)

    span1a = create_test_span_with_content(
        trace1_id,
        name="search_web",
        span_id=111,
        span_type="TOOL",
        status=trace_api.StatusCode.ERROR,
        custom_attributes={"query": "test"},
    )
    span1b = create_test_span_with_content(
        trace1_id,
        name="other_tool",
        span_id=112,
        span_type="TOOL",
        status=trace_api.StatusCode.OK,
        custom_attributes={"data": "value"},
    )

    trace2_id = "trace2"
    _create_trace(store, trace2_id, exp_id)

    span2 = create_test_span_with_content(
        trace2_id,
        name="search_web",
        span_id=222,
        span_type="TOOL",
        status=trace_api.StatusCode.OK,
        custom_attributes={"query": "test2"},
    )

    store.log_spans(exp_id, [span1a, span1b])
    store.log_spans(exp_id, [span2])

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.name = "search_web" AND span.status = "OK"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.name = "search_web" AND span.status = "ERROR"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.name = "other_tool" AND span.status = "OK"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    traces, _ = store.search_traces([exp_id], filter_string='span.name = "search_web"')
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace1_id, trace2_id}

    traces, _ = store.search_traces([exp_id], filter_string='span.status = "OK"')
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace1_id, trace2_id}


def test_search_traces_span_filters_with_no_results(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_no_results")

    # Create a trace with a span
    trace_id = "trace1"
    _create_trace(store, trace_id, exp_id)

    span = create_test_span_with_content(
        trace_id,
        name="test_span",
        span_id=111,
        span_type="LLM",
        status=trace_api.StatusCode.OK,
        custom_attributes={"model": "gpt-4"},
    )

    store.log_spans(exp_id, [span])

    # Test searching for non-existent type
    traces, _ = store.search_traces([exp_id], filter_string='span.type = "NONEXISTENT"')
    assert len(traces) == 0

    # Test searching for non-existent content
    traces, _ = store.search_traces(
        [exp_id], filter_string='span.content LIKE "%nonexistent_model%"'
    )
    assert len(traces) == 0

    # Test contradictory conditions
    traces, _ = store.search_traces(
        [exp_id], filter_string='span.type = "LLM" AND span.type = "RETRIEVER"'
    )
    assert len(traces) == 0


def test_search_traces_with_span_attributes_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_attributes_search")

    # Create traces with spans having custom attributes
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create spans with different custom attributes
    span1 = create_test_span_with_content(
        trace1_id,
        name="llm_span",
        span_id=111,
        span_type="LLM",
        custom_attributes={"model": "gpt-4", "temperature": 0.7, "max_tokens": 1000},
    )

    span2 = create_test_span_with_content(
        trace2_id,
        name="llm_span",
        span_id=222,
        span_type="LLM",
        custom_attributes={"model": "claude-3", "temperature": 0.5, "provider": "anthropic"},
    )

    span3 = create_test_span_with_content(
        trace3_id,
        name="retriever_span",
        span_id=333,
        span_type="RETRIEVER",
        custom_attributes={"database": "pinecone", "top_k": 10, "similarity.threshold": 0.8},
    )

    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])

    traces, _ = store.search_traces([exp_id], filter_string='span.attributes.model LIKE "%gpt-4%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.temperature LIKE "%0.7%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.provider LIKE "%anthropic%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.database LIKE "%pinecone%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.nonexistent LIKE "%value%"'
    )
    assert len(traces) == 0

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.similarity.threshold LIKE "%0.8%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id


def test_search_traces_with_feedback_and_expectation_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_feedback_expectation_search")

    # Create multiple traces
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)
    _create_trace(store, trace4_id, exp_id)

    # Create feedback for trace1 and trace2
    feedback1 = Feedback(
        trace_id=trace1_id,
        name="correctness",
        value=True,
        source=AssessmentSource(source_type="HUMAN", source_id="user1@example.com"),
        rationale="The response is accurate",
    )

    feedback2 = Feedback(
        trace_id=trace2_id,
        name="correctness",
        value=False,
        source=AssessmentSource(source_type="LLM_JUDGE", source_id="gpt-4"),
        rationale="The response contains errors",
    )

    feedback3 = Feedback(
        trace_id=trace2_id,
        name="helpfulness",
        value=5,
        source=AssessmentSource(source_type="HUMAN", source_id="user2@example.com"),
    )

    feedback4 = Feedback(
        trace_id=trace1_id,
        name="quality",
        value="high",
        source=AssessmentSource(source_type="HUMAN", source_id="user1@example.com"),
    )

    # Create expectations for trace3 and trace4
    expectation1 = Expectation(
        trace_id=trace3_id,
        name="response_length",
        value=150,
        source=AssessmentSource(source_type="CODE", source_id="length_checker"),
    )

    expectation2 = Expectation(
        trace_id=trace4_id,
        name="response_length",
        value=200,
        source=AssessmentSource(source_type="CODE", source_id="length_checker"),
    )

    expectation3 = Expectation(
        trace_id=trace4_id,
        name="latency_ms",
        value=1000,
        source=AssessmentSource(source_type="CODE", source_id="latency_monitor"),
    )

    expectation4 = Expectation(
        trace_id=trace3_id,
        name="priority",
        value="urgent",
        source=AssessmentSource(source_type="CODE", source_id="priority_checker"),
    )

    # Store assessments
    store.create_assessment(feedback1)
    store.create_assessment(feedback2)
    store.create_assessment(feedback3)
    store.create_assessment(feedback4)
    store.create_assessment(expectation1)
    store.create_assessment(expectation2)
    store.create_assessment(expectation3)
    store.create_assessment(expectation4)

    # Test: Search for traces with correctness feedback = True
    traces, _ = store.search_traces([exp_id], filter_string='feedback.correctness = "true"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: Search for traces with correctness feedback = False
    traces, _ = store.search_traces([exp_id], filter_string='feedback.correctness = "false"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: Search for traces with helpfulness feedback = 5
    traces, _ = store.search_traces([exp_id], filter_string='feedback.helpfulness = "5"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: Search for traces with string-valued feedback
    traces, _ = store.search_traces([exp_id], filter_string='feedback.quality = "high"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: Search for traces with response_length expectation = 150
    traces, _ = store.search_traces([exp_id], filter_string='expectation.response_length = "150"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: Search for traces with response_length expectation = 200
    traces, _ = store.search_traces([exp_id], filter_string='expectation.response_length = "200"')
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id

    # Test: Search for traces with latency_ms expectation = 1000
    traces, _ = store.search_traces([exp_id], filter_string='expectation.latency_ms = "1000"')
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id

    # Test: Search for traces with string-valued expectation
    traces, _ = store.search_traces([exp_id], filter_string='expectation.priority = "urgent"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: Combined filter with AND - trace with multiple expectations
    traces, _ = store.search_traces(
        [exp_id],
        filter_string='expectation.response_length = "200" AND expectation.latency_ms = "1000"',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id

    # Test: Search for non-existent feedback
    traces, _ = store.search_traces([exp_id], filter_string='feedback.nonexistent = "value"')
    assert len(traces) == 0

    # Test: Search for non-existent expectation
    traces, _ = store.search_traces([exp_id], filter_string='expectation.nonexistent = "value"')
    assert len(traces) == 0


def test_search_traces_with_run_id(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_run_id")
    run1_id = "run1"
    run2_id = "run2"
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, trace_metadata={"mlflow.sourceRun": run1_id})
    _create_trace(store, trace2_id, exp_id, trace_metadata={"mlflow.sourceRun": run2_id})
    _create_trace(store, trace3_id, exp_id)

    traces, _ = store.search_traces([exp_id], filter_string='trace.run_id = "run1"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    traces, _ = store.search_traces([exp_id], filter_string='trace.run_id = "run2"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    traces, _ = store.search_traces([exp_id], filter_string='trace.run_id = "run3"')
    assert len(traces) == 0


def test_search_traces_with_client_request_id_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_client_request_id")

    # Create traces with different client_request_ids
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, client_request_id="client-req-abc")
    _create_trace(store, trace2_id, exp_id, client_request_id="client-req-xyz")
    _create_trace(store, trace3_id, exp_id, client_request_id=None)

    # Test: Exact match with =
    traces, _ = store.search_traces(
        [exp_id], filter_string='trace.client_request_id = "client-req-abc"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: Not equal with !=
    traces, _ = store.search_traces(
        [exp_id], filter_string='trace.client_request_id != "client-req-abc"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: LIKE pattern matching
    traces, _ = store.search_traces([exp_id], filter_string='trace.client_request_id LIKE "%abc%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: ILIKE case-insensitive pattern matching
    traces, _ = store.search_traces([exp_id], filter_string='trace.client_request_id ILIKE "%ABC%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id


def test_search_traces_with_name_like_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_name_like")

    # Create traces with different names
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, tags={TraceTagKey.TRACE_NAME: "GenerateResponse"})
    _create_trace(store, trace2_id, exp_id, tags={TraceTagKey.TRACE_NAME: "QueryDatabase"})
    _create_trace(store, trace3_id, exp_id, tags={TraceTagKey.TRACE_NAME: "GenerateEmbedding"})

    # Test: LIKE with prefix
    traces, _ = store.search_traces([exp_id], filter_string='trace.name LIKE "Generate%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test: LIKE with suffix
    traces, _ = store.search_traces([exp_id], filter_string='trace.name LIKE "%Database"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: ILIKE case-insensitive
    traces, _ = store.search_traces([exp_id], filter_string='trace.name ILIKE "%response%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: ILIKE with wildcard in middle
    traces, _ = store.search_traces([exp_id], filter_string='trace.name ILIKE "%generate%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}


def test_search_traces_with_tag_like_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_tag_like")

    # Create traces with different tag values
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, tags={"environment": "production-us-east-1"})
    _create_trace(store, trace2_id, exp_id, tags={"environment": "production-us-west-2"})
    _create_trace(store, trace3_id, exp_id, tags={"environment": "staging-us-east-1"})

    # Test: LIKE with prefix
    traces, _ = store.search_traces([exp_id], filter_string='tag.environment LIKE "production%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: LIKE with suffix
    traces, _ = store.search_traces([exp_id], filter_string='tag.environment LIKE "%-us-east-1"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test: ILIKE case-insensitive
    traces, _ = store.search_traces([exp_id], filter_string='tag.environment ILIKE "%PRODUCTION%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}


def test_search_traces_with_feedback_like_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_feedback_like")

    # Create traces with different feedback values
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create feedback with string values that can be pattern matched
    feedback1 = Feedback(
        trace_id=trace1_id,
        name="comment",
        value="Great response! Very helpful.",
        source=AssessmentSource(source_type="HUMAN", source_id="user1@example.com"),
    )

    feedback2 = Feedback(
        trace_id=trace2_id,
        name="comment",
        value="Response was okay but could be better.",
        source=AssessmentSource(source_type="HUMAN", source_id="user2@example.com"),
    )

    feedback3 = Feedback(
        trace_id=trace3_id,
        name="comment",
        value="Not helpful at all.",
        source=AssessmentSource(source_type="HUMAN", source_id="user3@example.com"),
    )

    store.create_assessment(feedback1)
    store.create_assessment(feedback2)
    store.create_assessment(feedback3)

    # Test: LIKE pattern matching
    traces, _ = store.search_traces([exp_id], filter_string='feedback.comment LIKE "%helpful%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test: ILIKE case-insensitive pattern matching
    traces, _ = store.search_traces([exp_id], filter_string='feedback.comment ILIKE "%GREAT%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: LIKE with negation - response was okay
    traces, _ = store.search_traces([exp_id], filter_string='feedback.comment LIKE "%okay%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id


def test_search_traces_with_assessment_is_null_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_null_filters")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"
    trace5_id = "trace5"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)
    _create_trace(store, trace4_id, exp_id)
    _create_trace(store, trace5_id, exp_id)

    feedback1 = Feedback(
        trace_id=trace1_id,
        name="quality",
        value="good",
        source=AssessmentSource(source_type="HUMAN", source_id="user1@example.com"),
    )
    feedback2 = Feedback(
        trace_id=trace2_id,
        name="quality",
        value="bad",
        source=AssessmentSource(source_type="HUMAN", source_id="user2@example.com"),
    )

    expectation1 = Expectation(
        trace_id=trace4_id,
        name="score",
        value=85,
        source=AssessmentSource(source_type="CODE", source_id="scorer"),
    )

    store.create_assessment(feedback1)
    store.create_assessment(feedback2)
    store.create_assessment(expectation1)

    traces, _ = store.search_traces([exp_id], filter_string="feedback.quality IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    traces, _ = store.search_traces([exp_id], filter_string="feedback.quality IS NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id, trace5_id}

    traces, _ = store.search_traces([exp_id], filter_string="expectation.score IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace4_id}

    traces, _ = store.search_traces([exp_id], filter_string="expectation.score IS NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id, trace5_id}

    traces, _ = store.search_traces(
        [exp_id],
        filter_string='feedback.quality IS NOT NULL AND feedback.quality = "good"',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id


def test_search_traces_with_feedback_filters_excludes_invalid_assessments(
    store: SqlAlchemyStore,
):
    exp_id = store.create_experiment("test_feedback_filters_excludes_invalid")

    trace1_id = "trace1"
    trace2_id = "trace2"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)

    # trace1: overridden from "no" to "yes" - only "yes" is valid
    original_feedback = Feedback(
        trace_id=trace1_id,
        name="correctness",
        value="no",
        source=AssessmentSource(source_type="HUMAN", source_id="user@example.com"),
    )
    created_original = store.create_assessment(original_feedback)

    override_feedback = Feedback(
        trace_id=trace1_id,
        name="correctness",
        value="yes",
        source=AssessmentSource(source_type="HUMAN", source_id="user@example.com"),
        overrides=created_original.assessment_id,
    )
    store.create_assessment(override_feedback)

    # trace2: "no" assessment, never overridden
    feedback2 = Feedback(
        trace_id=trace2_id,
        name="correctness",
        value="no",
        source=AssessmentSource(source_type="HUMAN", source_id="user@example.com"),
    )
    store.create_assessment(feedback2)

    # Filtering by "yes" should return only trace1 (current valid assessment)
    traces, _ = store.search_traces([exp_id], filter_string='feedback.correctness = "yes"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Filtering by "no" should return only trace2 (trace1's "no" is invalid/overridden)
    traces, _ = store.search_traces([exp_id], filter_string='feedback.correctness = "no"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # IS NOT NULL should return both (both have a valid assessment)
    traces, _ = store.search_traces([exp_id], filter_string="feedback.correctness IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}


def test_search_traces_session_scoped_assessment_expands_to_all_session_traces(
    store: SqlAlchemyStore,
):
    exp_id = store.create_experiment("test_session_assessment_expansion")

    # Session A: 3 traces
    _create_trace(
        store, "sa-t1", exp_id, trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-a"}
    )
    _create_trace(
        store, "sa-t2", exp_id, trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-a"}
    )
    _create_trace(
        store, "sa-t3", exp_id, trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-a"}
    )

    # Session B: 3 traces
    _create_trace(
        store, "sb-t1", exp_id, trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-b"}
    )
    _create_trace(
        store, "sb-t2", exp_id, trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-b"}
    )
    _create_trace(
        store, "sb-t3", exp_id, trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-b"}
    )

    # Add a session-scoped assessment on the first trace of session A
    session_feedback = Feedback(
        trace_id="sa-t1",
        name="session_quality",
        value="good",
        source=AssessmentSource(source_type="HUMAN", source_id="user@example.com"),
        metadata={TraceMetadataKey.TRACE_SESSION: "session-a"},
    )
    store.create_assessment(session_feedback)

    # Add a non-session assessment on a trace in session B (no session metadata)
    non_session_feedback = Feedback(
        trace_id="sb-t1",
        name="trace_quality",
        value="bad",
        source=AssessmentSource(source_type="HUMAN", source_id="user@example.com"),
    )
    store.create_assessment(non_session_feedback)

    # Searching with the session-scoped assessment filter should return all 3 traces
    # from session A (not just the one with the assessment)
    traces, _ = store.search_traces([exp_id], filter_string='feedback.session_quality = "good"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {"sa-t1", "sa-t2", "sa-t3"}

    # IS NOT NULL with session-scoped assessment should also expand
    traces, _ = store.search_traces([exp_id], filter_string="feedback.session_quality IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {"sa-t1", "sa-t2", "sa-t3"}

    # Searching with non-session assessment should return only the single matching trace
    traces, _ = store.search_traces([exp_id], filter_string='feedback.trace_quality = "bad"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {"sb-t1"}

    # IS NOT NULL with non-session assessment should return only the single trace
    traces, _ = store.search_traces([exp_id], filter_string="feedback.trace_quality IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {"sb-t1"}

    # IS NULL with session-scoped assessment should exclude all session siblings too
    traces, _ = store.search_traces([exp_id], filter_string="feedback.session_quality IS NULL")
    trace_ids = {t.request_id for t in traces}
    # All session-A traces are excluded (session-scoped assessment covers the whole session)
    assert trace_ids == {"sb-t1", "sb-t2", "sb-t3"}


def test_search_traces_with_expectation_like_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_expectation_like")

    # Create traces with different expectation values
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create expectations with string values
    expectation1 = Expectation(
        trace_id=trace1_id,
        name="output_format",
        value="JSON with nested structure",
        source=AssessmentSource(source_type="CODE", source_id="validator"),
    )

    expectation2 = Expectation(
        trace_id=trace2_id,
        name="output_format",
        value="XML document",
        source=AssessmentSource(source_type="CODE", source_id="validator"),
    )

    expectation3 = Expectation(
        trace_id=trace3_id,
        name="output_format",
        value="JSON array",
        source=AssessmentSource(source_type="CODE", source_id="validator"),
    )

    store.create_assessment(expectation1)
    store.create_assessment(expectation2)
    store.create_assessment(expectation3)

    # Test: LIKE pattern matching
    traces, _ = store.search_traces(
        [exp_id], filter_string='expectation.output_format LIKE "%JSON%"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test: ILIKE case-insensitive
    traces, _ = store.search_traces(
        [exp_id], filter_string='expectation.output_format ILIKE "%xml%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: LIKE with specific pattern
    traces, _ = store.search_traces(
        [exp_id], filter_string='expectation.output_format LIKE "%nested%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id


def test_search_traces_with_metadata_like_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_metadata_like")

    # Create traces with different metadata values
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(
        store, trace1_id, exp_id, trace_metadata={"custom_field": "production-deployment-v1"}
    )
    _create_trace(
        store, trace2_id, exp_id, trace_metadata={"custom_field": "production-deployment-v2"}
    )
    _create_trace(
        store, trace3_id, exp_id, trace_metadata={"custom_field": "staging-deployment-v1"}
    )

    # Test: LIKE with prefix
    traces, _ = store.search_traces(
        [exp_id], filter_string='metadata.custom_field LIKE "production%"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: LIKE with suffix
    traces, _ = store.search_traces([exp_id], filter_string='metadata.custom_field LIKE "%-v1"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test: ILIKE case-insensitive
    traces, _ = store.search_traces(
        [exp_id], filter_string='metadata.custom_field ILIKE "%PRODUCTION%"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}


def test_search_traces_with_combined_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_combined_filters")

    # Create traces with various attributes
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(
        store,
        trace1_id,
        exp_id,
        tags={TraceTagKey.TRACE_NAME: "GenerateResponse", "env": "production"},
        client_request_id="req-prod-001",
    )
    _create_trace(
        store,
        trace2_id,
        exp_id,
        tags={TraceTagKey.TRACE_NAME: "GenerateResponse", "env": "staging"},
        client_request_id="req-staging-001",
    )
    _create_trace(
        store,
        trace3_id,
        exp_id,
        tags={TraceTagKey.TRACE_NAME: "QueryDatabase", "env": "production"},
        client_request_id="req-prod-002",
    )
    _create_trace(
        store,
        trace4_id,
        exp_id,
        tags={TraceTagKey.TRACE_NAME: "QueryDatabase", "env": "staging"},
        client_request_id="req-staging-002",
    )

    # Test: Combine LIKE filters with AND
    traces, _ = store.search_traces(
        [exp_id],
        filter_string='trace.name LIKE "Generate%" AND tag.env = "production"',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: Combine ILIKE with exact match
    traces, _ = store.search_traces(
        [exp_id],
        filter_string='trace.client_request_id ILIKE "%PROD%" AND trace.name = "QueryDatabase"',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: Multiple LIKE conditions
    traces, _ = store.search_traces(
        [exp_id],
        filter_string='trace.name LIKE "%Response%" AND trace.client_request_id LIKE "%-staging-%"',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: ILIKE on tag with exact match on another field
    traces, _ = store.search_traces(
        [exp_id],
        filter_string='tag.env ILIKE "%STAGING%" AND trace.name != "GenerateResponse"',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id


def test_search_traces_with_client_request_id_edge_cases(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_client_request_id_edge")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    # Various client_request_id formats
    _create_trace(store, trace1_id, exp_id, client_request_id="simple")
    _create_trace(store, trace2_id, exp_id, client_request_id="with-dashes-123")
    _create_trace(store, trace3_id, exp_id, client_request_id="WITH_UNDERSCORES_456")
    _create_trace(store, trace4_id, exp_id, client_request_id=None)

    # Test: LIKE with wildcard at start
    traces, _ = store.search_traces([exp_id], filter_string='trace.client_request_id LIKE "%123"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: LIKE with wildcard at end
    traces, _ = store.search_traces([exp_id], filter_string='trace.client_request_id LIKE "WITH%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: ILIKE finds case-insensitive match
    traces, _ = store.search_traces([exp_id], filter_string='trace.client_request_id ILIKE "with%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}

    # Test: Exact match still works
    traces, _ = store.search_traces([exp_id], filter_string='trace.client_request_id = "simple"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: != excludes matched trace
    traces, _ = store.search_traces([exp_id], filter_string='trace.client_request_id != "simple"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}


def test_search_traces_with_name_ilike_variations(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_name_ilike_variations")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id, tags={TraceTagKey.TRACE_NAME: "USER_LOGIN"})
    _create_trace(store, trace2_id, exp_id, tags={TraceTagKey.TRACE_NAME: "user_logout"})
    _create_trace(store, trace3_id, exp_id, tags={TraceTagKey.TRACE_NAME: "User_Profile_Update"})
    _create_trace(store, trace4_id, exp_id, tags={TraceTagKey.TRACE_NAME: "AdminDashboard"})

    # Test: ILIKE finds all user-related traces regardless of case
    traces, _ = store.search_traces([exp_id], filter_string='trace.name ILIKE "user%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}

    # Test: ILIKE with wildcard in middle
    traces, _ = store.search_traces([exp_id], filter_string='trace.name ILIKE "%_log%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: LIKE is case-sensitive (should not match)
    traces, _ = store.search_traces([exp_id], filter_string='trace.name LIKE "user%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id}  # Only lowercase match

    # Test: Exact match with !=
    traces, _ = store.search_traces([exp_id], filter_string='trace.name != "USER_LOGIN"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id, trace4_id}


def test_search_traces_with_span_name_like_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_name_like")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create spans with different names
    span1 = create_test_span_with_content(
        trace1_id, name="llm.generate_response", span_id=111, span_type="LLM"
    )
    span2 = create_test_span_with_content(
        trace2_id, name="llm.generate_embedding", span_id=222, span_type="LLM"
    )
    span3 = create_test_span_with_content(
        trace3_id, name="database.query_users", span_id=333, span_type="TOOL"
    )

    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])

    # Test: LIKE with prefix
    traces, _ = store.search_traces([exp_id], filter_string='span.name LIKE "llm.%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: LIKE with suffix
    traces, _ = store.search_traces([exp_id], filter_string='span.name LIKE "%_response"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: ILIKE case-insensitive
    traces, _ = store.search_traces([exp_id], filter_string='span.name ILIKE "%GENERATE%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: LIKE with wildcard in middle
    traces, _ = store.search_traces([exp_id], filter_string='span.name LIKE "%base.%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_name_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_name_rlike")

    # Create traces with different names
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id, tags={TraceTagKey.TRACE_NAME: "GenerateResponse"})
    _create_trace(store, trace2_id, exp_id, tags={TraceTagKey.TRACE_NAME: "QueryDatabase"})
    _create_trace(store, trace3_id, exp_id, tags={TraceTagKey.TRACE_NAME: "GenerateEmbedding"})
    _create_trace(store, trace4_id, exp_id, tags={TraceTagKey.TRACE_NAME: "api_v1_call"})

    # Test: RLIKE with regex pattern matching "Generate" at start
    traces, _ = store.search_traces([exp_id], filter_string='trace.name RLIKE "^Generate"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test: RLIKE with regex pattern matching "Database" at end
    traces, _ = store.search_traces([exp_id], filter_string='trace.name RLIKE "Database$"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: RLIKE with character class [RE]
    traces, _ = store.search_traces([exp_id], filter_string='trace.name RLIKE "^Generate[RE]"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test: RLIKE with alternation (OR)
    traces, _ = store.search_traces([exp_id], filter_string='trace.name RLIKE "(Query|Embedding)"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}

    # Test: RLIKE with digit pattern
    traces, _ = store.search_traces([exp_id], filter_string='trace.name RLIKE "v[0-9]+"')
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_tag_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_tag_rlike")

    # Create traces with different tag values
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id, tags={"environment": "production-us-east-1"})
    _create_trace(store, trace2_id, exp_id, tags={"environment": "production-us-west-2"})
    _create_trace(store, trace3_id, exp_id, tags={"environment": "staging-us-east-1"})
    _create_trace(store, trace4_id, exp_id, tags={"environment": "dev-local"})

    # Test: RLIKE with regex pattern for production environments
    traces, _ = store.search_traces([exp_id], filter_string='tag.environment RLIKE "^production"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: RLIKE with pattern matching regions
    traces, _ = store.search_traces(
        [exp_id], filter_string='tag.environment RLIKE "us-(east|west)-[0-9]"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}

    # Test: RLIKE with negation pattern (not starting with production/staging)
    traces, _ = store.search_traces([exp_id], filter_string='tag.environment RLIKE "^dev"')
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_span_name_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_name_rlike")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"
    trace5_id = "trace5"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)
    _create_trace(store, trace4_id, exp_id)
    _create_trace(store, trace5_id, exp_id)

    # Create spans with different names
    span1 = create_test_span_with_content(
        trace1_id, name="llm.generate_response", span_id=111, span_type="LLM"
    )
    span2 = create_test_span_with_content(
        trace2_id, name="llm.generate_embedding", span_id=222, span_type="LLM"
    )
    span3 = create_test_span_with_content(
        trace3_id, name="database.query_users", span_id=333, span_type="TOOL"
    )
    span4 = create_test_span_with_content(
        trace4_id, name="api_v2_endpoint", span_id=444, span_type="TOOL"
    )
    span5 = create_test_span_with_content(
        trace5_id, name="base.query_users", span_id=444, span_type="TOOL"
    )

    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])
    store.log_spans(exp_id, [span4])
    store.log_spans(exp_id, [span5])

    # Test: RLIKE with pattern matching llm namespace
    traces, _ = store.search_traces([exp_id], filter_string='span.name RLIKE "^llm\\."')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: RLIKE with alternation for different operations
    traces, _ = store.search_traces([exp_id], filter_string='span.name RLIKE "(response|users)"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id, trace5_id}

    # Test: RLIKE with version pattern
    traces, _ = store.search_traces([exp_id], filter_string='span.name RLIKE "v[0-9]+_"')
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id

    # Test: RLIKE matching embedded substring
    traces, _ = store.search_traces([exp_id], filter_string='span.name RLIKE "query"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace5_id}

    traces, _ = store.search_traces([exp_id], filter_string='span.name RLIKE "query_users"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace5_id}

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.name RLIKE "^database\\.query_users$"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_feedback_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_feedback_rlike")

    # Create traces with different feedback values
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create feedback with string values that can be pattern matched
    from mlflow.entities.assessment import AssessmentSource, Feedback

    feedback1 = Feedback(
        trace_id=trace1_id,
        name="comment",
        value="Great response! Very helpful.",
        source=AssessmentSource(source_type="HUMAN", source_id="user1@example.com"),
    )

    feedback2 = Feedback(
        trace_id=trace2_id,
        name="comment",
        value="Response was okay but could be better.",
        source=AssessmentSource(source_type="HUMAN", source_id="user2@example.com"),
    )

    feedback3 = Feedback(
        trace_id=trace3_id,
        name="comment",
        value="Not helpful at all.",
        source=AssessmentSource(source_type="HUMAN", source_id="user3@example.com"),
    )

    store.create_assessment(feedback1)
    store.create_assessment(feedback2)
    store.create_assessment(feedback3)

    # Test: RLIKE pattern matching response patterns
    traces, _ = store.search_traces(
        [exp_id], filter_string='feedback.comment RLIKE "Great.*helpful"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: RLIKE with alternation
    traces, _ = store.search_traces(
        [exp_id], filter_string='feedback.comment RLIKE "(okay|better)"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: RLIKE matching negative feedback
    traces, _ = store.search_traces([exp_id], filter_string='feedback.comment RLIKE "Not.*all"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id


def test_search_traces_with_metadata_is_null_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_metadata_is_null")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, trace_metadata={"env": "production", "region": "us"})
    _create_trace(store, trace2_id, exp_id, trace_metadata={"env": "staging"})
    _create_trace(store, trace3_id, exp_id, trace_metadata={})

    traces, _ = store.search_traces([exp_id], filter_string="metadata.region IS NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}

    traces, _ = store.search_traces([exp_id], filter_string="metadata.env IS NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id}

    traces, _ = store.search_traces(
        [exp_id], filter_string='metadata.region IS NULL AND metadata.env = "staging"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id}


def test_search_traces_with_metadata_is_not_null_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_metadata_is_not_null")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, trace_metadata={"env": "production", "region": "us"})
    _create_trace(store, trace2_id, exp_id, trace_metadata={"env": "staging"})
    _create_trace(store, trace3_id, exp_id, trace_metadata={})

    traces, _ = store.search_traces([exp_id], filter_string="metadata.region IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id}

    traces, _ = store.search_traces([exp_id], filter_string="metadata.env IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    traces, _ = store.search_traces(
        [exp_id], filter_string='metadata.region IS NOT NULL AND metadata.env = "production"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id}


def test_search_traces_with_tag_is_null_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_tag_is_null")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, tags={"env": "production", "region": "us"})
    _create_trace(store, trace2_id, exp_id, tags={"env": "staging"})
    _create_trace(store, trace3_id, exp_id, tags={})

    traces, _ = store.search_traces([exp_id], filter_string="tag.region IS NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}

    traces, _ = store.search_traces([exp_id], filter_string="tag.env IS NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id}

    traces, _ = store.search_traces(
        [exp_id], filter_string='tag.region IS NULL AND tag.env = "staging"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id}


def test_search_traces_with_tag_is_not_null_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_tag_is_not_null")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, tags={"env": "production", "region": "us"})
    _create_trace(store, trace2_id, exp_id, tags={"env": "staging"})
    _create_trace(store, trace3_id, exp_id, tags={})

    traces, _ = store.search_traces([exp_id], filter_string="tag.region IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id}

    traces, _ = store.search_traces([exp_id], filter_string="tag.env IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    traces, _ = store.search_traces(
        [exp_id], filter_string='tag.region IS NOT NULL AND tag.env = "production"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id}


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_metadata_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_metadata_rlike")

    # Create traces with different metadata values
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, trace_metadata={"version": "v1.2.3"})
    _create_trace(store, trace2_id, exp_id, trace_metadata={"version": "v2.0.0-beta"})
    _create_trace(store, trace3_id, exp_id, trace_metadata={"version": "v2.1.5"})

    # Test: RLIKE with semantic version pattern (no anchors for SQLite compatibility)
    traces, _ = store.search_traces(
        [exp_id], filter_string='metadata.version RLIKE "v[0-9]+\\.[0-9]+\\.[0-9]"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}

    # Test: RLIKE with version 2.x pattern
    traces, _ = store.search_traces([exp_id], filter_string='metadata.version RLIKE "v2\\.[0-9]"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}

    # Test: RLIKE matching beta versions
    traces, _ = store.search_traces([exp_id], filter_string='metadata.version RLIKE "beta"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_client_request_id_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_client_request_id_rlike")

    # Create traces with different client_request_id patterns
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id, client_request_id="req-prod-us-east-123")
    _create_trace(store, trace2_id, exp_id, client_request_id="req-prod-us-west-456")
    _create_trace(store, trace3_id, exp_id, client_request_id="req-staging-eu-789")
    _create_trace(store, trace4_id, exp_id, client_request_id="req-dev-local-001")

    # Test: RLIKE with pattern matching production requests
    traces, _ = store.search_traces(
        [exp_id], filter_string='trace.client_request_id RLIKE "^req-prod"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: RLIKE with pattern matching US regions
    traces, _ = store.search_traces(
        [exp_id], filter_string='trace.client_request_id RLIKE "us-(east|west)"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: RLIKE with digit pattern - all traces end with 3 digits
    traces, _ = store.search_traces(
        [exp_id], filter_string='trace.client_request_id RLIKE "[0-9]{3}$"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id, trace4_id}

    # Test: RLIKE matching staging or dev environments
    traces, _ = store.search_traces(
        [exp_id], filter_string='trace.client_request_id RLIKE "(staging|dev)"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id}


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_span_type_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_type_rlike")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)
    _create_trace(store, trace4_id, exp_id)

    # Create spans with different types
    span1 = create_test_span_with_content(trace1_id, name="generate", span_id=111, span_type="LLM")
    span2 = create_test_span_with_content(
        trace2_id, name="embed", span_id=222, span_type="LLM_EMBEDDING"
    )
    span3 = create_test_span_with_content(
        trace3_id, name="retrieve", span_id=333, span_type="RETRIEVER"
    )
    span4 = create_test_span_with_content(
        trace4_id, name="chain", span_id=444, span_type="CHAIN_PARENT"
    )

    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])
    store.log_spans(exp_id, [span4])

    # Test: RLIKE with pattern matching LLM types (LLM or LLM_*)
    traces, _ = store.search_traces([exp_id], filter_string='span.type RLIKE "^LLM"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: RLIKE with pattern matching types ending with specific suffix
    traces, _ = store.search_traces([exp_id], filter_string='span.type RLIKE "PARENT$"')
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id

    # Test: RLIKE with character class for embedding or retriever
    traces, _ = store.search_traces(
        [exp_id], filter_string='span.type RLIKE "(EMBEDDING|RETRIEVER)"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}

    # Test: RLIKE matching underscore patterns
    traces, _ = store.search_traces([exp_id], filter_string='span.type RLIKE "_"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace4_id}


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_span_attributes_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_attributes_rlike")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)
    _create_trace(store, trace4_id, exp_id)

    # Create spans with different custom attributes
    span1 = create_test_span_with_content(
        trace1_id,
        name="call1",
        span_id=111,
        span_type="LLM",
        custom_attributes={"model": "gpt-4-turbo-preview", "provider": "openai"},
    )
    span2 = create_test_span_with_content(
        trace2_id,
        name="call2",
        span_id=222,
        span_type="LLM",
        custom_attributes={"model": "gpt-3.5-turbo", "provider": "openai"},
    )
    span3 = create_test_span_with_content(
        trace3_id,
        name="call3",
        span_id=333,
        span_type="LLM",
        custom_attributes={"model": "claude-3-opus-20240229", "provider": "anthropic"},
    )
    span4 = create_test_span_with_content(
        trace4_id,
        name="call4",
        span_id=444,
        span_type="LLM",
        custom_attributes={"model": "claude-3-sonnet-20240229", "provider": "anthropic"},
    )

    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])
    store.log_spans(exp_id, [span4])

    traces, _ = store.search_traces([exp_id], filter_string='span.attributes.model RLIKE "^gpt"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    traces, _ = store.search_traces([exp_id], filter_string='span.attributes.model RLIKE "^claude"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id}

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.model RLIKE "(preview|[0-9]{8})"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id, trace4_id}

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.provider RLIKE "^openai"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    traces, _ = store.search_traces([exp_id], filter_string='span.attributes.model RLIKE "turbo"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}


def test_search_traces_with_empty_and_special_characters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_special_chars")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(
        store,
        trace1_id,
        exp_id,
        tags={"special": "test@example.com"},
        client_request_id="req-123",
    )
    _create_trace(
        store,
        trace2_id,
        exp_id,
        tags={"special": "user#admin"},
        client_request_id="req-456",
    )
    _create_trace(
        store,
        trace3_id,
        exp_id,
        tags={"special": "path/to/file"},
        client_request_id="req-789",
    )

    # Test: LIKE with @ character
    traces, _ = store.search_traces([exp_id], filter_string='tag.special LIKE "%@%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: LIKE with # character
    traces, _ = store.search_traces([exp_id], filter_string='tag.special LIKE "%#%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: LIKE with / character
    traces, _ = store.search_traces([exp_id], filter_string='tag.special LIKE "%/%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: ILIKE case-insensitive with special chars
    traces, _ = store.search_traces([exp_id], filter_string='tag.special ILIKE "%ADMIN%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id


def test_search_traces_with_timestamp_ms_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_timestamp_ms")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    base_time = 1000000  # Use a fixed base time for consistency

    _create_trace(store, trace1_id, exp_id, request_time=base_time)
    _create_trace(store, trace2_id, exp_id, request_time=base_time + 5000)
    _create_trace(store, trace3_id, exp_id, request_time=base_time + 10000)
    _create_trace(store, trace4_id, exp_id, request_time=base_time + 15000)

    # Test: = (equals)
    traces, _ = store.search_traces([exp_id], filter_string=f"trace.timestamp_ms = {base_time}")
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: != (not equals)
    traces, _ = store.search_traces([exp_id], filter_string=f"trace.timestamp_ms != {base_time}")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id, trace4_id}

    # Test: > (greater than)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.timestamp_ms > {base_time + 5000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id}

    # Test: >= (greater than or equal)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.timestamp_ms >= {base_time + 5000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id, trace4_id}

    # Test: < (less than)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.timestamp_ms < {base_time + 10000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: <= (less than or equal)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.timestamp_ms <= {base_time + 10000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}

    # Test: Combined conditions (range query)
    traces, _ = store.search_traces(
        [exp_id],
        filter_string=f"trace.timestamp_ms >= {base_time + 5000} "
        f"AND trace.timestamp_ms < {base_time + 15000}",
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}


def test_search_traces_with_execution_time_ms_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_execution_time_ms")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"
    trace5_id = "trace5"

    base_time = 1000000

    # Create traces with different execution times
    _create_trace(store, trace1_id, exp_id, request_time=base_time, execution_duration=100)
    _create_trace(store, trace2_id, exp_id, request_time=base_time, execution_duration=500)
    _create_trace(store, trace3_id, exp_id, request_time=base_time, execution_duration=1000)
    _create_trace(store, trace4_id, exp_id, request_time=base_time, execution_duration=2000)
    _create_trace(store, trace5_id, exp_id, request_time=base_time, execution_duration=5000)

    # Test: = (equals)
    traces, _ = store.search_traces([exp_id], filter_string="trace.execution_time_ms = 1000")
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: != (not equals)
    traces, _ = store.search_traces([exp_id], filter_string="trace.execution_time_ms != 1000")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace4_id, trace5_id}

    # Test: > (greater than)
    traces, _ = store.search_traces([exp_id], filter_string="trace.execution_time_ms > 1000")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace4_id, trace5_id}

    # Test: >= (greater than or equal)
    traces, _ = store.search_traces([exp_id], filter_string="trace.execution_time_ms >= 1000")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id, trace5_id}

    # Test: < (less than)
    traces, _ = store.search_traces([exp_id], filter_string="trace.execution_time_ms < 1000")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: <= (less than or equal)
    traces, _ = store.search_traces([exp_id], filter_string="trace.execution_time_ms <= 1000")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}

    # Test: Combined conditions (find traces with execution time between 500ms and 2000ms)
    traces, _ = store.search_traces(
        [exp_id],
        filter_string="trace.execution_time_ms >= 500 AND trace.execution_time_ms <= 2000",
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id, trace4_id}


def test_search_traces_with_end_time_ms_all_operators(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_end_time_ms_all_ops")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    base_time = 1000000

    # end_time_ms = timestamp_ms + execution_time_ms
    # trace1: starts at base_time, runs 1000ms -> ends at base_time + 1000
    # trace2: starts at base_time, runs 3000ms -> ends at base_time + 3000
    # trace3: starts at base_time, runs 5000ms -> ends at base_time + 5000
    # trace4: starts at base_time, runs 10000ms -> ends at base_time + 10000
    _create_trace(store, trace1_id, exp_id, request_time=base_time, execution_duration=1000)
    _create_trace(store, trace2_id, exp_id, request_time=base_time, execution_duration=3000)
    _create_trace(store, trace3_id, exp_id, request_time=base_time, execution_duration=5000)
    _create_trace(store, trace4_id, exp_id, request_time=base_time, execution_duration=10000)

    # Test: = (equals)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.end_time_ms = {base_time + 3000}"
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: != (not equals)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.end_time_ms != {base_time + 3000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id, trace4_id}

    # Test: > (greater than)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.end_time_ms > {base_time + 3000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id}

    # Test: >= (greater than or equal)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.end_time_ms >= {base_time + 3000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id, trace4_id}

    # Test: < (less than)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.end_time_ms < {base_time + 5000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: <= (less than or equal)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.end_time_ms <= {base_time + 5000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}

    # Test: Combined conditions (range query)
    traces, _ = store.search_traces(
        [exp_id],
        filter_string=f"trace.end_time_ms > {base_time + 1000} "
        f"AND trace.end_time_ms < {base_time + 10000}",
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}


def test_search_traces_with_status_operators(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_status_operators")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    # Create traces with different statuses
    _create_trace(store, trace1_id, exp_id, state=TraceState.OK)
    _create_trace(store, trace2_id, exp_id, state=TraceState.OK)
    _create_trace(store, trace3_id, exp_id, state=TraceState.ERROR)
    _create_trace(store, trace4_id, exp_id, state=TraceState.IN_PROGRESS)

    # Test: = (equals) for OK status
    traces, _ = store.search_traces([exp_id], filter_string="trace.status = 'OK'")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: = (equals) for ERROR status
    traces, _ = store.search_traces([exp_id], filter_string="trace.status = 'ERROR'")
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: != (not equals)
    traces, _ = store.search_traces([exp_id], filter_string="trace.status != 'OK'")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id}

    # Test: LIKE operator
    traces, _ = store.search_traces([exp_id], filter_string="trace.status LIKE 'OK'")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: ILIKE operator
    traces, _ = store.search_traces([exp_id], filter_string="trace.status ILIKE 'error'")
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: Using different aliases (attributes.status and status)
    traces, _ = store.search_traces([exp_id], filter_string="attributes.status = 'OK'")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    traces, _ = store.search_traces([exp_id], filter_string="status = 'IN_PROGRESS'")
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id


def test_search_traces_with_combined_numeric_and_string_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_combined_numeric_string")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    base_time = 1000000

    _create_trace(
        store,
        trace1_id,
        exp_id,
        request_time=base_time,
        execution_duration=100,
        tags={TraceTagKey.TRACE_NAME: "FastQuery"},
        state=TraceState.OK,
    )
    _create_trace(
        store,
        trace2_id,
        exp_id,
        request_time=base_time + 1000,
        execution_duration=500,
        tags={TraceTagKey.TRACE_NAME: "SlowQuery"},
        state=TraceState.OK,
    )
    _create_trace(
        store,
        trace3_id,
        exp_id,
        request_time=base_time + 2000,
        execution_duration=2000,
        tags={TraceTagKey.TRACE_NAME: "FastQuery"},
        state=TraceState.ERROR,
    )
    _create_trace(
        store,
        trace4_id,
        exp_id,
        request_time=base_time + 3000,
        execution_duration=5000,
        tags={TraceTagKey.TRACE_NAME: "SlowQuery"},
        state=TraceState.ERROR,
    )

    # Test: Fast queries (execution time < 1000ms) with OK status
    traces, _ = store.search_traces(
        [exp_id],
        filter_string="trace.execution_time_ms < 1000 AND trace.status = 'OK'",
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: Slow queries (execution time >= 2000ms)
    traces, _ = store.search_traces(
        [exp_id],
        filter_string="trace.execution_time_ms >= 2000",
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id}

    # Test: Traces that started after base_time + 1000 with ERROR status
    traces, _ = store.search_traces(
        [exp_id],
        filter_string=f"trace.timestamp_ms > {base_time + 1000} AND trace.status = 'ERROR'",
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id}

    # Test: FastQuery traces with execution time < 500ms
    traces, _ = store.search_traces(
        [exp_id],
        filter_string='trace.name = "FastQuery" AND trace.execution_time_ms < 500',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: Complex query with time range and name pattern
    traces, _ = store.search_traces(
        [exp_id],
        filter_string=(
            f"trace.timestamp_ms >= {base_time} "
            f"AND trace.timestamp_ms <= {base_time + 2000} "
            'AND trace.name LIKE "%Query%"'
        ),
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}


def test_search_traces_with_prompts_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_prompts_exact")

    # Create traces with different linked prompts
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    # Trace 1: linked to qa-agent-system-prompt version 4
    _create_trace(store, trace1_id, exp_id)
    store.link_prompts_to_trace(
        trace1_id, [PromptVersion(name="qa-agent-system-prompt", version=4, template="")]
    )

    # Trace 2: linked to qa-agent-system-prompt version 5
    _create_trace(store, trace2_id, exp_id)
    store.link_prompts_to_trace(
        trace2_id, [PromptVersion(name="qa-agent-system-prompt", version=5, template="")]
    )

    # Trace 3: linked to chat-assistant-prompt version 1
    _create_trace(store, trace3_id, exp_id)
    store.link_prompts_to_trace(
        trace3_id, [PromptVersion(name="chat-assistant-prompt", version=1, template="")]
    )

    # Trace 4: linked to multiple prompts
    _create_trace(store, trace4_id, exp_id)
    store.link_prompts_to_trace(
        trace4_id,
        [
            PromptVersion(name="qa-agent-system-prompt", version=4, template=""),
            PromptVersion(name="chat-assistant-prompt", version=2, template=""),
        ],
    )

    # Test: Filter by exact prompt name/version
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "qa-agent-system-prompt/4"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace4_id}

    # Test: Filter by another exact prompt name/version
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "qa-agent-system-prompt/5"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: Filter by chat assistant prompt
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "chat-assistant-prompt/1"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: Filter by prompt that appears in multiple trace
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "chat-assistant-prompt/2"')
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id


@pytest.mark.parametrize(
    ("comparator", "filter_string"),
    [
        ("LIKE", 'prompt LIKE "%qa-agent%"'),
        ("ILIKE", 'prompt ILIKE "%CHAT%"'),
        ("RLIKE", 'prompt RLIKE "version.*1"'),
        ("!=", 'prompt != "test/1"'),
    ],
)
def test_search_traces_with_prompts_filter_invalid_comparator(
    store: SqlAlchemyStore, comparator: str, filter_string: str
):
    exp_id = store.create_experiment("test_prompts_invalid")

    with pytest.raises(
        MlflowException,
        match=f"Invalid comparator '{comparator}' for prompts filter. "
        "Only '=' is supported with format: prompt = \"name/version\"",
    ):
        store.search_traces([exp_id], filter_string=filter_string)


@pytest.mark.parametrize(
    ("filter_string", "invalid_value"),
    [
        ('prompt = "qa-agent-system-prompt"', "qa-agent-system-prompt"),
        ('prompt = "foo/1/baz"', "foo/1/baz"),
        ('prompt = ""', ""),
    ],
)
def test_search_traces_with_prompts_filter_invalid_format(
    store: SqlAlchemyStore, filter_string: str, invalid_value: str
):
    exp_id = store.create_experiment("test_prompts_invalid_format")

    with pytest.raises(
        MlflowException,
        match=f"Invalid prompts filter value '{invalid_value}'. "
        'Expected format: prompt = "name/version"',
    ):
        store.search_traces([exp_id], filter_string=filter_string)


def test_search_traces_with_prompts_filter_no_matches(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_prompts_no_match")

    # Create traces with linked prompts
    trace1_id = "trace1"
    _create_trace(store, trace1_id, exp_id)
    store.link_prompts_to_trace(
        trace1_id, [PromptVersion(name="qa-agent-system-prompt", version=4, template="")]
    )

    # Test: Filter by non-existent prompt
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "non-existent-prompt/999"')
    assert len(traces) == 0

    # Test: Filter by correct name but wrong version
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "qa-agent-system-prompt/999"')
    assert len(traces) == 0


def test_search_traces_with_prompts_filter_multiple_prompts(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_prompts_multiple")

    # Create traces with multiple linked prompts
    trace1_id = "trace1"
    trace2_id = "trace2"

    # Trace 1: Single prompt
    _create_trace(store, trace1_id, exp_id)
    store.link_prompts_to_trace(trace1_id, [PromptVersion(name="prompt-a", version=1, template="")])

    # Trace 2: Multiple prompts
    _create_trace(store, trace2_id, exp_id)
    store.link_prompts_to_trace(
        trace2_id,
        [
            PromptVersion(name="prompt-a", version=1, template=""),
            PromptVersion(name="prompt-b", version=2, template=""),
            PromptVersion(name="prompt-c", version=3, template=""),
        ],
    )

    # Test: Filter by first prompt - should match both
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "prompt-a/1"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: Filter by second prompt - should only match trace2
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "prompt-b/2"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: Filter by third prompt - should only match trace2
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "prompt-c/3"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id


def test_search_traces_with_span_attributute_backticks(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_attribute_backticks")
    trace_info_1 = _create_trace(store, "trace_1", exp_id)
    trace_info_2 = _create_trace(store, "trace_2", exp_id)

    span1 = create_mlflow_span(
        OTelReadableSpan(
            name="span_trace1",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={
                "mlflow.traceRequestId": json.dumps(trace_info_1.trace_id, cls=TraceJSONEncoder),
                "mlflow.experimentId": json.dumps(exp_id, cls=TraceJSONEncoder),
                "mlflow.spanInputs": json.dumps({"input": "test1"}, cls=TraceJSONEncoder),
            },
            start_time=1000000000,
            end_time=2000000000,
            resource=_OTelResource.get_empty(),
        ),
        trace_info_1.trace_id,
        "LLM",
    )

    span2 = create_mlflow_span(
        OTelReadableSpan(
            name="span_trace2",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={
                "mlflow.traceRequestId": json.dumps(trace_info_2.trace_id, cls=TraceJSONEncoder),
                "mlflow.experimentId": json.dumps(exp_id, cls=TraceJSONEncoder),
                "mlflow.spanInputs": json.dumps({"input": "test2"}, cls=TraceJSONEncoder),
            },
            start_time=1000000000,
            end_time=2000000000,
            resource=_OTelResource.get_empty(),
        ),
        trace_info_2.trace_id,
        "LLM",
    )

    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])

    traces, _ = store.search_traces([exp_id], filter_string='trace.text ILIKE "%test1%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace_info_1.trace_id

    traces, _ = store.search_traces([exp_id], filter_string='trace.text ILIKE "%test2%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace_info_2.trace_id


def test_set_and_delete_tags(store: SqlAlchemyStore):
    exp1 = store.create_experiment("exp1")
    trace_id = "tr-123"
    _create_trace(store, trace_id, experiment_id=exp1)

    # Delete system tag for easier testing
    store.delete_trace_tag(trace_id, MLFLOW_ARTIFACT_LOCATION)

    assert store.get_trace_info(trace_id).tags == {}

    store.set_trace_tag(trace_id, "tag1", "apple")
    assert store.get_trace_info(trace_id).tags == {"tag1": "apple"}

    store.set_trace_tag(trace_id, "tag1", "grape")
    assert store.get_trace_info(trace_id).tags == {"tag1": "grape"}

    store.set_trace_tag(trace_id, "tag2", "orange")
    assert store.get_trace_info(trace_id).tags == {"tag1": "grape", "tag2": "orange"}

    store.delete_trace_tag(trace_id, "tag1")
    assert store.get_trace_info(trace_id).tags == {"tag2": "orange"}

    # test value length
    store.set_trace_tag(trace_id, "key", "v" * MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE)
    assert store.get_trace_info(trace_id).tags["key"] == "v" * MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE

    with pytest.raises(MlflowException, match="No trace tag with key 'tag1'"):
        store.delete_trace_tag(trace_id, "tag1")


@pytest.mark.parametrize(
    ("key", "value", "expected_error"),
    [
        (None, "value", "Missing value for required parameter 'key'"),
        (
            "invalid?tag!name:(",
            "value",
            "Invalid value \"invalid\\?tag!name:\\(\" for parameter 'key' supplied",
        ),
        (
            "/.:\\.",
            "value",
            "Invalid value \"/\\.:\\\\\\\\.\" for parameter 'key' supplied",
        ),
        ("../", "value", "Invalid value \"\\.\\./\" for parameter 'key' supplied"),
        ("a" * 251, "value", "'key' exceeds the maximum length of 250 characters"),
    ],
    # Name each test case too avoid including the long string arguments in the test name
    ids=["null-key", "bad-key-1", "bad-key-2", "bad-key-3", "too-long-key"],
)
def test_set_invalid_tag(key, value, expected_error, store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match=expected_error):
        store.set_trace_tag("tr-123", key, value)


def test_set_tag_truncate_too_long_tag(store: SqlAlchemyStore):
    exp1 = store.create_experiment("exp1")
    trace_id = "tr-123"
    _create_trace(store, trace_id, experiment_id=exp1)

    store.set_trace_tag(trace_id, "key", "123" + "a" * 8000)
    tags = store.get_trace_info(trace_id).tags
    assert len(tags["key"]) == 8000
    assert tags["key"] == "123" + "a" * 7997


def test_delete_traces(store):
    exp1 = store.create_experiment("exp1")
    exp2 = store.create_experiment("exp2")
    now = int(time.time() * 1000)

    for i in range(10):
        _create_trace(
            store,
            f"tr-exp1-{i}",
            exp1,
            tags={"tag": "apple"},
            trace_metadata={"rq": "foo"},
        )
        _create_trace(
            store,
            f"tr-exp2-{i}",
            exp2,
            tags={"tag": "orange"},
            trace_metadata={"rq": "bar"},
        )

    traces, _ = store.search_traces([exp1, exp2])
    assert len(traces) == 20

    deleted = store.delete_traces(experiment_id=exp1, max_timestamp_millis=now)
    assert deleted == 10
    traces, _ = store.search_traces([exp1, exp2])
    assert len(traces) == 10
    for trace in traces:
        assert trace.experiment_id == exp2

    deleted = store.delete_traces(experiment_id=exp2, max_timestamp_millis=now)
    assert deleted == 10
    traces, _ = store.search_traces([exp1, exp2])
    assert len(traces) == 0

    deleted = store.delete_traces(experiment_id=exp1, max_timestamp_millis=now)
    assert deleted == 0


def test_delete_traces_with_max_timestamp(store):
    exp1 = store.create_experiment("exp1")
    for i in range(10):
        _create_trace(store, f"tr-{i}", exp1, request_time=i)

    deleted = store.delete_traces(exp1, max_timestamp_millis=3)
    assert deleted == 4  # inclusive (0, 1, 2, 3)
    traces, _ = store.search_traces([exp1])
    assert len(traces) == 6
    for trace in traces:
        assert trace.timestamp_ms >= 4

    deleted = store.delete_traces(exp1, max_timestamp_millis=10)
    assert deleted == 6
    traces, _ = store.search_traces([exp1])
    assert len(traces) == 0


def test_delete_traces_with_max_count(store):
    exp1 = store.create_experiment("exp1")
    for i in range(10):
        _create_trace(store, f"tr-{i}", exp1, request_time=i)

    deleted = store.delete_traces(exp1, max_traces=4, max_timestamp_millis=10)
    assert deleted == 4
    traces, _ = store.search_traces([exp1])
    assert len(traces) == 6
    # Traces should be deleted from the oldest
    for trace in traces:
        assert trace.timestamp_ms >= 4

    deleted = store.delete_traces(exp1, max_traces=10, max_timestamp_millis=8)
    assert deleted == 5
    traces, _ = store.search_traces([exp1])
    assert len(traces) == 1


def test_delete_traces_with_trace_ids(store):
    exp1 = store.create_experiment("exp1")
    for i in range(10):
        _create_trace(store, f"tr-{i}", exp1, request_time=i)

    deleted = store.delete_traces(exp1, trace_ids=[f"tr-{i}" for i in range(8)])
    assert deleted == 8
    traces, _ = store.search_traces([exp1])
    assert len(traces) == 2
    assert [trace.trace_id for trace in traces] == ["tr-9", "tr-8"]


def test_delete_traces_raises_error(store):
    exp_id = store.create_experiment("test")

    with pytest.raises(
        MlflowException,
        match=r"Either `max_timestamp_millis` or `trace_ids` must be specified.",
    ):
        store.delete_traces(exp_id)
    with pytest.raises(
        MlflowException,
        match=r"Only one of `max_timestamp_millis` and `trace_ids` can be specified.",
    ):
        store.delete_traces(exp_id, max_timestamp_millis=100, trace_ids=["trace_id"])
    with pytest.raises(
        MlflowException,
        match=r"`max_traces` can't be specified if `trace_ids` is specified.",
    ):
        store.delete_traces(exp_id, max_traces=2, trace_ids=["trace_id"])
    with pytest.raises(
        MlflowException, match=r"`max_traces` must be a positive integer, received 0"
    ):
        store.delete_traces(exp_id, 100, max_traces=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans(store: SqlAlchemyStore, is_async: bool):
    # Create an experiment and trace first
    experiment_id = store.create_experiment("test_span_experiment")
    trace_info = TraceInfo(
        trace_id="tr-span-test-123",
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1234,
        execution_duration=100,
        state=TraceState.OK,
    )
    trace_info = store.start_trace(trace_info)

    # Create a mock OpenTelemetry span

    # Create mock context
    mock_context = mock.Mock()
    mock_context.trace_id = 12345
    mock_context.span_id = 222 if not is_async else 333
    mock_context.is_remote = False
    mock_context.trace_flags = trace_api.TraceFlags(1)
    mock_context.trace_state = trace_api.TraceState()  # Empty TraceState

    parent_mock_context = mock.Mock()
    parent_mock_context.trace_id = 12345
    parent_mock_context.span_id = 111
    parent_mock_context.is_remote = False
    parent_mock_context.trace_flags = trace_api.TraceFlags(1)
    parent_mock_context.trace_state = trace_api.TraceState()  # Empty TraceState

    readable_span = OTelReadableSpan(
        name="test_span",
        context=mock_context,
        parent=parent_mock_context if not is_async else None,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_info.trace_id),
            "mlflow.spanInputs": json.dumps({"input": "test_input"}, cls=TraceJSONEncoder),
            "mlflow.spanOutputs": json.dumps({"output": "test_output"}, cls=TraceJSONEncoder),
            "mlflow.spanType": json.dumps("LLM" if not is_async else "CHAIN", cls=TraceJSONEncoder),
            "custom_attr": json.dumps("custom_value", cls=TraceJSONEncoder),
        },
        start_time=1000000000 if not is_async else 3000000000,
        end_time=2000000000 if not is_async else 4000000000,
        resource=_OTelResource.get_empty(),
    )

    # Create MLflow span from OpenTelemetry span
    span = create_mlflow_span(readable_span, trace_info.trace_id, "LLM")
    assert isinstance(span, Span)

    # Test logging the span using sync or async method
    if is_async:
        logged_spans = await store.log_spans_async(experiment_id, [span])
    else:
        logged_spans = store.log_spans(experiment_id, [span])

    # Verify the returned spans are the same
    assert len(logged_spans) == 1
    assert logged_spans[0] == span
    assert logged_spans[0].trace_id == trace_info.trace_id
    assert logged_spans[0].span_id == span.span_id

    # Verify the span was saved to the database
    with store.ManagedSessionMaker() as session:
        saved_span = (
            session
            .query(SqlSpan)
            .filter(SqlSpan.trace_id == trace_info.trace_id, SqlSpan.span_id == span.span_id)
            .first()
        )

        assert saved_span is not None
        assert saved_span.experiment_id == int(experiment_id)
        assert saved_span.parent_span_id == span.parent_id
        assert saved_span.status == "UNSET"  # Default OpenTelemetry status
        assert saved_span.status == span.status.status_code
        assert saved_span.start_time_unix_nano == span.start_time_ns
        assert saved_span.end_time_unix_nano == span.end_time_ns
        # Check the computed duration
        assert saved_span.duration_ns == (span.end_time_ns - span.start_time_ns)

        # Verify the content is properly serialized
        content_dict = json.loads(saved_span.content)
        assert content_dict["name"] == "test_span"
        # Inputs and outputs are stored in attributes as strings
        assert content_dict["attributes"]["mlflow.spanInputs"] == json.dumps(
            {"input": "test_input"}, cls=TraceJSONEncoder
        )
        assert content_dict["attributes"]["mlflow.spanOutputs"] == json.dumps(
            {"output": "test_output"}, cls=TraceJSONEncoder
        )
        expected_type = "LLM" if not is_async else "CHAIN"
        assert content_dict["attributes"]["mlflow.spanType"] == json.dumps(
            expected_type, cls=TraceJSONEncoder
        )


def test_log_spans_multiple_traces(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_multi_trace_experiment")

    span1 = create_mlflow_span(
        OTelReadableSpan(
            name="span_trace1",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps("tr-multi-1", cls=TraceJSONEncoder)},
            start_time=1000000000,
            end_time=2000000000,
            resource=_OTelResource.get_empty(),
        ),
        "tr-multi-1",
    )

    span2 = create_mlflow_span(
        OTelReadableSpan(
            name="span_trace2",
            context=trace_api.SpanContext(
                trace_id=67890,
                span_id=222,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps("tr-multi-2", cls=TraceJSONEncoder)},
            start_time=3000000000,
            end_time=4000000000,
            resource=_OTelResource.get_empty(),
        ),
        "tr-multi-2",
    )

    # Multi-trace log_spans should succeed in a single call
    result = store.log_spans(experiment_id, [span1, span2])
    assert len(result) == 2

    # Verify both traces were created with correct spans in the database
    with store.ManagedSessionMaker() as session:
        trace1 = session.query(SqlTraceInfo).filter_by(request_id="tr-multi-1").one()
        assert trace1.experiment_id == int(experiment_id)

        trace2 = session.query(SqlTraceInfo).filter_by(request_id="tr-multi-2").one()
        assert trace2.experiment_id == int(experiment_id)

        span_row1 = session.query(SqlSpan).filter_by(trace_id="tr-multi-1").one()
        assert span_row1.name == "span_trace1"

        span_row2 = session.query(SqlSpan).filter_by(trace_id="tr-multi-2").one()
        assert span_row2.name == "span_trace2"


def test_log_spans_persists_links(store: SqlAlchemyStore):
    trace_id = "tr-links-test"
    experiment_id = store.create_experiment("test_links_experiment")

    span = create_test_span(
        trace_id=trace_id,
        links=[
            Link(trace_id="tr-abc123", span_id="aabbccddeeff0011", attributes={"type": "causal"}),
            Link(trace_id="tr-def456", span_id="1122334455667788"),
        ],
    )

    store.log_spans(experiment_id, [span])

    # Verify links survive the full round-trip via get_trace
    trace = store.get_trace(trace_id)
    retrieved_span = trace.data.spans[0]
    assert len(retrieved_span.links) == 2
    assert retrieved_span.links[0].trace_id == "tr-abc123"
    assert retrieved_span.links[0].span_id == "aabbccddeeff0011"
    assert retrieved_span.links[0].attributes == {"type": "causal"}
    assert retrieved_span.links[1].trace_id == "tr-def456"
    assert retrieved_span.links[1].span_id == "1122334455667788"
    assert retrieved_span.links[1].attributes is None


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_creates_trace_if_not_exists(store: SqlAlchemyStore, is_async: bool):
    # Create an experiment but no trace
    experiment_id = store.create_experiment("test_auto_trace_experiment")

    # Create a span without a pre-existing trace
    trace_id = "tr-auto-created-trace"
    readable_span = OTelReadableSpan(
        name="auto_trace_span",
        context=trace_api.SpanContext(
            trace_id=98765,
            span_id=555,
            is_remote=False,
            trace_flags=trace_api.TraceFlags(1),
        ),
        parent=None,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
            "mlflow.experimentId": json.dumps(experiment_id, cls=TraceJSONEncoder),
        },
        start_time=5000000000,
        end_time=6000000000,
        resource=_OTelResource.get_empty(),
    )

    span = create_mlflow_span(readable_span, trace_id)

    # Log the span - should create the trace automatically
    if is_async:
        logged_spans = await store.log_spans_async(experiment_id, [span])
    else:
        logged_spans = store.log_spans(experiment_id, [span])

    assert len(logged_spans) == 1
    assert logged_spans[0] == span

    # Verify the trace was created
    with store.ManagedSessionMaker() as session:
        created_trace = (
            session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).first()
        )

        assert created_trace is not None
        assert created_trace.experiment_id == int(experiment_id)
        assert created_trace.timestamp_ms == 5000000000 // 1_000_000
        assert created_trace.execution_time_ms == 1000000000 // 1_000_000
        # When root span status is UNSET (unexpected), we assume trace status is OK
        assert created_trace.status == "OK"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_empty_list(store: SqlAlchemyStore, is_async: bool):
    experiment_id = store.create_experiment("test_empty_experiment")

    if is_async:
        result = await store.log_spans_async(experiment_id, [])
    else:
        result = store.log_spans(experiment_id, [])
    assert result == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_concurrent_trace_creation(store: SqlAlchemyStore, is_async: bool):
    # Create an experiment
    experiment_id = store.create_experiment("test_concurrent_trace")
    trace_id = "tr-concurrent-test"

    # Create a span
    readable_span = OTelReadableSpan(
        name="concurrent_span",
        context=trace_api.SpanContext(
            trace_id=12345,
            span_id=999,
            is_remote=False,
            trace_flags=trace_api.TraceFlags(1),
        ),
        parent=None,
        resource=_OTelResource.get_empty(),
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        },
        start_time=1000000000,
        end_time=2000000000,
        status=trace_api.Status(trace_api.StatusCode.OK),
        events=[],
        links=[],
    )

    span = create_mlflow_span(readable_span, trace_id)

    # Simulate a race condition where flush() raises IntegrityError
    # This tests that the code properly handles concurrent trace creation
    original_flush = None
    call_count = 0

    def mock_flush(self):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call to flush (for trace creation) raises IntegrityError
            raise IntegrityError("UNIQUE constraint failed", None, None)
        else:
            # Subsequent calls work normally
            return original_flush()

    with store.ManagedSessionMaker() as session:
        original_flush = session.flush
        with mock.patch.object(session, "flush", mock_flush):
            # This should handle the IntegrityError and still succeed
            if is_async:
                result = await store.log_spans_async(experiment_id, [span])
            else:
                result = store.log_spans(experiment_id, [span])

    # Verify the span was logged successfully despite the race condition
    assert len(result) == 1
    assert result[0] == span

    # Verify the trace and span exist in the database
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.experiment_id == int(experiment_id)

        saved_span = (
            session
            .query(SqlSpan)
            .filter(SqlSpan.trace_id == trace_id, SqlSpan.span_id == span.span_id)
            .one()
        )
        assert saved_span is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_updates_trace_time_range(store: SqlAlchemyStore, is_async: bool):
    experiment_id = _create_experiments(store, "test_log_spans_updates_trace")
    trace_id = "tr-time-update-test-123"

    # Create first span from 1s to 2s
    span1 = create_mlflow_span(
        OTelReadableSpan(
            name="early_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=1_000_000_000,  # 1 second in nanoseconds
            end_time=2_000_000_000,  # 2 seconds
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log first span - creates trace with 1s start, 1s duration
    if is_async:
        await store.log_spans_async(experiment_id, [span1])
    else:
        store.log_spans(experiment_id, [span1])

    # Verify initial trace times
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 1_000  # 1 second
        assert trace.execution_time_ms == 1_000  # 1 second duration

    # Create second span that starts earlier (0.5s) and ends later (3s)
    span2 = create_mlflow_span(
        OTelReadableSpan(
            name="extended_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=222,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=500_000_000,  # 0.5 seconds
            end_time=3_000_000_000,  # 3 seconds
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log second span - should update trace to 0.5s start, 2.5s duration
    if is_async:
        await store.log_spans_async(experiment_id, [span2])
    else:
        store.log_spans(experiment_id, [span2])

    # Verify trace times were updated
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 500  # 0.5 seconds (earlier start)
        assert trace.execution_time_ms == 2_500  # 2.5 seconds duration (0.5s to 3s)

    # Create third span that only extends the end time (2.5s to 4s)
    span3 = create_mlflow_span(
        OTelReadableSpan(
            name="later_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=333,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=2_500_000_000,  # 2.5 seconds
            end_time=4_000_000_000,  # 4 seconds
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log third span - should only update end time
    if is_async:
        await store.log_spans_async(experiment_id, [span3])
    else:
        store.log_spans(experiment_id, [span3])

    # Verify trace times were updated again
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 500  # Still 0.5 seconds (no earlier start)
        assert trace.execution_time_ms == 3_500  # 3.5 seconds duration (0.5s to 4s)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_no_end_time(store: SqlAlchemyStore, is_async: bool):
    experiment_id = _create_experiments(store, "test_log_spans_no_end_time")
    trace_id = "tr-no-end-time-test-123"

    # Create span without end time (in-progress span)
    span1 = create_mlflow_span(
        OTelReadableSpan(
            name="in_progress_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=1_000_000_000,  # 1 second in nanoseconds
            end_time=None,  # No end time - span still in progress
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log span with no end time
    if is_async:
        await store.log_spans_async(experiment_id, [span1])
    else:
        store.log_spans(experiment_id, [span1])

    # Verify trace has timestamp but no execution_time
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 1_000  # 1 second
        assert trace.execution_time_ms is None  # No execution time since span not ended

    # Add a second span that also has no end time
    span2 = create_mlflow_span(
        OTelReadableSpan(
            name="another_in_progress_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=222,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=500_000_000,  # 0.5 seconds - earlier start
            end_time=None,  # No end time
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log second span with no end time
    if is_async:
        await store.log_spans_async(experiment_id, [span2])
    else:
        store.log_spans(experiment_id, [span2])

    # Verify trace timestamp updated but execution_time still None
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 500  # Updated to earlier time
        assert trace.execution_time_ms is None  # Still no execution time

    # Now add a span with an end time
    span3 = create_mlflow_span(
        OTelReadableSpan(
            name="completed_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=333,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=2_000_000_000,  # 2 seconds
            end_time=3_000_000_000,  # 3 seconds
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log span with end time
    if is_async:
        await store.log_spans_async(experiment_id, [span3])
    else:
        store.log_spans(experiment_id, [span3])

    # Verify trace now has execution_time
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 500  # Still earliest start
        assert trace.execution_time_ms == 2_500  # 3s - 0.5s = 2.5s


def test_create_logged_model(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    model = store.create_logged_model(experiment_id=exp_id)
    assert model.experiment_id == exp_id
    assert model.name is not None
    assert model.metrics is None
    assert model.tags == {}
    assert model.params == {}

    # name
    model = store.create_logged_model(experiment_id=exp_id, name="my_model")
    assert model.name == "my_model"

    # source_run_id
    run = store.create_run(
        experiment_id=exp_id, user_id="user", start_time=0, run_name="test", tags=[]
    )
    model = store.create_logged_model(experiment_id=exp_id, source_run_id=run.info.run_id)
    assert model.source_run_id == run.info.run_id

    # model_type
    model = store.create_logged_model(experiment_id=exp_id, model_type="my_model_type")
    assert model.model_type == "my_model_type"

    # tags
    model = store.create_logged_model(
        experiment_id=exp_id,
        name="my_model",
        tags=[LoggedModelTag("tag1", "apple")],
    )
    assert model.tags == {"tag1": "apple"}

    # params
    model = store.create_logged_model(
        experiment_id=exp_id,
        name="my_model",
        params=[LoggedModelParameter("param1", "apple")],
    )
    assert model.params == {"param1": "apple"}

    # Should not be able to create a logged model in a non-active experiment
    store.delete_experiment(exp_id)
    with pytest.raises(MlflowException, match="must be in the 'active' state"):
        store.create_logged_model(experiment_id=exp_id)


def test_log_logged_model_params(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    model = store.create_logged_model(experiment_id=exp_id)
    assert not model.params
    store.log_logged_model_params(
        model_id=model.model_id, params=[LoggedModelParameter("param1", "apple")]
    )
    loaded_model = store.get_logged_model(model_id=model.model_id)
    assert loaded_model.params == {"param1": "apple"}


def test_log_model_metrics_use_run_experiment_id(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    run = store.create_run(exp_id, "user", 0, [], "test_run")
    model = store.create_logged_model(experiment_id=exp_id, source_run_id=run.info.run_id)

    metric = Metric(
        key="metric",
        value=1.0,
        timestamp=get_current_time_millis(),
        step=0,
        model_id=model.model_id,
        run_id=run.info.run_id,
    )

    store.log_metric(run.info.run_id, metric)

    with store.ManagedSessionMaker() as session:
        logged_metrics = (
            session
            .query(SqlLoggedModelMetric)
            .filter(SqlLoggedModelMetric.model_id == model.model_id)
            .all()
        )
        assert len(logged_metrics) == 1
        assert logged_metrics[0].experiment_id == int(exp_id)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "my/model",
        "my.model",
        "my:model",
        "my%model",
        "my'model",
        'my"model',
    ],
)
def test_create_logged_model_invalid_name(store: SqlAlchemyStore, name: str):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    with pytest.raises(MlflowException, match="Invalid model name"):
        store.create_logged_model(exp_id, name=name)


def test_get_logged_model(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    model = store.create_logged_model(experiment_id=exp_id)
    fetched_model = store.get_logged_model(model.model_id)
    assert fetched_model.name == model.name
    assert fetched_model.model_id == model.model_id

    with pytest.raises(MlflowException, match="not found"):
        store.get_logged_model("does-not-exist")


def test_delete_logged_model(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    run = store.create_run(exp_id, "user", 0, [], "test_run")
    model = store.create_logged_model(experiment_id=exp_id, source_run_id=run.info.run_id)
    metric = Metric(
        key="metric",
        value=0,
        timestamp=0,
        step=0,
        model_id=model.model_id,
        run_id=run.info.run_id,
    )
    store.log_metric(run.info.run_id, metric)
    store.delete_logged_model(model.model_id)
    with pytest.raises(MlflowException, match="not found"):
        store.get_logged_model(model.model_id)

    models = store.search_logged_models(experiment_ids=[exp_id])
    assert len(models) == 0


def test_delete_run_does_not_delete_logged_model(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    run = store.create_run(exp_id, "user", 0, [], "run")
    model = store.create_logged_model(experiment_id=exp_id, source_run_id=run.info.run_id)
    store.delete_run(run.info.run_id)
    retrieved = store.get_logged_model(model.model_id)
    assert retrieved.model_id == model.model_id


def test_hard_delete_logged_model(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    model = store.create_logged_model(experiment_id=exp_id)
    store.delete_logged_model(model.model_id)
    store._hard_delete_logged_model(model.model_id)
    with store.ManagedSessionMaker() as session:
        actual_model = (
            session.query(models.SqlLoggedModel).filter_by(model_id=model.model_id).first()
        )
        assert actual_model is None


def test_get_deleted_logged_models(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    model = store.create_logged_model(experiment_id=exp_id)
    assert store._get_deleted_logged_models() == []
    store.delete_logged_model(model.model_id)
    assert store._get_deleted_logged_models(older_than=1000000) == []
    assert store._get_deleted_logged_models() == [model.model_id]


def test_finalize_logged_model(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    model = store.create_logged_model(experiment_id=exp_id)
    store.finalize_logged_model(model.model_id, status=LoggedModelStatus.READY)
    assert store.get_logged_model(model.model_id).status == LoggedModelStatus.READY

    store.finalize_logged_model(model.model_id, status=LoggedModelStatus.FAILED)
    assert store.get_logged_model(model.model_id).status == LoggedModelStatus.FAILED

    with pytest.raises(MlflowException, match="not found"):
        store.finalize_logged_model("does-not-exist", status=LoggedModelStatus.READY)


def test_set_logged_model_tags(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    model = store.create_logged_model(experiment_id=exp_id)
    store.set_logged_model_tags(model.model_id, [LoggedModelTag("tag1", "apple")])
    assert store.get_logged_model(model.model_id).tags == {"tag1": "apple"}

    # New tag
    store.set_logged_model_tags(model.model_id, [LoggedModelTag("tag2", "orange")])
    assert store.get_logged_model(model.model_id).tags == {"tag1": "apple", "tag2": "orange"}

    # Exieting tag
    store.set_logged_model_tags(model.model_id, [LoggedModelTag("tag2", "grape")])
    assert store.get_logged_model(model.model_id).tags == {"tag1": "apple", "tag2": "grape"}

    with pytest.raises(MlflowException, match="not found"):
        store.set_logged_model_tags("does-not-exist", [LoggedModelTag("tag1", "apple")])

    # Multiple tags
    store.set_logged_model_tags(
        model.model_id, [LoggedModelTag("tag3", "val3"), LoggedModelTag("tag4", "val4")]
    )
    assert store.get_logged_model(model.model_id).tags == {
        "tag1": "apple",
        "tag2": "grape",
        "tag3": "val3",
        "tag4": "val4",
    }


def test_delete_logged_model_tag(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    model = store.create_logged_model(experiment_id=exp_id)
    store.set_logged_model_tags(model.model_id, [LoggedModelTag("tag1", "apple")])
    store.delete_logged_model_tag(model.model_id, "tag1")
    assert store.get_logged_model(model.model_id).tags == {}

    with pytest.raises(MlflowException, match="not found"):
        store.delete_logged_model_tag("does-not-exist", "tag1")

    with pytest.raises(MlflowException, match="No tag with key"):
        store.delete_logged_model_tag(model.model_id, "tag1")


def test_search_logged_models(store: SqlAlchemyStore):
    exp_id_1 = store.create_experiment(f"exp-{uuid.uuid4()}")

    model_1 = store.create_logged_model(experiment_id=exp_id_1)
    time.sleep(0.001)  # Ensure the next model has a different timestamp
    models = store.search_logged_models(experiment_ids=[exp_id_1])
    assert [m.name for m in models] == [model_1.name]

    model_2 = store.create_logged_model(experiment_id=exp_id_1)
    time.sleep(0.001)
    models = store.search_logged_models(experiment_ids=[exp_id_1])
    assert [m.name for m in models] == [model_2.name, model_1.name]

    exp_id_2 = store.create_experiment(f"exp-{uuid.uuid4()}")
    model_3 = store.create_logged_model(experiment_id=exp_id_2)
    models = store.search_logged_models(experiment_ids=[exp_id_2])
    assert [m.name for m in models] == [model_3.name]

    models = store.search_logged_models(experiment_ids=[exp_id_1, exp_id_2])
    assert [m.name for m in models] == [model_3.name, model_2.name, model_1.name]


def test_search_logged_models_filter_string(store: SqlAlchemyStore):
    exp_id_1 = store.create_experiment(f"exp-{uuid.uuid4()}")
    model_1 = store.create_logged_model(experiment_id=exp_id_1)
    time.sleep(0.001)  # Ensure the next model has a different timestamp
    models = store.search_logged_models(experiment_ids=[exp_id_1])

    # Search by string attribute
    models = store.search_logged_models(
        experiment_ids=[exp_id_1],
        filter_string=f"name = '{model_1.name}'",
    )
    assert [m.name for m in models] == [model_1.name]
    assert models.token is None

    models = store.search_logged_models(
        experiment_ids=[exp_id_1],
        filter_string=f"attributes.name = '{model_1.name}'",
    )
    assert [m.name for m in models] == [model_1.name]
    assert models.token is None

    models = store.search_logged_models(
        experiment_ids=[exp_id_1],
        filter_string=f"name LIKE '{model_1.name[:3]}%'",
    )
    assert [m.name for m in models] == [model_1.name]
    assert models.token is None

    for val in (
        # A single item without a comma
        f"('{model_1.name}')",
        # A single item with a comma
        f"('{model_1.name}',)",
        # Multiple items
        f"('{model_1.name}', 'foo')",
    ):
        # IN
        models = store.search_logged_models(
            experiment_ids=[exp_id_1],
            filter_string=f"name IN {val}",
        )
        assert [m.name for m in models] == [model_1.name]
        assert models.token is None
        # NOT IN
        models = store.search_logged_models(
            experiment_ids=[exp_id_1],
            filter_string=f"name NOT IN {val}",
        )
        assert [m.name for m in models] == []

    # Search by numeric attribute
    models = store.search_logged_models(
        experiment_ids=[exp_id_1],
        filter_string="creation_timestamp > 0",
    )
    assert [m.name for m in models] == [model_1.name]
    assert models.token is None
    models = store.search_logged_models(
        experiment_ids=[exp_id_1],
        filter_string="creation_timestamp = 0",
    )
    assert models == []
    assert models.token is None

    # Search by param
    model_2 = store.create_logged_model(
        experiment_id=exp_id_1, params=[LoggedModelParameter("param1", "val1")]
    )
    time.sleep(0.001)
    models = store.search_logged_models(
        experiment_ids=[exp_id_1],
        filter_string="params.param1 = 'val1'",
    )
    assert [m.name for m in models] == [model_2.name]
    assert models.token is None

    # Search by tag
    model_3 = store.create_logged_model(
        experiment_id=exp_id_1, tags=[LoggedModelTag("tag1", "val1")]
    )
    time.sleep(0.001)
    models = store.search_logged_models(
        experiment_ids=[exp_id_1],
        filter_string="tags.tag1 = 'val1'",
    )
    assert [m.name for m in models] == [model_3.name]
    assert models.token is None

    # Search by metric
    model_4 = store.create_logged_model(experiment_id=exp_id_1)
    run = store.create_run(
        experiment_id=exp_id_1, user_id="user", start_time=0, run_name="test", tags=[]
    )
    store.log_batch(
        run.info.run_id,
        metrics=[
            Metric(
                key="metric",
                value=1,
                timestamp=int(time.time() * 1000),
                step=0,
                model_id=model_4.model_id,
                dataset_name="dataset_name",
                dataset_digest="dataset_digest",
                run_id=run.info.run_id,
            )
        ],
        params=[],
        tags=[],
    )
    time.sleep(0.001)
    models = store.search_logged_models(
        experiment_ids=[exp_id_1],
        filter_string="metrics.metric = 1",
    )
    assert [m.name for m in models] == [model_4.name]
    assert models.token is None

    models = store.search_logged_models(
        experiment_ids=[exp_id_1],
        filter_string="metrics.metric > 0.5",
    )
    assert [m.name for m in models] == [model_4.name]
    assert models.token is None

    models = store.search_logged_models(
        experiment_ids=[exp_id_1],
        filter_string="metrics.metric < 3",
    )
    assert [m.name for m in models] == [model_4.name]
    assert models.token is None

    # Search by multiple entities
    model_5 = store.create_logged_model(
        experiment_id=exp_id_1,
        params=[LoggedModelParameter("param2", "val2")],
        tags=[LoggedModelTag("tag2", "val2")],
    )
    time.sleep(0.001)
    models = store.search_logged_models(
        experiment_ids=[exp_id_1],
        filter_string="params.param2 = 'val2' AND tags.tag2 = 'val2'",
    )
    assert [m.name for m in models] == [model_5.name]
    assert models.token is None

    # Search by tag with key containing whitespace
    model_6 = store.create_logged_model(
        experiment_id=exp_id_1, tags=[LoggedModelTag("tag 3", "val3")]
    )
    time.sleep(0.001)
    models = store.search_logged_models(
        experiment_ids=[exp_id_1],
        filter_string="tags.`tag 3` = 'val3'",
    )
    assert [m.name for m in models] == [model_6.name]
    assert models.token is None

    # Pagination with filter_string
    first_page = store.search_logged_models(
        experiment_ids=[exp_id_1], max_results=2, filter_string="creation_timestamp > 0"
    )
    assert [m.name for m in first_page] == [model_6.name, model_5.name]
    assert first_page.token is not None
    second_page = store.search_logged_models(
        experiment_ids=[exp_id_1],
        filter_string="creation_timestamp > 0",
        page_token=first_page.token,
    )
    assert [m.name for m in second_page] == [model_4.name, model_3.name, model_2.name, model_1.name]
    assert second_page.token is None


def test_search_logged_models_invalid_filter_string(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    with pytest.raises(MlflowException, match="Invalid filter string"):
        store.search_logged_models(
            experiment_ids=[exp_id],
            filter_string="Foo",
        )

    with pytest.raises(MlflowException, match="Invalid filter string"):
        store.search_logged_models(
            experiment_ids=[exp_id],
            filter_string="name = 'foo' OR name = 'bar'",
        )

    with pytest.raises(MlflowException, match="Invalid entity type"):
        store.search_logged_models(
            experiment_ids=[exp_id],
            filter_string="foo.bar = 'a'",
        )

    with pytest.raises(MlflowException, match="Invalid comparison operator"):
        store.search_logged_models(
            experiment_ids=[exp_id],
            filter_string="name > 'foo'",
        )

    with pytest.raises(MlflowException, match="Invalid comparison operator"):
        store.search_logged_models(
            experiment_ids=[exp_id],
            filter_string="metrics.foo LIKE 0",
        )


def test_search_logged_models_order_by(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    model_1 = store.create_logged_model(name="model_1", experiment_id=exp_id)
    time.sleep(0.001)  # Ensure the next model has a different timestamp
    model_2 = store.create_logged_model(name="model_2", experiment_id=exp_id)
    time.sleep(0.001)  # Ensure the next model has a different timestamp
    run = store.create_run(
        experiment_id=exp_id, user_id="user", start_time=0, run_name="test", tags=[]
    )

    store.log_batch(
        run.info.run_id,
        metrics=[
            Metric(
                key="metric",
                value=1,
                timestamp=int(time.time() * 1000),
                step=0,
                model_id=model_1.model_id,
                dataset_name="dataset_name",
                dataset_digest="dataset_digest",
                run_id=run.info.run_id,
            ),
            Metric(
                key="metric",
                value=1,
                timestamp=int(time.time() * 1000),
                step=0,
                model_id=model_1.model_id,
                dataset_name="dataset_name",
                dataset_digest="dataset_digest",
                run_id=run.info.run_id,
            ),
            Metric(
                key="metric_2",
                value=1,
                timestamp=int(time.time() * 1000),
                step=0,
                model_id=model_1.model_id,
                dataset_name="dataset_name",
                dataset_digest="dataset_digest",
                run_id=run.info.run_id,
            ),
        ],
        params=[],
        tags=[],
    )
    store.log_batch(
        run.info.run_id,
        metrics=[
            Metric(
                key="metric",
                value=2,
                timestamp=int(time.time() * 1000),
                step=0,
                model_id=model_2.model_id,
                dataset_name="dataset_name",
                dataset_digest="dataset_digest",
                run_id=run.info.run_id,
            )
        ],
        params=[],
        tags=[],
    )

    # Should be sorted by creation time in descending order by default
    models = store.search_logged_models(experiment_ids=[exp_id])
    assert [m.name for m in models] == [model_2.name, model_1.name]

    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[{"field_name": "creation_timestamp", "ascending": True}],
    )
    assert [m.name for m in models] == [model_1.name, model_2.name]

    # Alias for creation_timestamp
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[{"field_name": "creation_time", "ascending": True}],
    )
    assert [m.name for m in models] == [model_1.name, model_2.name]

    # Sort by name
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[{"field_name": "name"}],
    )
    assert [m.name for m in models] == [model_1.name, model_2.name]

    # Sort by metric
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[{"field_name": "metrics.metric"}],
    )
    assert [m.name for m in models] == [model_1.name, model_2.name]

    # Sort by metric in descending order
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[{"field_name": "metrics.metric", "ascending": False}],
    )
    assert [m.name for m in models] == [model_2.name, model_1.name]

    # model 2 doesn't have metric_2, should be sorted last
    for ascending in (True, False):
        models = store.search_logged_models(
            experiment_ids=[exp_id],
            order_by=[{"field_name": "metrics.metric_2", "ascending": ascending}],
        )
        assert [m.name for m in models] == [model_1.name, model_2.name]


@dataclass
class DummyDataset:
    name: str
    digest: str


def test_search_logged_models_order_by_dataset(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    model_1 = store.create_logged_model(experiment_id=exp_id)
    time.sleep(0.001)  # Ensure the next model has a different timestamp
    model_2 = store.create_logged_model(experiment_id=exp_id)
    time.sleep(0.001)  # Ensure the next model has a different timestamp
    run = store.create_run(
        experiment_id=exp_id, user_id="user", start_time=0, run_name="test", tags=[]
    )
    dataset_1 = DummyDataset("dataset1", "digest1")
    dataset_2 = DummyDataset("dataset2", "digest2")

    # For dataset_1, model_1 has a higher accuracy
    # For dataset_2, model_2 has a higher accuracy
    store.log_batch(
        run.info.run_id,
        metrics=[
            Metric(
                key="accuracy",
                value=0.9,
                timestamp=1,
                step=0,
                model_id=model_1.model_id,
                dataset_name=dataset_1.name,
                dataset_digest=dataset_1.digest,
                run_id=run.info.run_id,
            ),
            Metric(
                key="accuracy",
                value=0.8,
                timestamp=2,
                step=0,
                model_id=model_1.model_id,
                dataset_name=dataset_2.name,
                dataset_digest=dataset_2.digest,
                run_id=run.info.run_id,
            ),
        ],
        params=[],
        tags=[],
    )
    store.log_batch(
        run.info.run_id,
        metrics=[
            Metric(
                key="accuracy",
                value=0.8,
                timestamp=3,
                step=0,
                model_id=model_2.model_id,
                dataset_name=dataset_1.name,
                dataset_digest=dataset_1.digest,
                run_id=run.info.run_id,
            ),
            Metric(
                key="accuracy",
                value=0.9,
                timestamp=4,
                step=0,
                model_id=model_2.model_id,
                dataset_name=dataset_2.name,
                dataset_digest=dataset_2.digest,
                run_id=run.info.run_id,
            ),
        ],
        params=[],
        tags=[],
    )

    # Sorted by accuracy for dataset_1
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[
            {
                "field_name": "metrics.accuracy",
                "dataset_name": dataset_1.name,
                "dataset_digest": dataset_1.digest,
            }
        ],
    )
    assert [m.name for m in models] == [model_2.name, model_1.name]

    # Sorted by accuracy for dataset_2
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[
            {
                "field_name": "metrics.accuracy",
                "dataset_name": dataset_2.name,
                "dataset_digest": dataset_2.digest,
            }
        ],
    )
    assert [m.name for m in models] == [model_1.name, model_2.name]

    # Sort by accuracy with only name
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[
            {
                "field_name": "metrics.accuracy",
                "dataset_name": dataset_1.name,
            }
        ],
    )
    assert [m.name for m in models] == [model_2.name, model_1.name]

    # Sort by accuracy with only digest
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[
            {
                "field_name": "metrics.accuracy",
                "dataset_digest": dataset_1.digest,
            }
        ],
    )
    assert [m.name for m in models] == [model_2.name, model_1.name]


def test_search_logged_models_pagination(store: SqlAlchemyStore):
    exp_id_1 = store.create_experiment(f"exp-{uuid.uuid4()}")

    model_1 = store.create_logged_model(experiment_id=exp_id_1)
    time.sleep(0.001)  # Ensure the next model has a different timestamp
    model_2 = store.create_logged_model(experiment_id=exp_id_1)

    page = store.search_logged_models(experiment_ids=[exp_id_1], max_results=3)
    assert [m.name for m in page] == [model_2.name, model_1.name]
    assert page.token is None

    page_1 = store.search_logged_models(experiment_ids=[exp_id_1], max_results=1)
    assert [m.name for m in page_1] == [model_2.name]
    assert page_1.token is not None

    page_2 = store.search_logged_models(
        experiment_ids=[exp_id_1], max_results=1, page_token=page_1.token
    )
    assert [m.name for m in page_2] == [model_1.name]
    assert page_2.token is None

    page_2 = store.search_logged_models(
        experiment_ids=[exp_id_1], max_results=100, page_token=page_1.token
    )
    assert [m.name for m in page_2] == [model_1.name]
    assert page_2.token is None

    # Search params must match the page token
    exp_id_2 = store.create_experiment(f"exp-{uuid.uuid4()}")
    with pytest.raises(MlflowException, match="Experiment IDs in the page token do not match"):
        store.search_logged_models(experiment_ids=[exp_id_2], page_token=page_1.token)

    with pytest.raises(MlflowException, match="Order by in the page token does not match"):
        store.search_logged_models(
            experiment_ids=[exp_id_1],
            order_by=[{"field_name": "creation_time"}],
            page_token=page_1.token,
        )

    with pytest.raises(MlflowException, match="Filter string in the page token does not match"):
        store.search_logged_models(
            experiment_ids=[exp_id_1],
            filter_string=f"name = '{model_1.name}'",
            page_token=page_1.token,
        )


def test_search_logged_models_datasets_filter(store):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    run_id = store.create_run(exp_id, "user", 0, [], "test_run").info.run_id
    model1 = store.create_logged_model(exp_id, source_run_id=run_id)
    model2 = store.create_logged_model(exp_id, source_run_id=run_id)
    model3 = store.create_logged_model(exp_id, source_run_id=run_id)
    store.log_batch(
        run_id,
        metrics=[
            Metric(
                key="metric1",
                value=0.1,
                timestamp=0,
                step=0,
                model_id=model1.model_id,
                dataset_name="dataset1",
                dataset_digest="digest1",
            ),
            Metric(
                key="metric1",
                value=0.2,
                timestamp=0,
                step=0,
                model_id=model2.model_id,
                dataset_name="dataset1",
                dataset_digest="digest2",
            ),
            Metric(key="metric2", value=0.1, timestamp=0, step=0, model_id=model3.model_id),
        ],
        params=[],
        tags=[],
    )

    # Restrict results to models with metrics on dataset1
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        filter_string="metrics.metric1 >= 0.1",
        datasets=[{"dataset_name": "dataset1"}],
    )
    assert {m.name for m in models} == {model1.name, model2.name}
    # Restrict results to models with metrics on dataset1 and digest1
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        filter_string="metrics.metric1 >= 0.1",
        datasets=[{"dataset_name": "dataset1", "dataset_digest": "digest1"}],
    )
    assert {m.name for m in models} == {model1.name}
    # No filter string, match models with any metrics on the dataset
    models = store.search_logged_models(
        experiment_ids=[exp_id], datasets=[{"dataset_name": "dataset1"}]
    )
    assert {m.name for m in models} == {model1.name, model2.name}


def test_log_batch_logged_model(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    run = store.create_run(
        experiment_id=exp_id, user_id="user", start_time=0, run_name="test", tags=[]
    )
    model = store.create_logged_model(experiment_id=exp_id)
    metric = Metric(
        key="metric1",
        value=1,
        timestamp=int(time.time() * 1000),
        step=3,
        model_id=model.model_id,
        dataset_name="dataset_name",
        dataset_digest="dataset_digest",
        run_id=run.info.run_id,
    )
    store.log_batch(run.info.run_id, metrics=[metric], params=[], tags=[])
    model = store.get_logged_model(model.model_id)
    assert model.metrics == [metric]

    # Log the same metric, should not throw
    store.log_batch(run.info.run_id, metrics=[metric], params=[], tags=[])
    assert model.metrics == [metric]

    # Log an empty batch, should not throw
    store.log_batch(run.info.run_id, metrics=[], params=[], tags=[])
    assert model.metrics == [metric]

    another_metric = Metric(
        key="metric2",
        value=2,
        timestamp=int(time.time() * 1000),
        step=4,
        model_id=model.model_id,
        dataset_name="dataset_name",
        dataset_digest="dataset_digest",
        run_id=run.info.run_id,
    )
    store.log_batch(run.info.run_id, metrics=[another_metric], params=[], tags=[])
    model = store.get_logged_model(model.model_id)
    actual_metrics = sorted(model.metrics, key=lambda m: m.key)
    expected_metrics = sorted([metric, another_metric], key=lambda m: m.key)
    assert actual_metrics == expected_metrics

    # Log multiple metrics
    metrics = [
        Metric(
            key=f"metric{i + 3}",
            value=3,
            timestamp=int(time.time() * 1000),
            step=5,
            model_id=model.model_id,
            dataset_name="dataset_name",
            dataset_digest="dataset_digest",
            run_id=run.info.run_id,
        )
        for i in range(3)
    ]

    store.log_batch(run.info.run_id, metrics=metrics, params=[], tags=[])
    model = store.get_logged_model(model.model_id)
    actual_metrics = sorted(model.metrics, key=lambda m: m.key)
    expected_metrics = sorted([metric, another_metric, *metrics], key=lambda m: m.key)
    assert actual_metrics == expected_metrics


def test_create_and_get_assessment(store_and_trace_info):
    store, trace_info = store_and_trace_info

    feedback = Feedback(
        trace_id=trace_info.request_id,
        name="correctness",
        value=True,
        rationale="The response is correct and well-formatted",
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN, source_id="evaluator@company.com"
        ),
        metadata={"project": "test-project", "version": "1.0"},
        span_id="span-123",
    )

    created_feedback = store.create_assessment(feedback)
    assert created_feedback.assessment_id is not None
    assert created_feedback.assessment_id.startswith("a-")
    assert created_feedback.trace_id == trace_info.request_id
    assert created_feedback.create_time_ms is not None
    assert created_feedback.name == "correctness"
    assert created_feedback.value is True
    assert created_feedback.rationale == "The response is correct and well-formatted"
    assert created_feedback.metadata == {"project": "test-project", "version": "1.0"}
    assert created_feedback.span_id == "span-123"
    assert created_feedback.valid

    expectation = Expectation(
        trace_id=trace_info.request_id,
        name="expected_response",
        value="The capital of France is Paris.",
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN, source_id="annotator@company.com"
        ),
        metadata={"context": "geography-qa", "difficulty": "easy"},
        span_id="span-456",
    )

    created_expectation = store.create_assessment(expectation)
    assert created_expectation.assessment_id != created_feedback.assessment_id
    assert created_expectation.trace_id == trace_info.request_id
    assert created_expectation.value == "The capital of France is Paris."
    assert created_expectation.metadata == {"context": "geography-qa", "difficulty": "easy"}
    assert created_expectation.span_id == "span-456"
    assert created_expectation.valid

    retrieved_feedback = store.get_assessment(trace_info.request_id, created_feedback.assessment_id)
    assert retrieved_feedback.name == "correctness"
    assert retrieved_feedback.value is True
    assert retrieved_feedback.rationale == "The response is correct and well-formatted"
    assert retrieved_feedback.metadata == {"project": "test-project", "version": "1.0"}
    assert retrieved_feedback.span_id == "span-123"
    assert retrieved_feedback.trace_id == trace_info.request_id
    assert retrieved_feedback.valid

    retrieved_expectation = store.get_assessment(
        trace_info.request_id, created_expectation.assessment_id
    )
    assert retrieved_expectation.value == "The capital of France is Paris."
    assert retrieved_expectation.metadata == {"context": "geography-qa", "difficulty": "easy"}
    assert retrieved_expectation.span_id == "span-456"
    assert retrieved_expectation.trace_id == trace_info.request_id
    assert retrieved_expectation.valid


def test_get_assessment_errors(store_and_trace_info):
    store, trace_info = store_and_trace_info

    with pytest.raises(MlflowException, match=r"Trace with (ID|request_id) 'fake_trace' not found"):
        store.get_assessment("fake_trace", "fake_assessment")

    with pytest.raises(
        MlflowException,
        match=r"Assessment with ID 'fake_assessment' not found for trace",
    ):
        store.get_assessment(trace_info.request_id, "fake_assessment")


def test_update_assessment_feedback(store_and_trace_info):
    store, trace_info = store_and_trace_info

    original_feedback = Feedback(
        trace_id=trace_info.request_id,
        name="correctness",
        value=True,
        rationale="Original rationale",
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN, source_id="evaluator@company.com"
        ),
        metadata={"project": "test-project", "version": "1.0"},
        span_id="span-123",
    )

    created_feedback = store.create_assessment(original_feedback)
    original_id = created_feedback.assessment_id

    updated_feedback = store.update_assessment(
        trace_id=trace_info.request_id,
        assessment_id=original_id,
        name="correctness_updated",
        feedback=FeedbackValue(value=False),
        rationale="Updated rationale",
        metadata={"project": "test-project", "version": "2.0", "new_field": "added"},
    )

    assert updated_feedback.assessment_id == original_id
    assert updated_feedback.name == "correctness_updated"
    assert updated_feedback.value is False
    assert updated_feedback.rationale == "Updated rationale"
    assert updated_feedback.metadata == {
        "project": "test-project",
        "version": "2.0",
        "new_field": "added",
    }
    assert updated_feedback.span_id == "span-123"
    assert updated_feedback.source.source_id == "evaluator@company.com"
    assert updated_feedback.valid is True

    retrieved = store.get_assessment(trace_info.request_id, original_id)
    assert retrieved.value is False
    assert retrieved.name == "correctness_updated"
    assert retrieved.rationale == "Updated rationale"


def test_update_assessment_expectation(store_and_trace_info):
    store, trace_info = store_and_trace_info

    original_expectation = Expectation(
        trace_id=trace_info.request_id,
        name="expected_response",
        value="The capital of France is Paris.",
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN, source_id="annotator@company.com"
        ),
        metadata={"context": "geography-qa"},
        span_id="span-456",
    )

    created_expectation = store.create_assessment(original_expectation)
    original_id = created_expectation.assessment_id

    updated_expectation = store.update_assessment(
        trace_id=trace_info.request_id,
        assessment_id=original_id,
        expectation=ExpectationValue(value="The capital and largest city of France is Paris."),
        metadata={"context": "geography-qa", "updated": "true"},
    )

    assert updated_expectation.assessment_id == original_id
    assert updated_expectation.name == "expected_response"
    assert updated_expectation.value == "The capital and largest city of France is Paris."
    assert updated_expectation.metadata == {"context": "geography-qa", "updated": "true"}
    assert updated_expectation.span_id == "span-456"
    assert updated_expectation.source.source_id == "annotator@company.com"


def test_update_assessment_partial_fields(store_and_trace_info):
    store, trace_info = store_and_trace_info

    original_feedback = Feedback(
        trace_id=trace_info.request_id,
        name="quality",
        value=5,
        rationale="Original rationale",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
        metadata={"scorer": "automated"},
    )

    created_feedback = store.create_assessment(original_feedback)
    original_id = created_feedback.assessment_id

    updated_feedback = store.update_assessment(
        trace_id=trace_info.request_id,
        assessment_id=original_id,
        rationale="Updated rationale only",
    )

    assert updated_feedback.assessment_id == original_id
    assert updated_feedback.name == "quality"
    assert updated_feedback.value == 5
    assert updated_feedback.rationale == "Updated rationale only"
    assert updated_feedback.metadata == {"scorer": "automated"}


def test_update_assessment_type_validation(store_and_trace_info):
    store, trace_info = store_and_trace_info

    feedback = Feedback(
        trace_id=trace_info.request_id,
        name="test_feedback",
        value="original",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
    )
    created_feedback = store.create_assessment(feedback)

    with pytest.raises(
        MlflowException, match=r"Cannot update expectation value on a Feedback assessment"
    ):
        store.update_assessment(
            trace_id=trace_info.request_id,
            assessment_id=created_feedback.assessment_id,
            expectation=ExpectationValue(value="This should fail"),
        )

    expectation = Expectation(
        trace_id=trace_info.request_id,
        name="test_expectation",
        value="original_expected",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )
    created_expectation = store.create_assessment(expectation)

    with pytest.raises(
        MlflowException, match=r"Cannot update feedback value on an Expectation assessment"
    ):
        store.update_assessment(
            trace_id=trace_info.request_id,
            assessment_id=created_expectation.assessment_id,
            feedback=FeedbackValue(value="This should fail"),
        )


def test_update_assessment_errors(store_and_trace_info):
    store, trace_info = store_and_trace_info

    with pytest.raises(MlflowException, match=r"Trace with (ID|request_id) 'fake_trace' not found"):
        store.update_assessment(
            trace_id="fake_trace", assessment_id="fake_assessment", rationale="This should fail"
        )

    with pytest.raises(
        MlflowException,
        match=r"Assessment with ID 'fake_assessment' not found for trace",
    ):
        store.update_assessment(
            trace_id=trace_info.request_id,
            assessment_id="fake_assessment",
            rationale="This should fail",
        )


def test_update_assessment_metadata_merging(store_and_trace_info):
    store, trace_info = store_and_trace_info

    original = Feedback(
        trace_id=trace_info.request_id,
        name="test",
        value="original",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
        metadata={"keep": "this", "override": "old_value", "remove_me": "will_stay"},
    )

    created = store.create_assessment(original)

    updated = store.update_assessment(
        trace_id=trace_info.request_id,
        assessment_id=created.assessment_id,
        metadata={"override": "new_value", "new_key": "new_value"},
    )

    expected_metadata = {
        "keep": "this",
        "override": "new_value",
        "remove_me": "will_stay",
        "new_key": "new_value",
    }
    assert updated.metadata == expected_metadata


def test_update_assessment_timestamps(store_and_trace_info):
    store, trace_info = store_and_trace_info

    original = Feedback(
        trace_id=trace_info.request_id,
        name="test",
        value="original",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
    )

    created = store.create_assessment(original)
    original_create_time = created.create_time_ms
    original_update_time = created.last_update_time_ms

    time.sleep(0.001)

    updated = store.update_assessment(
        trace_id=trace_info.request_id,
        assessment_id=created.assessment_id,
        name="updated_name",
    )

    assert updated.create_time_ms == original_create_time
    assert updated.last_update_time_ms > original_update_time


def test_create_assessment_with_overrides(store_and_trace_info):
    store, trace_info = store_and_trace_info

    original_feedback = Feedback(
        trace_id=trace_info.request_id,
        name="quality",
        value="poor",
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE),
    )

    created_original = store.create_assessment(original_feedback)

    override_feedback = Feedback(
        trace_id=trace_info.request_id,
        name="quality",
        value="excellent",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
        overrides=created_original.assessment_id,
    )

    created_override = store.create_assessment(override_feedback)

    assert created_override.overrides == created_original.assessment_id
    assert created_override.value == "excellent"
    assert created_override.valid is True

    retrieved_original = store.get_assessment(trace_info.request_id, created_original.assessment_id)
    assert retrieved_original.valid is False
    assert retrieved_original.value == "poor"


def test_create_assessment_override_nonexistent(store_and_trace_info):
    store, trace_info = store_and_trace_info

    override_feedback = Feedback(
        trace_id=trace_info.request_id,
        name="quality",
        value="excellent",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
        overrides="nonexistent-assessment-id",
    )

    with pytest.raises(
        MlflowException, match=r"Assessment with ID 'nonexistent-assessment-id' not found"
    ):
        store.create_assessment(override_feedback)


def test_delete_assessment_idempotent(store_and_trace_info):
    store, trace_info = store_and_trace_info

    feedback = Feedback(
        trace_id=trace_info.request_id,
        name="test",
        value="test_value",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
    )

    created_feedback = store.create_assessment(feedback)

    retrieved = store.get_assessment(trace_info.request_id, created_feedback.assessment_id)
    assert retrieved.assessment_id == created_feedback.assessment_id

    store.delete_assessment(trace_info.request_id, created_feedback.assessment_id)

    with pytest.raises(
        MlflowException,
        match=rf"Assessment with ID '{created_feedback.assessment_id}' not found for trace",
    ):
        store.get_assessment(trace_info.request_id, created_feedback.assessment_id)

    store.delete_assessment(trace_info.request_id, created_feedback.assessment_id)
    store.delete_assessment(trace_info.request_id, "fake_assessment_id")


def test_delete_assessment_override_behavior(store_and_trace_info):
    store, trace_info = store_and_trace_info

    original = store.create_assessment(
        Feedback(
            trace_id=trace_info.request_id,
            name="original",
            value="original_value",
            source=AssessmentSource(source_type=AssessmentSourceType.CODE),
        ),
    )

    override = store.create_assessment(
        Feedback(
            trace_id=trace_info.request_id,
            name="override",
            value="override_value",
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
            overrides=original.assessment_id,
        ),
    )

    assert store.get_assessment(trace_info.request_id, original.assessment_id).valid is False
    assert store.get_assessment(trace_info.request_id, override.assessment_id).valid is True

    store.delete_assessment(trace_info.request_id, override.assessment_id)

    with pytest.raises(MlflowException, match="not found"):
        store.get_assessment(trace_info.request_id, override.assessment_id)
    assert store.get_assessment(trace_info.request_id, original.assessment_id).valid is True


def test_assessment_with_run_id(store_and_trace_info):
    store, trace_info = store_and_trace_info

    run = store.create_run(
        experiment_id=trace_info.experiment_id,
        user_id="test_user",
        start_time=get_current_time_millis(),
        tags=[],
        run_name="test_run",
    )

    feedback = Feedback(
        trace_id=trace_info.request_id,
        name="run_feedback",
        value="excellent",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
    )
    feedback.run_id = run.info.run_id

    created_feedback = store.create_assessment(feedback)
    assert created_feedback.run_id == run.info.run_id

    retrieved_feedback = store.get_assessment(trace_info.request_id, created_feedback.assessment_id)
    assert retrieved_feedback.run_id == run.info.run_id


def test_assessment_with_error(store_and_trace_info):
    store, trace_info = store_and_trace_info

    try:
        raise ValueError("Test error message")
    except ValueError as test_error:
        feedback = Feedback(
            trace_id=trace_info.request_id,
            name="error_feedback",
            value=None,
            error=test_error,
            source=AssessmentSource(source_type=AssessmentSourceType.CODE),
        )

    created_feedback = store.create_assessment(feedback)
    assert created_feedback.error.error_message == "Test error message"
    assert created_feedback.error.error_code == "ValueError"
    assert created_feedback.error.stack_trace is not None
    assert "ValueError: Test error message" in created_feedback.error.stack_trace
    assert "test_assessment_with_error" in created_feedback.error.stack_trace

    retrieved_feedback = store.get_assessment(trace_info.request_id, created_feedback.assessment_id)
    assert retrieved_feedback.error.error_message == "Test error message"
    assert retrieved_feedback.error.error_code == "ValueError"
    assert retrieved_feedback.error.stack_trace is not None
    assert "ValueError: Test error message" in retrieved_feedback.error.stack_trace
    assert created_feedback.error.stack_trace == retrieved_feedback.error.stack_trace


def test_start_trace_with_assessments_missing_trace_id(store):
    """
    Regression test for NOT NULL constraint on assessments.trace_id during trace export.

    During normal trace export (MlflowV3SpanExporter), two things happen:

    1. log_spans() is called incrementally as each span completes. Internally this calls
       start_trace(), creating the trace row in the DB.
    2. When the root span finishes, _log_trace() calls start_trace() again with the full
       TraceInfo — including any assessments attached to the trace.

    Because the trace row already exists from step 1, the second start_trace() hits an
    IntegrityError and falls back to session.merge(). Assessments created standalone
    (e.g. returned by custom metric functions) have trace_id=None by design. Without
    backfilling trace_id before the merge, SQLAlchemy updates the assessment row with
    trace_id=NULL, violating the NOT NULL constraint on assessments.trace_id.
    """
    exp_id = store.create_experiment("test_assessment_trace_id")
    timestamp_ms = get_current_time_millis()
    trace_id = f"tr-{uuid.uuid4()}"

    # Step 1: log_spans() creates the trace row as spans are exported incrementally.
    store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms,
            execution_duration=0,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"cr-{uuid.uuid4()}",
            request_preview=None,
            response_preview=None,
        ),
    )

    # Assessment with trace_id=None, as returned by custom metric functions.
    assessment = Feedback(
        name="test_feedback",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user1"),
        trace_id=None,
        value="good",
    )

    # Step 2: _log_trace() calls start_trace() with the full TraceInfo (including
    # assessments) after the root span finishes. The trace already exists from step 1,
    # so this hits the IntegrityError -> session.merge() path. Before the fix, this
    # raised sqlite3.IntegrityError because assessment.trace_id was None.
    result = store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms,
            execution_duration=100,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"cr-{uuid.uuid4()}",
            request_preview="request",
            response_preview="response",
            assessments=[assessment],
        ),
    )

    assert len(result.assessments) == 1
    assert result.assessments[0].trace_id == trace_id
    assert result.assessments[0].name == "test_feedback"


def test_dataset_crud_operations(store):
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store):
        experiment_ids = _create_experiments(store, ["test_exp_1", "test_exp_2"])
        created_dataset = store.create_dataset(
            name="test_eval_dataset",
            tags={
                "purpose": "testing",
                "environment": "test",
                mlflow_tags.MLFLOW_USER: "test_user",
            },
            experiment_ids=experiment_ids,
        )

        assert created_dataset.dataset_id is not None
        assert created_dataset.dataset_id.startswith("d-")
        assert created_dataset.name == "test_eval_dataset"
        assert created_dataset.tags == {
            "purpose": "testing",
            "environment": "test",
            mlflow_tags.MLFLOW_USER: "test_user",
        }
        assert created_dataset.created_time > 0
        assert created_dataset.last_update_time > 0
        assert created_dataset.created_time == created_dataset.last_update_time
        assert created_dataset.schema is None  # Schema is computed when data is added
        assert created_dataset.profile is None  # Profile is computed when data is added
        assert created_dataset.created_by == "test_user"  # Extracted from mlflow.user tag

        retrieved_dataset = store.get_dataset(dataset_id=created_dataset.dataset_id)
        assert retrieved_dataset.dataset_id == created_dataset.dataset_id
        assert retrieved_dataset.name == created_dataset.name
        assert retrieved_dataset.tags == created_dataset.tags
        assert retrieved_dataset._experiment_ids is None
        assert retrieved_dataset.experiment_ids == experiment_ids
        assert not retrieved_dataset.has_records()

        with pytest.raises(
            MlflowException, match="Evaluation dataset with id 'd-nonexistent' not found"
        ):
            store.get_dataset(dataset_id="d-nonexistent")

        store.delete_dataset(created_dataset.dataset_id)
        with pytest.raises(MlflowException, match="not found"):
            store.get_dataset(dataset_id=created_dataset.dataset_id)

        # Verify idempotentcy
        store.delete_dataset("d-nonexistent")


def test_dataset_records_pagination(store):
    exp_id = _create_experiments(store, ["pagination_test_exp"])[0]

    dataset = store.create_dataset(
        name="pagination_test_dataset", experiment_ids=[exp_id], tags={"test": "pagination"}
    )

    records = [
        {
            "inputs": {"id": i, "question": f"Question {i}"},
            "expectations": {"answer": f"Answer {i}"},
            "tags": {"index": str(i)},
        }
        for i in range(25)
    ]

    store.upsert_dataset_records(dataset.dataset_id, records)

    page1, next_token1 = store._load_dataset_records(dataset.dataset_id, max_results=10)
    assert len(page1) == 10
    assert next_token1 is not None  # Token should exist for more pages

    # Collect all IDs from page1
    page1_ids = {r.inputs["id"] for r in page1}
    assert len(page1_ids) == 10  # All IDs should be unique

    page2, next_token2 = store._load_dataset_records(
        dataset.dataset_id, max_results=10, page_token=next_token1
    )
    assert len(page2) == 10
    assert next_token2 is not None  # Token should exist for more pages

    # Collect all IDs from page2
    page2_ids = {r.inputs["id"] for r in page2}
    assert len(page2_ids) == 10  # All IDs should be unique
    assert page1_ids.isdisjoint(page2_ids)  # No overlap between pages

    page3, next_token3 = store._load_dataset_records(
        dataset.dataset_id, max_results=10, page_token=next_token2
    )
    assert len(page3) == 5
    assert next_token3 is None  # No more pages

    # Collect all IDs from page3
    page3_ids = {r.inputs["id"] for r in page3}
    assert len(page3_ids) == 5  # All IDs should be unique
    assert page1_ids.isdisjoint(page3_ids)  # No overlap
    assert page2_ids.isdisjoint(page3_ids)  # No overlap

    # Verify we got all 25 records across all pages
    all_ids = page1_ids | page2_ids | page3_ids
    assert all_ids == set(range(25))

    all_records, no_token = store._load_dataset_records(dataset.dataset_id, max_results=None)
    assert len(all_records) == 25
    assert no_token is None

    # Verify we have all expected records (order doesn't matter)
    all_record_ids = {r.inputs["id"] for r in all_records}
    assert all_record_ids == set(range(25))


def test_dataset_search_comprehensive(store):
    test_prefix = "test_search_"
    exp_ids = _create_experiments(store, [f"{test_prefix}exp_{i}" for i in range(1, 4)])

    datasets = []
    for i in range(10):
        name = f"{test_prefix}dataset_{i:02d}"
        tags = {"priority": "high" if i % 2 == 0 else "low", "mlflow.user": f"user_{i % 3}"}

        if i < 3:
            created = store.create_dataset(
                name=name,
                experiment_ids=[exp_ids[0]],
                tags=tags,
            )
        elif i < 6:
            created = store.create_dataset(
                name=name,
                experiment_ids=[exp_ids[1], exp_ids[2]],
                tags=tags,
            )
        elif i < 8:
            created = store.create_dataset(
                name=name,
                experiment_ids=[exp_ids[2]],
                tags=tags,
            )
        else:
            created = store.create_dataset(
                name=name,
                experiment_ids=[],
                tags=tags,
            )
        datasets.append(created)
        time.sleep(0.001)

    results = store.search_datasets(experiment_ids=[exp_ids[0]])
    assert len([d for d in results if d.name.startswith(test_prefix)]) == 3

    results = store.search_datasets(experiment_ids=[exp_ids[1], exp_ids[2]])
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 5

    results = store.search_datasets(order_by=["name"])
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    names = [d.name for d in test_results]
    assert names == sorted(names)

    results = store.search_datasets(order_by=["name DESC"])
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    names = [d.name for d in test_results]
    assert names == sorted(names, reverse=True)

    page1 = store.search_datasets(max_results=3)
    assert len(page1) == 3
    assert page1.token is not None

    page2 = store.search_datasets(max_results=3, page_token=page1.token)
    assert len(page2) == 3
    assert all(d1.dataset_id != d2.dataset_id for d1 in page1 for d2 in page2)

    results = store.search_datasets(experiment_ids=None)
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 10

    results = store.search_datasets(filter_string=f"name LIKE '%{test_prefix}dataset_0%'")
    assert len(results) == 10
    assert all("dataset_0" in d.name for d in results)

    results = store.search_datasets(filter_string=f"name = '{test_prefix}dataset_05'")
    assert len(results) == 1
    assert results[0].name == f"{test_prefix}dataset_05"

    results = store.search_datasets(filter_string="tags.priority = 'high'")
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 5
    assert all(d.tags.get("priority") == "high" for d in test_results)

    results = store.search_datasets(filter_string="tags.priority != 'high'")
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 5
    assert all(d.tags.get("priority") == "low" for d in test_results)

    results = store.search_datasets(
        filter_string=f"name LIKE '%{test_prefix}%' AND tags.priority = 'low'"
    )
    assert len(results) == 5
    assert all(d.tags.get("priority") == "low" and test_prefix in d.name for d in results)

    mid_dataset = datasets[5]
    results = store.search_datasets(filter_string=f"created_time > {mid_dataset.created_time}")
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 4
    assert all(d.created_time > mid_dataset.created_time for d in test_results)

    results = store.search_datasets(
        experiment_ids=[exp_ids[0]], filter_string="tags.priority = 'high'"
    )
    assert len(results) == 2
    assert all(d.tags.get("priority") == "high" for d in results)

    results = store.search_datasets(filter_string="tags.priority = 'low'", order_by=["name ASC"])
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    names = [d.name for d in test_results]
    assert names == sorted(names)

    created_user = store.create_dataset(
        name=f"{test_prefix}_user_dataset",
        tags={"test": "user", mlflow_tags.MLFLOW_USER: "test_user_1"},
        experiment_ids=[exp_ids[0]],
    )

    results = store.search_datasets(filter_string="created_by = 'test_user_1'")
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 1
    assert test_results[0].created_by == "test_user_1"

    records_with_user = [
        {
            "inputs": {"test": "data"},
            "expectations": {"result": "expected"},
            "tags": {mlflow_tags.MLFLOW_USER: "test_user_2"},
        }
    ]
    store.upsert_dataset_records(created_user.dataset_id, records_with_user)

    results = store.search_datasets(filter_string="last_updated_by = 'test_user_2'")
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 1
    assert test_results[0].last_updated_by == "test_user_2"

    with pytest.raises(MlflowException, match="Invalid attribute key"):
        store.search_datasets(filter_string="invalid_field = 'value'")


def test_dataset_schema_and_profile_computation(store):
    test_prefix = "test_schema_profile_"
    exp_ids = _create_experiments(store, [f"{test_prefix}exp"])

    dataset = store.create_dataset(name=f"{test_prefix}dataset", experiment_ids=exp_ids)

    assert dataset.schema is None
    assert dataset.profile is None

    records = [
        {
            "inputs": {
                "question": "What is MLflow?",
                "temperature": 0.7,
                "max_tokens": 100,
                "use_cache": True,
                "tags": ["ml", "tools"],
            },
            "expectations": {
                "accuracy": 0.95,
                "contains_key_info": True,
                "response": "MLflow is an open source platform",
            },
            "source": {"source_type": "TRACE", "source_data": {"trace_id": "trace1"}},
        },
        {
            "inputs": {
                "question": "What is Python?",
                "temperature": 0.5,
                "max_tokens": 150,
                "metadata": {"user": "test", "session": 123},
            },
            "expectations": {"accuracy": 0.9},
            "source": {"source_type": "TRACE", "source_data": {"trace_id": "trace2"}},
        },
        {
            "inputs": {"question": "What is Docker?", "temperature": 0.8},
            "source": {"source_type": "HUMAN", "source_data": {"user": "human"}},
        },
    ]

    store.upsert_dataset_records(dataset.dataset_id, records)

    updated_dataset = store.get_dataset(dataset.dataset_id)

    assert updated_dataset.schema is not None
    schema = json.loads(updated_dataset.schema)
    assert "inputs" in schema
    assert "expectations" in schema
    assert schema["inputs"]["question"] == "string"
    assert schema["inputs"]["temperature"] == "float"
    assert schema["inputs"]["max_tokens"] == "integer"
    assert schema["inputs"]["use_cache"] == "boolean"
    assert schema["inputs"]["tags"] == "array"
    assert schema["inputs"]["metadata"] == "object"
    assert schema["expectations"]["accuracy"] == "float"
    assert schema["expectations"]["contains_key_info"] == "boolean"
    assert schema["expectations"]["response"] == "string"

    assert updated_dataset.profile is not None
    profile = json.loads(updated_dataset.profile)
    assert profile["num_records"] == 3


def test_dataset_schema_and_profile_incremental_updates(store):
    test_prefix = "test_incremental_"
    exp_ids = _create_experiments(store, [f"{test_prefix}exp"])

    dataset = store.create_dataset(name=f"{test_prefix}dataset", experiment_ids=exp_ids)

    initial_records = [
        {
            "inputs": {"question": "What is MLflow?", "temperature": 0.7},
            "expectations": {"accuracy": 0.95},
            "source": {"source_type": "TRACE", "source_data": {"trace_id": "trace1"}},
        }
    ]

    store.upsert_dataset_records(dataset.dataset_id, initial_records)

    dataset1 = store.get_dataset(dataset.dataset_id)
    schema1 = json.loads(dataset1.schema)
    profile1 = json.loads(dataset1.profile)

    assert schema1["inputs"] == {"question": "string", "temperature": "float"}
    assert schema1["expectations"] == {"accuracy": "float"}
    assert profile1["num_records"] == 1

    additional_records = [
        {
            "inputs": {
                "question": "What is Python?",
                "temperature": 0.5,
                "max_tokens": 100,
                "use_cache": True,
            },
            "expectations": {"accuracy": 0.9, "relevance": 0.85},
            "source": {"source_type": "HUMAN", "source_data": {"user": "test_user"}},
        }
    ]

    store.upsert_dataset_records(dataset.dataset_id, additional_records)

    dataset2 = store.get_dataset(dataset.dataset_id)
    schema2 = json.loads(dataset2.schema)
    profile2 = json.loads(dataset2.profile)

    assert schema2["inputs"]["question"] == "string"
    assert schema2["inputs"]["temperature"] == "float"
    assert schema2["inputs"]["max_tokens"] == "integer"
    assert schema2["inputs"]["use_cache"] == "boolean"
    assert schema2["expectations"]["accuracy"] == "float"
    assert schema2["expectations"]["relevance"] == "float"

    assert profile2["num_records"] == 2


def test_dataset_user_detection(store):
    test_prefix = "test_user_detection_"
    exp_ids = _create_experiments(store, [f"{test_prefix}exp"])

    dataset1 = store.create_dataset(
        name=f"{test_prefix}dataset1",
        tags={mlflow_tags.MLFLOW_USER: "john_doe", "other": "tag"},
        experiment_ids=exp_ids,
    )
    assert dataset1.created_by == "john_doe"
    assert dataset1.tags[mlflow_tags.MLFLOW_USER] == "john_doe"

    dataset2 = store.create_dataset(
        name=f"{test_prefix}dataset2", tags={"other": "tag"}, experiment_ids=exp_ids
    )
    assert dataset2.created_by is None
    assert mlflow_tags.MLFLOW_USER not in dataset2.tags

    results = store.search_datasets(filter_string="created_by = 'john_doe'")
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 1
    assert test_results[0].dataset_id == dataset1.dataset_id


def test_dataset_filtering_ordering_pagination(store):
    test_prefix = "test_filter_order_page_"
    exp_ids = _create_experiments(store, [f"{test_prefix}exp_{i}" for i in range(3)])

    datasets = []
    for i in range(10):
        time.sleep(0.01)
        tags = {
            "priority": "high" if i < 3 else ("medium" if i < 7 else "low"),
            "model": f"model_{i % 3}",
            "environment": "production" if i % 2 == 0 else "staging",
        }
        created = store.create_dataset(
            name=f"{test_prefix}_dataset_{i:02d}",
            tags=tags,
            experiment_ids=[exp_ids[i % len(exp_ids)]],
        )
        datasets.append(created)

    results = store.search_datasets(
        filter_string="tags.priority = 'high'", order_by=["name ASC"], max_results=2
    )
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 2
    assert all(d.tags.get("priority") == "high" for d in test_results)
    assert test_results[0].name < test_results[1].name

    results_all = store.search_datasets(
        filter_string="tags.priority = 'high'", order_by=["name ASC"]
    )
    test_results_all = [d for d in results_all if d.name.startswith(test_prefix)]
    assert len(test_results_all) == 3

    mid_time = datasets[5].created_time
    results = store.search_datasets(
        filter_string=f"tags.environment = 'production' AND created_time > {mid_time}",
        order_by=["created_time DESC"],
        max_results=3,
    )
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert all(d.tags.get("environment") == "production" for d in test_results)
    assert all(d.created_time > mid_time for d in test_results)

    for i in range(1, len(test_results)):
        assert test_results[i - 1].created_time >= test_results[i].created_time

    results = store.search_datasets(
        experiment_ids=[exp_ids[0]],
        filter_string="tags.model = 'model_0' AND tags.priority != 'low'",
        order_by=["last_update_time DESC"],
        max_results=5,
    )
    for d in results:
        assert d.tags.get("model") == "model_0"
        assert d.tags.get("priority") != "low"

    all_production = store.search_datasets(
        filter_string="tags.environment = 'production'", order_by=["name ASC"]
    )
    test_all_production = [d for d in all_production if d.name.startswith(test_prefix)]

    limited_results = store.search_datasets(
        filter_string="tags.environment = 'production'", order_by=["name ASC"], max_results=3
    )
    test_limited = [d for d in limited_results if d.name.startswith(test_prefix)]

    assert len(test_limited) == 3
    assert len(test_all_production) == 5
    for i in range(3):
        assert test_limited[i].dataset_id == test_all_production[i].dataset_id


def test_dataset_upsert_comprehensive(store):
    created_dataset = store.create_dataset(name="upsert_comprehensive")

    records_batch1 = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is a platform", "score": 0.8},
            "tags": {"version": "v1", "quality": "high"},
            "source": {
                "source_type": "TRACE",
                "source_data": {"trace_id": "trace-001", "span_id": "span-001"},
            },
        },
        {
            "inputs": {"question": "What is Python?"},
            "expectations": {"answer": "Python is a language"},
            "tags": {"category": "programming"},
        },
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is an ML platform", "confidence": 0.9},
            "tags": {"version": "v2", "reviewed": "true"},
            "source": {
                "source_type": "TRACE",
                "source_data": {"trace_id": "trace-002", "span_id": "span-002"},
            },
        },
    ]

    result = store.upsert_dataset_records(created_dataset.dataset_id, records_batch1)
    assert result["inserted"] == 2
    assert result["updated"] == 1

    loaded_records, next_token = store._load_dataset_records(created_dataset.dataset_id)
    assert len(loaded_records) == 2
    assert next_token is None

    mlflow_record = next(r for r in loaded_records if r.inputs["question"] == "What is MLflow?")
    assert mlflow_record.expectations == {
        "answer": "MLflow is an ML platform",
        "score": 0.8,
        "confidence": 0.9,
    }
    assert mlflow_record.tags == {"version": "v2", "quality": "high", "reviewed": "true"}

    assert mlflow_record.source.source_type == "TRACE"
    assert mlflow_record.source.source_data["trace_id"] == "trace-001"
    assert mlflow_record.source_id == "trace-001"

    initial_update_time = mlflow_record.last_update_time
    time.sleep(0.01)

    records_batch2 = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is the best ML platform", "rating": 5},
            "tags": {"version": "v3"},
        },
        {
            "inputs": {"question": "What is Spark?"},
            "expectations": {"answer": "Spark is a data processing engine"},
        },
    ]

    result = store.upsert_dataset_records(created_dataset.dataset_id, records_batch2)
    assert result["inserted"] == 1
    assert result["updated"] == 1

    loaded_records, next_token = store._load_dataset_records(created_dataset.dataset_id)
    assert len(loaded_records) == 3
    assert next_token is None

    updated_mlflow_record = next(
        r for r in loaded_records if r.inputs["question"] == "What is MLflow?"
    )
    assert updated_mlflow_record.expectations == {
        "answer": "MLflow is the best ML platform",
        "score": 0.8,
        "confidence": 0.9,
        "rating": 5,
    }
    assert updated_mlflow_record.tags == {
        "version": "v3",
        "quality": "high",
        "reviewed": "true",
    }
    assert updated_mlflow_record.last_update_time > initial_update_time
    assert updated_mlflow_record.source.source_data["trace_id"] == "trace-001"

    records_batch3 = [
        {"inputs": {"minimal": "input"}, "expectations": {"result": "minimal test"}},
        {"inputs": {"question": "Empty expectations"}, "expectations": {}},
        {"inputs": {"question": "No tags"}, "expectations": {"answer": "No tags"}, "tags": {}},
    ]

    result = store.upsert_dataset_records(created_dataset.dataset_id, records_batch3)
    assert result["inserted"] == 3
    assert result["updated"] == 0

    result = store.upsert_dataset_records(
        created_dataset.dataset_id,
        [{"inputs": {}, "expectations": {"result": "empty inputs allowed"}}],
    )
    assert result["inserted"] == 1
    assert result["updated"] == 0

    empty_result = store.upsert_dataset_records(created_dataset.dataset_id, [])
    assert empty_result["inserted"] == 0
    assert empty_result["updated"] == 0


def test_dataset_delete_records(store):
    test_prefix = "test_delete_records_"
    exp_ids = _create_experiments(store, [f"{test_prefix}exp"])

    dataset = store.create_dataset(name=f"{test_prefix}dataset", experiment_ids=exp_ids)

    records = [
        {
            "inputs": {"id": 1, "question": "What is MLflow?"},
            "expectations": {"answer": "ML platform"},
        },
        {
            "inputs": {"id": 2, "question": "What is Python?"},
            "expectations": {"answer": "Programming language"},
        },
        {
            "inputs": {"id": 3, "question": "What is Docker?"},
            "expectations": {"answer": "Container platform"},
        },
    ]
    store.upsert_dataset_records(dataset.dataset_id, records)

    loaded_records, _ = store._load_dataset_records(dataset.dataset_id)
    assert len(loaded_records) == 3

    record_ids = [r.dataset_record_id for r in loaded_records]

    deleted_count = store.delete_dataset_records(dataset.dataset_id, [record_ids[0]])
    assert deleted_count == 1

    remaining_records, _ = store._load_dataset_records(dataset.dataset_id)
    assert len(remaining_records) == 2

    updated_dataset = store.get_dataset(dataset.dataset_id)
    profile = json.loads(updated_dataset.profile)
    assert profile["num_records"] == 2

    deleted_count = store.delete_dataset_records(dataset.dataset_id, [record_ids[1], record_ids[2]])
    assert deleted_count == 2

    final_records, _ = store._load_dataset_records(dataset.dataset_id)
    assert len(final_records) == 0


def test_dataset_delete_records_idempotent(store):
    test_prefix = "test_delete_idempotent_"
    exp_ids = _create_experiments(store, [f"{test_prefix}exp"])

    dataset = store.create_dataset(name=f"{test_prefix}dataset", experiment_ids=exp_ids)

    deleted_count = store.delete_dataset_records(dataset.dataset_id, ["nonexistent-record-id"])
    assert deleted_count == 0


def test_dataset_associations_and_lazy_loading(store):
    experiment_ids = _create_experiments(store, ["test_exp_1", "test_exp_2", "test_exp_3"])
    created_dataset = store.create_dataset(
        name="multi_exp_dataset",
        experiment_ids=experiment_ids,
    )

    retrieved = store.get_dataset(dataset_id=created_dataset.dataset_id)
    assert retrieved._experiment_ids is None
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store):
        assert set(retrieved.experiment_ids) == set(experiment_ids)

    results = store.search_datasets(experiment_ids=[experiment_ids[1]])
    assert any(d.dataset_id == created_dataset.dataset_id for d in results)

    results = store.search_datasets(experiment_ids=[experiment_ids[0], experiment_ids[2]])
    matching = [d for d in results if d.dataset_id == created_dataset.dataset_id]
    assert len(matching) == 1
    assert matching[0]._experiment_ids is None
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store):
        assert set(matching[0].experiment_ids) == set(experiment_ids)

    records = [{"inputs": {"q": f"Q{i}"}, "expectations": {"a": f"A{i}"}} for i in range(5)]
    store.upsert_dataset_records(created_dataset.dataset_id, records)

    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store):
        retrieved = store.get_dataset(dataset_id=created_dataset.dataset_id)
        assert not retrieved.has_records()

        df = retrieved.to_df()
        assert len(df) == 5
        assert retrieved.has_records()

        assert list(df.columns) == [
            "inputs",
            "outputs",
            "expectations",
            "tags",
            "source_type",
            "source_id",
            "source",
            "created_time",
            "dataset_record_id",
        ]


def test_dataset_get_experiment_ids(store):
    experiment_ids = _create_experiments(store, ["exp_1", "exp_2", "exp_3"])
    created_dataset = store.create_dataset(
        name="test_get_experiment_ids",
        experiment_ids=experiment_ids,
    )

    fetched_experiment_ids = store.get_dataset_experiment_ids(created_dataset.dataset_id)
    assert set(fetched_experiment_ids) == set(experiment_ids)

    created_dataset2 = store.create_dataset(
        name="test_no_experiments",
        experiment_ids=[],
    )
    fetched_experiment_ids2 = store.get_dataset_experiment_ids(created_dataset2.dataset_id)
    assert fetched_experiment_ids2 == []

    result = store.get_dataset_experiment_ids("d-nonexistent")
    assert result == []

    result = store.get_dataset_experiment_ids("")
    assert result == []


def test_dataset_tags_with_sql_backend(store):
    tags = {"environment": "production", "version": "2.0", "team": "ml-ops"}

    created = store.create_dataset(
        name="tagged_dataset",
        tags=tags,
    )
    assert created.tags == tags

    retrieved = store.get_dataset(created.dataset_id)
    assert retrieved.tags == tags
    assert retrieved.tags["environment"] == "production"
    assert retrieved.tags["version"] == "2.0"
    assert retrieved.tags["team"] == "ml-ops"

    created_none = store.create_dataset(
        name="no_tags_dataset",
        tags=None,
    )
    retrieved_none = store.get_dataset(created_none.dataset_id)
    assert retrieved_none.tags == {}

    created_empty = store.create_dataset(
        name="empty_tags_dataset",
        tags={},
        experiment_ids=None,
    )
    retrieved_empty = store.get_dataset(created_empty.dataset_id)
    assert retrieved_empty.tags == {}


def test_dataset_update_tags(store):
    initial_tags = {"environment": "development", "version": "1.0", "deprecated": "true"}
    created = store.create_dataset(
        name="test_update_tags",
        tags=initial_tags,
        experiment_ids=None,
    )

    retrieved = store.get_dataset(created.dataset_id)
    assert retrieved.tags == initial_tags

    update_tags = {
        "environment": "production",
        "team": "ml-ops",
        "deprecated": None,  # This will be ignored, not delete the tag
    }
    store.set_dataset_tags(created.dataset_id, update_tags)

    updated = store.get_dataset(created.dataset_id)
    expected_tags = {
        "environment": "production",  # Updated
        "version": "1.0",  # Preserved
        "deprecated": "true",  # Preserved (None didn't delete it)
        "team": "ml-ops",  # Added
    }
    assert updated.tags == expected_tags
    assert updated.last_update_time == created.last_update_time
    assert updated.last_updated_by == created.last_updated_by

    created_no_tags = store.create_dataset(
        name="test_no_initial_tags",
        tags=None,
        experiment_ids=None,
    )

    store.set_dataset_tags(
        created_no_tags.dataset_id, {"new_tag": "value", "mlflow.user": "test_user2"}
    )

    updated_no_tags = store.get_dataset(created_no_tags.dataset_id)
    assert updated_no_tags.tags == {"new_tag": "value", "mlflow.user": "test_user2"}
    assert updated_no_tags.last_update_time == created_no_tags.last_update_time
    assert updated_no_tags.last_updated_by == created_no_tags.last_updated_by


def test_dataset_digest_updates_with_changes(store):
    experiment_id = store.create_experiment("test_exp")

    dataset = store.create_dataset(
        name="test_dataset",
        tags={"env": "test"},
        experiment_ids=[experiment_id],
    )

    initial_digest = dataset.digest
    assert initial_digest is not None

    time.sleep(0.01)  # Ensure time difference

    records = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"accuracy": 0.95},
        }
    ]

    store.upsert_dataset_records(dataset.dataset_id, records)

    updated_dataset = store.get_dataset(dataset.dataset_id)

    assert updated_dataset.digest != initial_digest

    prev_digest = updated_dataset.digest
    time.sleep(0.01)  # Ensure time difference

    more_records = [
        {
            "inputs": {"question": "How to track experiments?"},
            "expectations": {"accuracy": 0.9},
        }
    ]

    store.upsert_dataset_records(dataset.dataset_id, more_records)

    final_dataset = store.get_dataset(dataset.dataset_id)

    assert final_dataset.digest != prev_digest
    assert final_dataset.digest != initial_digest

    store.set_dataset_tags(dataset.dataset_id, {"new_tag": "value"})
    dataset_after_tags = store.get_dataset(dataset.dataset_id)

    assert dataset_after_tags.digest == final_dataset.digest


def test_sql_dataset_record_merge():
    with mock.patch("mlflow.store.tracking.dbmodels.models.get_current_time_millis") as mock_time:
        mock_time.return_value = 2000

        record = SqlEvaluationDatasetRecord()
        record.expectations = {"accuracy": 0.8, "relevance": 0.7}
        record.tags = {"env": "test"}
        record.created_time = 1000
        record.last_update_time = 1000
        record.created_by = "user1"
        record.last_updated_by = "user1"

        new_data = {
            "expectations": {"accuracy": 0.9, "completeness": 0.95},
            "tags": {"version": "2.0"},
        }

        record.merge(new_data)

        assert record.expectations == {
            "accuracy": 0.9,  # Updated
            "relevance": 0.7,  # Preserved
            "completeness": 0.95,  # Added
        }

        assert record.tags == {
            "env": "test",  # Preserved
            "version": "2.0",  # Added
        }

        assert record.created_time == 1000  # Preserved
        assert record.last_update_time == 2000  # Updated

        assert record.created_by == "user1"  # Preserved
        assert record.last_updated_by == "user1"  # No mlflow.user in tags

        record2 = SqlEvaluationDatasetRecord()
        record2.expectations = None
        record2.tags = None

        new_data2 = {"expectations": {"accuracy": 0.9}, "tags": {"env": "prod"}}

        record2.merge(new_data2)

        assert record2.expectations == {"accuracy": 0.9}
        assert record2.tags == {"env": "prod"}
        assert record2.last_update_time == 2000

        record3 = SqlEvaluationDatasetRecord()
        record3.created_by = "user1"
        record3.last_updated_by = "user1"

        new_data3 = {"tags": {"mlflow.user": "user2", "env": "prod"}}

        record3.merge(new_data3)

        assert record3.created_by == "user1"  # Preserved
        assert record3.last_updated_by == "user2"  # Updated from mlflow.user tag

        record4 = SqlEvaluationDatasetRecord()
        record4.expectations = {"accuracy": 0.8}
        record4.tags = {"env": "test"}
        record4.last_update_time = 1000

        record4.merge({})

        assert record4.expectations == {"accuracy": 0.8}
        assert record4.tags == {"env": "test"}
        assert record4.last_update_time == 2000

        record5 = SqlEvaluationDatasetRecord()
        record5.expectations = {"accuracy": 0.8}
        record5.tags = {"env": "test"}

        record5.merge({"expectations": {"relevance": 0.9}})

        assert record5.expectations == {"accuracy": 0.8, "relevance": 0.9}
        assert record5.tags == {"env": "test"}  # Unchanged

        record6 = SqlEvaluationDatasetRecord()
        record6.expectations = {"accuracy": 0.8}
        record6.tags = {"env": "test"}

        record6.merge({"tags": {"version": "1.0"}})

        assert record6.expectations == {"accuracy": 0.8}  # Unchanged
        assert record6.tags == {"env": "test", "version": "1.0"}


def test_sql_dataset_record_wrapping_unwrapping():
    from mlflow.entities.dataset_record import DATASET_RECORD_WRAPPED_OUTPUT_KEY

    entity = DatasetRecord(
        dataset_record_id="rec1",
        dataset_id="ds1",
        inputs={"q": "test"},
        outputs="string output",
        created_time=1000,
        last_update_time=1000,
    )

    sql_record = SqlEvaluationDatasetRecord.from_mlflow_entity(entity, "input_hash_123")

    assert sql_record.outputs == {DATASET_RECORD_WRAPPED_OUTPUT_KEY: "string output"}

    unwrapped_entity = sql_record.to_mlflow_entity()
    assert unwrapped_entity.outputs == "string output"

    entity2 = DatasetRecord(
        dataset_record_id="rec2",
        dataset_id="ds1",
        inputs={"q": "test"},
        outputs=[1, 2, 3],
        created_time=1000,
        last_update_time=1000,
    )

    sql_record2 = SqlEvaluationDatasetRecord.from_mlflow_entity(entity2, "input_hash_456")
    assert sql_record2.outputs == {DATASET_RECORD_WRAPPED_OUTPUT_KEY: [1, 2, 3]}

    unwrapped_entity2 = sql_record2.to_mlflow_entity()
    assert unwrapped_entity2.outputs == [1, 2, 3]

    entity3 = DatasetRecord(
        dataset_record_id="rec3",
        dataset_id="ds1",
        inputs={"q": "test"},
        outputs=42,
        created_time=1000,
        last_update_time=1000,
    )

    sql_record3 = SqlEvaluationDatasetRecord.from_mlflow_entity(entity3, "input_hash_789")
    assert sql_record3.outputs == {DATASET_RECORD_WRAPPED_OUTPUT_KEY: 42}

    unwrapped_entity3 = sql_record3.to_mlflow_entity()
    assert unwrapped_entity3.outputs == 42

    entity4 = DatasetRecord(
        dataset_record_id="rec4",
        dataset_id="ds1",
        inputs={"q": "test"},
        outputs={"result": "answer"},
        created_time=1000,
        last_update_time=1000,
    )

    sql_record4 = SqlEvaluationDatasetRecord.from_mlflow_entity(entity4, "input_hash_abc")
    assert sql_record4.outputs == {DATASET_RECORD_WRAPPED_OUTPUT_KEY: {"result": "answer"}}

    unwrapped_entity4 = sql_record4.to_mlflow_entity()
    assert unwrapped_entity4.outputs == {"result": "answer"}

    entity5 = DatasetRecord(
        dataset_record_id="rec5",
        dataset_id="ds1",
        inputs={"q": "test"},
        outputs=None,
        created_time=1000,
        last_update_time=1000,
    )

    sql_record5 = SqlEvaluationDatasetRecord.from_mlflow_entity(entity5, "input_hash_def")
    assert sql_record5.outputs is None

    unwrapped_entity5 = sql_record5.to_mlflow_entity()
    assert unwrapped_entity5.outputs is None

    sql_record6 = SqlEvaluationDatasetRecord()
    sql_record6.outputs = {"old": "data"}

    sql_record6.merge({"outputs": "new string output"})
    assert sql_record6.outputs == {DATASET_RECORD_WRAPPED_OUTPUT_KEY: "new string output"}

    sql_record7 = SqlEvaluationDatasetRecord()
    sql_record7.outputs = None

    sql_record7.merge({"outputs": {"new": "dict"}})
    assert sql_record7.outputs == {DATASET_RECORD_WRAPPED_OUTPUT_KEY: {"new": "dict"}}


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


def test_batch_get_traces_basic(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_batch_get_traces")
    trace_id = f"tr-{uuid.uuid4().hex}"

    spans = [
        create_test_span(
            trace_id=trace_id,
            name="root_span",
            span_id=111,
            status=trace_api.StatusCode.OK,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
            trace_num=12345,
        ),
        create_test_span(
            trace_id=trace_id,
            name="child_span",
            span_id=222,
            parent_id=111,
            status=trace_api.StatusCode.UNSET,
            start_ns=1_500_000_000,
            end_ns=1_800_000_000,
            trace_num=12345,
        ),
    ]

    store.log_spans(experiment_id, spans)
    traces = store.batch_get_traces([trace_id])

    assert len(traces) == 1
    loaded_spans = traces[0].data.spans

    assert len(loaded_spans) == 2

    root_span = next(s for s in loaded_spans if s.name == "root_span")
    child_span = next(s for s in loaded_spans if s.name == "child_span")

    assert root_span.trace_id == trace_id
    assert root_span.span_id == "000000000000006f"
    assert root_span.parent_id is None
    assert root_span.start_time_ns == 1_000_000_000
    assert root_span.end_time_ns == 2_000_000_000

    assert child_span.trace_id == trace_id
    assert child_span.span_id == "00000000000000de"
    assert child_span.parent_id == "000000000000006f"
    assert child_span.start_time_ns == 1_500_000_000
    assert child_span.end_time_ns == 1_800_000_000


def test_batch_get_traces_empty_trace(store: SqlAlchemyStore) -> None:
    trace_id = f"tr-{uuid.uuid4().hex}"
    traces = store.batch_get_traces([trace_id])
    assert traces == []


def test_batch_get_traces_ordering(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_load_spans_ordering")
    trace_id = f"tr-{uuid.uuid4().hex}"

    spans = [
        create_test_span(
            trace_id=trace_id,
            name="second_span",
            span_id=222,
            start_ns=2_000_000_000,
            end_ns=3_000_000_000,
            trace_num=12345,
        ),
        create_test_span(
            trace_id=trace_id,
            name="first_span",
            span_id=111,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
            trace_num=12345,
        ),
        create_test_span(
            trace_id=trace_id,
            name="third_span",
            span_id=333,
            start_ns=3_000_000_000,
            end_ns=4_000_000_000,
            trace_num=12345,
        ),
    ]

    store.log_spans(experiment_id, spans)
    traces = store.batch_get_traces([trace_id])

    assert len(traces) == 1
    loaded_spans = traces[0].data.spans

    assert len(loaded_spans) == 3
    assert loaded_spans[0].name == "first_span"
    assert loaded_spans[1].name == "second_span"
    assert loaded_spans[2].name == "third_span"

    assert loaded_spans[0].start_time_ns == 1_000_000_000
    assert loaded_spans[1].start_time_ns == 2_000_000_000
    assert loaded_spans[2].start_time_ns == 3_000_000_000


def test_batch_get_traces_with_complex_attributes(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_load_spans_complex")
    trace_id = f"tr-{uuid.uuid4().hex}"

    otel_span = create_test_otel_span(
        trace_id=trace_id,
        name="complex_span",
        status_code=trace_api.StatusCode.ERROR,
        status_description="Test error",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )

    otel_span._attributes = {
        "llm.model_name": "gpt-4",
        "llm.input_tokens": 100,
        "llm.output_tokens": 50,
        "custom.key": "custom_value",
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
    }

    span = create_mlflow_span(otel_span, trace_id, "LLM")

    store.log_spans(experiment_id, [span])
    traces = store.batch_get_traces([trace_id])

    assert len(traces) == 1
    loaded_spans = traces[0].data.spans

    assert len(loaded_spans) == 1
    loaded_span = loaded_spans[0]

    assert loaded_span.status.status_code == "ERROR"
    assert loaded_span.status.description == "Test error"

    assert loaded_span.attributes.get("llm.model_name") == "gpt-4"
    assert loaded_span.attributes.get("llm.input_tokens") == 100
    assert loaded_span.attributes.get("llm.output_tokens") == 50
    assert loaded_span.attributes.get("custom.key") == "custom_value"


def test_batch_get_traces_multiple_traces(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_load_spans_multiple")
    trace_id_1 = f"tr-{uuid.uuid4().hex}"
    trace_id_2 = f"tr-{uuid.uuid4().hex}"

    spans_trace_1 = [
        create_test_span(
            trace_id=trace_id_1,
            name="trace1_span1",
            span_id=111,
            trace_num=12345,
        ),
        create_test_span(
            trace_id=trace_id_1,
            name="trace1_span2",
            span_id=112,
            trace_num=12345,
        ),
    ]

    spans_trace_2 = [
        create_test_span(
            trace_id=trace_id_2,
            name="trace2_span1",
            span_id=221,
            trace_num=67890,
        ),
    ]

    store.log_spans(experiment_id, spans_trace_1)
    store.log_spans(experiment_id, spans_trace_2)
    traces = store.batch_get_traces([trace_id_1, trace_id_2])

    assert len(traces) == 2

    # Find traces by ID since order might not be guaranteed
    trace_1 = next(t for t in traces if t.info.trace_id == trace_id_1)
    trace_2 = next(t for t in traces if t.info.trace_id == trace_id_2)

    loaded_spans_1 = trace_1.data.spans
    loaded_spans_2 = trace_2.data.spans

    assert len(loaded_spans_1) == 2
    assert len(loaded_spans_2) == 1

    trace_1_spans = [span.to_dict() for span in loaded_spans_1]
    trace_2_spans = [span.to_dict() for span in loaded_spans_2]

    assert [span.to_dict() for span in loaded_spans_1] == trace_1_spans
    assert [span.to_dict() for span in loaded_spans_2] == trace_2_spans


def test_batch_get_traces_preserves_json_serialization(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_load_spans_json")
    trace_id = f"tr-{uuid.uuid4().hex}"

    original_span = create_test_span(
        trace_id=trace_id,
        name="json_test_span",
        span_id=111,
        status=trace_api.StatusCode.OK,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        trace_num=12345,
    )

    store.log_spans(experiment_id, [original_span])
    traces = store.batch_get_traces([trace_id])

    assert len(traces) == 1
    loaded_spans = traces[0].data.spans

    assert len(loaded_spans) == 1
    loaded_span = loaded_spans[0]

    assert loaded_span.name == original_span.name
    assert loaded_span.trace_id == original_span.trace_id
    assert loaded_span.span_id == original_span.span_id
    assert loaded_span.start_time_ns == original_span.start_time_ns
    assert loaded_span.end_time_ns == original_span.end_time_ns


def test_batch_get_traces_integration_with_trace_handler(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_integration")
    trace_id = f"tr-{uuid.uuid4().hex}"

    spans = [
        create_test_span(
            trace_id=trace_id,
            name="integration_span",
            span_id=111,
            status=trace_api.StatusCode.OK,
            trace_num=12345,
        ),
    ]

    store.log_spans(experiment_id, spans)

    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags.get(TraceTagKey.SPANS_LOCATION) == SpansLocation.TRACKING_STORE.value

    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    loaded_spans = traces[0].data.spans
    assert len(loaded_spans) == 1
    assert loaded_spans[0].name == "integration_span"


def test_batch_get_traces_with_incomplete_trace(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_incomplete_trace")
    trace_id = f"tr-{uuid.uuid4().hex}"

    spans = [
        create_test_span(
            trace_id=trace_id,
            name="incomplete_span",
            span_id=111,
            status=trace_api.StatusCode.OK,
            trace_num=12345,
        ),
    ]

    store.log_spans(experiment_id, spans)
    store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=1234,
            execution_duration=100,
            state=TraceState.OK,
            trace_metadata={
                TraceMetadataKey.SIZE_STATS: json.dumps({
                    TraceSizeStatsKey.NUM_SPANS: 2,
                }),
            },
        )
    )
    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 0

    # add another complete trace
    trace_id_2 = f"tr-{uuid.uuid4().hex}"
    spans = [
        create_test_span(
            trace_id=trace_id_2,
            name="incomplete_span",
            span_id=111,
            status=trace_api.StatusCode.OK,
            trace_num=12345,
        ),
    ]
    store.log_spans(experiment_id, spans)
    store.start_trace(
        TraceInfo(
            trace_id=trace_id_2,
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=1234,
            execution_duration=100,
            state=TraceState.OK,
        )
    )
    traces = store.batch_get_traces([trace_id, trace_id_2])
    assert len(traces) == 1
    assert traces[0].info.trace_id == trace_id_2
    assert traces[0].info.status == TraceState.OK
    assert len(traces[0].data.spans) == 1
    assert traces[0].data.spans[0].name == "incomplete_span"
    assert traces[0].data.spans[0].status.status_code == "OK"


def test_batch_get_traces_raises_for_artifact_repo_traces(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_artifact_repo_traces")

    # Create a trace via start_trace only (no log_spans call),
    # so it has no SPANS_LOCATION tag — simulating spans stored in artifact repo.
    artifact_trace_id = f"tr-{uuid.uuid4().hex}"
    store.start_trace(
        TraceInfo(
            trace_id=artifact_trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=1000,
            execution_duration=500,
            state=TraceState.OK,
        )
    )

    with pytest.raises(MlflowTracingException, match="not stored in tracking store"):
        store.batch_get_traces([artifact_trace_id])


def test_log_spans_token_usage(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_log_spans_token_usage")
    trace_id = f"tr-{uuid.uuid4().hex}"

    otel_span = create_test_otel_span(
        trace_id=trace_id,
        name="llm_call",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )

    otel_span._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }),
    }

    span = create_mlflow_span(otel_span, trace_id, "LLM")
    store.log_spans(experiment_id, [span])

    # verify token usage is stored in the trace info
    trace_info = store.get_trace_info(trace_id)
    assert trace_info.token_usage == {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }

    # verify loaded trace has same token usage
    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.token_usage is not None
    assert trace.info.token_usage["input_tokens"] == 100
    assert trace.info.token_usage["output_tokens"] == 50
    assert trace.info.token_usage["total_tokens"] == 150


def test_log_spans_update_token_usage_incrementally(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_log_spans_update_token_usage")
    trace_id = f"tr-{uuid.uuid4().hex}"

    otel_span1 = create_test_otel_span(
        trace_id=trace_id,
        name="first_llm_call",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )
    otel_span1._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }),
    }
    span1 = create_mlflow_span(otel_span1, trace_id, "LLM")
    store.log_spans(experiment_id, [span1])

    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.token_usage["input_tokens"] == 100
    assert trace.info.token_usage["output_tokens"] == 50
    assert trace.info.token_usage["total_tokens"] == 150

    otel_span2 = create_test_otel_span(
        trace_id=trace_id,
        name="second_llm_call",
        start_time=3_000_000_000,
        end_time=4_000_000_000,
        trace_id_num=12345,
        span_id_num=222,
    )
    otel_span2._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 200,
            "output_tokens": 75,
            "total_tokens": 275,
        }),
    }
    span2 = create_mlflow_span(otel_span2, trace_id, "LLM")
    store.log_spans(experiment_id, [span2])

    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.token_usage["input_tokens"] == 300
    assert trace.info.token_usage["output_tokens"] == 125
    assert trace.info.token_usage["total_tokens"] == 425


def test_log_spans_cost(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_log_spans_cost")
    trace_id = f"tr-{uuid.uuid4().hex}"

    otel_span = create_test_otel_span(
        trace_id=trace_id,
        name="llm_call",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )

    otel_span._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.LLM_COST: json.dumps({
            "input_cost": 0.01,
            "output_cost": 0.02,
            "total_cost": 0.03,
        }),
    }

    span = create_mlflow_span(otel_span, trace_id, "LLM")
    store.log_spans(experiment_id, [span])

    # verify cost is stored in the trace info
    trace_info = store.get_trace_info(trace_id)
    assert trace_info.cost == {
        "input_cost": 0.01,
        "output_cost": 0.02,
        "total_cost": 0.03,
    }

    # verify loaded trace has same cost
    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.cost is not None
    assert trace.info.cost["input_cost"] == 0.01
    assert trace.info.cost["output_cost"] == 0.02
    assert trace.info.cost["total_cost"] == 0.03


def test_log_spans_update_cost_incrementally(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_log_spans_update_cost")
    trace_id = f"tr-{uuid.uuid4().hex}"

    otel_span1 = create_test_otel_span(
        trace_id=trace_id,
        name="first_llm_call",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )
    otel_span1._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.LLM_COST: json.dumps({
            "input_cost": 0.01,
            "output_cost": 0.02,
            "total_cost": 0.03,
        }),
    }
    span1 = create_mlflow_span(otel_span1, trace_id, "LLM")
    store.log_spans(experiment_id, [span1])

    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.cost["input_cost"] == 0.01
    assert trace.info.cost["output_cost"] == 0.02
    assert trace.info.cost["total_cost"] == 0.03

    otel_span2 = create_test_otel_span(
        trace_id=trace_id,
        name="second_llm_call",
        start_time=3_000_000_000,
        end_time=4_000_000_000,
        trace_id_num=12345,
        span_id_num=222,
    )
    otel_span2._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.LLM_COST: json.dumps({
            "input_cost": 0.005,
            "output_cost": 0.01,
            "total_cost": 0.015,
        }),
    }
    span2 = create_mlflow_span(otel_span2, trace_id, "LLM")
    store.log_spans(experiment_id, [span2])

    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.cost["input_cost"] == 0.015
    assert trace.info.cost["output_cost"] == 0.03
    assert trace.info.cost["total_cost"] == 0.045


def test_log_spans_does_not_overwrite_finalized_trace_info(store: SqlAlchemyStore) -> None:
    """start_trace() sets TRACE_INFO_FINALIZED; subsequent log_spans() must not overwrite
    request_time, execution_duration, session_id, token_usage, or cost.
    """
    experiment_id = store.create_experiment("test_trace_info_finalized")
    trace_id = f"tr-{uuid.uuid4().hex}"

    # start_trace() writes authoritative trace-level values and sets TRACE_INFO_FINALIZED.
    authoritative_request_time = 1_000
    authoritative_duration = 500
    authoritative_session = "session-from-start-trace"
    authoritative_token_usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    authoritative_cost = {"input_cost": 0.001, "output_cost": 0.0005, "total_cost": 0.0015}

    _create_trace(
        store,
        trace_id,
        experiment_id,
        request_time=authoritative_request_time,
        execution_duration=authoritative_duration,
        trace_metadata={
            TraceMetadataKey.TRACE_SESSION: authoritative_session,
            TraceMetadataKey.TOKEN_USAGE: json.dumps(authoritative_token_usage),
            TraceMetadataKey.COST: json.dumps(authoritative_cost),
        },
    )

    # log_spans() arrives with different values that should all be ignored.
    otel_span = create_test_otel_span(
        trace_id=trace_id,
        name="llm_call",
        start_time=1_000_000,  # earlier start (ms=1) — should NOT update request_time
        end_time=9_000_000_000,  # later end — should NOT update execution_duration
        trace_id_num=99999,
        span_id_num=1,
    )
    otel_span._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        "session.id": "session-from-log-spans",
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 999,
            "output_tokens": 999,
            "total_tokens": 1998,
        }),
        SpanAttributeKey.LLM_COST: json.dumps({
            "input_cost": 9.99,
            "output_cost": 9.99,
            "total_cost": 19.98,
        }),
    }
    span = create_mlflow_span(otel_span, trace_id, "LLM")
    store.log_spans(experiment_id, [span])

    trace_info = store.get_trace_info(trace_id)

    # Timestamp and duration unchanged
    assert trace_info.request_time == authoritative_request_time
    assert trace_info.execution_duration == authoritative_duration

    # Session ID unchanged
    assert trace_info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == authoritative_session

    # Token usage unchanged
    assert trace_info.token_usage == authoritative_token_usage

    # Cost unchanged
    stored_cost = json.loads(trace_info.trace_metadata[TraceMetadataKey.COST])
    assert stored_cost == authoritative_cost


def test_batch_get_traces_token_usage(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_batch_get_traces_token_usage")

    trace_id_1 = f"tr-{uuid.uuid4().hex}"
    otel_span1 = create_test_otel_span(
        trace_id=trace_id_1,
        name="trace1_span",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )
    otel_span1._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id_1, cls=TraceJSONEncoder),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }),
    }
    span1 = create_mlflow_span(otel_span1, trace_id_1, "LLM")
    store.log_spans(experiment_id, [span1])

    trace_id_2 = f"tr-{uuid.uuid4().hex}"
    otel_span2 = create_test_otel_span(
        trace_id=trace_id_2,
        name="trace2_span",
        start_time=3_000_000_000,
        end_time=4_000_000_000,
        trace_id_num=67890,
        span_id_num=222,
    )
    otel_span2._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id_2, cls=TraceJSONEncoder),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
        }),
    }
    span2 = create_mlflow_span(otel_span2, trace_id_2, "LLM")
    store.log_spans(experiment_id, [span2])

    trace_id_3 = f"tr-{uuid.uuid4().hex}"
    otel_span3 = create_test_otel_span(
        trace_id=trace_id_3,
        name="trace3_span",
        start_time=5_000_000_000,
        end_time=6_000_000_000,
        trace_id_num=11111,
        span_id_num=333,
    )
    otel_span3._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id_3, cls=TraceJSONEncoder),
    }
    span3 = create_mlflow_span(otel_span3, trace_id_3, "UNKNOWN")
    store.log_spans(experiment_id, [span3])

    trace_infos = [
        store.get_trace_info(trace_id) for trace_id in [trace_id_1, trace_id_2, trace_id_3]
    ]
    assert trace_infos[0].token_usage == {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    assert trace_infos[1].token_usage == {
        "input_tokens": 200,
        "output_tokens": 100,
        "total_tokens": 300,
    }
    assert trace_infos[2].token_usage is None

    traces = store.batch_get_traces([trace_id_1, trace_id_2, trace_id_3])
    assert len(traces) == 3

    traces_by_id = {trace.info.trace_id: trace for trace in traces}

    trace1 = traces_by_id[trace_id_1]
    assert trace1.info.token_usage is not None
    assert trace1.info.token_usage["input_tokens"] == 100
    assert trace1.info.token_usage["output_tokens"] == 50
    assert trace1.info.token_usage["total_tokens"] == 150

    trace2 = traces_by_id[trace_id_2]
    assert trace2.info.token_usage is not None
    assert trace2.info.token_usage["input_tokens"] == 200
    assert trace2.info.token_usage["output_tokens"] == 100
    assert trace2.info.token_usage["total_tokens"] == 300

    trace3 = traces_by_id[trace_id_3]
    assert trace3.info.token_usage is None


def test_batch_get_trace_infos_basic(store: SqlAlchemyStore) -> None:
    from mlflow.tracing.constant import TraceMetadataKey

    experiment_id = store.create_experiment("test_batch_get_trace_infos")
    trace_id_1 = f"tr-{uuid.uuid4().hex}"
    trace_id_2 = f"tr-{uuid.uuid4().hex}"
    session_id = "session-123"

    # Create traces with session metadata
    trace_info_1 = TraceInfo(
        trace_id=trace_id_1,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceState.OK,
        trace_metadata={TraceMetadataKey.TRACE_SESSION: session_id},
    )
    store.start_trace(trace_info_1)

    trace_info_2 = TraceInfo(
        trace_id=trace_id_2,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=get_current_time_millis(),
        execution_duration=200,
        state=TraceState.OK,
        trace_metadata={TraceMetadataKey.TRACE_SESSION: session_id},
    )
    store.start_trace(trace_info_2)

    # Batch fetch trace infos
    trace_infos = store.batch_get_trace_infos([trace_id_1, trace_id_2])

    assert len(trace_infos) == 2
    trace_infos_by_id = {ti.trace_id: ti for ti in trace_infos}

    # Verify we got the trace infos
    assert trace_id_1 in trace_infos_by_id
    assert trace_id_2 in trace_infos_by_id

    # Verify metadata is present
    ti1 = trace_infos_by_id[trace_id_1]
    assert ti1.trace_id == trace_id_1
    assert ti1.timestamp_ms is not None
    assert ti1.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == session_id

    ti2 = trace_infos_by_id[trace_id_2]
    assert ti2.trace_id == trace_id_2
    assert ti2.timestamp_ms is not None
    assert ti2.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == session_id


def test_batch_get_trace_infos_empty(store: SqlAlchemyStore) -> None:
    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_infos = store.batch_get_trace_infos([trace_id])
    assert trace_infos == []


def test_batch_get_trace_infos_ordering(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_batch_get_trace_infos_ordering")
    trace_ids = [f"tr-{uuid.uuid4().hex}" for _ in range(3)]

    # Create traces in reverse order
    for i, trace_id in enumerate(reversed(trace_ids)):
        spans = [
            create_test_span(
                trace_id=trace_id,
                name=f"span_{i}",
                span_id=100 + i,
                status=trace_api.StatusCode.OK,
                start_ns=1_000_000_000 + i * 1_000_000_000,
                end_ns=2_000_000_000 + i * 1_000_000_000,
                trace_num=12345 + i,
            ),
        ]
        store.log_spans(experiment_id, spans)

    # Fetch in original order
    trace_infos = store.batch_get_trace_infos(trace_ids)

    # Verify order is preserved
    assert len(trace_infos) == 3
    for i, trace_info in enumerate(trace_infos):
        assert trace_info.trace_id == trace_ids[i]


def test_start_trace_creates_trace_metrics(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_start_trace_metrics")
    trace_id = f"tr-{uuid.uuid4().hex}"

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        trace_metadata={
            TraceMetadataKey.TOKEN_USAGE: json.dumps({
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            })
        },
    )
    store.start_trace(trace_info)

    with store.ManagedSessionMaker() as session:
        metrics = (
            session
            .query(SqlTraceMetrics)
            .filter(SqlTraceMetrics.request_id == trace_id)
            .order_by(SqlTraceMetrics.key)
            .all()
        )

        metrics_by_key = {metric.key: metric.value for metric in metrics}
        assert metrics_by_key == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }


def test_start_trace_merge_preserves_existing_metrics(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_merge_preserves_metrics")
    trace_id = f"tr-{uuid.uuid4().hex}"
    loc = trace_location.TraceLocation.from_experiment_id(experiment_id)
    ts = get_current_time_millis()

    store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=loc,
            request_time=ts,
            execution_duration=100,
            state=TraceStatus.OK,
            trace_metadata={
                TraceMetadataKey.TOKEN_USAGE: json.dumps({
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                })
            },
        )
    )

    # Second start_trace with a subset of metric keys triggers the merge path.
    result = store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=loc,
            request_time=ts,
            execution_duration=200,
            state=TraceStatus.OK,
            trace_metadata={
                TraceMetadataKey.TOKEN_USAGE: json.dumps({
                    "total_tokens": 110,
                    "cache_read_input_tokens": 5,
                })
            },
        )
    )

    assert result.trace_id == trace_id

    with store.ManagedSessionMaker() as session:
        metrics = (
            session
            .query(SqlTraceMetrics)
            .filter(SqlTraceMetrics.request_id == trace_id)
            .order_by(SqlTraceMetrics.key)
            .all()
        )
        metrics_by_key = {m.key: m.value for m in metrics}
        assert metrics_by_key == {
            "cache_read_input_tokens": 5,
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 110,
        }


def test_log_spans_creates_span_metrics(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_log_spans_metrics")
    trace_id = f"tr-{uuid.uuid4().hex}"

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=get_current_time_millis(),
        state=TraceStatus.OK,
    )
    store.start_trace(trace_info)

    otel_span = create_test_otel_span(
        trace_id=trace_id,
        name="llm_call",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )
    otel_span._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.LLM_COST: json.dumps({
            CostKey.INPUT_COST: 0.01,
            CostKey.OUTPUT_COST: 0.02,
            CostKey.TOTAL_COST: 0.03,
        }),
        SpanAttributeKey.MODEL: json.dumps("gpt-4-turbo"),
        SpanAttributeKey.MODEL_PROVIDER: json.dumps("openai"),
    }
    span = create_mlflow_span(otel_span, trace_id, "LLM")
    store.log_spans(experiment_id, [span])

    with store.ManagedSessionMaker() as session:
        metrics = (
            session
            .query(SqlSpanMetrics)
            .filter(SqlSpanMetrics.trace_id == trace_id, SqlSpanMetrics.span_id == span.span_id)
            .order_by(SqlSpanMetrics.key)
            .all()
        )
        metrics_by_key = {metric.key: metric.value for metric in metrics}
        assert metrics_by_key == {
            CostKey.INPUT_COST: 0.01,
            CostKey.OUTPUT_COST: 0.02,
            CostKey.TOTAL_COST: 0.03,
        }

        # Check that dimension_attributes is stored on the span
        sql_span = (
            session
            .query(SqlSpan)
            .filter(SqlSpan.trace_id == trace_id, SqlSpan.span_id == span.span_id)
            .one()
        )
        assert sql_span.dimension_attributes[SpanAttributeKey.MODEL] == "gpt-4-turbo"
        assert sql_span.dimension_attributes[SpanAttributeKey.MODEL_PROVIDER] == "openai"


def test_log_spans_updates_trace_metrics_incrementally(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_log_spans_incremental_metrics")
    trace_id = f"tr-{uuid.uuid4().hex}"

    otel_span1 = create_test_otel_span(
        trace_id=trace_id,
        name="first_llm_call",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )
    otel_span1._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }),
    }
    span1 = create_mlflow_span(otel_span1, trace_id, "LLM")
    store.log_spans(experiment_id, [span1])

    with store.ManagedSessionMaker() as session:
        metrics = (
            session
            .query(SqlTraceMetrics)
            .filter(SqlTraceMetrics.request_id == trace_id)
            .order_by(SqlTraceMetrics.key)
            .all()
        )

        metrics_by_key = {metric.key: metric.value for metric in metrics}
        assert metrics_by_key == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

    otel_span2 = create_test_otel_span(
        trace_id=trace_id,
        name="second_llm_call",
        start_time=3_000_000_000,
        end_time=4_000_000_000,
        trace_id_num=12345,
        span_id_num=222,
    )
    otel_span2._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 200,
            "output_tokens": 75,
            "total_tokens": 275,
        }),
    }
    span2 = create_mlflow_span(otel_span2, trace_id, "LLM")
    store.log_spans(experiment_id, [span2])

    with store.ManagedSessionMaker() as session:
        metrics = (
            session
            .query(SqlTraceMetrics)
            .filter(SqlTraceMetrics.request_id == trace_id)
            .order_by(SqlTraceMetrics.key)
            .all()
        )
        metrics_by_key = {metric.key: metric.value for metric in metrics}
        assert metrics_by_key == {
            "input_tokens": 300,
            "output_tokens": 125,
            "total_tokens": 425,
        }


def test_log_spans_stores_span_metrics_per_span(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_log_spans_metrics_per_span")
    trace_id = f"tr-{uuid.uuid4().hex}"

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=get_current_time_millis(),
        state=TraceStatus.OK,
    )
    store.start_trace(trace_info)

    otel_span1 = create_test_otel_span(
        trace_id=trace_id,
        name="first_llm_call",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )
    otel_span1._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.LLM_COST: json.dumps({
            CostKey.INPUT_COST: 0.001,
            CostKey.OUTPUT_COST: 0.002,
            CostKey.TOTAL_COST: 0.003,
        }),
    }
    span1 = create_mlflow_span(otel_span1, trace_id, "LLM")

    otel_span2 = create_test_otel_span(
        trace_id=trace_id,
        name="second_llm_call",
        start_time=3_000_000_000,
        end_time=4_000_000_000,
        trace_id_num=12345,
        span_id_num=222,
    )
    otel_span2._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.LLM_COST: json.dumps({
            CostKey.INPUT_COST: 0.01,
            CostKey.OUTPUT_COST: 0.02,
            CostKey.TOTAL_COST: 0.03,
        }),
    }
    span2 = create_mlflow_span(otel_span2, trace_id, "LLM")

    store.log_spans(experiment_id, [span1, span2])

    with store.ManagedSessionMaker() as session:
        all_metrics = (
            session
            .query(SqlSpanMetrics)
            .filter(SqlSpanMetrics.trace_id == trace_id)
            .order_by(SqlSpanMetrics.span_id, SqlSpanMetrics.key)
            .all()
        )

        span1_metrics = {m.key: m.value for m in all_metrics if m.span_id == span1.span_id}
        assert span1_metrics == {
            CostKey.INPUT_COST: 0.001,
            CostKey.OUTPUT_COST: 0.002,
            CostKey.TOTAL_COST: 0.003,
        }

        span2_metrics = {m.key: m.value for m in all_metrics if m.span_id == span2.span_id}
        assert span2_metrics == {
            CostKey.INPUT_COST: 0.01,
            CostKey.OUTPUT_COST: 0.02,
            CostKey.TOTAL_COST: 0.03,
        }


def test_get_trace_basic(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_get_trace")
    trace_id = f"tr-{uuid.uuid4().hex}"

    spans = [
        create_test_span(
            trace_id=trace_id,
            name="root_span",
            span_id=111,
            status=trace_api.StatusCode.OK,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
            trace_num=12345,
        ),
        create_test_span(
            trace_id=trace_id,
            name="child_span",
            span_id=222,
            parent_id=111,
            status=trace_api.StatusCode.UNSET,
            start_ns=1_500_000_000,
            end_ns=1_800_000_000,
            trace_num=12345,
        ),
    ]

    store.log_spans(experiment_id, spans)
    trace = store.get_trace(trace_id)

    assert trace is not None
    loaded_spans = trace.data.spans

    assert len(loaded_spans) == 2

    root_span = next(s for s in loaded_spans if s.name == "root_span")
    child_span = next(s for s in loaded_spans if s.name == "child_span")

    assert root_span.trace_id == trace_id
    assert root_span.span_id == "000000000000006f"
    assert root_span.parent_id is None
    assert root_span.start_time_ns == 1_000_000_000
    assert root_span.end_time_ns == 2_000_000_000

    assert child_span.trace_id == trace_id
    assert child_span.span_id == "00000000000000de"
    assert child_span.parent_id == "000000000000006f"
    assert child_span.start_time_ns == 1_500_000_000
    assert child_span.end_time_ns == 1_800_000_000


def test_get_trace_not_found(store: SqlAlchemyStore) -> None:
    trace_id = f"tr-{uuid.uuid4().hex}"
    with pytest.raises(MlflowException, match=f"Trace with ID {trace_id} is not found."):
        store.get_trace(trace_id)


def test_start_trace_only_no_spans_location_tag(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_start_trace_only")
    trace_id = f"tr-{uuid.uuid4().hex}"

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1000,
        execution_duration=1000,
        state=TraceState.OK,
        tags={"custom_tag": "value"},
        trace_metadata={"source": "test"},
    )
    created_trace_info = store.start_trace(trace_info)

    assert TraceTagKey.SPANS_LOCATION not in created_trace_info.tags


def test_start_trace_then_log_spans_adds_tag(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_start_trace_then_log_spans")
    trace_id = f"tr-{uuid.uuid4().hex}"

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1000,
        execution_duration=1000,
        state=TraceState.OK,
        tags={"custom_tag": "value"},
        trace_metadata={"source": "test"},
    )
    store.start_trace(trace_info)

    span = create_test_span(
        trace_id=trace_id,
        name="test_span",
        span_id=111,
        status=trace_api.StatusCode.OK,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        trace_num=12345,
    )
    store.log_spans(experiment_id, [span])

    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.TRACKING_STORE.value


def test_log_spans_then_start_trace_preserves_tag(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_log_spans_then_start_trace")
    trace_id = f"tr-{uuid.uuid4().hex}"

    span = create_test_span(
        trace_id=trace_id,
        name="test_span",
        span_id=111,
        status=trace_api.StatusCode.OK,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        trace_num=12345,
    )
    store.log_spans(experiment_id, [span])

    trace_info_for_start = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1000,
        execution_duration=1000,
        state=TraceState.OK,
        tags={"custom_tag": "value"},
        trace_metadata={"source": "test"},
    )
    store.start_trace(trace_info_for_start)

    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.TRACKING_STORE.value


def test_log_spans_then_start_trace_preserves_preview(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_preview_preserved")
    trace_id = f"tr-{uuid.uuid4().hex}"

    span = create_test_span(
        trace_id=trace_id,
        name="llm_call",
        span_id=111,
        status=trace_api.StatusCode.OK,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        trace_num=12345,
        attributes={
            "input.value": '{"messages": [{"role": "user", "content": "Hello"}]}',
            "output.value": '{"choices": [{"message": {"role": "assistant", "content": "Hi"}}]}',
            "openinference.span.kind": "LLM",
        },
    )
    store.log_spans(experiment_id, [span])

    trace_info_for_start = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1000,
        execution_duration=1000,
        state=TraceState.OK,
        tags={"custom_tag": "value"},
        trace_metadata={"source": "test"},
    )
    store.start_trace(trace_info_for_start)

    trace_info = store.get_trace_info(trace_id)
    assert trace_info.request_preview is not None
    assert trace_info.response_preview is not None
    assert "Hello" in trace_info.request_preview
    assert "Hi" in trace_info.response_preview


@pytest.mark.skipif(
    mlflow.get_tracking_uri().startswith("mysql"),
    reason="MySQL does not support concurrent log_spans calls for now",
)
def test_concurrent_log_spans_spans_location_tag(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_concurrent_log_spans")
    trace_id = f"tr-{uuid.uuid4().hex}"

    def log_span_worker(span_id):
        span = create_test_span(
            trace_id=trace_id,
            name=f"concurrent_span_{span_id}",
            span_id=span_id,
            parent_id=111 if span_id != 111 else None,
            status=trace_api.StatusCode.OK,
            start_ns=1_000_000_000 + span_id * 1000,
            end_ns=2_000_000_000 + span_id * 1000,
            trace_num=12345,
        )
        store.log_spans(experiment_id, [span])
        return span_id

    # Simulate client-side workspace selection and ensure it propagates to worker threads.
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        # Launch multiple concurrent log_spans calls
        with ThreadPoolExecutor(
            max_workers=5, thread_name_prefix="test-sqlalchemy-log-spans"
        ) as executor:
            futures = [executor.submit(log_span_worker, i) for i in range(111, 116)]

            # Wait for all to complete
            results = [future.result() for future in futures]

        # All workers should complete successfully
        assert len(results) == 5
        assert set(results) == {111, 112, 113, 114, 115}

        # Verify the SPANS_LOCATION tag was created correctly
        trace_info = store.get_trace_info(trace_id)
        assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.TRACKING_STORE.value

        # Verify all spans were logged
        trace = store.get_trace(trace_id)
        assert len(trace.data.spans) == 5
        span_names = {span.name for span in trace.data.spans}
        expected_names = {f"concurrent_span_{i}" for i in range(111, 116)}
        assert span_names == expected_names


@pytest.mark.parametrize("allow_partial", [True, False])
def test_get_trace_with_partial_trace(store: SqlAlchemyStore, allow_partial: bool) -> None:
    experiment_id = store.create_experiment("test_partial_trace")
    trace_id = f"tr-{uuid.uuid4().hex}"

    # Log only 1 span but indicate trace should have 2 spans
    spans = [
        create_test_span(
            trace_id=trace_id,
            name="span_1",
            span_id=111,
            status=trace_api.StatusCode.OK,
            trace_num=12345,
        ),
    ]

    store.log_spans(experiment_id, spans)
    store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=1234,
            execution_duration=100,
            state=TraceState.OK,
            trace_metadata={
                TraceMetadataKey.SIZE_STATS: json.dumps({
                    TraceSizeStatsKey.NUM_SPANS: 2,  # Expecting 2 spans
                }),
            },
        )
    )

    if allow_partial:
        trace = store.get_trace(trace_id, allow_partial=allow_partial)
        assert trace is not None
        assert len(trace.data.spans) == 1
        assert trace.data.spans[0].name == "span_1"
    else:
        with pytest.raises(
            MlflowException,
            match=f"Trace with ID {trace_id} is not fully exported yet",
        ):
            store.get_trace(trace_id, allow_partial=allow_partial)


@pytest.mark.parametrize("allow_partial", [True, False])
def test_get_trace_with_complete_trace(store: SqlAlchemyStore, allow_partial: bool) -> None:
    experiment_id = store.create_experiment("test_complete_trace")
    trace_id = f"tr-{uuid.uuid4().hex}"

    # Log 2 spans matching the expected count
    spans = [
        create_test_span(
            trace_id=trace_id,
            name="span_1",
            span_id=111,
            status=trace_api.StatusCode.OK,
            trace_num=12345,
        ),
        create_test_span(
            trace_id=trace_id,
            name="span_2",
            span_id=222,
            parent_id=111,
            status=trace_api.StatusCode.OK,
            trace_num=12345,
        ),
    ]

    store.log_spans(experiment_id, spans)
    store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=1234,
            execution_duration=100,
            state=TraceState.OK,
            trace_metadata={
                TraceMetadataKey.SIZE_STATS: json.dumps({
                    TraceSizeStatsKey.NUM_SPANS: 2,  # Expecting 2 spans
                }),
            },
        )
    )

    # should always return the trace
    trace = store.get_trace(trace_id, allow_partial=allow_partial)
    assert trace is not None
    assert len(trace.data.spans) == 2


def test_log_spans_session_id_handling(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_session_id")

    # Session ID gets stored from span attributes
    trace_id1 = f"tr-{uuid.uuid4().hex}"
    otel_span1 = create_test_otel_span(trace_id=trace_id1)
    otel_span1._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id1, cls=TraceJSONEncoder),
        "session.id": "session-123",
    }
    span1 = create_mlflow_span(otel_span1, trace_id1, "LLM")
    store.log_spans(experiment_id, [span1])

    trace_info1 = store.get_trace_info(trace_id1)
    assert trace_info1.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "session-123"

    # Existing session ID is preserved
    trace_id2 = f"tr-{uuid.uuid4().hex}"
    trace_with_session = TraceInfo(
        trace_id=trace_id2,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1234,
        execution_duration=100,
        state=TraceState.IN_PROGRESS,
        trace_metadata={TraceMetadataKey.TRACE_SESSION: "existing-session"},
    )
    store.start_trace(trace_with_session)

    otel_span2 = create_test_otel_span(trace_id=trace_id2)
    otel_span2._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id2, cls=TraceJSONEncoder),
        "session.id": "different-session",
    }
    span2 = create_mlflow_span(otel_span2, trace_id2, "LLM")
    store.log_spans(experiment_id, [span2])

    trace_info2 = store.get_trace_info(trace_id2)
    assert trace_info2.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "existing-session"

    # No session ID means no metadata
    trace_id3 = f"tr-{uuid.uuid4().hex}"
    otel_span3 = create_test_otel_span(trace_id=trace_id3)
    span3 = create_mlflow_span(otel_span3, trace_id3, "LLM")
    store.log_spans(experiment_id, [span3])

    trace_info3 = store.get_trace_info(trace_id3)
    assert TraceMetadataKey.TRACE_SESSION not in trace_info3.trace_metadata


def test_log_spans_user_id_handling(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_user_id")

    # User ID gets stored from span attributes
    trace_id1 = f"tr-{uuid.uuid4().hex}"
    otel_span1 = create_test_otel_span(trace_id=trace_id1)
    otel_span1._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id1, cls=TraceJSONEncoder),
        "user.id": "alice",
    }
    span1 = create_mlflow_span(otel_span1, trace_id1, "LLM")
    store.log_spans(experiment_id, [span1])

    trace_info1 = store.get_trace_info(trace_id1)
    assert trace_info1.trace_metadata.get(TraceMetadataKey.TRACE_USER) == "alice"

    # Existing user ID is preserved
    trace_id2 = f"tr-{uuid.uuid4().hex}"
    trace_with_user = TraceInfo(
        trace_id=trace_id2,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1234,
        execution_duration=100,
        state=TraceState.IN_PROGRESS,
        trace_metadata={TraceMetadataKey.TRACE_USER: "existing-user"},
    )
    store.start_trace(trace_with_user)

    otel_span2 = create_test_otel_span(trace_id=trace_id2)
    otel_span2._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id2, cls=TraceJSONEncoder),
        "user.id": "different-user",
    }
    span2 = create_mlflow_span(otel_span2, trace_id2, "LLM")
    store.log_spans(experiment_id, [span2])

    trace_info2 = store.get_trace_info(trace_id2)
    assert trace_info2.trace_metadata.get(TraceMetadataKey.TRACE_USER) == "existing-user"

    # No user ID means no metadata
    trace_id3 = f"tr-{uuid.uuid4().hex}"
    otel_span3 = create_test_otel_span(trace_id=trace_id3)
    span3 = create_mlflow_span(otel_span3, trace_id3, "LLM")
    store.log_spans(experiment_id, [span3])

    trace_info3 = store.get_trace_info(trace_id3)
    assert TraceMetadataKey.TRACE_USER not in trace_info3.trace_metadata

    # Both session and user ID work together
    trace_id4 = f"tr-{uuid.uuid4().hex}"
    otel_span4 = create_test_otel_span(trace_id=trace_id4)
    otel_span4._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id4, cls=TraceJSONEncoder),
        "session.id": "session-456",
        "user.id": "bob",
    }
    span4 = create_mlflow_span(otel_span4, trace_id4, "LLM")
    store.log_spans(experiment_id, [span4])

    trace_info4 = store.get_trace_info(trace_id4)
    assert trace_info4.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "session-456"
    assert trace_info4.trace_metadata.get(TraceMetadataKey.TRACE_USER) == "bob"


def test_find_completed_sessions(store: SqlAlchemyStore):
    """
    Test finding completed sessions based on their last trace timestamp.
    Sessions with last trace in time window are returned, ordered by last_trace_timestamp.
    """
    exp_id = store.create_experiment("test_find_completed_sessions")

    # Session A: last trace at t=2000
    for timestamp, trace_id in [(1000, "trace_a1"), (2000, "trace_a2")]:
        _create_trace(
            store,
            trace_id,
            exp_id,
            request_time=timestamp,
            trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-a"},
        )

    # Session B: last trace at t=4000
    for timestamp, trace_id in [(3000, "trace_b1"), (4000, "trace_b2")]:
        _create_trace(
            store,
            trace_id,
            exp_id,
            request_time=timestamp,
            trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-b"},
        )

    # Session C: last trace at t=10000 (outside query window)
    for timestamp, trace_id in [(5000, "trace_c1"), (10000, "trace_c2")]:
        _create_trace(
            store,
            trace_id,
            exp_id,
            request_time=timestamp,
            trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-c"},
        )

    _create_trace(store, "trace_no_session", exp_id, request_time=2500)

    # Query window [0, 5000] should return session-a and session-b
    completed = store.find_completed_sessions(
        experiment_id=exp_id,
        min_last_trace_timestamp_ms=0,
        max_last_trace_timestamp_ms=5000,
    )

    assert len(completed) == 2
    assert {s.session_id for s in completed} == {"session-a", "session-b"}
    assert completed[0].session_id == "session-a"
    assert completed[0].first_trace_timestamp_ms == 1000
    assert completed[0].last_trace_timestamp_ms == 2000
    assert completed[1].session_id == "session-b"
    assert completed[1].first_trace_timestamp_ms == 3000
    assert completed[1].last_trace_timestamp_ms == 4000

    # Narrower window [3000, 5000] should only return session-b
    completed = store.find_completed_sessions(
        experiment_id=exp_id,
        min_last_trace_timestamp_ms=3000,
        max_last_trace_timestamp_ms=5000,
    )
    assert len(completed) == 1
    assert completed[0].session_id == "session-b"

    # Test max_results pagination
    completed = store.find_completed_sessions(
        experiment_id=exp_id,
        min_last_trace_timestamp_ms=0,
        max_last_trace_timestamp_ms=5000,
        max_results=1,
    )
    assert len(completed) == 1
    assert completed[0].session_id == "session-a"


def test_find_completed_sessions_aggregates_across_all_traces(store: SqlAlchemyStore):
    """
    Regression test: first/last timestamps should be computed across ALL session traces,
    not just those matching the min_last_trace_timestamp_ms filter.
    """
    exp_id = store.create_experiment("test_session_timestamp_aggregation")

    _create_trace(
        store,
        "trace1",
        exp_id,
        request_time=1000,
        trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-a"},
    )
    _create_trace(
        store,
        "trace2",
        exp_id,
        request_time=3000,
        trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-a"},
    )

    completed = store.find_completed_sessions(
        experiment_id=exp_id, min_last_trace_timestamp_ms=2000, max_last_trace_timestamp_ms=4000
    )

    assert len(completed) == 1
    assert completed[0].first_trace_timestamp_ms == 1000
    assert completed[0].last_trace_timestamp_ms == 3000


def test_find_completed_sessions_with_filter_string(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_find_completed_sessions_with_filter")

    # Session A: first trace env="prod", second env="dev" - should match prod filter
    # Session B: first trace env="dev", second env="prod" - should NOT match prod filter
    for session_id, times, envs in [
        ("session-a", [1000, 2000], ["prod", "dev"]),
        ("session-b", [3000, 4000], ["dev", "prod"]),
    ]:
        for timestamp, env in zip(times, envs):
            _create_trace(
                store,
                f"trace_{session_id}_{timestamp}",
                exp_id,
                request_time=timestamp,
                trace_metadata={TraceMetadataKey.TRACE_SESSION: session_id},
                tags={"env": env},
            )

    # Tag filter should only match session-a (first trace has env=prod)
    completed = store.find_completed_sessions(
        experiment_id=exp_id,
        min_last_trace_timestamp_ms=0,
        max_last_trace_timestamp_ms=10000,
        filter_string="tag.env = 'prod'",
    )
    assert len(completed) == 1
    assert completed[0].session_id == "session-a"
    assert completed[0].first_trace_timestamp_ms == 1000
    assert completed[0].last_trace_timestamp_ms == 2000

    # Session C: test metadata filter (first trace user_id="alice", second user_id="bob")
    for timestamp, user in [(5000, "alice"), (6000, "bob")]:
        _create_trace(
            store,
            f"trace_c_{timestamp}",
            exp_id,
            request_time=timestamp,
            trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-c", "user_id": user},
        )

    # Metadata filter should match session-c (first trace has user_id=alice)
    completed = store.find_completed_sessions(
        experiment_id=exp_id,
        min_last_trace_timestamp_ms=0,
        max_last_trace_timestamp_ms=10000,
        filter_string="metadata.user_id = 'alice'",
    )
    assert len(completed) == 1
    assert completed[0].session_id == "session-c"


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
