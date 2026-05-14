import json
import math
import os
import shutil
import time
import uuid
from dataclasses import dataclass
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
    Expectation,
    Feedback,
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
