import json
import math
import os
import pathlib
import random
import re
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
from packaging.version import Version
from sqlalchemy.exc import IntegrityError

import mlflow
import mlflow.db
import mlflow.store.db.base_sql_model
from mlflow import entities
from mlflow.entities import (
    AssessmentSource,
    AssessmentSourceType,
    Expectation,
    Experiment,
    ExperimentTag,
    Feedback,
    Metric,
    Param,
    RunStatus,
    RunTag,
    SourceType,
    ViewType,
    _DatasetSummary,
    trace_location,
)
from mlflow.entities.assessment import ExpectationValue, FeedbackValue
from mlflow.entities.dataset_record import DatasetRecord
from mlflow.entities.logged_model_output import LoggedModelOutput
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.span import Span, create_mlflow_span
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_state import TraceState
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import (
    MLFLOW_TRACKING_URI,
)
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    TEMPORARILY_UNAVAILABLE,
    ErrorCode,
)
from mlflow.store.db.db_types import MSSQL, MYSQL, POSTGRES, SQLITE
from mlflow.store.db.utils import (
    _get_latest_schema_revision,
    _get_schema_version,
)
from mlflow.store.entities import PagedList
from mlflow.store.tracking import (
    SEARCH_MAX_RESULTS_DEFAULT,
    SEARCH_MAX_RESULTS_THRESHOLD,
)
from mlflow.store.tracking.dbmodels import models
from mlflow.store.tracking.dbmodels.models import (
    SqlDataset,
    SqlEntityAssociation,
    SqlEvaluationDataset,
    SqlEvaluationDatasetRecord,
    SqlExperiment,
    SqlExperimentTag,
    SqlInput,
    SqlInputTag,
    SqlLatestMetric,
    SqlLoggedModel,
    SqlLoggedModelMetric,
    SqlLoggedModelParam,
    SqlLoggedModelTag,
    SqlMetric,
    SqlParam,
    SqlRun,
    SqlSpan,
    SqlTag,
    SqlTraceInfo,
    SqlTraceMetadata,
    SqlTraceMetrics,
    SqlTraceTag,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore, _get_orderby_clauses
from mlflow.tracing.constant import (
    MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE,
    SpanAttributeKey,
    SpansLocation,
    TraceMetadataKey,
    TraceSizeStatsKey,
    TraceTagKey,
)
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils import mlflow_tags
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_DATASET_CONTEXT,
    MLFLOW_RUN_NAME,
)
from mlflow.utils.name_utils import _GENERATOR_PREDICATES
from mlflow.utils.os import is_windows
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.validation import (
    MAX_DATASET_DIGEST_SIZE,
    MAX_DATASET_NAME_SIZE,
    MAX_DATASET_PROFILE_SIZE,
    MAX_DATASET_SCHEMA_SIZE,
    MAX_DATASET_SOURCE_SIZE,
    MAX_EXPERIMENT_NAME_LENGTH,
    MAX_INPUT_TAG_KEY_SIZE,
    MAX_INPUT_TAG_VALUE_SIZE,
    MAX_TAG_VAL_LENGTH,
)

from tests.integration.utils import invoke_cli_runner
from tests.store.tracking.test_file_store import assert_dataset_inputs_equal

DB_URI = "sqlite:///"
ARTIFACT_URI = "artifact_folder"

pytestmark = pytest.mark.notrackingurimock

IS_MSSQL = MLFLOW_TRACKING_URI.get() and MLFLOW_TRACKING_URI.get().startswith("mssql+pyodbc")


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
    return create_mlflow_span(otel_span, trace_id, span_type)


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
            SqlExperiment,
        ):
            session.query(model).delete()

        # Reset experiment_id to start at 1
        if reset_experiment_id := _get_query_to_reset_experiment_id(store):
            session.execute(sqlalchemy.sql.text(reset_experiment_id))


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


def test_default_experiment(store: SqlAlchemyStore):
    experiments = store.search_experiments()
    assert len(experiments) == 1

    first = experiments[0]
    assert first.experiment_id == "0"
    assert first.name == "Default"


def test_default_experiment_lifecycle(store: SqlAlchemyStore, tmp_path):
    default_experiment = store.get_experiment(experiment_id=0)
    assert default_experiment.name == Experiment.DEFAULT_EXPERIMENT_NAME
    assert default_experiment.lifecycle_stage == entities.LifecycleStage.ACTIVE

    _create_experiments(store, "aNothEr")
    all_experiments = [e.name for e in store.search_experiments()]
    assert set(all_experiments) == {"aNothEr", "Default"}

    store.delete_experiment(0)

    assert [e.name for e in store.search_experiments()] == ["aNothEr"]
    another = store.get_experiment(1)
    assert another.name == "aNothEr"

    default_experiment = store.get_experiment(experiment_id=0)
    assert default_experiment.name == Experiment.DEFAULT_EXPERIMENT_NAME
    assert default_experiment.lifecycle_stage == entities.LifecycleStage.DELETED

    # destroy SqlStore and make a new one
    db_uri = store.db_uri
    artifact_uri = store.artifact_root_uri
    del store
    store = SqlAlchemyStore(db_uri, artifact_uri)

    # test that default experiment is not reactivated
    default_experiment = store.get_experiment(experiment_id=0)
    assert default_experiment.name == Experiment.DEFAULT_EXPERIMENT_NAME
    assert default_experiment.lifecycle_stage == entities.LifecycleStage.DELETED

    assert [e.name for e in store.search_experiments()] == ["aNothEr"]
    all_experiments = [e.name for e in store.search_experiments(ViewType.ALL)]
    assert set(all_experiments) == {"aNothEr", "Default"}

    # ensure that experiment ID dor active experiment is unchanged
    another = store.get_experiment(1)
    assert another.name == "aNothEr"

    if MLFLOW_TRACKING_URI.get():
        with store.ManagedSessionMaker() as session:
            default_exp = (
                session.query(SqlExperiment)
                .filter(SqlExperiment.experiment_id == store.DEFAULT_EXPERIMENT_ID)
                .first()
            )
            if default_exp:
                default_exp.lifecycle_stage = entities.LifecycleStage.ACTIVE
                session.commit()


def test_raise_duplicate_experiments(store: SqlAlchemyStore):
    with pytest.raises(Exception, match=r"Experiment\(name=.+\) already exists"):
        _create_experiments(store, ["test", "test"])


def test_duplicate_experiment_with_artifact_location_returns_resource_already_exists(
    store: SqlAlchemyStore, tmp_path: Path
):
    exp_name = "test_duplicate_with_artifact_location"
    artifact_location = str(tmp_path / "test_artifacts")

    # First creation should succeed
    store.create_experiment(exp_name, artifact_location=artifact_location)

    # Second creation should raise MlflowException with RESOURCE_ALREADY_EXISTS error code
    with pytest.raises(MlflowException, match="already exists") as exc_info:
        store.create_experiment(exp_name, artifact_location=artifact_location)

    # Verify that the error code is RESOURCE_ALREADY_EXISTS, not BAD_REQUEST
    assert exc_info.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_raise_experiment_dont_exist(store: SqlAlchemyStore):
    with pytest.raises(Exception, match=r"No Experiment with id=.+ exists"):
        store.get_experiment(experiment_id=100)


def test_delete_experiment(store: SqlAlchemyStore):
    experiments = _create_experiments(store, ["morty", "rick", "rick and morty"])

    all_experiments = store.search_experiments()
    assert len(all_experiments) == len(experiments) + 1  # default

    exp_id = experiments[0]
    exp = store.get_experiment(exp_id)
    time.sleep(0.01)
    store.delete_experiment(exp_id)

    updated_exp = store.get_experiment(exp_id)
    assert updated_exp.lifecycle_stage == entities.LifecycleStage.DELETED

    assert len(store.search_experiments()) == len(all_experiments) - 1
    assert updated_exp.last_update_time > exp.last_update_time


def test_delete_restore_experiment_with_runs(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "test exp")
    run1 = _run_factory(store, config=_get_run_configs(experiment_id)).info.run_id
    run2 = _run_factory(store, config=_get_run_configs(experiment_id)).info.run_id
    store.delete_run(run1)
    run_ids = [run1, run2]

    store.delete_experiment(experiment_id)

    updated_exp = store.get_experiment(experiment_id)
    assert updated_exp.lifecycle_stage == entities.LifecycleStage.DELETED

    deleted_run_list = store.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        run_view_type=ViewType.DELETED_ONLY,
    )

    assert len(deleted_run_list) == 2
    for deleted_run in deleted_run_list:
        assert deleted_run.info.lifecycle_stage == entities.LifecycleStage.DELETED
        assert deleted_run.info.experiment_id in experiment_id
        assert deleted_run.info.run_id in run_ids
        with store.ManagedSessionMaker() as session:
            assert store._get_run(session, deleted_run.info.run_id).deleted_time is not None

    store.restore_experiment(experiment_id)

    updated_exp = store.get_experiment(experiment_id)
    assert updated_exp.lifecycle_stage == entities.LifecycleStage.ACTIVE

    restored_run_list = store.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
    )

    assert len(restored_run_list) == 2
    for restored_run in restored_run_list:
        assert restored_run.info.lifecycle_stage == entities.LifecycleStage.ACTIVE
        with store.ManagedSessionMaker() as session:
            assert store._get_run(session, restored_run.info.run_id).deleted_time is None
        assert restored_run.info.experiment_id in experiment_id
        assert restored_run.info.run_id in run_ids


def test_get_experiment(store: SqlAlchemyStore):
    name = "goku"
    experiment_id = _create_experiments(store, name)
    actual = store.get_experiment(experiment_id)
    assert actual.name == name
    assert actual.experiment_id == experiment_id

    actual_by_name = store.get_experiment_by_name(name)
    assert actual_by_name.name == name
    assert actual_by_name.experiment_id == experiment_id
    assert store.get_experiment_by_name("idontexist") is None

    store.delete_experiment(experiment_id)
    assert store.get_experiment_by_name(name).experiment_id == experiment_id


def test_search_experiments_view_type(store: SqlAlchemyStore):
    experiment_names = ["a", "b"]
    experiment_ids = _create_experiments(store, experiment_names)
    store.delete_experiment(experiment_ids[1])

    experiments = store.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    assert [e.name for e in experiments] == ["a", "Default"]
    experiments = store.search_experiments(view_type=ViewType.DELETED_ONLY)
    assert [e.name for e in experiments] == ["b"]
    experiments = store.search_experiments(view_type=ViewType.ALL)
    assert [e.name for e in experiments] == ["b", "a", "Default"]


def test_search_experiments_filter_by_attribute(store: SqlAlchemyStore):
    experiment_names = ["a", "ab", "Abc"]
    _create_experiments(store, experiment_names)

    experiments = store.search_experiments(filter_string="name = 'a'")
    assert [e.name for e in experiments] == ["a"]
    experiments = store.search_experiments(filter_string="attribute.name = 'a'")
    assert [e.name for e in experiments] == ["a"]
    experiments = store.search_experiments(filter_string="attribute.`name` = 'a'")
    assert [e.name for e in experiments] == ["a"]
    experiments = store.search_experiments(filter_string="attribute.`name` != 'a'")
    assert [e.name for e in experiments] == ["Abc", "ab", "Default"]
    experiments = store.search_experiments(filter_string="name LIKE 'a%'")
    assert [e.name for e in experiments] == ["ab", "a"]
    experiments = store.search_experiments(filter_string="name ILIKE 'a%'")
    assert [e.name for e in experiments] == ["Abc", "ab", "a"]
    experiments = store.search_experiments(filter_string="name ILIKE 'a%' AND name ILIKE '%b'")
    assert [e.name for e in experiments] == ["ab"]


def test_search_experiments_filter_by_time_attribute(store: SqlAlchemyStore):
    # Sleep to ensure that the first experiment has a different creation_time than the default
    # experiment and eliminate flakiness.
    time.sleep(0.001)
    time_before_create1 = get_current_time_millis()
    exp_id1 = store.create_experiment("1")
    exp1 = store.get_experiment(exp_id1)
    time.sleep(0.001)
    time_before_create2 = get_current_time_millis()
    exp_id2 = store.create_experiment("2")
    exp2 = store.get_experiment(exp_id2)

    experiments = store.search_experiments(filter_string=f"creation_time = {exp1.creation_time}")
    assert [e.experiment_id for e in experiments] == [exp_id1]

    experiments = store.search_experiments(filter_string=f"creation_time != {exp1.creation_time}")
    assert [e.experiment_id for e in experiments] == [
        exp_id2,
        store.DEFAULT_EXPERIMENT_ID,
    ]

    experiments = store.search_experiments(filter_string=f"creation_time >= {time_before_create1}")
    assert [e.experiment_id for e in experiments] == [exp_id2, exp_id1]

    experiments = store.search_experiments(filter_string=f"creation_time < {time_before_create2}")
    assert [e.experiment_id for e in experiments] == [
        exp_id1,
        store.DEFAULT_EXPERIMENT_ID,
    ]

    # To avoid that the creation_time equals `now`, we wait one additional millisecond.
    time.sleep(0.001)
    now = get_current_time_millis()
    experiments = store.search_experiments(filter_string=f"creation_time >= {now}")
    assert experiments == []

    time.sleep(0.001)
    time_before_rename = get_current_time_millis()
    store.rename_experiment(exp_id1, "new_name")
    experiments = store.search_experiments(
        filter_string=f"last_update_time >= {time_before_rename}"
    )
    assert [e.experiment_id for e in experiments] == [exp_id1]

    experiments = store.search_experiments(
        filter_string=f"last_update_time <= {get_current_time_millis()}"
    )
    assert {e.experiment_id for e in experiments} == {
        exp_id1,
        exp_id2,
        store.DEFAULT_EXPERIMENT_ID,
    }

    experiments = store.search_experiments(
        filter_string=f"last_update_time = {exp2.last_update_time}"
    )
    assert [e.experiment_id for e in experiments] == [exp_id2]


def test_search_experiments_filter_by_tag(store: SqlAlchemyStore):
    experiments = [
        ("exp1", [ExperimentTag("key1", "value"), ExperimentTag("key2", "value")]),
        ("exp2", [ExperimentTag("key1", "vaLue"), ExperimentTag("key2", "vaLue")]),
        ("exp3", [ExperimentTag("k e y 1", "value")]),
    ]
    for name, tags in experiments:
        time.sleep(0.001)
        store.create_experiment(name, tags=tags)

    experiments = store.search_experiments(filter_string="tag.key1 = 'value'")
    assert [e.name for e in experiments] == ["exp1"]
    experiments = store.search_experiments(filter_string="tag.`k e y 1` = 'value'")
    assert [e.name for e in experiments] == ["exp3"]
    experiments = store.search_experiments(filter_string="tag.\"k e y 1\" = 'value'")
    assert [e.name for e in experiments] == ["exp3"]
    experiments = store.search_experiments(filter_string="tag.key1 != 'value'")
    assert [e.name for e in experiments] == ["exp2"]
    experiments = store.search_experiments(filter_string="tag.key1 != 'VALUE'")
    assert [e.name for e in experiments] == ["exp2", "exp1"]
    experiments = store.search_experiments(filter_string="tag.key1 LIKE 'val%'")
    assert [e.name for e in experiments] == ["exp1"]
    experiments = store.search_experiments(filter_string="tag.key1 LIKE '%Lue'")
    assert [e.name for e in experiments] == ["exp2"]
    experiments = store.search_experiments(filter_string="tag.key1 ILIKE '%alu%'")
    assert [e.name for e in experiments] == ["exp2", "exp1"]
    experiments = store.search_experiments(
        filter_string="tag.key1 LIKE 'va%' AND tag.key2 LIKE '%Lue'"
    )
    assert [e.name for e in experiments] == ["exp2"]
    experiments = store.search_experiments(filter_string="tag.KEY = 'value'")
    assert len(experiments) == 0


def test_search_experiments_filter_by_attribute_and_tag(store: SqlAlchemyStore):
    store.create_experiment("exp1", tags=[ExperimentTag("a", "1"), ExperimentTag("b", "2")])
    store.create_experiment("exp2", tags=[ExperimentTag("a", "3"), ExperimentTag("b", "4")])
    experiments = store.search_experiments(filter_string="name ILIKE 'exp%' AND tags.a = '1'")
    assert [e.name for e in experiments] == ["exp1"]


def test_search_experiments_order_by(store: SqlAlchemyStore):
    experiment_names = ["x", "y", "z"]
    _create_experiments(store, experiment_names)

    experiments = store.search_experiments(order_by=["name"])
    assert [e.name for e in experiments] == ["Default", "x", "y", "z"]

    experiments = store.search_experiments(order_by=["name ASC"])
    assert [e.name for e in experiments] == ["Default", "x", "y", "z"]

    experiments = store.search_experiments(order_by=["name DESC"])
    assert [e.name for e in experiments] == ["z", "y", "x", "Default"]

    experiments = store.search_experiments(order_by=["experiment_id DESC"])
    assert [e.name for e in experiments] == ["z", "y", "x", "Default"]

    experiments = store.search_experiments(order_by=["name", "experiment_id"])
    assert [e.name for e in experiments] == ["Default", "x", "y", "z"]


def test_search_experiments_order_by_time_attribute(store: SqlAlchemyStore):
    # Sleep to ensure that the first experiment has a different creation_time than the default
    # experiment and eliminate flakiness.
    time.sleep(0.001)
    exp_id1 = store.create_experiment("1")
    time.sleep(0.001)
    exp_id2 = store.create_experiment("2")

    experiments = store.search_experiments(order_by=["creation_time"])
    assert [e.experiment_id for e in experiments] == [
        store.DEFAULT_EXPERIMENT_ID,
        exp_id1,
        exp_id2,
    ]

    experiments = store.search_experiments(order_by=["creation_time DESC"])
    assert [e.experiment_id for e in experiments] == [
        exp_id2,
        exp_id1,
        store.DEFAULT_EXPERIMENT_ID,
    ]

    experiments = store.search_experiments(order_by=["last_update_time"])
    assert [e.experiment_id for e in experiments] == [
        store.DEFAULT_EXPERIMENT_ID,
        exp_id1,
        exp_id2,
    ]

    store.rename_experiment(exp_id1, "new_name")
    experiments = store.search_experiments(order_by=["last_update_time"])
    assert [e.experiment_id for e in experiments] == [
        store.DEFAULT_EXPERIMENT_ID,
        exp_id2,
        exp_id1,
    ]


def test_search_experiments_max_results(store: SqlAlchemyStore):
    experiment_names = list(map(str, range(9)))
    _create_experiments(store, experiment_names)
    reversed_experiment_names = experiment_names[::-1]

    experiments = store.search_experiments()
    assert [e.name for e in experiments] == reversed_experiment_names + ["Default"]
    experiments = store.search_experiments(max_results=3)
    assert [e.name for e in experiments] == reversed_experiment_names[:3]


def test_search_experiments_max_results_validation(store: SqlAlchemyStore):
    with pytest.raises(
        MlflowException,
        match=r"Invalid value None for parameter 'max_results' supplied. "
        r"It must be a positive integer",
    ):
        store.search_experiments(max_results=None)
    with pytest.raises(
        MlflowException,
        match=r"Invalid value 0 for parameter 'max_results' supplied. "
        r"It must be a positive integer",
    ):
        store.search_experiments(max_results=0)
    with pytest.raises(
        MlflowException,
        match=r"Invalid value 1000000 for parameter 'max_results' supplied. "
        r"It must be at most 50000",
    ):
        store.search_experiments(max_results=1_000_000)


def test_search_experiments_pagination(store: SqlAlchemyStore):
    experiment_names = list(map(str, range(9)))
    _create_experiments(store, experiment_names)
    reversed_experiment_names = experiment_names[::-1]

    experiments = store.search_experiments(max_results=4)
    assert [e.name for e in experiments] == reversed_experiment_names[:4]
    assert experiments.token is not None

    experiments = store.search_experiments(max_results=4, page_token=experiments.token)
    assert [e.name for e in experiments] == reversed_experiment_names[4:8]
    assert experiments.token is not None

    experiments = store.search_experiments(max_results=4, page_token=experiments.token)
    assert [e.name for e in experiments] == reversed_experiment_names[8:] + ["Default"]
    assert experiments.token is None


def test_create_experiments(store: SqlAlchemyStore):
    with store.ManagedSessionMaker() as session:
        result = session.query(models.SqlExperiment).all()
        assert len(result) == 1
    time_before_create = get_current_time_millis()
    experiment_id = store.create_experiment(name="test exp")
    assert experiment_id == "1"
    with store.ManagedSessionMaker() as session:
        result = session.query(models.SqlExperiment).all()
        assert len(result) == 2

        test_exp = session.query(models.SqlExperiment).filter_by(name="test exp").first()
        assert str(test_exp.experiment_id) == experiment_id
        assert test_exp.name == "test exp"

    actual = store.get_experiment(experiment_id)
    assert actual.experiment_id == experiment_id
    assert actual.name == "test exp"
    assert actual.creation_time >= time_before_create
    assert actual.last_update_time == actual.creation_time

    with pytest.raises(MlflowException, match=r"'name' exceeds the maximum length"):
        store.create_experiment(name="x" * (MAX_EXPERIMENT_NAME_LENGTH + 1))


def test_create_experiment_with_tags_works_correctly(store: SqlAlchemyStore):
    experiment_id = store.create_experiment(
        name="test exp",
        artifact_location="some location",
        tags=[ExperimentTag("key1", "val1"), ExperimentTag("key2", "val2")],
    )
    experiment = store.get_experiment(experiment_id)
    assert len(experiment.tags) == 2
    assert experiment.tags["key1"] == "val1"
    assert experiment.tags["key2"] == "val2"


def test_run_tag_model(store: SqlAlchemyStore):
    # Create a run whose UUID we can reference when creating tag models.
    # `run_id` is a foreign key in the tags table; therefore, in order
    # to insert a tag with a given run UUID, the UUID must be present in
    # the runs table
    run = _run_factory(store)
    with store.ManagedSessionMaker() as session:
        new_tag = models.SqlTag(run_uuid=run.info.run_id, key="test", value="val")
        session.add(new_tag)
        session.commit()
        added_tags = [tag for tag in session.query(models.SqlTag).all() if tag.key == new_tag.key]
        assert len(added_tags) == 1
        added_tag = added_tags[0].to_mlflow_entity()
        assert added_tag.value == new_tag.value


def test_metric_model(store: SqlAlchemyStore):
    # Create a run whose UUID we can reference when creating metric models.
    # `run_id` is a foreign key in the tags table; therefore, in order
    # to insert a metric with a given run UUID, the UUID must be present in
    # the runs table
    run = _run_factory(store)
    with store.ManagedSessionMaker() as session:
        new_metric = models.SqlMetric(run_uuid=run.info.run_id, key="accuracy", value=0.89)
        session.add(new_metric)
        session.commit()
        metrics = session.query(models.SqlMetric).all()
        assert len(metrics) == 1

        added_metric = metrics[0].to_mlflow_entity()
        assert added_metric.value == new_metric.value
        assert added_metric.key == new_metric.key


def test_param_model(store: SqlAlchemyStore):
    # Create a run whose UUID we can reference when creating parameter models.
    # `run_id` is a foreign key in the tags table; therefore, in order
    # to insert a parameter with a given run UUID, the UUID must be present in
    # the runs table
    run = _run_factory(store)
    with store.ManagedSessionMaker() as session:
        new_param = models.SqlParam(run_uuid=run.info.run_id, key="accuracy", value="test param")
        session.add(new_param)
        session.commit()
        params = session.query(models.SqlParam).all()
        assert len(params) == 1

        added_param = params[0].to_mlflow_entity()
        assert added_param.value == new_param.value
        assert added_param.key == new_param.key


def test_run_needs_uuid(store: SqlAlchemyStore):
    regex = {
        SQLITE: r"NOT NULL constraint failed",
        POSTGRES: r"null value in column .+ of relation .+ violates not-null constrain",
        MYSQL: r"(Field .+ doesn't have a default value|Instance .+ has a NULL identity key)",
        MSSQL: r"Cannot insert the value NULL into column .+, table .+",
    }[store._get_dialect()]
    # Depending on the implementation, a NULL identity key may result in different
    # exceptions, including IntegrityError (sqlite) and FlushError (MysQL).
    # Therefore, we check for the more generic 'SQLAlchemyError'
    with pytest.raises(MlflowException, match=regex) as exception_context:
        with store.ManagedSessionMaker() as session:
            session.add(models.SqlRun())
    assert exception_context.value.error_code == ErrorCode.Name(BAD_REQUEST)


def test_run_data_model(store: SqlAlchemyStore):
    with store.ManagedSessionMaker() as session:
        run_id = uuid.uuid4().hex
        m1 = models.SqlMetric(run_uuid=run_id, key="accuracy", value=0.89)
        m2 = models.SqlMetric(run_uuid=run_id, key="recall", value=0.89)
        p1 = models.SqlParam(run_uuid=run_id, key="loss", value="test param")
        p2 = models.SqlParam(run_uuid=run_id, key="blue", value="test param")
        run_data = models.SqlRun(run_uuid=run_id)

        session.add_all([m1, m2, p1, p2])
        session.add(run_data)
        session.commit()

        run_datums = session.query(models.SqlRun).all()
        actual = run_datums[0]
        assert len(run_datums) == 1
        assert len(actual.params) == 2
        assert len(actual.metrics) == 2


def test_run_info(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "test exp")
    config = {
        "experiment_id": experiment_id,
        "name": "test run",
        "user_id": "Anderson",
        "run_uuid": "test",
        "status": RunStatus.to_string(RunStatus.SCHEDULED),
        "source_type": SourceType.to_string(SourceType.LOCAL),
        "source_name": "Python application",
        "entry_point_name": "main.py",
        "start_time": get_current_time_millis(),
        "end_time": get_current_time_millis(),
        "source_version": mlflow.__version__,
        "lifecycle_stage": entities.LifecycleStage.ACTIVE,
        "artifact_uri": "//",
    }
    run = models.SqlRun(**config).to_mlflow_entity()

    for k, v in config.items():
        # These keys were removed from RunInfo.
        if k in [
            "source_name",
            "source_type",
            "source_version",
            "name",
            "entry_point_name",
        ]:
            continue

        if k == "run_uuid":
            k = "run_id"

        v2 = getattr(run.info, k)
        if k == "source_type":
            assert v == SourceType.to_string(v2)
        else:
            assert v == v2


def test_create_run_with_tags(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "test_create_run")
    tags = [RunTag("3", "4"), RunTag("1", "2")]
    expected = _get_run_configs(experiment_id=experiment_id, tags=tags)

    actual = store.create_run(**expected)

    # run name should be added as a tag by the store
    tags.append(RunTag(mlflow_tags.MLFLOW_RUN_NAME, expected["run_name"]))

    assert actual.info.experiment_id == experiment_id
    assert actual.info.user_id == expected["user_id"]
    assert actual.info.run_name == expected["run_name"]
    assert actual.info.start_time == expected["start_time"]
    assert len(actual.data.tags) == len(tags)
    assert actual.data.tags == {tag.key: tag.value for tag in tags}
    assert actual.inputs.dataset_inputs == []


def test_create_run_sets_name(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "test_create_run_run_name")
    configs = _get_run_configs(experiment_id=experiment_id)
    run_id = store.create_run(**configs).info.run_id
    run = store.get_run(run_id)
    assert run.info.run_name == configs["run_name"]
    assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == configs["run_name"]

    run_id = store.create_run(
        experiment_id=experiment_id,
        user_id="user",
        start_time=0,
        run_name=None,
        tags=[RunTag(mlflow_tags.MLFLOW_RUN_NAME, "test")],
    ).info.run_id
    run = store.get_run(run_id)
    assert run.info.run_name == "test"
    assert run.inputs.dataset_inputs == []

    with pytest.raises(
        MlflowException,
        match=re.escape(
            "Both 'run_name' argument and 'mlflow.runName' tag are specified, but with "
            "different values (run_name='test', mlflow.runName='test_2').",
        ),
    ):
        store.create_run(
            experiment_id=experiment_id,
            user_id="user",
            start_time=0,
            run_name="test",
            tags=[RunTag(mlflow_tags.MLFLOW_RUN_NAME, "test_2")],
        )


def test_get_run_with_name(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "test_get_run")
    configs = _get_run_configs(experiment_id=experiment_id)
    run_id = store.create_run(**configs).info.run_id

    run = store.get_run(run_id)

    assert run.info.experiment_id == experiment_id
    assert run.info.run_name == configs["run_name"]

    no_run_configs = {
        "experiment_id": experiment_id,
        "user_id": "Anderson",
        "start_time": get_current_time_millis(),
        "tags": [],
        "run_name": None,
    }
    run_id = store.create_run(**no_run_configs).info.run_id

    run = store.get_run(run_id)

    assert run.info.run_name.split("-")[0] in _GENERATOR_PREDICATES

    name_empty_str_run = store.create_run(**{**configs, **{"run_name": ""}})
    run_name = name_empty_str_run.info.run_name
    assert run_name.split("-")[0] in _GENERATOR_PREDICATES


def test_to_mlflow_entity_and_proto(store: SqlAlchemyStore):
    # Create a run and log metrics, params, tags to the run
    created_run = _run_factory(store)
    run_id = created_run.info.run_id
    store.log_metric(
        run_id=run_id,
        metric=entities.Metric(key="my-metric", value=3.4, timestamp=0, step=0),
    )
    store.log_param(run_id=run_id, param=Param(key="my-param", value="param-val"))
    store.set_tag(run_id=run_id, tag=RunTag(key="my-tag", value="tag-val"))

    # Verify that we can fetch the run & convert it to proto - Python protobuf bindings
    # will perform type-checking to ensure all values have the right types
    run = store.get_run(run_id)
    run.to_proto()

    # Verify attributes of the Python run entity
    assert isinstance(run.info, entities.RunInfo)
    assert isinstance(run.data, entities.RunData)

    assert run.data.metrics == {"my-metric": 3.4}
    assert run.data.params == {"my-param": "param-val"}
    assert run.data.tags["my-tag"] == "tag-val"

    # Get the parent experiment of the run, verify it can be converted to protobuf
    exp = store.get_experiment(run.info.experiment_id)
    exp.to_proto()


def test_delete_run(store: SqlAlchemyStore):
    run = _run_factory(store)

    store.delete_run(run.info.run_id)

    with store.ManagedSessionMaker() as session:
        actual = session.query(models.SqlRun).filter_by(run_uuid=run.info.run_id).first()
        assert actual.lifecycle_stage == entities.LifecycleStage.DELETED
        assert (
            actual.deleted_time is not None
        )  # deleted time should be updated and thus not None anymore

        deleted_run = store.get_run(run.info.run_id)
        assert actual.run_uuid == deleted_run.info.run_id


def test_hard_delete_run(store: SqlAlchemyStore):
    run = _run_factory(store)
    metric = entities.Metric("blahmetric", 100.0, get_current_time_millis(), 0)
    store.log_metric(run.info.run_id, metric)
    param = entities.Param("blahparam", "100.0")
    store.log_param(run.info.run_id, param)
    tag = entities.RunTag("test tag", "a boogie")
    store.set_tag(run.info.run_id, tag)

    store._hard_delete_run(run.info.run_id)

    with store.ManagedSessionMaker() as session:
        actual_run = session.query(models.SqlRun).filter_by(run_uuid=run.info.run_id).first()
        assert actual_run is None
        actual_metric = session.query(models.SqlMetric).filter_by(run_uuid=run.info.run_id).first()
        assert actual_metric is None
        actual_param = session.query(models.SqlParam).filter_by(run_uuid=run.info.run_id).first()
        assert actual_param is None
        actual_tag = session.query(models.SqlTag).filter_by(run_uuid=run.info.run_id).first()
        assert actual_tag is None


def test_get_deleted_runs(store: SqlAlchemyStore):
    run = _run_factory(store)
    deleted_run_ids = store._get_deleted_runs()
    assert deleted_run_ids == []

    store.delete_run(run.info.run_id)
    deleted_run_ids = store._get_deleted_runs()
    assert deleted_run_ids == [run.info.run_id]


def test_log_metric(store: SqlAlchemyStore):
    run = _run_factory(store)

    tkey = "blahmetric"
    tval = 100.0
    metric = entities.Metric(tkey, tval, get_current_time_millis(), 0)
    metric2 = entities.Metric(tkey, tval, get_current_time_millis() + 2, 0)
    nan_metric = entities.Metric("NaN", float("nan"), 0, 0)
    pos_inf_metric = entities.Metric("PosInf", float("inf"), 0, 0)
    neg_inf_metric = entities.Metric("NegInf", -float("inf"), 0, 0)
    store.log_metric(run.info.run_id, metric)
    store.log_metric(run.info.run_id, metric2)
    store.log_metric(run.info.run_id, nan_metric)
    store.log_metric(run.info.run_id, pos_inf_metric)
    store.log_metric(run.info.run_id, neg_inf_metric)

    run = store.get_run(run.info.run_id)
    assert tkey in run.data.metrics
    assert run.data.metrics[tkey] == tval

    # SQL store _get_run method returns full history of recorded metrics.
    # Should return duplicates as well
    # MLflow RunData contains only the last reported values for metrics.
    with store.ManagedSessionMaker() as session:
        sql_run_metrics = store._get_run(session, run.info.run_id).metrics
        assert len(sql_run_metrics) == 5
        assert len(run.data.metrics) == 4
        assert math.isnan(run.data.metrics["NaN"])
        assert run.data.metrics["PosInf"] == 1.7976931348623157e308
        assert run.data.metrics["NegInf"] == -1.7976931348623157e308


@pytest.mark.skipif(
    is_windows(),
    reason="Flaky on Windows due to SQLite database locking issues with concurrent writes",
)
def test_log_metric_concurrent_logging_succeeds(store: SqlAlchemyStore):
    """
    Verifies that concurrent logging succeeds without deadlock, which has been an issue
    in previous MLflow releases
    """
    experiment_id = _create_experiments(store, "concurrency_exp")
    run_config = _get_run_configs(experiment_id=experiment_id)
    run1 = _run_factory(store, run_config)
    run2 = _run_factory(store, run_config)

    def log_metrics(run):
        for metric_val in range(100):
            store.log_metric(
                run.info.run_id,
                Metric("metric_key", metric_val, get_current_time_millis(), 0),
            )
        for batch_idx in range(5):
            store.log_batch(
                run.info.run_id,
                metrics=[
                    Metric(
                        f"metric_batch_{batch_idx}",
                        (batch_idx * 100) + val_offset,
                        get_current_time_millis(),
                        0,
                    )
                    for val_offset in range(100)
                ],
                params=[],
                tags=[],
            )
        for metric_val in range(100):
            store.log_metric(
                run.info.run_id,
                Metric("metric_key", metric_val, get_current_time_millis(), 0),
            )
        return "success"

    log_metrics_futures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Log metrics to two runs across four threads
        log_metrics_futures = [
            executor.submit(log_metrics, run) for run in [run1, run2, run1, run2]
        ]

    for future in log_metrics_futures:
        assert future.result() == "success"

    for run in [run1, run2, run1, run2]:
        # We visit each run twice, logging 100 metric entries for 6 metric names; the same entry
        # may be written multiple times concurrently; we assert that at least 100 metric entries
        # are present because at least 100 unique entries must have been written
        assert len(store.get_metric_history(run.info.run_id, "metric_key")) >= 100
        for batch_idx in range(5):
            assert (
                len(store.get_metric_history(run.info.run_id, f"metric_batch_{batch_idx}")) >= 100
            )


def test_record_logged_model(
    store: SqlAlchemyStore,
):
    run = _run_factory(store)
    flavors_with_config = {
        "tf": "flavor body",
        "python_function": {"config": {"a": 1}, "code": "code"},
    }
    m_with_config = Model(artifact_path="model/path", run_id="run_id", flavors=flavors_with_config)
    store.record_logged_model(run.info.run_id, m_with_config)
    with store.ManagedSessionMaker() as session:
        run = store._get_run(run_uuid=run.info.run_id, session=session)
        tags = [t.value for t in run.tags if t.key == mlflow_tags.MLFLOW_LOGGED_MODELS]
        flavors = m_with_config.get_tags_dict().get("flavors", {})
        assert all("config" not in v for v in flavors.values())
        assert tags[0] == json.dumps([m_with_config.get_tags_dict()])


def test_log_metric_allows_multiple_values_at_same_ts_and_run_data_uses_max_ts_value(
    store: SqlAlchemyStore,
):
    run = _run_factory(store)
    run_id = run.info.run_id
    metric_name = "test-metric-1"
    # Check that we get the max of (step, timestamp, value) in that order
    tuples_to_log = [
        (0, 100, 1000),
        (3, 40, 100),  # larger step wins even though it has smaller value
        (3, 50, 10),  # larger timestamp wins even though it has smaller value
        (3, 50, 20),  # tiebreak by max value
        (3, 50, 20),  # duplicate metrics with same (step, timestamp, value) are ok
        # verify that we can log steps out of order / negative steps
        (-3, 900, 900),
        (-1, 800, 800),
    ]
    for step, timestamp, value in reversed(tuples_to_log):
        store.log_metric(run_id, Metric(metric_name, value, timestamp, step))

    metric_history = store.get_metric_history(run_id, metric_name)
    logged_tuples = [(m.step, m.timestamp, m.value) for m in metric_history]
    assert set(logged_tuples) == set(tuples_to_log)

    run_data = store.get_run(run_id).data
    run_metrics = run_data.metrics
    assert len(run_metrics) == 1
    assert run_metrics[metric_name] == 20
    metric_obj = run_data._metric_objs[0]
    assert metric_obj.key == metric_name
    assert metric_obj.step == 3
    assert metric_obj.timestamp == 50
    assert metric_obj.value == 20


def test_log_null_metric(store: SqlAlchemyStore):
    run = _run_factory(store)

    tkey = "blahmetric"
    tval = None
    metric = entities.Metric(tkey, tval, get_current_time_millis(), 0)

    with pytest.raises(
        MlflowException, match=r"Missing value for required parameter 'value'"
    ) as exception_context:
        store.log_metric(run.info.run_id, metric)
    assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_log_param(store: SqlAlchemyStore):
    run = _run_factory(store)

    tkey = "blahmetric"
    tval = "100.0"
    param = entities.Param(tkey, tval)
    param2 = entities.Param("new param", "new key")
    store.log_param(run.info.run_id, param)
    store.log_param(run.info.run_id, param2)
    store.log_param(run.info.run_id, param2)

    run = store.get_run(run.info.run_id)
    assert len(run.data.params) == 2
    assert tkey in run.data.params
    assert run.data.params[tkey] == tval


def test_log_param_uniqueness(store: SqlAlchemyStore):
    run = _run_factory(store)

    tkey = "blahmetric"
    tval = "100.0"
    param = entities.Param(tkey, tval)
    param2 = entities.Param(tkey, "newval")
    store.log_param(run.info.run_id, param)

    with pytest.raises(MlflowException, match=r"Changing param values is not allowed"):
        store.log_param(run.info.run_id, param2)


def test_log_empty_str(store: SqlAlchemyStore):
    run = _run_factory(store)

    tkey = "blahmetric"
    tval = ""
    param = entities.Param(tkey, tval)
    param2 = entities.Param("new param", "new key")
    store.log_param(run.info.run_id, param)
    store.log_param(run.info.run_id, param2)

    run = store.get_run(run.info.run_id)
    assert len(run.data.params) == 2
    assert tkey in run.data.params
    assert run.data.params[tkey] == tval


def test_log_null_param(store: SqlAlchemyStore):
    run = _run_factory(store)

    tkey = "blahmetric"
    tval = None
    param = entities.Param(tkey, tval)

    dialect = store._get_dialect()
    regex = {
        SQLITE: r"NOT NULL constraint failed",
        POSTGRES: r"null value in column .+ of relation .+ violates not-null constrain",
        MYSQL: r"Column .+ cannot be null",
        MSSQL: r"Cannot insert the value NULL into column .+, table .+",
    }[dialect]
    with pytest.raises(MlflowException, match=regex) as exception_context:
        store.log_param(run.info.run_id, param)
    if dialect != MYSQL:
        assert exception_context.value.error_code == ErrorCode.Name(BAD_REQUEST)
    else:
        # Some MySQL client packages (and there are several available, e.g.
        # PyMySQL, mysqlclient, mysql-connector-python... reports some
        # errors, including NULL constraint violations, as a SQLAlchemy
        # OperationalError, even though they should be reported as a more
        # generic SQLAlchemyError. If that is fixed, we can remove this
        # special case.
        assert exception_context.value.error_code == ErrorCode.Name(
            BAD_REQUEST
        ) or exception_context.value.error_code == ErrorCode.Name(TEMPORARILY_UNAVAILABLE)


@pytest.mark.skipif(
    Version(sqlalchemy.__version__) < Version("2.0")
    and mlflow.get_tracking_uri().startswith("mssql"),
    reason="large string parameters are sent as TEXT/NTEXT; see tests/db/compose.yml for details",
)
def test_log_param_max_length_value(store: SqlAlchemyStore, monkeypatch):
    run = _run_factory(store)
    tkey = "blahmetric"
    tval = "x" * 6000
    param = entities.Param(tkey, tval)
    store.log_param(run.info.run_id, param)
    run = store.get_run(run.info.run_id)
    assert run.data.params[tkey] == str(tval)
    monkeypatch.setenv("MLFLOW_TRUNCATE_LONG_VALUES", "false")
    with pytest.raises(MlflowException, match="exceeds the maximum length"):
        store.log_param(run.info.run_id, entities.Param(tkey, "x" * 6001))

    monkeypatch.setenv("MLFLOW_TRUNCATE_LONG_VALUES", "true")
    store.log_param(run.info.run_id, entities.Param(tkey, "x" * 6001))


def test_set_experiment_tag(store: SqlAlchemyStore):
    exp_id = _create_experiments(store, "setExperimentTagExp")
    tag = entities.ExperimentTag("tag0", "value0")
    new_tag = entities.RunTag("tag0", "value00000")
    store.set_experiment_tag(exp_id, tag)
    experiment = store.get_experiment(exp_id)
    assert experiment.tags["tag0"] == "value0"
    # test that updating a tag works
    store.set_experiment_tag(exp_id, new_tag)
    experiment = store.get_experiment(exp_id)
    assert experiment.tags["tag0"] == "value00000"
    # test that setting a tag on 1 experiment does not impact another experiment.
    exp_id_2 = _create_experiments(store, "setExperimentTagExp2")
    experiment2 = store.get_experiment(exp_id_2)
    assert len(experiment2.tags) == 0
    # setting a tag on different experiments maintains different values across experiments
    different_tag = entities.RunTag("tag0", "differentValue")
    store.set_experiment_tag(exp_id_2, different_tag)
    experiment = store.get_experiment(exp_id)
    assert experiment.tags["tag0"] == "value00000"
    experiment2 = store.get_experiment(exp_id_2)
    assert experiment2.tags["tag0"] == "differentValue"
    # test can set multi-line tags
    multi_line_Tag = entities.ExperimentTag("multiline tag", "value2\nvalue2\nvalue2")
    store.set_experiment_tag(exp_id, multi_line_Tag)
    experiment = store.get_experiment(exp_id)
    assert experiment.tags["multiline tag"] == "value2\nvalue2\nvalue2"
    # test cannot set tags that are too long
    long_tag = entities.ExperimentTag("longTagKey", "a" * 100_001)
    with pytest.raises(MlflowException, match="exceeds the maximum length of 5000"):
        store.set_experiment_tag(exp_id, long_tag)
    # test can set tags that are somewhat long
    long_tag = entities.ExperimentTag("longTagKey", "a" * 4999)
    store.set_experiment_tag(exp_id, long_tag)
    # test cannot set tags on deleted experiments
    store.delete_experiment(exp_id)
    with pytest.raises(MlflowException, match="must be in the 'active' state"):
        store.set_experiment_tag(exp_id, entities.ExperimentTag("should", "notset"))


def test_delete_experiment_tag(store: SqlAlchemyStore):
    exp_id = _create_experiments(store, "setExperimentTagExp")
    tag = entities.ExperimentTag("tag0", "value0")
    store.set_experiment_tag(exp_id, tag)
    experiment = store.get_experiment(exp_id)
    assert experiment.tags["tag0"] == "value0"
    # test that deleting a tag works
    store.delete_experiment_tag(exp_id, tag.key)
    experiment = store.get_experiment(exp_id)
    assert "tag0" not in experiment.tags


def test_set_tag(store: SqlAlchemyStore, monkeypatch):
    run = _run_factory(store)

    tkey = "test tag"
    tval = "a boogie"
    new_val = "new val"
    tag = entities.RunTag(tkey, tval)
    new_tag = entities.RunTag(tkey, new_val)
    store.set_tag(run.info.run_id, tag)
    # Overwriting tags is allowed
    store.set_tag(run.info.run_id, new_tag)
    # test setting tags that are too long fails.
    monkeypatch.setenv("MLFLOW_TRUNCATE_LONG_VALUES", "false")
    with pytest.raises(
        MlflowException, match=f"exceeds the maximum length of {MAX_TAG_VAL_LENGTH} characters"
    ):
        store.set_tag(
            run.info.run_id, entities.RunTag("longTagKey", "a" * (MAX_TAG_VAL_LENGTH + 1))
        )

    monkeypatch.setenv("MLFLOW_TRUNCATE_LONG_VALUES", "true")
    store.set_tag(run.info.run_id, entities.RunTag("longTagKey", "a" * (MAX_TAG_VAL_LENGTH + 1)))

    # test can set tags that are somewhat long
    store.set_tag(run.info.run_id, entities.RunTag("longTagKey", "a" * (MAX_TAG_VAL_LENGTH - 1)))
    run = store.get_run(run.info.run_id)
    assert tkey in run.data.tags
    assert run.data.tags[tkey] == new_val


def test_delete_tag(store: SqlAlchemyStore):
    run = _run_factory(store)
    k0 = "tag0"
    v0 = "val0"
    k1 = "tag1"
    v1 = "val1"
    tag0 = entities.RunTag(k0, v0)
    tag1 = entities.RunTag(k1, v1)
    store.set_tag(run.info.run_id, tag0)
    store.set_tag(run.info.run_id, tag1)
    # delete a tag and check whether it is correctly deleted.
    store.delete_tag(run.info.run_id, k0)
    run = store.get_run(run.info.run_id)
    assert k0 not in run.data.tags
    assert k1 in run.data.tags
    assert run.data.tags[k1] == v1

    # test that deleting a tag works correctly with multiple runs having the same tag.
    run2 = _run_factory(store, config=_get_run_configs(run.info.experiment_id))
    store.set_tag(run.info.run_id, tag0)
    store.set_tag(run2.info.run_id, tag0)
    store.delete_tag(run.info.run_id, k0)
    run = store.get_run(run.info.run_id)
    run2 = store.get_run(run2.info.run_id)
    assert k0 not in run.data.tags
    assert k0 in run2.data.tags
    # test that you cannot delete tags that don't exist.
    with pytest.raises(MlflowException, match="No tag with name"):
        store.delete_tag(run.info.run_id, "fakeTag")
    # test that you cannot delete tags for nonexistent runs
    with pytest.raises(MlflowException, match="Run with id=randomRunId not found"):
        store.delete_tag("randomRunId", k0)
    # test that you cannot delete tags for deleted runs.
    store.delete_run(run.info.run_id)
    with pytest.raises(MlflowException, match="must be in the 'active' state"):
        store.delete_tag(run.info.run_id, k1)


def test_get_metric_history(store: SqlAlchemyStore):
    run = _run_factory(store)

    key = "test"
    expected = [
        models.SqlMetric(key=key, value=0.6, timestamp=1, step=0).to_mlflow_entity(),
        models.SqlMetric(key=key, value=0.7, timestamp=2, step=0).to_mlflow_entity(),
    ]

    for metric in expected:
        store.log_metric(run.info.run_id, metric)

    actual = store.get_metric_history(run.info.run_id, key)

    assert sorted(
        [(m.key, m.value, m.timestamp) for m in expected],
    ) == sorted(
        [(m.key, m.value, m.timestamp) for m in actual],
    )


def test_get_metric_history_with_max_results(store: SqlAlchemyStore):
    run = _run_factory(store)
    run_id = run.info.run_id

    metric_key = "test_metric"
    expected_metrics = []
    for i in range(5):
        metric = models.SqlMetric(
            key=metric_key, value=float(i), timestamp=1000 + i, step=i
        ).to_mlflow_entity()
        store.log_metric(run_id, metric)
        expected_metrics.append(metric)

    # Test without max_results - should return all 5 metrics
    all_metrics = store.get_metric_history(run_id, metric_key)
    assert len(all_metrics) == 5

    # Test with max_results=3 - should return only first 3 metrics
    limited_metrics = store.get_metric_history(run_id, metric_key, max_results=3)
    assert len(limited_metrics) == 3

    all_metric_tuples = {(m.key, m.value, m.timestamp, m.step) for m in all_metrics}
    limited_metric_tuples = {(m.key, m.value, m.timestamp, m.step) for m in limited_metrics}
    assert limited_metric_tuples.issubset(all_metric_tuples)

    # Test with max_results=0 - should return no metrics
    no_metrics = store.get_metric_history(run_id, metric_key, max_results=0)
    assert len(no_metrics) == 0

    # Test with max_results larger than available metrics - should return all metrics
    more_metrics = store.get_metric_history(run_id, metric_key, max_results=10)
    assert len(more_metrics) == 5

    more_metric_tuples = {(m.key, m.value, m.timestamp, m.step) for m in more_metrics}
    assert more_metric_tuples == all_metric_tuples


def test_get_metric_history_with_page_token(store: SqlAlchemyStore):
    run = _run_factory(store)
    run_id = run.info.run_id

    metric_key = "test_metric"
    for i in range(10):
        metric = models.SqlMetric(
            key=metric_key, value=float(i), timestamp=1000 + i, step=i
        ).to_mlflow_entity()
        store.log_metric(run_id, metric)

    page_size = 4

    first_page = store.get_metric_history(
        run_id, metric_key, max_results=page_size, page_token=None
    )
    assert isinstance(first_page, PagedList)
    assert first_page.token is not None
    assert len(first_page) == 4

    second_page = store.get_metric_history(
        run_id, metric_key, max_results=page_size, page_token=first_page.token
    )
    assert isinstance(first_page, PagedList)
    assert second_page.token is not None
    assert len(second_page) == 4

    third_page = store.get_metric_history(
        run_id, metric_key, max_results=page_size, page_token=second_page.token
    )
    assert isinstance(first_page, PagedList)
    assert third_page.token is None
    assert len(third_page) == 2

    all_paginated_metrics = list(first_page) + list(second_page) + list(third_page)
    assert len(all_paginated_metrics) == 10

    metric_values = [m.value for m in all_paginated_metrics]
    expected_values = [float(i) for i in range(10)]
    assert sorted(metric_values) == sorted(expected_values)

    # Test with invalid page_token
    with pytest.raises(MlflowException, match="Invalid page token"):
        store.get_metric_history(run_id, metric_key, page_token="invalid_token")

    # Test pagination without max_results (should return all in one page)
    result = store.get_metric_history(run_id, metric_key, page_token=None)
    assert len(result) == 10
    assert result.token is None  # No next page


def test_rename_experiment(store: SqlAlchemyStore):
    new_name = "new name"
    experiment_id = _create_experiments(store, "test name")
    experiment = store.get_experiment(experiment_id)
    time.sleep(0.01)
    store.rename_experiment(experiment_id, new_name)

    renamed_experiment = store.get_experiment(experiment_id)

    assert renamed_experiment.name == new_name
    assert renamed_experiment.last_update_time > experiment.last_update_time


def test_update_run_info(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "test_update_run_info")
    for new_status_string in models.RunStatusTypes:
        run = _run_factory(store, config=_get_run_configs(experiment_id=experiment_id))
        endtime = get_current_time_millis()
        actual = store.update_run_info(
            run.info.run_id, RunStatus.from_string(new_status_string), endtime, None
        )
        assert actual.status == new_status_string
        assert actual.end_time == endtime

    # test updating run name without changing other attributes.
    origin_run_info = store.get_run(run.info.run_id).info
    updated_info = store.update_run_info(run.info.run_id, None, None, "name_abc2")
    assert updated_info.run_name == "name_abc2"
    assert updated_info.status == origin_run_info.status
    assert updated_info.end_time == origin_run_info.end_time


def test_update_run_name(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "test_update_run_name")
    configs = _get_run_configs(experiment_id=experiment_id)

    run_id = store.create_run(**configs).info.run_id
    run = store.get_run(run_id)
    assert run.info.run_name == configs["run_name"]

    store.update_run_info(run_id, RunStatus.FINISHED, 1000, "new name")
    run = store.get_run(run_id)
    assert run.info.run_name == "new name"
    assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "new name"

    store.update_run_info(run_id, RunStatus.FINISHED, 1000, None)
    run = store.get_run(run_id)
    assert run.info.run_name == "new name"
    assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "new name"

    store.update_run_info(run_id, RunStatus.FINISHED, 1000, "")
    run = store.get_run(run_id)
    assert run.info.run_name == "new name"
    assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "new name"

    store.delete_tag(run_id, mlflow_tags.MLFLOW_RUN_NAME)
    run = store.get_run(run_id)
    assert run.info.run_name == "new name"
    assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) is None

    store.update_run_info(run_id, RunStatus.FINISHED, 1000, "newer name")
    run = store.get_run(run_id)
    assert run.info.run_name == "newer name"
    assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "newer name"

    store.set_tag(run_id, entities.RunTag(mlflow_tags.MLFLOW_RUN_NAME, "newest name"))
    run = store.get_run(run_id)
    assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "newest name"
    assert run.info.run_name == "newest name"

    store.log_batch(
        run_id,
        metrics=[],
        params=[],
        tags=[entities.RunTag(mlflow_tags.MLFLOW_RUN_NAME, "batch name")],
    )
    run = store.get_run(run_id)
    assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "batch name"
    assert run.info.run_name == "batch name"


def test_restore_experiment(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "helloexp")
    exp = store.get_experiment(experiment_id)
    assert exp.lifecycle_stage == entities.LifecycleStage.ACTIVE

    experiment_id = exp.experiment_id
    store.delete_experiment(experiment_id)

    deleted = store.get_experiment(experiment_id)
    assert deleted.experiment_id == experiment_id
    assert deleted.lifecycle_stage == entities.LifecycleStage.DELETED
    time.sleep(0.01)
    store.restore_experiment(exp.experiment_id)
    restored = store.get_experiment(exp.experiment_id)
    assert restored.experiment_id == experiment_id
    assert restored.lifecycle_stage == entities.LifecycleStage.ACTIVE
    assert restored.last_update_time > deleted.last_update_time


def test_delete_restore_run(store: SqlAlchemyStore):
    run = _run_factory(store)
    assert run.info.lifecycle_stage == entities.LifecycleStage.ACTIVE

    # Verify that active runs can be restored (run restoration is idempotent)
    store.restore_run(run.info.run_id)

    # Verify that run deletion is idempotent
    store.delete_run(run.info.run_id)
    store.delete_run(run.info.run_id)

    deleted = store.get_run(run.info.run_id)
    assert deleted.info.run_id == run.info.run_id
    assert deleted.info.lifecycle_stage == entities.LifecycleStage.DELETED
    with store.ManagedSessionMaker() as session:
        assert store._get_run(session, deleted.info.run_id).deleted_time is not None
    # Verify that restoration of a deleted run is idempotent
    store.restore_run(run.info.run_id)
    store.restore_run(run.info.run_id)
    restored = store.get_run(run.info.run_id)
    assert restored.info.run_id == run.info.run_id
    assert restored.info.lifecycle_stage == entities.LifecycleStage.ACTIVE
    with store.ManagedSessionMaker() as session:
        assert store._get_run(session, restored.info.run_id).deleted_time is None


def test_error_logging_to_deleted_run(store: SqlAlchemyStore):
    exp = _create_experiments(store, "error_logging")
    run_id = _run_factory(store, _get_run_configs(experiment_id=exp)).info.run_id

    store.delete_run(run_id)
    assert store.get_run(run_id).info.lifecycle_stage == entities.LifecycleStage.DELETED
    with pytest.raises(MlflowException, match=r"The run .+ must be in the 'active' state"):
        store.log_param(run_id, entities.Param("p1345", "v1"))

    with pytest.raises(MlflowException, match=r"The run .+ must be in the 'active' state"):
        store.log_metric(run_id, entities.Metric("m1345", 1.0, 123, 0))

    with pytest.raises(MlflowException, match=r"The run .+ must be in the 'active' state"):
        store.set_tag(run_id, entities.RunTag("t1345", "tv1"))

    # restore this run and try again
    store.restore_run(run_id)
    assert store.get_run(run_id).info.lifecycle_stage == entities.LifecycleStage.ACTIVE
    store.log_param(run_id, entities.Param("p1345", "v22"))
    store.log_metric(run_id, entities.Metric("m1345", 34.0, 85, 1))  # earlier timestamp
    store.set_tag(run_id, entities.RunTag("t1345", "tv44"))

    run = store.get_run(run_id)
    assert run.data.params == {"p1345": "v22"}
    assert run.data.metrics == {"m1345": 34.0}
    metric_history = store.get_metric_history(run_id, "m1345")
    assert len(metric_history) == 1
    metric_obj = metric_history[0]
    assert metric_obj.key == "m1345"
    assert metric_obj.value == 34.0
    assert metric_obj.timestamp == 85
    assert metric_obj.step == 1
    assert {("t1345", "tv44")} <= set(run.data.tags.items())


def test_order_by_metric_tag_param(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("order_by_metric")

    def create_and_log_run(names):
        name = str(names[0]) + "/" + names[1]
        run_id = store.create_run(
            experiment_id,
            user_id="MrDuck",
            start_time=123,
            tags=[entities.RunTag("metric", names[1])],
            run_name=name,
        ).info.run_id
        if names[0] is not None:
            store.log_metric(run_id, entities.Metric("x", float(names[0]), 1, 0))
            store.log_metric(run_id, entities.Metric("y", float(names[1]), 1, 0))
        store.log_param(run_id, entities.Param("metric", names[1]))
        return run_id

    # the expected order in ascending sort is :
    # inf > number > -inf > None > nan
    for names in zip(
        [None, "nan", "inf", "-inf", "-1000", "0", "0", "1000"],
        ["1", "2", "3", "4", "5", "6", "7", "8"],
    ):
        create_and_log_run(names)

    # asc/asc
    assert _get_ordered_runs(store, ["metrics.x asc", "metrics.y asc"], experiment_id) == [
        "-inf/4",
        "-1000/5",
        "0/6",
        "0/7",
        "1000/8",
        "inf/3",
        "nan/2",
        "None/1",
    ]

    assert _get_ordered_runs(store, ["metrics.x asc", "tag.metric asc"], experiment_id) == [
        "-inf/4",
        "-1000/5",
        "0/6",
        "0/7",
        "1000/8",
        "inf/3",
        "nan/2",
        "None/1",
    ]

    # asc/desc
    assert _get_ordered_runs(store, ["metrics.x asc", "metrics.y desc"], experiment_id) == [
        "-inf/4",
        "-1000/5",
        "0/7",
        "0/6",
        "1000/8",
        "inf/3",
        "nan/2",
        "None/1",
    ]

    assert _get_ordered_runs(store, ["metrics.x asc", "tag.metric desc"], experiment_id) == [
        "-inf/4",
        "-1000/5",
        "0/7",
        "0/6",
        "1000/8",
        "inf/3",
        "nan/2",
        "None/1",
    ]

    # desc / asc
    assert _get_ordered_runs(store, ["metrics.x desc", "metrics.y asc"], experiment_id) == [
        "inf/3",
        "1000/8",
        "0/6",
        "0/7",
        "-1000/5",
        "-inf/4",
        "nan/2",
        "None/1",
    ]

    # desc / desc
    assert _get_ordered_runs(store, ["metrics.x desc", "param.metric desc"], experiment_id) == [
        "inf/3",
        "1000/8",
        "0/7",
        "0/6",
        "-1000/5",
        "-inf/4",
        "nan/2",
        "None/1",
    ]


def test_order_by_attributes(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("order_by_attributes")

    def create_run(start_time, end):
        return store.create_run(
            experiment_id,
            user_id="MrDuck",
            start_time=start_time,
            tags=[],
            run_name=str(end),
        ).info.run_id

    start_time = 123
    for end in [234, None, 456, -123, 789, 123]:
        run_id = create_run(start_time, end)
        store.update_run_info(run_id, run_status=RunStatus.FINISHED, end_time=end, run_name=None)
        start_time += 1

    # asc
    assert _get_ordered_runs(store, ["attribute.end_time asc"], experiment_id) == [
        "-123",
        "123",
        "234",
        "456",
        "789",
        "None",
    ]

    # desc
    assert _get_ordered_runs(store, ["attribute.end_time desc"], experiment_id) == [
        "789",
        "456",
        "234",
        "123",
        "-123",
        "None",
    ]

    # Sort priority correctly handled
    assert _get_ordered_runs(
        store, ["attribute.start_time asc", "attribute.end_time desc"], experiment_id
    ) == ["234", "None", "456", "-123", "789", "123"]


def test_search_vanilla(store: SqlAlchemyStore):
    exp = _create_experiments(store, "search_vanilla")
    runs = [_run_factory(store, _get_run_configs(exp)).info.run_id for r in range(3)]

    assert sorted(
        runs,
    ) == sorted(_search_runs(store, exp, run_view_type=ViewType.ALL))
    assert sorted(
        runs,
    ) == sorted(_search_runs(store, exp, run_view_type=ViewType.ACTIVE_ONLY))
    assert _search_runs(store, exp, run_view_type=ViewType.DELETED_ONLY) == []

    first = runs[0]

    store.delete_run(first)
    assert sorted(
        runs,
    ) == sorted(_search_runs(store, exp, run_view_type=ViewType.ALL))
    assert sorted(
        runs[1:],
    ) == sorted(_search_runs(store, exp, run_view_type=ViewType.ACTIVE_ONLY))
    assert _search_runs(store, exp, run_view_type=ViewType.DELETED_ONLY) == [first]

    store.restore_run(first)
    assert sorted(
        runs,
    ) == sorted(_search_runs(store, exp, run_view_type=ViewType.ALL))
    assert sorted(
        runs,
    ) == sorted(_search_runs(store, exp, run_view_type=ViewType.ACTIVE_ONLY))
    assert _search_runs(store, exp, run_view_type=ViewType.DELETED_ONLY) == []


def test_search_params(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "search_params")
    r1 = _run_factory(store, _get_run_configs(experiment_id)).info.run_id
    r2 = _run_factory(store, _get_run_configs(experiment_id)).info.run_id

    store.log_param(r1, entities.Param("generic_param", "p_val"))
    store.log_param(r2, entities.Param("generic_param", "p_val"))

    store.log_param(r1, entities.Param("generic_2", "some value"))
    store.log_param(r2, entities.Param("generic_2", "another value"))

    store.log_param(r1, entities.Param("p_a", "abc"))
    store.log_param(r2, entities.Param("p_b", "ABC"))

    # test search returns both runs
    filter_string = "params.generic_param = 'p_val'"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, experiment_id, filter_string))

    # test search returns appropriate run (same key different values per run)
    filter_string = "params.generic_2 = 'some value'"
    assert _search_runs(store, experiment_id, filter_string) == [r1]
    filter_string = "params.generic_2 = 'another value'"
    assert _search_runs(store, experiment_id, filter_string) == [r2]

    filter_string = "params.generic_param = 'wrong_val'"
    assert _search_runs(store, experiment_id, filter_string) == []

    filter_string = "params.generic_param != 'p_val'"
    assert _search_runs(store, experiment_id, filter_string) == []

    filter_string = "params.generic_param != 'wrong_val'"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, experiment_id, filter_string))
    filter_string = "params.generic_2 != 'wrong_val'"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, experiment_id, filter_string))

    filter_string = "params.p_a = 'abc'"
    assert _search_runs(store, experiment_id, filter_string) == [r1]

    filter_string = "params.p_a = 'ABC'"
    assert _search_runs(store, experiment_id, filter_string) == []

    filter_string = "params.p_a != 'ABC'"
    assert _search_runs(store, experiment_id, filter_string) == [r1]

    filter_string = "params.p_b = 'ABC'"
    assert _search_runs(store, experiment_id, filter_string) == [r2]

    filter_string = "params.generic_2 LIKE '%other%'"
    assert _search_runs(store, experiment_id, filter_string) == [r2]

    filter_string = "params.generic_2 LIKE 'other%'"
    assert _search_runs(store, experiment_id, filter_string) == []

    filter_string = "params.generic_2 LIKE '%other'"
    assert _search_runs(store, experiment_id, filter_string) == []

    filter_string = "params.generic_2 LIKE 'other'"
    assert _search_runs(store, experiment_id, filter_string) == []

    filter_string = "params.generic_2 LIKE '%Other%'"
    assert _search_runs(store, experiment_id, filter_string) == []

    filter_string = "params.generic_2 ILIKE '%Other%'"
    assert _search_runs(store, experiment_id, filter_string) == [r2]


def test_search_tags(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "search_tags")
    r1 = _run_factory(store, _get_run_configs(experiment_id)).info.run_id
    r2 = _run_factory(store, _get_run_configs(experiment_id)).info.run_id

    store.set_tag(r1, entities.RunTag("generic_tag", "p_val"))
    store.set_tag(r2, entities.RunTag("generic_tag", "p_val"))

    store.set_tag(r1, entities.RunTag("generic_2", "some value"))
    store.set_tag(r2, entities.RunTag("generic_2", "another value"))

    store.set_tag(r1, entities.RunTag("p_a", "abc"))
    store.set_tag(r2, entities.RunTag("p_b", "ABC"))

    # test search returns both runs
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, experiment_id, filter_string="tags.generic_tag = 'p_val'"))
    assert _search_runs(store, experiment_id, filter_string="tags.generic_tag = 'P_VAL'") == []
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, experiment_id, filter_string="tags.generic_tag != 'P_VAL'"))
    # test search returns appropriate run (same key different values per run)
    assert _search_runs(store, experiment_id, filter_string="tags.generic_2 = 'some value'") == [r1]
    assert _search_runs(store, experiment_id, filter_string="tags.generic_2 = 'another value'") == [
        r2
    ]
    assert _search_runs(store, experiment_id, filter_string="tags.generic_tag = 'wrong_val'") == []
    assert _search_runs(store, experiment_id, filter_string="tags.generic_tag != 'p_val'") == []
    assert sorted(
        [r1, r2],
    ) == sorted(
        _search_runs(store, experiment_id, filter_string="tags.generic_tag != 'wrong_val'"),
    )
    assert sorted(
        [r1, r2],
    ) == sorted(
        _search_runs(store, experiment_id, filter_string="tags.generic_2 != 'wrong_val'"),
    )
    assert _search_runs(store, experiment_id, filter_string="tags.p_a = 'abc'") == [r1]
    assert _search_runs(store, experiment_id, filter_string="tags.p_b = 'ABC'") == [r2]
    assert _search_runs(store, experiment_id, filter_string="tags.generic_2 LIKE '%other%'") == [r2]
    assert _search_runs(store, experiment_id, filter_string="tags.generic_2 LIKE '%Other%'") == []
    assert _search_runs(store, experiment_id, filter_string="tags.generic_2 LIKE 'other%'") == []
    assert _search_runs(store, experiment_id, filter_string="tags.generic_2 LIKE '%other'") == []
    assert _search_runs(store, experiment_id, filter_string="tags.generic_2 LIKE 'other'") == []
    assert _search_runs(store, experiment_id, filter_string="tags.generic_2 ILIKE '%Other%'") == [
        r2
    ]
    assert _search_runs(
        store,
        experiment_id,
        filter_string="tags.generic_2 ILIKE '%Other%' and tags.generic_tag = 'p_val'",
    ) == [r2]
    assert _search_runs(
        store,
        experiment_id,
        filter_string="tags.generic_2 ILIKE '%Other%' and tags.generic_tag ILIKE 'p_val'",
    ) == [r2]


def test_search_metrics(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "search_metric")
    r1 = _run_factory(store, _get_run_configs(experiment_id)).info.run_id
    r2 = _run_factory(store, _get_run_configs(experiment_id)).info.run_id

    store.log_metric(r1, entities.Metric("common", 1.0, 1, 0))
    store.log_metric(r2, entities.Metric("common", 1.0, 1, 0))

    store.log_metric(r1, entities.Metric("measure_a", 1.0, 1, 0))
    store.log_metric(r2, entities.Metric("measure_a", 200.0, 2, 0))
    store.log_metric(r2, entities.Metric("measure_a", 400.0, 3, 0))

    store.log_metric(r1, entities.Metric("m_a", 2.0, 2, 0))
    store.log_metric(r2, entities.Metric("m_b", 3.0, 2, 0))
    store.log_metric(r2, entities.Metric("m_b", 4.0, 8, 0))  # this is last timestamp
    store.log_metric(r2, entities.Metric("m_b", 8.0, 3, 0))

    filter_string = "metrics.common = 1.0"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, experiment_id, filter_string))

    filter_string = "metrics.common > 0.0"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, experiment_id, filter_string))

    filter_string = "metrics.common >= 0.0"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, experiment_id, filter_string))

    filter_string = "metrics.common < 4.0"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, experiment_id, filter_string))

    filter_string = "metrics.common <= 4.0"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, experiment_id, filter_string))

    filter_string = "metrics.common != 1.0"
    assert _search_runs(store, experiment_id, filter_string) == []

    filter_string = "metrics.common >= 3.0"
    assert _search_runs(store, experiment_id, filter_string) == []

    filter_string = "metrics.common <= 0.75"
    assert _search_runs(store, experiment_id, filter_string) == []

    # tests for same metric name across runs with different values and timestamps
    filter_string = "metrics.measure_a > 0.0"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, experiment_id, filter_string))

    filter_string = "metrics.measure_a < 50.0"
    assert _search_runs(store, experiment_id, filter_string) == [r1]

    filter_string = "metrics.measure_a < 1000.0"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, experiment_id, filter_string))

    filter_string = "metrics.measure_a != -12.0"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, experiment_id, filter_string))

    filter_string = "metrics.measure_a > 50.0"
    assert _search_runs(store, experiment_id, filter_string) == [r2]

    filter_string = "metrics.measure_a = 1.0"
    assert _search_runs(store, experiment_id, filter_string) == [r1]

    filter_string = "metrics.measure_a = 400.0"
    assert _search_runs(store, experiment_id, filter_string) == [r2]

    # test search with unique metric keys
    filter_string = "metrics.m_a > 1.0"
    assert _search_runs(store, experiment_id, filter_string) == [r1]

    filter_string = "metrics.m_b > 1.0"
    assert _search_runs(store, experiment_id, filter_string) == [r2]

    # there is a recorded metric this threshold but not last timestamp
    filter_string = "metrics.m_b > 5.0"
    assert _search_runs(store, experiment_id, filter_string) == []

    # metrics matches last reported timestamp for 'm_b'
    filter_string = "metrics.m_b = 4.0"
    assert _search_runs(store, experiment_id, filter_string) == [r2]


def test_search_attrs(store: SqlAlchemyStore, tmp_path):
    e1 = _create_experiments(store, "search_attributes_1")
    r1 = _run_factory(store, _get_run_configs(experiment_id=e1)).info.run_id

    e2 = _create_experiments(store, "search_attrs_2")
    r2 = _run_factory(store, _get_run_configs(experiment_id=e2)).info.run_id

    filter_string = ""
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, [e1, e2], filter_string))

    filter_string = "attribute.status != 'blah'"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, [e1, e2], filter_string))

    filter_string = f"attribute.status = '{RunStatus.to_string(RunStatus.RUNNING)}'"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, [e1, e2], filter_string))

    # change status for one of the runs
    store.update_run_info(r2, RunStatus.FAILED, 300, None)

    filter_string = "attribute.status = 'RUNNING'"
    assert _search_runs(store, [e1, e2], filter_string) == [r1]

    filter_string = "attribute.status = 'FAILED'"
    assert _search_runs(store, [e1, e2], filter_string) == [r2]

    filter_string = "attribute.status != 'SCHEDULED'"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, [e1, e2], filter_string))

    filter_string = "attribute.status = 'SCHEDULED'"
    assert _search_runs(store, [e1, e2], filter_string) == []

    filter_string = "attribute.status = 'KILLED'"
    assert _search_runs(store, [e1, e2], filter_string) == []

    expected_artifact_uri = (
        pathlib.Path.cwd().joinpath(tmp_path, "artifacts", e1, r1, "artifacts").as_uri()
    )
    filter_string = f"attr.artifact_uri = '{expected_artifact_uri}'"
    assert _search_runs(store, [e1, e2], filter_string) == [r1]

    filter_string = (
        f"attr.artifact_uri = '{tmp_path}/artifacts/{e1.upper()}/{r1.upper()}/artifacts'"
    )
    assert _search_runs(store, [e1, e2], filter_string) == []

    filter_string = (
        f"attr.artifact_uri != '{tmp_path}/artifacts/{e1.upper()}/{r1.upper()}/artifacts'"
    )
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, [e1, e2], filter_string))

    filter_string = f"attr.artifact_uri = '{tmp_path}/artifacts/{e2}/{r1}/artifacts'"
    assert _search_runs(store, [e1, e2], filter_string) == []

    filter_string = "attribute.artifact_uri = 'random_artifact_path'"
    assert _search_runs(store, [e1, e2], filter_string) == []

    filter_string = "attribute.artifact_uri != 'random_artifact_path'"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, [e1, e2], filter_string))

    filter_string = f"attribute.artifact_uri LIKE '%{r1}%'"
    assert _search_runs(store, [e1, e2], filter_string) == [r1]

    filter_string = f"attribute.artifact_uri LIKE '%{r1[:16]}%'"
    assert _search_runs(store, [e1, e2], filter_string) == [r1]

    filter_string = f"attribute.artifact_uri LIKE '%{r1[-16:]}%'"
    assert _search_runs(store, [e1, e2], filter_string) == [r1]

    filter_string = f"attribute.artifact_uri LIKE '%{r1.upper()}%'"
    assert _search_runs(store, [e1, e2], filter_string) == []

    filter_string = f"attribute.artifact_uri ILIKE '%{r1.upper()}%'"
    assert _search_runs(store, [e1, e2], filter_string) == [r1]

    filter_string = f"attribute.artifact_uri ILIKE '%{r1[:16].upper()}%'"
    assert _search_runs(store, [e1, e2], filter_string) == [r1]

    filter_string = f"attribute.artifact_uri ILIKE '%{r1[-16:].upper()}%'"
    assert _search_runs(store, [e1, e2], filter_string) == [r1]

    for k, v in {"experiment_id": e1, "lifecycle_stage": "ACTIVE"}.items():
        with pytest.raises(MlflowException, match=r"Invalid attribute key '.+' specified"):
            _search_runs(store, [e1, e2], f"attribute.{k} = '{v}'")


def test_search_full(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "search_params")
    r1 = _run_factory(store, _get_run_configs(experiment_id)).info.run_id
    r2 = _run_factory(store, _get_run_configs(experiment_id)).info.run_id

    store.log_param(r1, entities.Param("generic_param", "p_val"))
    store.log_param(r2, entities.Param("generic_param", "p_val"))

    store.log_param(r1, entities.Param("p_a", "abc"))
    store.log_param(r2, entities.Param("p_b", "ABC"))

    store.log_metric(r1, entities.Metric("common", 1.0, 1, 0))
    store.log_metric(r2, entities.Metric("common", 1.0, 1, 0))

    store.log_metric(r1, entities.Metric("m_a", 2.0, 2, 0))
    store.log_metric(r2, entities.Metric("m_b", 3.0, 2, 0))
    store.log_metric(r2, entities.Metric("m_b", 4.0, 8, 0))
    store.log_metric(r2, entities.Metric("m_b", 8.0, 3, 0))

    filter_string = "params.generic_param = 'p_val' and metrics.common = 1.0"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, experiment_id, filter_string))

    # all params and metrics match
    filter_string = "params.generic_param = 'p_val' and metrics.common = 1.0 and metrics.m_a > 1.0"
    assert _search_runs(store, experiment_id, filter_string) == [r1]

    filter_string = (
        "params.generic_param = 'p_val' and metrics.common = 1.0 "
        "and metrics.m_a > 1.0 and params.p_a LIKE 'a%'"
    )
    assert _search_runs(store, experiment_id, filter_string) == [r1]

    filter_string = (
        "params.generic_param = 'p_val' and metrics.common = 1.0 "
        "and metrics.m_a > 1.0 and params.p_a LIKE 'A%'"
    )
    assert _search_runs(store, experiment_id, filter_string) == []

    filter_string = (
        "params.generic_param = 'p_val' and metrics.common = 1.0 "
        "and metrics.m_a > 1.0 and params.p_a ILIKE 'A%'"
    )
    assert _search_runs(store, experiment_id, filter_string) == [r1]

    # test with mismatch param
    filter_string = (
        "params.random_bad_name = 'p_val' and metrics.common = 1.0 and metrics.m_a > 1.0"
    )
    assert _search_runs(store, experiment_id, filter_string) == []

    # test with mismatch metric
    filter_string = (
        "params.generic_param = 'p_val' and metrics.common = 1.0 and metrics.m_a > 100.0"
    )
    assert _search_runs(store, experiment_id, filter_string) == []


def test_search_with_max_results(store: SqlAlchemyStore):
    exp = _create_experiments(store, "search_with_max_results")
    # Bulk insert runs using SQLAlchemy for performance
    run_uuids = [uuid.uuid4().hex for _ in range(1200)]
    with store.ManagedSessionMaker() as session:
        session.add_all(
            SqlRun(
                run_uuid=run_uuid,
                name="name",
                experiment_id=int(exp),
                user_id="Anderson",
                status=RunStatus.to_string(RunStatus.RUNNING),
                start_time=i,
                lifecycle_stage=entities.LifecycleStage.ACTIVE,
            )
            for i, run_uuid in enumerate(run_uuids)
        )
    # reverse the ordering, since we created in increasing order of start_time
    runs = list(reversed(run_uuids))

    assert runs[:1000] == _search_runs(store, exp)
    for n in [1, 2, 4, 8, 10, 20, 50, 100, 500, 1000, 1200, 2000]:
        assert runs[: min(1200, n)] == _search_runs(store, exp, max_results=n)

    maxPlusOne = SEARCH_MAX_RESULTS_THRESHOLD + 1

    with pytest.raises(
        MlflowException,
        match=rf"Invalid value {maxPlusOne} for parameter 'max_results'",
    ):
        _search_runs(store, exp, max_results=maxPlusOne)


def test_search_with_deterministic_max_results(store: SqlAlchemyStore):
    exp = _create_experiments(store, "test_search_with_deterministic_max_results")
    # Create 10 runs with the same start_time.
    # Sort based on run_id
    runs = sorted(
        [_run_factory(store, _get_run_configs(exp, start_time=10)).info.run_id for r in range(10)]
    )
    for n in [1, 2, 4, 8, 10, 20]:
        assert runs[: min(10, n)] == _search_runs(store, exp, max_results=n)


def test_search_runs_pagination(store: SqlAlchemyStore):
    exp = _create_experiments(store, "test_search_runs_pagination")
    # test returned token behavior
    runs = sorted(
        [_run_factory(store, _get_run_configs(exp, start_time=10)).info.run_id for r in range(10)]
    )
    result = store.search_runs([exp], None, ViewType.ALL, max_results=4)
    assert [r.info.run_id for r in result] == runs[0:4]
    assert result.token is not None
    result = store.search_runs([exp], None, ViewType.ALL, max_results=4, page_token=result.token)
    assert [r.info.run_id for r in result] == runs[4:8]
    assert result.token is not None
    result = store.search_runs([exp], None, ViewType.ALL, max_results=4, page_token=result.token)
    assert [r.info.run_id for r in result] == runs[8:]
    assert result.token is None


def test_search_runs_run_name(store: SqlAlchemyStore):
    exp_id = _create_experiments(store, "test_search_runs_pagination")
    run1 = _run_factory(store, dict(_get_run_configs(exp_id), run_name="run_name1"))
    run2 = _run_factory(store, dict(_get_run_configs(exp_id), run_name="run_name2"))
    result = store.search_runs(
        [exp_id],
        filter_string="attributes.run_name = 'run_name1'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run1.info.run_id]
    result = store.search_runs(
        [exp_id],
        filter_string="attributes.`Run name` = 'run_name1'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run1.info.run_id]
    result = store.search_runs(
        [exp_id],
        filter_string="attributes.`run name` = 'run_name2'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run2.info.run_id]
    result = store.search_runs(
        [exp_id],
        filter_string="attributes.`Run Name` = 'run_name2'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run2.info.run_id]
    result = store.search_runs(
        [exp_id],
        filter_string="tags.`mlflow.runName` = 'run_name2'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run2.info.run_id]

    store.update_run_info(
        run1.info.run_id,
        RunStatus.FINISHED,
        end_time=run1.info.end_time,
        run_name="new_run_name1",
    )
    result = store.search_runs(
        [exp_id],
        filter_string="attributes.run_name = 'new_run_name1'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run1.info.run_id]

    # TODO: Test attribute-based search after set_tag

    # Test run name filter works for runs logged in MLflow <= 1.29.0
    with store.ManagedSessionMaker() as session:
        sql_run1 = session.query(SqlRun).filter(SqlRun.run_uuid == run1.info.run_id).one()
        sql_run1.name = ""

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.run_name = 'new_run_name1'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run1.info.run_id]

    result = store.search_runs(
        [exp_id],
        filter_string="tags.`mlflow.runName` = 'new_run_name1'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run1.info.run_id]


def test_search_runs_run_id(store: SqlAlchemyStore):
    exp_id = _create_experiments(store, "test_search_runs_run_id")
    # Set start_time to ensure the search result is deterministic
    run1 = _run_factory(store, dict(_get_run_configs(exp_id), start_time=1))
    run2 = _run_factory(store, dict(_get_run_configs(exp_id), start_time=2))
    run_id1 = run1.info.run_id
    run_id2 = run2.info.run_id

    result = store.search_runs(
        [exp_id],
        filter_string=f"attributes.run_id = '{run_id1}'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run_id1]

    result = store.search_runs(
        [exp_id],
        filter_string=f"attributes.run_id != '{run_id1}'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run_id2]

    result = store.search_runs(
        [exp_id],
        filter_string=f"attributes.run_id IN ('{run_id1}')",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run_id1]

    result = store.search_runs(
        [exp_id],
        filter_string=f"attributes.run_id NOT IN ('{run_id1}')",
        run_view_type=ViewType.ACTIVE_ONLY,
    )

    result = store.search_runs(
        [exp_id],
        filter_string=f"run_name = '{run1.info.run_name}' AND run_id IN ('{run_id1}')",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run_id1]

    for filter_string in [
        f"attributes.run_id IN ('{run_id1}','{run_id2}')",
        f"attributes.run_id IN ('{run_id1}', '{run_id2}')",
        f"attributes.run_id IN ('{run_id1}',  '{run_id2}')",
    ]:
        result = store.search_runs(
            [exp_id], filter_string=filter_string, run_view_type=ViewType.ACTIVE_ONLY
        )
        assert [r.info.run_id for r in result] == [run_id2, run_id1]

    result = store.search_runs(
        [exp_id],
        filter_string=f"attributes.run_id NOT IN ('{run_id1}', '{run_id2}')",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert result == []


def test_search_runs_start_time_alias(store: SqlAlchemyStore):
    exp_id = _create_experiments(store, "test_search_runs_start_time_alias")
    # Set start_time to ensure the search result is deterministic
    run1 = _run_factory(store, dict(_get_run_configs(exp_id), start_time=1))
    run2 = _run_factory(store, dict(_get_run_configs(exp_id), start_time=2))
    run_id1 = run1.info.run_id
    run_id2 = run2.info.run_id

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.run_name = 'name'",
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attributes.start_time DESC"],
    )
    assert [r.info.run_id for r in result] == [run_id2, run_id1]

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.run_name = 'name'",
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attributes.created ASC"],
    )
    assert [r.info.run_id for r in result] == [run_id1, run_id2]

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.run_name = 'name'",
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attributes.Created DESC"],
    )
    assert [r.info.run_id for r in result] == [run_id2, run_id1]

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.start_time > 0",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id1, run_id2}

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.created > 1",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run_id2]

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.Created > 2",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert result == []


def test_search_runs_datasets(store: SqlAlchemyStore):
    exp_id = _create_experiments(store, "test_search_runs_datasets")
    # Set start_time to ensure the search result is deterministic
    run1 = _run_factory(store, dict(_get_run_configs(exp_id), start_time=1))
    run2 = _run_factory(store, dict(_get_run_configs(exp_id), start_time=3))
    run3 = _run_factory(store, dict(_get_run_configs(exp_id), start_time=2))

    dataset1 = entities.Dataset(
        name="name1",
        digest="digest1",
        source_type="st1",
        source="source1",
        schema="schema1",
        profile="profile1",
    )
    dataset2 = entities.Dataset(
        name="name2",
        digest="digest2",
        source_type="st2",
        source="source2",
        schema="schema2",
        profile="profile2",
    )
    dataset3 = entities.Dataset(
        name="name3",
        digest="digest3",
        source_type="st3",
        source="source3",
        schema="schema3",
        profile="profile3",
    )

    test_tag = [entities.InputTag(key=MLFLOW_DATASET_CONTEXT, value="test")]
    train_tag = [entities.InputTag(key=MLFLOW_DATASET_CONTEXT, value="train")]
    eval_tag = [entities.InputTag(key=MLFLOW_DATASET_CONTEXT, value="eval")]

    inputs_run1 = [
        entities.DatasetInput(dataset1, train_tag),
        entities.DatasetInput(dataset2, eval_tag),
    ]
    inputs_run2 = [
        entities.DatasetInput(dataset1, train_tag),
        entities.DatasetInput(dataset3, eval_tag),
    ]
    inputs_run3 = [entities.DatasetInput(dataset2, test_tag)]

    store.log_inputs(run1.info.run_id, inputs_run1)
    store.log_inputs(run2.info.run_id, inputs_run2)
    store.log_inputs(run3.info.run_id, inputs_run3)
    run_id1 = run1.info.run_id
    run_id2 = run2.info.run_id
    run_id3 = run3.info.run_id

    result = store.search_runs(
        [exp_id],
        filter_string="dataset.name = 'name1'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id2, run_id1}

    result = store.search_runs(
        [exp_id],
        filter_string="dataset.digest = 'digest2'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3, run_id1}

    result = store.search_runs(
        [exp_id],
        filter_string="dataset.name = 'name4'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert set(result) == set()

    result = store.search_runs(
        [exp_id],
        filter_string="dataset.context = 'train'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id2, run_id1}

    result = store.search_runs(
        [exp_id],
        filter_string="dataset.context = 'test'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3}

    result = store.search_runs(
        [exp_id],
        filter_string="dataset.context = 'test' and dataset.name = 'name2'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3}

    result = store.search_runs(
        [exp_id],
        filter_string="dataset.name = 'name2' and dataset.context = 'test'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3}

    result = store.search_runs(
        [exp_id],
        filter_string="datasets.name IN ('name1', 'name2')",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3, run_id1, run_id2}

    result = store.search_runs(
        [exp_id],
        filter_string="datasets.digest IN ('digest1', 'digest2')",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3, run_id1, run_id2}

    result = store.search_runs(
        [exp_id],
        filter_string="datasets.name LIKE 'Name%'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == set()

    result = store.search_runs(
        [exp_id],
        filter_string="datasets.name ILIKE 'Name%'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3, run_id1, run_id2}

    result = store.search_runs(
        [exp_id],
        filter_string="datasets.context ILIKE 'test%'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3}

    result = store.search_runs(
        [exp_id],
        filter_string="datasets.context IN ('test', 'train')",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3, run_id1, run_id2}


def test_search_runs_datasets_with_param_filters(store: SqlAlchemyStore):
    """Test that combining param/tag filters with dataset filters works correctly.

    This is a regression test for https://github.com/mlflow/mlflow/pull/19498
    where combining non-attribute filters (params, tags, metrics) with dataset
    filters caused SQLAlchemy alias conflicts.
    """
    exp_id = _create_experiments(store, "test_search_runs_datasets_with_param_filters")
    run1 = _run_factory(store, _get_run_configs(exp_id))
    run2 = _run_factory(store, _get_run_configs(exp_id))

    # Log params to runs
    store.log_param(run1.info.run_id, Param("learning_rate", "0.01"))
    store.log_param(run1.info.run_id, Param("batch_size", "32"))
    store.log_param(run2.info.run_id, Param("learning_rate", "0.02"))

    # Log datasets to runs
    dataset1 = entities.Dataset(
        name="train_data",
        digest="digest1",
        source_type="local",
        source="source1",
    )
    train_tag = [entities.InputTag(key=MLFLOW_DATASET_CONTEXT, value="training")]
    store.log_inputs(run1.info.run_id, [entities.DatasetInput(dataset1, train_tag)])
    store.log_inputs(run2.info.run_id, [entities.DatasetInput(dataset1, train_tag)])

    run_id1 = run1.info.run_id

    # Test: param filter + dataset name filter
    result = store.search_runs(
        [exp_id],
        filter_string="params.learning_rate = '0.01' AND dataset.name = 'train_data'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id1}

    # Test: param filter + dataset context filter
    result = store.search_runs(
        [exp_id],
        filter_string="params.learning_rate = '0.01' AND dataset.context = 'training'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id1}

    # Test: multiple param filters + dataset filter
    result = store.search_runs(
        [exp_id],
        filter_string=(
            "params.learning_rate = '0.01' AND params.batch_size = '32' "
            "AND dataset.name = 'train_data'"
        ),
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id1}

    # Test: param filter + multiple dataset filters
    result = store.search_runs(
        [exp_id],
        filter_string=(
            "params.learning_rate = '0.01' AND dataset.name = 'train_data' "
            "AND dataset.context = 'training'"
        ),
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id1}


def test_search_datasets(store: SqlAlchemyStore):
    exp_id1 = _create_experiments(store, "test_search_datasets_1")
    # Create an additional experiment to ensure we filter on specified experiment
    # and search works on multiple experiments.
    exp_id2 = _create_experiments(store, "test_search_datasets_2")

    run1 = _run_factory(store, dict(_get_run_configs(exp_id1), start_time=1))
    run2 = _run_factory(store, dict(_get_run_configs(exp_id1), start_time=2))
    run3 = _run_factory(store, dict(_get_run_configs(exp_id2), start_time=3))

    dataset1 = entities.Dataset(
        name="name1",
        digest="digest1",
        source_type="st1",
        source="source1",
        schema="schema1",
        profile="profile1",
    )
    dataset2 = entities.Dataset(
        name="name2",
        digest="digest2",
        source_type="st2",
        source="source2",
        schema="schema2",
        profile="profile2",
    )
    dataset3 = entities.Dataset(
        name="name3",
        digest="digest3",
        source_type="st3",
        source="source3",
        schema="schema3",
        profile="profile3",
    )
    dataset4 = entities.Dataset(
        name="name4",
        digest="digest4",
        source_type="st4",
        source="source4",
        schema="schema4",
        profile="profile4",
    )

    test_tag = [entities.InputTag(key=MLFLOW_DATASET_CONTEXT, value="test")]
    train_tag = [entities.InputTag(key=MLFLOW_DATASET_CONTEXT, value="train")]
    eval_tag = [entities.InputTag(key=MLFLOW_DATASET_CONTEXT, value="eval")]
    no_context_tag = [entities.InputTag(key="not_context", value="test")]

    inputs_run1 = [
        entities.DatasetInput(dataset1, train_tag),
        entities.DatasetInput(dataset2, eval_tag),
        entities.DatasetInput(dataset4, no_context_tag),
    ]
    inputs_run2 = [
        entities.DatasetInput(dataset1, train_tag),
        entities.DatasetInput(dataset2, test_tag),
    ]
    inputs_run3 = [entities.DatasetInput(dataset3, train_tag)]

    store.log_inputs(run1.info.run_id, inputs_run1)
    store.log_inputs(run2.info.run_id, inputs_run2)
    store.log_inputs(run3.info.run_id, inputs_run3)

    # Verify actual and expected results are same size and that all elements are equal.
    def assert_has_same_elements(actual_list, expected_list):
        assert len(actual_list) == len(expected_list)
        for actual in actual_list:
            # Verify the expected results list contains same element.
            isEqual = False
            for expected in expected_list:
                isEqual = actual == expected
                if isEqual:
                    break
            assert isEqual

    # Verify no results from exp_id2 are returned.
    results = store._search_datasets([exp_id1])
    expected_results = [
        _DatasetSummary(exp_id1, dataset1.name, dataset1.digest, "train"),
        _DatasetSummary(exp_id1, dataset2.name, dataset2.digest, "eval"),
        _DatasetSummary(exp_id1, dataset2.name, dataset2.digest, "test"),
        _DatasetSummary(exp_id1, dataset4.name, dataset4.digest, None),
    ]
    assert_has_same_elements(results, expected_results)

    # Verify results from both experiment are returned.
    results = store._search_datasets([exp_id1, exp_id2])
    expected_results.append(_DatasetSummary(exp_id2, dataset3.name, dataset3.digest, "train"))
    assert_has_same_elements(results, expected_results)


def test_search_datasets_returns_no_more_than_max_results(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_search_datasets")
    run = _run_factory(store, dict(_get_run_configs(exp_id), start_time=1))
    inputs = []
    # We intentionally add more than 1000 datasets here to test we only return 1000.
    for i in range(1010):
        dataset = entities.Dataset(
            name="name" + str(i),
            digest="digest" + str(i),
            source_type="st" + str(i),
            source="source" + str(i),
            schema="schema" + str(i),
            profile="profile" + str(i),
        )
        input_tag = [entities.InputTag(key=MLFLOW_DATASET_CONTEXT, value=str(i))]
        inputs.append(entities.DatasetInput(dataset, input_tag))

    store.log_inputs(run.info.run_id, inputs)

    results = store._search_datasets([exp_id])
    assert len(results) == 1000


def test_log_batch(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "log_batch")
    run_id = _run_factory(store, _get_run_configs(experiment_id)).info.run_id
    metric_entities = [Metric("m1", 0.87, 12345, 0), Metric("m2", 0.49, 12345, 1)]
    param_entities = [Param("p1", "p1val"), Param("p2", "p2val")]
    tag_entities = [
        RunTag("t1", "t1val"),
        RunTag("t2", "t2val"),
        RunTag(MLFLOW_RUN_NAME, "my_run"),
    ]
    store.log_batch(
        run_id=run_id, metrics=metric_entities, params=param_entities, tags=tag_entities
    )
    run = store.get_run(run_id)
    assert run.data.tags == {"t1": "t1val", "t2": "t2val", MLFLOW_RUN_NAME: "my_run"}
    assert run.data.params == {"p1": "p1val", "p2": "p2val"}
    metric_histories = sum((store.get_metric_history(run_id, key) for key in run.data.metrics), [])
    metrics = [(m.key, m.value, m.timestamp, m.step) for m in metric_histories]
    assert set(metrics) == {("m1", 0.87, 12345, 0), ("m2", 0.49, 12345, 1)}


def test_log_batch_limits(store: SqlAlchemyStore):
    # Test that log batch at the maximum allowed request size succeeds (i.e doesn't hit
    # SQL limitations, etc)
    experiment_id = _create_experiments(store, "log_batch_limits")
    run_id = _run_factory(store, _get_run_configs(experiment_id)).info.run_id
    metric_tuples = [(f"m{i}", i, 12345, i * 2) for i in range(1000)]
    metric_entities = [Metric(*metric_tuple) for metric_tuple in metric_tuples]
    store.log_batch(run_id=run_id, metrics=metric_entities, params=[], tags=[])
    run = store.get_run(run_id)
    metric_histories = sum((store.get_metric_history(run_id, key) for key in run.data.metrics), [])
    metrics = [(m.key, m.value, m.timestamp, m.step) for m in metric_histories]
    assert set(metrics) == set(metric_tuples)


def test_log_batch_param_overwrite_disallowed(store: SqlAlchemyStore):
    # Test that attempting to overwrite a param via log_batch results in an exception and that
    # no partial data is logged
    run = _run_factory(store)
    tkey = "my-param"
    param = entities.Param(tkey, "orig-val")
    store.log_param(run.info.run_id, param)

    overwrite_param = entities.Param(tkey, "newval")
    tag = entities.RunTag("tag-key", "tag-val")
    metric = entities.Metric("metric-key", 3.0, 12345, 0)
    with pytest.raises(
        MlflowException, match=r"Changing param values is not allowed"
    ) as exception_context:
        store.log_batch(run.info.run_id, metrics=[metric], params=[overwrite_param], tags=[tag])
    assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    _verify_logged(store, run.info.run_id, metrics=[], params=[param], tags=[])


def test_log_batch_with_unchanged_and_new_params(store: SqlAlchemyStore):
    """
    Test case to ensure the following code works:
    ---------------------------------------------
    mlflow.log_params({"a": 0, "b": 1})
    mlflow.log_params({"a": 0, "c": 2})
    ---------------------------------------------
    """
    run = _run_factory(store)
    store.log_batch(
        run.info.run_id,
        metrics=[],
        params=[entities.Param("a", "0"), entities.Param("b", "1")],
        tags=[],
    )
    store.log_batch(
        run.info.run_id,
        metrics=[],
        params=[entities.Param("a", "0"), entities.Param("c", "2")],
        tags=[],
    )
    _verify_logged(
        store,
        run.info.run_id,
        metrics=[],
        params=[
            entities.Param("a", "0"),
            entities.Param("b", "1"),
            entities.Param("c", "2"),
        ],
        tags=[],
    )


def test_log_batch_param_overwrite_disallowed_single_req(store: SqlAlchemyStore):
    # Test that attempting to overwrite a param via log_batch results in an exception
    run = _run_factory(store)
    pkey = "common-key"
    param0 = entities.Param(pkey, "orig-val")
    param1 = entities.Param(pkey, "newval")
    tag = entities.RunTag("tag-key", "tag-val")
    metric = entities.Metric("metric-key", 3.0, 12345, 0)
    with pytest.raises(
        MlflowException, match=r"Duplicate parameter keys have been submitted"
    ) as exception_context:
        store.log_batch(run.info.run_id, metrics=[metric], params=[param0, param1], tags=[tag])
    assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    _verify_logged(store, run.info.run_id, metrics=[], params=[], tags=[])


def test_log_batch_accepts_empty_payload(store: SqlAlchemyStore):
    run = _run_factory(store)
    store.log_batch(run.info.run_id, metrics=[], params=[], tags=[])
    _verify_logged(store, run.info.run_id, metrics=[], params=[], tags=[])


def test_log_batch_internal_error(store: SqlAlchemyStore):
    # Verify that internal errors during the DB save step for log_batch result in
    # MlflowExceptions
    run = _run_factory(store)

    def _raise_exception_fn(*args, **kwargs):
        raise Exception("Some internal error")

    package = "mlflow.store.tracking.sqlalchemy_store.SqlAlchemyStore"
    with (
        mock.patch(package + "._log_metrics") as metric_mock,
        mock.patch(package + "._log_params") as param_mock,
        mock.patch(package + "._set_tags") as tags_mock,
    ):
        metric_mock.side_effect = _raise_exception_fn
        param_mock.side_effect = _raise_exception_fn
        tags_mock.side_effect = _raise_exception_fn
        for kwargs in [
            {"metrics": [Metric("a", 3, 1, 0)]},
            {"params": [Param("b", "c")]},
            {"tags": [RunTag("c", "d")]},
        ]:
            log_batch_kwargs = {"metrics": [], "params": [], "tags": []}
            log_batch_kwargs.update(kwargs)
            with pytest.raises(MlflowException, match=r"Some internal error"):
                store.log_batch(run.info.run_id, **log_batch_kwargs)


def test_log_batch_nonexistent_run(store: SqlAlchemyStore):
    nonexistent_run_id = uuid.uuid4().hex
    with pytest.raises(
        MlflowException, match=rf"Run with id={nonexistent_run_id} not found"
    ) as exception_context:
        store.log_batch(nonexistent_run_id, [], [], [])
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_log_batch_params_idempotency(store: SqlAlchemyStore):
    run = _run_factory(store)
    params = [Param("p-key", "p-val")]
    store.log_batch(run.info.run_id, metrics=[], params=params, tags=[])
    store.log_batch(run.info.run_id, metrics=[], params=params, tags=[])
    _verify_logged(store, run.info.run_id, metrics=[], params=params, tags=[])


def test_log_batch_tags_idempotency(store: SqlAlchemyStore):
    run = _run_factory(store)
    store.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")])
    store.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")])
    _verify_logged(store, run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")])


def test_log_batch_allows_tag_overwrite(store: SqlAlchemyStore):
    run = _run_factory(store)
    store.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "val")])
    store.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "newval")])
    _verify_logged(store, run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "newval")])


def test_log_batch_allows_tag_overwrite_single_req(store: SqlAlchemyStore):
    run = _run_factory(store)
    tags = [RunTag("t-key", "val"), RunTag("t-key", "newval")]
    store.log_batch(run.info.run_id, metrics=[], params=[], tags=tags)
    _verify_logged(store, run.info.run_id, metrics=[], params=[], tags=[tags[-1]])


def test_log_batch_metrics(store: SqlAlchemyStore):
    run = _run_factory(store)

    tkey = "blahmetric"
    tval = 100.0
    metric = entities.Metric(tkey, tval, get_current_time_millis(), 0)
    metric2 = entities.Metric(tkey, tval, get_current_time_millis() + 2, 0)
    nan_metric = entities.Metric("NaN", float("nan"), 0, 0)
    pos_inf_metric = entities.Metric("PosInf", float("inf"), 0, 0)
    neg_inf_metric = entities.Metric("NegInf", -float("inf"), 0, 0)

    # duplicate metric and metric2 values should be eliminated
    metrics = [
        metric,
        metric2,
        nan_metric,
        pos_inf_metric,
        neg_inf_metric,
        metric,
        metric2,
    ]
    store._log_metrics(run.info.run_id, metrics)

    run = store.get_run(run.info.run_id)
    assert tkey in run.data.metrics
    assert run.data.metrics[tkey] == tval

    # SQL store _get_run method returns full history of recorded metrics.
    # Should return duplicates as well
    # MLflow RunData contains only the last reported values for metrics.
    with store.ManagedSessionMaker() as session:
        sql_run_metrics = store._get_run(session, run.info.run_id).metrics
        assert len(sql_run_metrics) == 5
        assert len(run.data.metrics) == 4
        assert math.isnan(run.data.metrics["NaN"])
        assert run.data.metrics["PosInf"] == 1.7976931348623157e308
        assert run.data.metrics["NegInf"] == -1.7976931348623157e308


def test_log_batch_same_metric_repeated_single_req(store: SqlAlchemyStore):
    run = _run_factory(store)
    metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
    metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
    store.log_batch(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])
    _verify_logged(store, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])


def test_log_batch_same_metric_repeated_multiple_reqs(store: SqlAlchemyStore):
    run = _run_factory(store)
    metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
    metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
    store.log_batch(run.info.run_id, params=[], metrics=[metric0], tags=[])
    _verify_logged(store, run.info.run_id, params=[], metrics=[metric0], tags=[])
    store.log_batch(run.info.run_id, params=[], metrics=[metric1], tags=[])
    _verify_logged(store, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])


def test_log_batch_same_metrics_repeated_multiple_reqs(store: SqlAlchemyStore):
    run = _run_factory(store)
    metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
    metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
    store.log_batch(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])
    _verify_logged(store, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])
    store.log_batch(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])
    _verify_logged(store, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])


def test_log_batch_null_metrics(store: SqlAlchemyStore):
    run = _run_factory(store)

    tkey = "blahmetric"
    tval = None
    metric_1 = entities.Metric(tkey, tval, get_current_time_millis(), 0)

    tkey = "blahmetric2"
    tval = None
    metric_2 = entities.Metric(tkey, tval, get_current_time_millis(), 0)

    metrics = [metric_1, metric_2]

    with pytest.raises(
        MlflowException,
        match=r"Missing value for required parameter 'metrics\[0\]\.value'",
    ) as exception_context:
        store.log_batch(run.info.run_id, metrics=metrics, params=[], tags=[])
    assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_log_batch_params_max_length_value(store: SqlAlchemyStore, monkeypatch):
    run = _run_factory(store)
    param_entities = [Param("long param", "x" * 6000), Param("short param", "xyz")]
    expected_param_entities = [
        Param("long param", "x" * 6000),
        Param("short param", "xyz"),
    ]
    store.log_batch(run.info.run_id, [], param_entities, [])
    _verify_logged(store, run.info.run_id, [], expected_param_entities, [])
    param_entities = [Param("long param", "x" * 6001)]
    monkeypatch.setenv("MLFLOW_TRUNCATE_LONG_VALUES", "false")
    with pytest.raises(MlflowException, match="exceeds the maximum length"):
        store.log_batch(run.info.run_id, [], param_entities, [])

    monkeypatch.setenv("MLFLOW_TRUNCATE_LONG_VALUES", "true")
    store.log_batch(run.info.run_id, [], param_entities, [])


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
        os.path.join(current_dir, os.pardir, os.pardir, "resources", "db")
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


def get_ordered_runs(store, order_clauses, experiment_id):
    return [
        r.data.tags[mlflow_tags.MLFLOW_RUN_NAME]
        for r in store.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            run_view_type=ViewType.ALL,
            order_by=order_clauses,
        )
    ]


def _generate_large_data(store, nb_runs=1000):
    experiment_id = store.create_experiment("test_experiment")

    current_run = 0

    run_ids = []
    runs_list = []
    metrics_list = []
    tags_list = []
    params_list = []
    latest_metrics_list = []

    for _ in range(nb_runs):
        run_id = uuid.uuid4().hex
        run_ids.append(run_id)
        run_data = {
            "run_uuid": run_id,
            "user_id": "Anderson",
            "start_time": current_run,
            "artifact_uri": f"file:///tmp/artifacts/{run_id}",
            "experiment_id": experiment_id,
        }
        runs_list.append(run_data)

        for i in range(100):
            metric = {
                "key": f"mkey_{i}",
                "value": i,
                "timestamp": i * 2,
                "step": i * 3,
                "is_nan": False,
                "run_uuid": run_id,
            }
            metrics_list.append(metric)
            tag = {
                "key": f"tkey_{i}",
                "value": "tval_%s" % (current_run % 10),
                "run_uuid": run_id,
            }
            tags_list.append(tag)
            param = {
                "key": f"pkey_{i}",
                "value": "pval_%s" % ((current_run + 1) % 11),
                "run_uuid": run_id,
            }
            params_list.append(param)
        latest_metrics_list.append(
            {
                "key": "mkey_0",
                "value": current_run,
                "timestamp": 100 * 2,
                "step": 100 * 3,
                "is_nan": False,
                "run_uuid": run_id,
            }
        )
        current_run += 1

    # Bulk insert all data in a single transaction
    with store.engine.begin() as conn:
        conn.execute(sqlalchemy.insert(SqlRun), runs_list)
        conn.execute(sqlalchemy.insert(SqlParam), params_list)
        conn.execute(sqlalchemy.insert(SqlMetric), metrics_list)
        conn.execute(sqlalchemy.insert(SqlLatestMetric), latest_metrics_list)
        conn.execute(sqlalchemy.insert(SqlTag), tags_list)

    return experiment_id, run_ids


def test_search_runs_returns_expected_results_with_large_experiment(
    store: SqlAlchemyStore,
):
    """
    This case tests the SQLAlchemyStore implementation of the SearchRuns API to ensure
    that search queries over an experiment containing many runs, each with a large number
    of metrics, parameters, and tags, are performant and return the expected results.
    """
    experiment_id, run_ids = _generate_large_data(store)

    run_results = store.search_runs([experiment_id], None, ViewType.ALL, max_results=100)
    assert len(run_results) == 100
    # runs are sorted by desc start_time
    assert [run.info.run_id for run in run_results] == list(reversed(run_ids[900:]))


def test_search_runs_correctly_filters_large_data(store: SqlAlchemyStore):
    experiment_id, _ = _generate_large_data(store, 1000)

    run_results = store.search_runs(
        [experiment_id],
        "metrics.mkey_0 < 26 and metrics.mkey_0 > 5 ",
        ViewType.ALL,
        max_results=50,
    )
    assert len(run_results) == 20

    run_results = store.search_runs(
        [experiment_id],
        "metrics.mkey_0 < 26 and metrics.mkey_0 > 5 and tags.tkey_0 = 'tval_0' ",
        ViewType.ALL,
        max_results=10,
    )
    assert len(run_results) == 2  # 20 runs between 9 and 26, 2 of which have a 0 tkey_0 value

    run_results = store.search_runs(
        [experiment_id],
        "metrics.mkey_0 < 26 and metrics.mkey_0 > 5 "
        "and tags.tkey_0 = 'tval_0' "
        "and params.pkey_0 = 'pval_0'",
        ViewType.ALL,
        max_results=5,
    )
    assert len(run_results) == 1  # 2 runs on previous request, 1 of which has a 0 pkey_0 value


def test_search_runs_keep_all_runs_when_sorting(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_experiment1")

    r1 = store.create_run(
        experiment_id=experiment_id,
        start_time=0,
        tags=[],
        user_id="Me",
        run_name="name",
    ).info.run_id
    r2 = store.create_run(
        experiment_id=experiment_id,
        start_time=0,
        tags=[],
        user_id="Me",
        run_name="name",
    ).info.run_id
    store.set_tag(r1, RunTag(key="t1", value="1"))
    store.set_tag(r1, RunTag(key="t2", value="1"))
    store.set_tag(r2, RunTag(key="t2", value="1"))

    run_results = store.search_runs(
        [experiment_id], None, ViewType.ALL, max_results=1000, order_by=["tag.t1"]
    )
    assert len(run_results) == 2


def test_try_get_run_tag(store: SqlAlchemyStore):
    run = _run_factory(store)
    store.set_tag(run.info.run_id, entities.RunTag("k1", "v1"))
    store.set_tag(run.info.run_id, entities.RunTag("k2", "v2"))

    with store.ManagedSessionMaker() as session:
        tag = store._try_get_run_tag(session, run.info.run_id, "k0")
        assert tag is None

        tag = store._try_get_run_tag(session, run.info.run_id, "k1")
        assert tag.key == "k1"
        assert tag.value == "v1"

        tag = store._try_get_run_tag(session, run.info.run_id, "k2")
        assert tag.key == "k2"
        assert tag.value == "v2"


def test_get_metric_history_on_non_existent_metric_key(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "test_exp")[0]
    run = store.create_run(
        experiment_id=experiment_id,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )
    run_id = run.info.run_id
    metrics = store.get_metric_history(run_id, "test_metric")
    assert metrics == []


def test_insert_large_text_in_dataset_table(store: SqlAlchemyStore):
    with store.engine.begin() as conn:
        # cursor = conn.cursor()
        dataset_source = "a" * 65535  # 65535 is the max size for a TEXT column
        dataset_profile = "a" * 16777215  # 16777215 is the max size for a MEDIUMTEXT column
        conn.execute(
            sqlalchemy.sql.text(
                f"""
            INSERT INTO datasets
                (dataset_uuid,
                experiment_id,
                name,
                digest,
                dataset_source_type,
                dataset_source,
                dataset_schema,
                dataset_profile)
            VALUES
                ('test_uuid',
                0,
                'test_name',
                'test_digest',
                'test_source_type',
                '{dataset_source}', '
                test_schema',
                '{dataset_profile}')
            """
            )
        )
        results = conn.execute(
            sqlalchemy.sql.text("SELECT dataset_source, dataset_profile from datasets")
        ).first()
        dataset_source_from_db = results[0]
        assert len(dataset_source_from_db) == len(dataset_source)
        dataset_profile_from_db = results[1]
        assert len(dataset_profile_from_db) == len(dataset_profile)

        # delete contents of datasets table
        conn.execute(sqlalchemy.sql.text("DELETE FROM datasets"))


def test_log_inputs_and_retrieve_runs_behaves_as_expected(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "test exp")
    run1 = _run_factory(store, config=_get_run_configs(experiment_id, start_time=1))
    run2 = _run_factory(store, config=_get_run_configs(experiment_id, start_time=3))
    run3 = _run_factory(store, config=_get_run_configs(experiment_id, start_time=2))

    dataset1 = entities.Dataset(
        name="name1",
        digest="digest1",
        source_type="st1",
        source="source1",
        schema="schema1",
        profile="profile1",
    )
    dataset2 = entities.Dataset(
        name="name2",
        digest="digest2",
        source_type="st2",
        source="source2",
        schema="schema2",
        profile="profile2",
    )
    dataset3 = entities.Dataset(
        name="name3",
        digest="digest3",
        source_type="st3",
        source="source3",
        schema="schema3",
        profile="profile3",
    )

    tags1 = [
        entities.InputTag(key="key1", value="value1"),
        entities.InputTag(key="key2", value="value2"),
    ]
    tags2 = [
        entities.InputTag(key="key3", value="value3"),
        entities.InputTag(key="key4", value="value4"),
    ]
    tags3 = [
        entities.InputTag(key="key5", value="value5"),
        entities.InputTag(key="key6", value="value6"),
    ]

    inputs_run1 = [
        entities.DatasetInput(dataset1, tags1),
        entities.DatasetInput(dataset2, tags1),
    ]
    inputs_run2 = [
        entities.DatasetInput(dataset1, tags2),
        entities.DatasetInput(dataset3, tags3),
    ]
    inputs_run3 = [entities.DatasetInput(dataset2, tags3)]

    store.log_inputs(run1.info.run_id, inputs_run1)
    store.log_inputs(run2.info.run_id, inputs_run2)
    store.log_inputs(run3.info.run_id, inputs_run3)

    run1 = store.get_run(run1.info.run_id)
    assert_dataset_inputs_equal(run1.inputs.dataset_inputs, inputs_run1)
    run2 = store.get_run(run2.info.run_id)
    assert_dataset_inputs_equal(run2.inputs.dataset_inputs, inputs_run2)
    run3 = store.get_run(run3.info.run_id)
    assert_dataset_inputs_equal(run3.inputs.dataset_inputs, inputs_run3)

    search_results_1 = store.search_runs(
        [experiment_id], None, ViewType.ALL, max_results=4, order_by=["start_time ASC"]
    )
    run1 = search_results_1[0]
    assert_dataset_inputs_equal(run1.inputs.dataset_inputs, inputs_run1)
    run2 = search_results_1[2]
    assert_dataset_inputs_equal(run2.inputs.dataset_inputs, inputs_run2)
    run3 = search_results_1[1]
    assert_dataset_inputs_equal(run3.inputs.dataset_inputs, inputs_run3)

    search_results_2 = store.search_runs(
        [experiment_id], None, ViewType.ALL, max_results=4, order_by=["start_time DESC"]
    )
    run1 = search_results_2[2]
    assert_dataset_inputs_equal(run1.inputs.dataset_inputs, inputs_run1)
    run2 = search_results_2[0]
    assert_dataset_inputs_equal(run2.inputs.dataset_inputs, inputs_run2)
    run3 = search_results_2[1]
    assert_dataset_inputs_equal(run3.inputs.dataset_inputs, inputs_run3)


def test_log_input_multiple_times_does_not_overwrite_tags_or_dataset(
    store: SqlAlchemyStore,
):
    experiment_id = _create_experiments(store, "test exp")
    run = _run_factory(store, config=_get_run_configs(experiment_id))
    dataset = entities.Dataset(
        name="name",
        digest="digest",
        source_type="st",
        source="source",
        schema="schema",
        profile="profile",
    )
    tags = [
        entities.InputTag(key="key1", value="value1"),
        entities.InputTag(key="key2", value="value2"),
    ]
    store.log_inputs(run.info.run_id, [entities.DatasetInput(dataset, tags)])

    for i in range(3):
        # Since the dataset name and digest are the same as the previously logged dataset,
        # no changes should be made
        overwrite_dataset = entities.Dataset(
            name="name",
            digest="digest",
            source_type="st{i}",
            source=f"source{i}",
            schema=f"schema{i}",
            profile=f"profile{i}",
        )
        # Since the dataset has already been logged as an input to the run, no changes should be
        # made to the input tags
        overwrite_tags = [
            entities.InputTag(key=f"key{i}", value=f"value{i}"),
            entities.InputTag(key=f"key{i + 1}", value=f"value{i + 1}"),
        ]
        store.log_inputs(
            run.info.run_id, [entities.DatasetInput(overwrite_dataset, overwrite_tags)]
        )

    run = store.get_run(run.info.run_id)
    assert_dataset_inputs_equal(run.inputs.dataset_inputs, [entities.DatasetInput(dataset, tags)])

    # Logging a dataset with a different name or digest to the original run should result
    # in the addition of another dataset input
    other_name_dataset = entities.Dataset(
        name="other_name",
        digest="digest",
        source_type="st",
        source="source",
        schema="schema",
        profile="profile",
    )
    other_name_input_tags = [entities.InputTag(key="k1", value="v1")]
    store.log_inputs(
        run.info.run_id,
        [entities.DatasetInput(other_name_dataset, other_name_input_tags)],
    )

    other_digest_dataset = entities.Dataset(
        name="name",
        digest="other_digest",
        source_type="st",
        source="source",
        schema="schema",
        profile="profile",
    )
    other_digest_input_tags = [entities.InputTag(key="k2", value="v2")]
    store.log_inputs(
        run.info.run_id,
        [entities.DatasetInput(other_digest_dataset, other_digest_input_tags)],
    )

    run = store.get_run(run.info.run_id)
    assert_dataset_inputs_equal(
        run.inputs.dataset_inputs,
        [
            entities.DatasetInput(dataset, tags),
            entities.DatasetInput(other_name_dataset, other_name_input_tags),
            entities.DatasetInput(other_digest_dataset, other_digest_input_tags),
        ],
    )

    # Logging the same dataset with different tags to new runs should result in each run
    # having its own new input tags and the same dataset input
    for i in range(3):
        new_run = store.create_run(
            experiment_id=experiment_id,
            user_id="user",
            start_time=0,
            tags=[],
            run_name=None,
        )
        new_tags = [
            entities.InputTag(key=f"key{i}", value=f"value{i}"),
            entities.InputTag(key=f"key{i + 1}", value=f"value{i + 1}"),
        ]
        store.log_inputs(new_run.info.run_id, [entities.DatasetInput(dataset, new_tags)])
        new_run = store.get_run(new_run.info.run_id)
        assert_dataset_inputs_equal(
            new_run.inputs.dataset_inputs, [entities.DatasetInput(dataset, new_tags)]
        )


def test_log_inputs_handles_case_when_no_datasets_are_specified(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "test exp")
    run = _run_factory(store, config=_get_run_configs(experiment_id))
    store.log_inputs(run.info.run_id)
    store.log_inputs(run.info.run_id, datasets=None)


def test_log_inputs_fails_with_missing_inputs(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "test exp")
    run = _run_factory(store, config=_get_run_configs(experiment_id))

    dataset = entities.Dataset(name="name1", digest="digest1", source_type="type", source="source")

    tags = [entities.InputTag(key="key", value="train")]

    # Test input key missing
    with pytest.raises(MlflowException, match="InputTag key cannot be None"):
        store.log_inputs(
            run.info.run_id,
            [
                entities.DatasetInput(
                    tags=[entities.InputTag(key=None, value="train")], dataset=dataset
                )
            ],
        )

    # Test input value missing
    with pytest.raises(MlflowException, match="InputTag value cannot be None"):
        store.log_inputs(
            run.info.run_id,
            [
                entities.DatasetInput(
                    tags=[entities.InputTag(key="key", value=None)], dataset=dataset
                )
            ],
        )

    # Test dataset name missing
    with pytest.raises(MlflowException, match="Dataset name cannot be None"):
        store.log_inputs(
            run.info.run_id,
            [
                entities.DatasetInput(
                    tags=tags,
                    dataset=entities.Dataset(
                        name=None, digest="digest1", source_type="type", source="source"
                    ),
                )
            ],
        )

    # Test dataset digest missing
    with pytest.raises(MlflowException, match="Dataset digest cannot be None"):
        store.log_inputs(
            run.info.run_id,
            [
                entities.DatasetInput(
                    tags=tags,
                    dataset=entities.Dataset(
                        name="name", digest=None, source_type="type", source="source"
                    ),
                )
            ],
        )

    # Test dataset source type missing
    with pytest.raises(MlflowException, match="Dataset source_type cannot be None"):
        store.log_inputs(
            run.info.run_id,
            [
                entities.DatasetInput(
                    tags=tags,
                    dataset=entities.Dataset(
                        name="name", digest="digest1", source_type=None, source="source"
                    ),
                )
            ],
        )

    # Test dataset source missing
    with pytest.raises(MlflowException, match="Dataset source cannot be None"):
        store.log_inputs(
            run.info.run_id,
            [
                entities.DatasetInput(
                    tags=tags,
                    dataset=entities.Dataset(
                        name="name", digest="digest1", source_type="type", source=None
                    ),
                )
            ],
        )


def _validate_log_inputs(
    store: SqlAlchemyStore,
    exp_name,
    dataset_inputs,
):
    run = _run_factory(store, _get_run_configs(_create_experiments(store, exp_name)))
    store.log_inputs(run.info.run_id, dataset_inputs)
    run1 = store.get_run(run.info.run_id)
    assert_dataset_inputs_equal(run1.inputs.dataset_inputs, dataset_inputs)


def _validate_invalid_log_inputs(store: SqlAlchemyStore, run_id, dataset_inputs, error_message):
    with pytest.raises(MlflowException, match=error_message):
        store.log_inputs(run_id, dataset_inputs)


def test_log_inputs_with_large_inputs_limit_check(store: SqlAlchemyStore):
    run = _run_factory(store, _get_run_configs(_create_experiments(store, "test_invalid_inputs")))
    run_id = run.info.run_id

    # Test input key
    dataset = entities.Dataset(name="name1", digest="digest1", source_type="type", source="source")
    _validate_log_inputs(
        store,
        "test_input_key",
        [
            entities.DatasetInput(
                tags=[entities.InputTag(key="a" * MAX_INPUT_TAG_KEY_SIZE, value="train")],
                dataset=dataset,
            )
        ],
    )
    _validate_invalid_log_inputs(
        store,
        run_id,
        [
            entities.DatasetInput(
                tags=[entities.InputTag(key="a" * (MAX_INPUT_TAG_KEY_SIZE + 1), value="train")],
                dataset=dataset,
            )
        ],
        f"'key' exceeds the maximum length of {MAX_INPUT_TAG_KEY_SIZE}",
    )

    # Test input value
    dataset = entities.Dataset(name="name2", digest="digest1", source_type="type", source="source")
    _validate_log_inputs(
        store,
        "test_input_value",
        [
            entities.DatasetInput(
                tags=[entities.InputTag(key="key", value="a" * MAX_INPUT_TAG_VALUE_SIZE)],
                dataset=dataset,
            )
        ],
    )
    _validate_invalid_log_inputs(
        store,
        run_id,
        [
            entities.DatasetInput(
                tags=[entities.InputTag(key="key", value="a" * (MAX_INPUT_TAG_VALUE_SIZE + 1))],
                dataset=dataset,
            )
        ],
        f"'value' exceeds the maximum length of {MAX_INPUT_TAG_VALUE_SIZE}",
    )

    # Test dataset name
    tags = [entities.InputTag(key="key", value="train")]
    _validate_log_inputs(
        store,
        "test_dataset_name",
        [
            entities.DatasetInput(
                tags=tags,
                dataset=entities.Dataset(
                    name="a" * MAX_DATASET_NAME_SIZE,
                    digest="digest1",
                    source_type="type",
                    source="source",
                ),
            )
        ],
    )
    _validate_invalid_log_inputs(
        store,
        run_id,
        [
            entities.DatasetInput(
                tags=tags,
                dataset=entities.Dataset(
                    name="a" * (MAX_DATASET_NAME_SIZE + 1),
                    digest="digest1",
                    source_type="type",
                    source="source",
                ),
            )
        ],
        f"'name' exceeds the maximum length of {MAX_DATASET_NAME_SIZE}",
    )

    # Test dataset digest
    _validate_log_inputs(
        store,
        "test_dataset_digest",
        [
            entities.DatasetInput(
                tags=tags,
                dataset=entities.Dataset(
                    name="name1",
                    digest="a" * MAX_DATASET_DIGEST_SIZE,
                    source_type="type",
                    source="source",
                ),
            )
        ],
    )
    _validate_invalid_log_inputs(
        store,
        run_id,
        [
            entities.DatasetInput(
                tags=tags,
                dataset=entities.Dataset(
                    name="name1",
                    digest="a" * (MAX_DATASET_DIGEST_SIZE + 1),
                    source_type="type",
                    source="source",
                ),
            )
        ],
        f"'digest' exceeds the maximum length of {MAX_DATASET_DIGEST_SIZE}",
    )

    # Test dataset source
    _validate_log_inputs(
        store,
        "test_dataset_source",
        [
            entities.DatasetInput(
                tags=tags,
                dataset=entities.Dataset(
                    name="name3",
                    digest="digest1",
                    source_type="type",
                    source="a" * MAX_DATASET_SOURCE_SIZE,
                ),
            )
        ],
    )
    _validate_invalid_log_inputs(
        store,
        run_id,
        [
            entities.DatasetInput(
                tags=tags,
                dataset=entities.Dataset(
                    name="name3",
                    digest="digest1",
                    source_type="type",
                    source="a" * (MAX_DATASET_SOURCE_SIZE + 1),
                ),
            )
        ],
        f"'source' exceeds the maximum length of {MAX_DATASET_SOURCE_SIZE}",
    )

    # Test dataset schema
    _validate_log_inputs(
        store,
        "test_dataset_schema",
        [
            entities.DatasetInput(
                tags=tags,
                dataset=entities.Dataset(
                    name="name4",
                    digest="digest1",
                    source_type="type",
                    source="source",
                    schema="a" * MAX_DATASET_SCHEMA_SIZE,
                ),
            )
        ],
    )
    _validate_invalid_log_inputs(
        store,
        run_id,
        [
            entities.DatasetInput(
                tags=tags,
                dataset=entities.Dataset(
                    name="name4",
                    digest="digest1",
                    source_type="type",
                    source="source",
                    schema="a" * (MAX_DATASET_SCHEMA_SIZE + 1),
                ),
            )
        ],
        f"'schema' exceeds the maximum length of {MAX_DATASET_SCHEMA_SIZE}",
    )

    # Test dataset profile
    _validate_log_inputs(
        store,
        "test_dataset_profile",
        [
            entities.DatasetInput(
                tags=tags,
                dataset=entities.Dataset(
                    name="name5",
                    digest="digest1",
                    source_type="type",
                    source="source",
                    profile="a" * MAX_DATASET_PROFILE_SIZE,
                ),
            )
        ],
    )
    _validate_invalid_log_inputs(
        store,
        run_id,
        [
            entities.DatasetInput(
                tags=tags,
                dataset=entities.Dataset(
                    name="name5",
                    digest="digest1",
                    source_type="type",
                    source="source",
                    profile="a" * (MAX_DATASET_PROFILE_SIZE + 1),
                ),
            )
        ],
        f"'profile' exceeds the maximum length of {MAX_DATASET_PROFILE_SIZE}",
    )


def test_log_inputs_with_duplicates_in_single_request(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "test exp")
    run1 = _run_factory(store, config=_get_run_configs(experiment_id, start_time=1))

    dataset1 = entities.Dataset(
        name="name1",
        digest="digest1",
        source_type="st1",
        source="source1",
        schema="schema1",
        profile="profile1",
    )

    tags1 = [
        entities.InputTag(key="key1", value="value1"),
        entities.InputTag(key="key2", value="value2"),
    ]

    inputs_run1 = [
        entities.DatasetInput(dataset1, tags1),
        entities.DatasetInput(dataset1, tags1),
    ]

    store.log_inputs(run1.info.run_id, inputs_run1)
    run1 = store.get_run(run1.info.run_id)
    assert_dataset_inputs_equal(
        run1.inputs.dataset_inputs, [entities.DatasetInput(dataset1, tags1)]
    )


def test_sqlalchemy_store_behaves_as_expected_with_inmemory_sqlite_db(monkeypatch):
    monkeypatch.setenv("MLFLOW_SQLALCHEMYSTORE_POOLCLASS", "SingletonThreadPool")
    store = SqlAlchemyStore("sqlite:///:memory:", ARTIFACT_URI)
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


def test_sqlalchemy_store_can_be_initialized_when_default_experiment_has_been_deleted(
    tmp_sqlite_uri,
):
    store = SqlAlchemyStore(tmp_sqlite_uri, ARTIFACT_URI)
    store.delete_experiment("0")
    assert store.get_experiment("0").lifecycle_stage == entities.LifecycleStage.DELETED
    SqlAlchemyStore(tmp_sqlite_uri, ARTIFACT_URI)


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


def _assert_create_experiment_appends_to_artifact_uri_path_correctly(
    artifact_root_uri, expected_artifact_uri_format
):
    # Patch `is_local_uri` to prevent the SqlAlchemy store from attempting to create local
    # filesystem directories for file URI and POSIX path test cases
    with mock.patch("mlflow.store.tracking.sqlalchemy_store.is_local_uri", return_value=False):
        with TempDir() as tmp:
            dbfile_path = tmp.path("db")
            store = SqlAlchemyStore(
                db_uri="sqlite:///" + dbfile_path,
                default_artifact_root=artifact_root_uri,
            )
            exp_id = store.create_experiment(name="exp")
            exp = store.get_experiment(exp_id)

            if hasattr(store, "__del__"):
                store.__del__()

            cwd = Path.cwd().as_posix()
            drive = Path.cwd().drive
            if is_windows() and expected_artifact_uri_format.startswith("file:"):
                cwd = f"/{cwd}"
                drive = f"{drive}/"
            assert exp.artifact_location == expected_artifact_uri_format.format(
                e=exp_id, cwd=cwd, drive=drive
            )


@pytest.mark.skipif(not is_windows(), reason="This test only passes on Windows")
@pytest.mark.parametrize(
    ("input_uri", "expected_uri"),
    [
        (
            "\\my_server/my_path/my_sub_path",
            "file:///{drive}my_server/my_path/my_sub_path/{e}",
        ),
        ("path/to/local/folder", "file://{cwd}/path/to/local/folder/{e}"),
        ("/path/to/local/folder", "file:///{drive}path/to/local/folder/{e}"),
        ("#path/to/local/folder?", "file://{cwd}/{e}#path/to/local/folder?"),
        ("file:path/to/local/folder", "file://{cwd}/path/to/local/folder/{e}"),
        ("file:///path/to/local/folder", "file:///{drive}path/to/local/folder/{e}"),
        (
            "file:path/to/local/folder?param=value",
            "file://{cwd}/path/to/local/folder/{e}?param=value",
        ),
        (
            "file:///path/to/local/folder?param=value#fragment",
            "file:///{drive}path/to/local/folder/{e}?param=value#fragment",
        ),
    ],
)
def test_create_experiment_appends_to_artifact_local_path_file_uri_correctly_on_windows(
    input_uri, expected_uri
):
    _assert_create_experiment_appends_to_artifact_uri_path_correctly(input_uri, expected_uri)


@pytest.mark.skipif(is_windows(), reason="This test fails on Windows")
@pytest.mark.parametrize(
    ("input_uri", "expected_uri"),
    [
        ("path/to/local/folder", "{cwd}/path/to/local/folder/{e}"),
        ("/path/to/local/folder", "/path/to/local/folder/{e}"),
        ("#path/to/local/folder?", "{cwd}/#path/to/local/folder?/{e}"),
        ("file:path/to/local/folder", "file://{cwd}/path/to/local/folder/{e}"),
        ("file:///path/to/local/folder", "file:///path/to/local/folder/{e}"),
        (
            "file:path/to/local/folder?param=value",
            "file://{cwd}/path/to/local/folder/{e}?param=value",
        ),
        (
            "file:///path/to/local/folder?param=value#fragment",
            "file:///path/to/local/folder/{e}?param=value#fragment",
        ),
    ],
)
def test_create_experiment_appends_to_artifact_local_path_file_uri_correctly(
    input_uri, expected_uri
):
    _assert_create_experiment_appends_to_artifact_uri_path_correctly(input_uri, expected_uri)


@pytest.mark.parametrize(
    ("input_uri", "expected_uri"),
    [
        ("s3://bucket/path/to/root", "s3://bucket/path/to/root/{e}"),
        (
            "s3://bucket/path/to/root?creds=mycreds",
            "s3://bucket/path/to/root/{e}?creds=mycreds",
        ),
        (
            "dbscheme+driver://root@host/dbname?creds=mycreds#myfragment",
            "dbscheme+driver://root@host/dbname/{e}?creds=mycreds#myfragment",
        ),
        (
            "dbscheme+driver://root:password@hostname.com?creds=mycreds#myfragment",
            "dbscheme+driver://root:password@hostname.com/{e}?creds=mycreds#myfragment",
        ),
        (
            "dbscheme+driver://root:password@hostname.com/mydb?creds=mycreds#myfragment",
            "dbscheme+driver://root:password@hostname.com/mydb/{e}?creds=mycreds#myfragment",
        ),
    ],
)
def test_create_experiment_appends_to_artifact_uri_path_correctly(input_uri, expected_uri):
    _assert_create_experiment_appends_to_artifact_uri_path_correctly(input_uri, expected_uri)


def _assert_create_run_appends_to_artifact_uri_path_correctly(
    artifact_root_uri, expected_artifact_uri_format
):
    # Patch `is_local_uri` to prevent the SqlAlchemy store from attempting to create local
    # filesystem directories for file URI and POSIX path test cases
    with mock.patch("mlflow.store.tracking.sqlalchemy_store.is_local_uri", return_value=False):
        with TempDir() as tmp:
            dbfile_path = tmp.path("db")
            store = SqlAlchemyStore(
                db_uri="sqlite:///" + dbfile_path,
                default_artifact_root=artifact_root_uri,
            )
            exp_id = store.create_experiment(name="exp")
            run = store.create_run(
                experiment_id=exp_id,
                user_id="user",
                start_time=0,
                tags=[],
                run_name="name",
            )

            if hasattr(store, "__del__"):
                store.__del__()

            cwd = Path.cwd().as_posix()
            drive = Path.cwd().drive
            if is_windows() and expected_artifact_uri_format.startswith("file:"):
                cwd = f"/{cwd}"
                drive = f"{drive}/"
            assert run.info.artifact_uri == expected_artifact_uri_format.format(
                e=exp_id, r=run.info.run_id, cwd=cwd, drive=drive
            )


@pytest.mark.skipif(not is_windows(), reason="This test only passes on Windows")
@pytest.mark.parametrize(
    ("input_uri", "expected_uri"),
    [
        (
            "\\my_server/my_path/my_sub_path",
            "file:///{drive}my_server/my_path/my_sub_path/{e}/{r}/artifacts",
        ),
        ("path/to/local/folder", "file://{cwd}/path/to/local/folder/{e}/{r}/artifacts"),
        (
            "/path/to/local/folder",
            "file:///{drive}path/to/local/folder/{e}/{r}/artifacts",
        ),
        (
            "#path/to/local/folder?",
            "file://{cwd}/{e}/{r}/artifacts#path/to/local/folder?",
        ),
        (
            "file:path/to/local/folder",
            "file://{cwd}/path/to/local/folder/{e}/{r}/artifacts",
        ),
        (
            "file:///path/to/local/folder",
            "file:///{drive}path/to/local/folder/{e}/{r}/artifacts",
        ),
        (
            "file:path/to/local/folder?param=value",
            "file://{cwd}/path/to/local/folder/{e}/{r}/artifacts?param=value",
        ),
        (
            "file:///path/to/local/folder?param=value#fragment",
            "file:///{drive}path/to/local/folder/{e}/{r}/artifacts?param=value#fragment",
        ),
    ],
)
def test_create_run_appends_to_artifact_local_path_file_uri_correctly_on_windows(
    input_uri, expected_uri
):
    _assert_create_run_appends_to_artifact_uri_path_correctly(input_uri, expected_uri)


@pytest.mark.skipif(is_windows(), reason="This test fails on Windows")
@pytest.mark.parametrize(
    ("input_uri", "expected_uri"),
    [
        ("path/to/local/folder", "{cwd}/path/to/local/folder/{e}/{r}/artifacts"),
        ("/path/to/local/folder", "/path/to/local/folder/{e}/{r}/artifacts"),
        ("#path/to/local/folder?", "{cwd}/#path/to/local/folder?/{e}/{r}/artifacts"),
        (
            "file:path/to/local/folder",
            "file://{cwd}/path/to/local/folder/{e}/{r}/artifacts",
        ),
        (
            "file:///path/to/local/folder",
            "file:///path/to/local/folder/{e}/{r}/artifacts",
        ),
        (
            "file:path/to/local/folder?param=value",
            "file://{cwd}/path/to/local/folder/{e}/{r}/artifacts?param=value",
        ),
        (
            "file:///path/to/local/folder?param=value#fragment",
            "file:///path/to/local/folder/{e}/{r}/artifacts?param=value#fragment",
        ),
    ],
)
def test_create_run_appends_to_artifact_local_path_file_uri_correctly(input_uri, expected_uri):
    _assert_create_run_appends_to_artifact_uri_path_correctly(input_uri, expected_uri)


@pytest.mark.parametrize(
    ("input_uri", "expected_uri"),
    [
        ("s3://bucket/path/to/root", "s3://bucket/path/to/root/{e}/{r}/artifacts"),
        (
            "s3://bucket/path/to/root?creds=mycreds",
            "s3://bucket/path/to/root/{e}/{r}/artifacts?creds=mycreds",
        ),
        (
            "dbscheme+driver://root@host/dbname?creds=mycreds#myfragment",
            "dbscheme+driver://root@host/dbname/{e}/{r}/artifacts?creds=mycreds#myfragment",
        ),
        (
            "dbscheme+driver://root:password@hostname.com?creds=mycreds#myfragment",
            "dbscheme+driver://root:password@hostname.com/{e}/{r}/artifacts"
            "?creds=mycreds#myfragment",
        ),
        (
            "dbscheme+driver://root:password@hostname.com/mydb?creds=mycreds#myfragment",
            "dbscheme+driver://root:password@hostname.com/mydb/{e}/{r}/artifacts"
            "?creds=mycreds#myfragment",
        ),
    ],
)
def test_create_run_appends_to_artifact_uri_path_correctly(input_uri, expected_uri):
    _assert_create_run_appends_to_artifact_uri_path_correctly(input_uri, expected_uri)


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
    assert trace_info.trace_metadata == {"rq1": "foo", "rq2": "bar"}
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
        experiment_ids=[exp1, exp2],
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
        experiment_ids=[exp1, exp2],
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
            experiment_ids=[exp1, exp2],
            filter_string=filter_string,
        )


def test_search_traces_raise_if_max_results_arg_is_invalid(store):
    with pytest.raises(
        MlflowException,
        match="Invalid value 50001 for parameter 'max_results' supplied.",
    ):
        store.search_traces(experiment_ids=[], max_results=50001)

    with pytest.raises(
        MlflowException, match="Invalid value -1 for parameter 'max_results' supplied."
    ):
        store.search_traces(experiment_ids=[], max_results=-1)


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

    # Store assessments
    store.create_assessment(feedback1)
    store.create_assessment(feedback2)
    store.create_assessment(feedback3)
    store.create_assessment(expectation1)
    store.create_assessment(expectation2)
    store.create_assessment(expectation3)

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

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.`mlflow.spanInputs` ILIKE "%test1%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace_info_1.trace_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.`mlflow.spanInputs` ILIKE "%test2%"'
    )
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
            session.query(SqlSpan)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_different_traces_raises_error(store: SqlAlchemyStore, is_async: bool):
    # Create two different traces
    experiment_id = store.create_experiment("test_multi_trace_experiment")
    trace_info1 = TraceInfo(
        trace_id="tr-span-test-789",
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1234,
        execution_duration=100,
        state=TraceState.OK,
    )
    trace_info2 = TraceInfo(
        trace_id="tr-span-test-999",
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=5678,
        execution_duration=200,
        state=TraceState.OK,
    )
    trace_info1 = store.start_trace(trace_info1)
    trace_info2 = store.start_trace(trace_info2)

    # Create spans for different traces
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
                "mlflow.traceRequestId": json.dumps(trace_info1.trace_id, cls=TraceJSONEncoder)
            },
            start_time=1000000000,
            end_time=2000000000,
            resource=_OTelResource.get_empty(),
        ),
        trace_info1.trace_id,
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
            attributes={
                "mlflow.traceRequestId": json.dumps(trace_info2.trace_id, cls=TraceJSONEncoder)
            },
            start_time=3000000000,
            end_time=4000000000,
            resource=_OTelResource.get_empty(),
        ),
        trace_info2.trace_id,
    )

    # Try to log spans from different traces - should raise MlflowException
    if is_async:
        with pytest.raises(MlflowException, match="All spans must belong to the same trace"):
            await store.log_spans_async(experiment_id, [span1, span2])
    else:
        with pytest.raises(MlflowException, match="All spans must belong to the same trace"):
            store.log_spans(experiment_id, [span1, span2])


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
            session.query(SqlSpan)
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


def test_log_outputs(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    run = store.create_run(
        experiment_id=exp_id, user_id="user", start_time=0, run_name="test", tags=[]
    )
    model = store.create_logged_model(experiment_id=exp_id)
    store.log_outputs(run.info.run_id, [LoggedModelOutput(model.model_id, 1)])
    run = store.get_run(run.info.run_id)
    assert run.outputs.model_outputs == [LoggedModelOutput(model.model_id, 1)]


def test_search_runs_returns_outputs(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")

    run1 = store.create_run(
        experiment_id=exp_id, user_id="user", start_time=0, run_name="run_with_model", tags=[]
    )
    model1 = store.create_logged_model(experiment_id=exp_id)
    store.log_outputs(run1.info.run_id, [LoggedModelOutput(model1.model_id, 0)])

    run2 = store.create_run(
        experiment_id=exp_id, user_id="user", start_time=1, run_name="run_without_model", tags=[]
    )

    result = store.search_runs(
        experiment_ids=[exp_id],
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
    )

    assert len(result) == 2

    run_with_model = next(r for r in result if r.info.run_id == run1.info.run_id)
    assert run_with_model.outputs is not None
    assert len(run_with_model.outputs.model_outputs) == 1
    assert run_with_model.outputs.model_outputs[0].model_id == model1.model_id
    assert run_with_model.outputs.model_outputs[0].step == 0

    run_without_model = next(r for r in result if r.info.run_id == run2.info.run_id)
    assert run_without_model.outputs is not None
    assert len(run_without_model.outputs.model_outputs) == 0


@pytest.mark.parametrize("tags_count", [0, 1, 2])
def test_get_run_inputs(store, tags_count):
    run = _run_factory(store)

    dataset = entities.Dataset(
        name="name1",
        digest="digest1",
        source_type="st1",
        source="source1",
        schema="schema1",
        profile="profile1",
    )

    tags = [entities.InputTag(key=f"foo{i}", value=f"bar{i}") for i in range(tags_count)]

    dataset_inputs = [entities.DatasetInput(dataset, tags)]

    store.log_inputs(run.info.run_id, dataset_inputs)

    with store.ManagedSessionMaker() as session:
        actual = store._get_run_inputs(session, [run.info.run_id])

    assert len(actual) == 1
    assert_dataset_inputs_equal(actual[0], dataset_inputs)


def test_get_run_inputs_run_order(store):
    exp_id = _create_experiments(store, "test_get_run_inputs_run_order")
    config = _get_run_configs(exp_id)

    run_with_one_input = _run_factory(store, config)
    run_with_no_inputs = _run_factory(store, config)
    run_with_two_inputs = _run_factory(store, config)

    dataset1 = entities.Dataset(
        name="name1",
        digest="digest1",
        source_type="st1",
        source="source1",
        schema="schema1",
        profile="profile1",
    )

    dataset2 = entities.Dataset(
        name="name2",
        digest="digest2",
        source_type="st2",
        source="source2",
        schema="schema2",
        profile="profile2",
    )

    tags_1 = [entities.InputTag(key="foo1", value="bar1")]

    tags_2 = [
        entities.InputTag(key="foo2", value="bar2"),
        entities.InputTag(key="foo3", value="bar3"),
    ]

    tags_3 = [
        entities.InputTag(key="foo4", value="bar4"),
        entities.InputTag(key="foo5", value="bar5"),
        entities.InputTag(key="foo6", value="bar6"),
    ]

    dataset_inputs_1 = [entities.DatasetInput(dataset1, tags_1)]
    dataset_inputs_2 = [
        entities.DatasetInput(dataset2, tags_2),
        entities.DatasetInput(dataset1, tags_3),
    ]

    store.log_inputs(run_with_one_input.info.run_id, dataset_inputs_1)
    store.log_inputs(run_with_two_inputs.info.run_id, dataset_inputs_2)

    expected = [dataset_inputs_1, [], dataset_inputs_2]

    runs = [run_with_one_input, run_with_no_inputs, run_with_two_inputs]
    run_uuids = [run.info.run_id for run in runs]

    with store.ManagedSessionMaker() as session:
        actual = store._get_run_inputs(session, run_uuids)

    assert len(expected) == len(actual)
    for expected_i, actual_i in zip(expected, actual):
        assert_dataset_inputs_equal(expected_i, actual_i)


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
            session.query(SqlLoggedModelMetric)
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

    with pytest.raises(MlflowException, match=r"Trace with request_id 'fake_trace' not found"):
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

    with pytest.raises(MlflowException, match=r"Trace with request_id 'fake_trace' not found"):
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


def _create_trace_info(trace_id: str, experiment_id: str):
    return TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1234,
        execution_duration=100,
        state=TraceState.OK,
        tags={"tag1": "apple", "tag2": "orange"},
        trace_metadata={"rq1": "foo", "rq2": "bar"},
    )


def test_link_traces_to_run(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    run = store.create_run(exp_id, user_id="user", start_time=0, tags=[], run_name="test_run")

    trace_ids = []
    for i in range(5):
        trace_info = _create_trace_info(f"trace-{i}", exp_id)
        store.start_trace(trace_info)
        trace_ids.append(trace_info.trace_id)

    store.link_traces_to_run(trace_ids, run.info.run_id)

    # search_traces should return traces linked to the run
    traces, _ = store.search_traces(
        experiment_ids=[exp_id], filter_string=f"run_id = '{run.info.run_id}'"
    )
    assert len(traces) == 5


def test_link_traces_to_run_100_limit(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    run = store.create_run(exp_id, user_id="user", start_time=0, tags=[], run_name="test_run")

    # Test exceeding the limit (101 traces)
    trace_ids = []
    for i in range(101):
        trace_info = _create_trace_info(f"trace-{i}", exp_id)
        store.start_trace(trace_info)
        trace_ids.append(trace_info.trace_id)

    with pytest.raises(MlflowException, match="Cannot link more than 100 traces to a run"):
        store.link_traces_to_run(trace_ids, run.info.run_id)


def test_link_traces_to_run_duplicate_trace_ids(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_ids = ["trace-1", "trace-2", "trace-3", "trace-4"]
    for trace_id in trace_ids:
        trace_info = _create_trace_info(trace_id, exp_id)
        store.start_trace(trace_info)
    run = store.create_run(exp_id, user_id="user", start_time=0, tags=[], run_name="test_run")
    search_args = {"experiment_ids": [exp_id], "filter_string": f"run_id = '{run.info.run_id}'"}

    store.link_traces_to_run(["trace-1", "trace-2", "trace-3"], run.info.run_id)

    assert len(store.search_traces(**search_args)[0]) == 3

    store.link_traces_to_run(["trace-3", "trace-4"], run.info.run_id)
    assert len(store.search_traces(**search_args)[0]) == 4

    store.link_traces_to_run(["trace-1", "trace-2"], run.info.run_id)
    assert len(store.search_traces(**search_args)[0]) == 4


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

    store.register_scorer(experiment_id, "accuracy_scorer", "serialized_accuracy_scorer1")
    store.register_scorer(experiment_id, "accuracy_scorer", "serialized_accuracy_scorer2")
    store.register_scorer(experiment_id, "accuracy_scorer", "serialized_accuracy_scorer3")

    store.register_scorer(experiment_id, "safety_scorer", "serialized_safety_scorer1")
    store.register_scorer(experiment_id, "safety_scorer", "serialized_safety_scorer2")

    store.register_scorer(experiment_id, "relevance_scorer", "relevance_scorer_scorer1")

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
            assert scorer._serialized_scorer == "serialized_accuracy_scorer3"
        elif scorer.scorer_name == "safety_scorer":
            assert scorer.scorer_version == 2, (
                f"Expected version 2 for safety_scorer, got {scorer.scorer_version}"
            )
            assert scorer._serialized_scorer == "serialized_safety_scorer2"
        elif scorer.scorer_name == "relevance_scorer":
            assert scorer.scorer_version == 1, (
                f"Expected version 1 for relevance_scorer, got {scorer.scorer_version}"
            )
            assert scorer._serialized_scorer == "relevance_scorer_scorer1"

    # Test list_scorer_versions
    accuracy_scorer_versions = store.list_scorer_versions(experiment_id, "accuracy_scorer")
    assert len(accuracy_scorer_versions) == 3, (
        f"Expected 3 versions, got {len(accuracy_scorer_versions)}"
    )

    # Verify versions are ordered by version number
    assert accuracy_scorer_versions[0].scorer_version == 1
    assert accuracy_scorer_versions[0]._serialized_scorer == "serialized_accuracy_scorer1"
    assert accuracy_scorer_versions[1].scorer_version == 2
    assert accuracy_scorer_versions[1]._serialized_scorer == "serialized_accuracy_scorer2"
    assert accuracy_scorer_versions[2].scorer_version == 3
    assert accuracy_scorer_versions[2]._serialized_scorer == "serialized_accuracy_scorer3"

    # Step 3: Test get_scorer with specific versions
    # Get accuracy_scorer version 1
    accuracy_v1 = store.get_scorer(experiment_id, "accuracy_scorer", version=1)
    assert accuracy_v1._serialized_scorer == "serialized_accuracy_scorer1"
    assert accuracy_v1.scorer_version == 1

    # Get accuracy_scorer version 2
    accuracy_v2 = store.get_scorer(experiment_id, "accuracy_scorer", version=2)
    assert accuracy_v2._serialized_scorer == "serialized_accuracy_scorer2"
    assert accuracy_v2.scorer_version == 2

    # Get accuracy_scorer version 3 (latest)
    accuracy_v3 = store.get_scorer(experiment_id, "accuracy_scorer", version=3)
    assert accuracy_v3._serialized_scorer == "serialized_accuracy_scorer3"
    assert accuracy_v3.scorer_version == 3

    # Step 4: Test get_scorer without version (should return latest)
    accuracy_latest = store.get_scorer(experiment_id, "accuracy_scorer")
    assert accuracy_latest._serialized_scorer == "serialized_accuracy_scorer3"
    assert accuracy_latest.scorer_version == 3

    safety_latest = store.get_scorer(experiment_id, "safety_scorer")
    assert safety_latest._serialized_scorer == "serialized_safety_scorer2"
    assert safety_latest.scorer_version == 2

    relevance_latest = store.get_scorer(experiment_id, "relevance_scorer")
    assert relevance_latest._serialized_scorer == "relevance_scorer_scorer1"
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
    assert accuracy_v2._serialized_scorer == "serialized_accuracy_scorer2"
    assert accuracy_v2.scorer_version == 2

    accuracy_v3 = store.get_scorer(experiment_id, "accuracy_scorer", version=3)
    assert accuracy_v3._serialized_scorer == "serialized_accuracy_scorer3"
    assert accuracy_v3.scorer_version == 3

    # Verify latest version still works
    accuracy_latest_after_partial_delete = store.get_scorer(experiment_id, "accuracy_scorer")
    assert accuracy_latest_after_partial_delete._serialized_scorer == "serialized_accuracy_scorer3"
    assert accuracy_latest_after_partial_delete.scorer_version == 3

    # Step 7: Test delete_scorer - delete all versions of accuracy_scorer
    store.delete_scorer(experiment_id, "accuracy_scorer")

    # Verify accuracy_scorer is completely deleted
    with pytest.raises(MlflowException, match="Scorer with name 'accuracy_scorer' not found"):
        store.get_scorer(experiment_id, "accuracy_scorer")

    # Verify other scorers still exist
    safety_latest_after_delete = store.get_scorer(experiment_id, "safety_scorer")
    assert safety_latest_after_delete._serialized_scorer == "serialized_safety_scorer2"
    assert safety_latest_after_delete.scorer_version == 2

    relevance_latest_after_delete = store.get_scorer(experiment_id, "relevance_scorer")
    assert relevance_latest_after_delete._serialized_scorer == "relevance_scorer_scorer1"
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
    store.register_scorer(experiment_id, "accuracy_scorer", "serialized_accuracy_scorer1")
    store.register_scorer(experiment_id, "accuracy_scorer", "serialized_accuracy_scorer2")
    store.register_scorer(experiment_id, "accuracy_scorer", "serialized_accuracy_scorer3")

    # Test list_scorer_versions for non-existent scorer
    with pytest.raises(MlflowException, match="Scorer with name 'non_existent_scorer' not found"):
        store.list_scorer_versions(experiment_id, "non_existent_scorer")


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

        with pytest.raises(MlflowException, match="not found"):
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
                TraceMetadataKey.SIZE_STATS: json.dumps(
                    {
                        TraceSizeStatsKey.NUM_SPANS: 2,
                    }
                ),
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
        SpanAttributeKey.CHAT_USAGE: json.dumps(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        ),
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
        SpanAttributeKey.CHAT_USAGE: json.dumps(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        ),
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
        SpanAttributeKey.CHAT_USAGE: json.dumps(
            {
                "input_tokens": 200,
                "output_tokens": 75,
                "total_tokens": 275,
            }
        ),
    }
    span2 = create_mlflow_span(otel_span2, trace_id, "LLM")
    store.log_spans(experiment_id, [span2])

    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.token_usage["input_tokens"] == 300
    assert trace.info.token_usage["output_tokens"] == 125
    assert trace.info.token_usage["total_tokens"] == 425


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
        SpanAttributeKey.CHAT_USAGE: json.dumps(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        ),
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
        SpanAttributeKey.CHAT_USAGE: json.dumps(
            {
                "input_tokens": 200,
                "output_tokens": 100,
                "total_tokens": 300,
            }
        ),
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
            TraceMetadataKey.TOKEN_USAGE: json.dumps(
                {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                }
            )
        },
    )
    store.start_trace(trace_info)

    with store.ManagedSessionMaker() as session:
        metrics = (
            session.query(SqlTraceMetrics)
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
        SpanAttributeKey.CHAT_USAGE: json.dumps(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        ),
    }
    span1 = create_mlflow_span(otel_span1, trace_id, "LLM")
    store.log_spans(experiment_id, [span1])

    with store.ManagedSessionMaker() as session:
        metrics = (
            session.query(SqlTraceMetrics)
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
        SpanAttributeKey.CHAT_USAGE: json.dumps(
            {
                "input_tokens": 200,
                "output_tokens": 75,
                "total_tokens": 275,
            }
        ),
    }
    span2 = create_mlflow_span(otel_span2, trace_id, "LLM")
    store.log_spans(experiment_id, [span2])

    with store.ManagedSessionMaker() as session:
        metrics = (
            session.query(SqlTraceMetrics)
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

    # Launch multiple concurrent log_spans calls
    with ThreadPoolExecutor(max_workers=5) as executor:
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
                TraceMetadataKey.SIZE_STATS: json.dumps(
                    {
                        TraceSizeStatsKey.NUM_SPANS: 2,  # Expecting 2 spans
                    }
                ),
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
                TraceMetadataKey.SIZE_STATS: json.dumps(
                    {
                        TraceSizeStatsKey.NUM_SPANS: 2,  # Expecting 2 spans
                    }
                ),
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
