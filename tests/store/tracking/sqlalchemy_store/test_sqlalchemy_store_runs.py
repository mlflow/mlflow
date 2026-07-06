import contextlib
import json
import math
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import pytest
import sqlalchemy
from packaging.version import Version

import mlflow
from mlflow import entities
from mlflow.entities import (
    Metric,
    Param,
    RunStatus,
    RunTag,
    SourceType,
    ViewType,
    _DatasetSummary,
    trace_location,
)
from mlflow.entities.logged_model_output import LoggedModelOutput
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_state import TraceState
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
from mlflow.store.entities import PagedList
from mlflow.store.tracking import (
    SEARCH_MAX_RESULTS_THRESHOLD,
)
from mlflow.store.tracking.dbmodels import models
from mlflow.store.tracking.dbmodels.models import (
    SqlLatestMetric,
    SqlMetric,
    SqlParam,
    SqlRun,
    SqlTag,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.utils import mlflow_tags
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATASET_CONTEXT,
    MLFLOW_RUN_NAME,
)
from mlflow.utils.name_utils import _GENERATOR_PREDICATES
from mlflow.utils.os import is_windows
from mlflow.utils.time import get_current_time_millis
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
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.store.tracking.sqlalchemy_store.conftest import (
    _create_experiments,
    _get_ordered_runs,
    _get_run_configs,
    _run_factory,
    _search_runs,
    _verify_logged,
)
from tests.store.tracking.test_file_store import assert_dataset_inputs_equal

pytestmark = pytest.mark.notrackingurimock


def _create_trace_info(trace_id: str, experiment_id) -> TraceInfo:
    return TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=0,
        execution_duration=0,
        state=TraceState.OK,
        tags={},
        trace_metadata={},
        client_request_id=None,
    )


def test_run_tag_model(store: SqlAlchemyStore):
    # Create a run whose UUID we can reference when creating tag models.
    # `run_id` is a foreign key in the tags table; therefore, in order
    # to insert a tag with a given run UUID, the UUID must be present in
    # the runs table
    run = _run_factory(store)
    with store.ManagedSessionMaker(read_only=False) as session:
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
    with store.ManagedSessionMaker(read_only=False) as session:
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
    with store.ManagedSessionMaker(read_only=False) as session:
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
        with store.ManagedSessionMaker(read_only=False) as session:
            session.add(models.SqlRun())
    assert exception_context.value.error_code == ErrorCode.Name(BAD_REQUEST)


def test_run_data_model(store: SqlAlchemyStore):
    with store.ManagedSessionMaker(read_only=False) as session:
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
        context_manager = (
            WorkspaceContext(DEFAULT_WORKSPACE_NAME)
            if isinstance(store, WorkspaceAwareSqlAlchemyStore)
            else contextlib.nullcontext()
        )
        with context_manager:
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
    with ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="test-sqlalchemy-log-metrics"
    ) as executor:
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

    with pytest.raises(MlflowException, match=r"'name' exceeds the maximum length"):
        store.rename_experiment(experiment_id, "x" * (MAX_EXPERIMENT_NAME_LENGTH + 1))


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


def test_search_attrs(store: SqlAlchemyStore):
    e1 = _create_experiments(store, "search_attributes_1")
    r1 = _run_factory(store, _get_run_configs(experiment_id=e1)).info.run_id

    e2 = _create_experiments(store, "search_attrs_2")
    r2 = _run_factory(store, _get_run_configs(experiment_id=e2)).info.run_id
    run1_artifact_uri = store.get_run(r1).info.artifact_uri
    uppercase_run1_artifact_uri = run1_artifact_uri.upper()
    mismatched_artifact_uri = run1_artifact_uri.replace(f"/{e1}/", f"/{e2}/", 1)
    if mismatched_artifact_uri == run1_artifact_uri:
        mismatched_artifact_uri = f"{run1_artifact_uri}/unexpected"

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

    filter_string = f"attr.artifact_uri = '{run1_artifact_uri}'"
    assert _search_runs(store, [e1, e2], filter_string) == [r1]

    filter_string = f"attr.artifact_uri = '{uppercase_run1_artifact_uri}'"
    assert _search_runs(store, [e1, e2], filter_string) == []

    filter_string = f"attr.artifact_uri != '{uppercase_run1_artifact_uri}'"
    assert sorted(
        [r1, r2],
    ) == sorted(_search_runs(store, [e1, e2], filter_string))

    filter_string = f"attr.artifact_uri = '{mismatched_artifact_uri}'"
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
    with store.ManagedSessionMaker(read_only=False) as session:
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
    runs = sorted([
        _run_factory(store, _get_run_configs(exp, start_time=10)).info.run_id for r in range(10)
    ])
    for n in [1, 2, 4, 8, 10, 20]:
        assert runs[: min(10, n)] == _search_runs(store, exp, max_results=n)


def test_search_runs_pagination(store: SqlAlchemyStore):
    exp = _create_experiments(store, "test_search_runs_pagination")
    # test returned token behavior
    runs = sorted([
        _run_factory(store, _get_run_configs(exp, start_time=10)).info.run_id for r in range(10)
    ])
    result = store.search_runs([exp], None, ViewType.ALL, max_results=4)
    assert [r.info.run_id for r in result] == runs[0:4]
    assert result.token is not None
    result = store.search_runs([exp], None, ViewType.ALL, max_results=4, page_token=result.token)
    assert [r.info.run_id for r in result] == runs[4:8]
    assert result.token is not None
    result = store.search_runs([exp], None, ViewType.ALL, max_results=4, page_token=result.token)
    assert [r.info.run_id for r in result] == runs[8:]
    assert result.token is None


def test_search_runs_pagination_last_page_exact(store: SqlAlchemyStore):
    exp = _create_experiments(store, "test_search_runs_pagination_last_page_exact")
    # Create exactly 8 runs (2 pages of 4 runs each)
    runs = sorted([
        _run_factory(store, _get_run_configs(exp, start_time=10)).info.run_id for _ in range(8)
    ])

    # First page: should return 4 runs and a token
    result = store.search_runs([exp], None, ViewType.ALL, max_results=4)
    assert [r.info.run_id for r in result] == runs[0:4]
    assert result.token is not None

    # Second page: should return exactly 4 runs (last page) with NO token
    # This is the key test case - with optimistic pagination, this would incorrectly
    # return a token
    result = store.search_runs([exp], None, ViewType.ALL, max_results=4, page_token=result.token)
    assert [r.info.run_id for r in result] == runs[4:8]
    assert result.token is None


def test_search_runs_pagination_with_max_results_none(store: SqlAlchemyStore):
    exp = _create_experiments(store, "test_search_runs_pagination_with_max_results_none")
    # Create 5 runs
    runs = sorted([
        _run_factory(store, _get_run_configs(exp, start_time=10)).info.run_id for _ in range(5)
    ])

    # Call search_runs with max_results=None - should return all runs with no token
    result = store.search_runs([exp], None, ViewType.ALL, max_results=None)
    assert len(result) == 5
    assert [r.info.run_id for r in result] == runs
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
    with store.ManagedSessionMaker(read_only=False) as session:
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


def test_log_batch_duplicate_metrics_across_key_batches(store: SqlAlchemyStore):
    """Test that duplicate metric detection works correctly when metric keys span multiple
    batches (batches of 100 keys). Previously, _insert_metrics was called inside the
    per-batch loop, causing metrics from unqueried batches to be inserted prematurely,
    which could raise an unhandled IntegrityError.
    See https://github.com/mlflow/mlflow/issues/19144
    """
    run = _run_factory(store)
    # Create >100 unique metric keys so they span multiple key batches
    num_keys = 150
    metrics = [
        Metric(key=f"metric-{i}", value=float(i), timestamp=1, step=0) for i in range(num_keys)
    ]
    # Log the metrics once
    store.log_batch(run.info.run_id, params=[], metrics=metrics, tags=[])
    _verify_logged(store, run.info.run_id, params=[], metrics=metrics, tags=[])
    # Log the same metrics again (all duplicates) — this should not raise
    store.log_batch(run.info.run_id, params=[], metrics=metrics, tags=[])
    _verify_logged(store, run.info.run_id, params=[], metrics=metrics, tags=[])


def test_log_batch_duplicate_metrics_mixed_with_new_across_key_batches(store: SqlAlchemyStore):
    # Test logging a mix of duplicate and new metrics when keys span multiple batches.
    run = _run_factory(store)
    num_keys = 150
    # Log initial metrics
    initial_metrics = [
        Metric(key=f"metric-{i}", value=float(i), timestamp=1, step=0) for i in range(num_keys)
    ]
    store.log_batch(run.info.run_id, params=[], metrics=initial_metrics, tags=[])
    # Log a mix: some duplicates from the initial batch + some new metrics
    duplicate_metrics = [
        Metric(key=f"metric-{i}", value=float(i), timestamp=1, step=0) for i in range(num_keys)
    ]
    new_metrics = [
        Metric(key=f"metric-{i}", value=float(i + num_keys), timestamp=2, step=1)
        for i in range(num_keys)
    ]
    mixed_metrics = duplicate_metrics + new_metrics
    store.log_batch(run.info.run_id, params=[], metrics=mixed_metrics, tags=[])
    _verify_logged(
        store, run.info.run_id, params=[], metrics=initial_metrics + new_metrics, tags=[]
    )


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


def _generate_large_data(store, nb_runs=1000):
    experiment_name = f"test_experiment_{uuid.uuid4().hex}"
    experiment_id = store.create_experiment(experiment_name)

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
        latest_metrics_list.append({
            "key": "mkey_0",
            "value": current_run,
            "timestamp": 100 * 2,
            "step": 100 * 3,
            "is_nan": False,
            "run_uuid": run_id,
        })
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


def test_get_metric_history_bulk_interval(store: SqlAlchemyStore):
    run = _run_factory(store)
    run_id = run.info.run_id

    metric_key = "test_metric"
    # Log 100 metric values at steps 0..99
    for i in range(100):
        store.log_metric(
            run_id,
            models.SqlMetric(
                key=metric_key, value=float(i), timestamp=1000 + i, step=i
            ).to_mlflow_entity(),
        )

    # Request downsampled to 10 results
    result = store.get_metric_history_bulk_interval(
        run_ids=[run_id],
        metric_key=metric_key,
        max_results=10,
        start_step=None,
        end_step=None,
    )

    # Should return roughly 10 sampled entries (plus min/max boundaries)
    assert len(result) <= 12
    assert len(result) >= 10

    # All results should have the correct run_id and metric key
    for m in result:
        assert m.run_id == run_id
        assert m.key == metric_key

    # Min and max steps should be preserved
    returned_steps = {m.step for m in result}
    assert 0 in returned_steps
    assert 99 in returned_steps


def test_get_metric_history_bulk_interval_no_metrics(store: SqlAlchemyStore):
    run = _run_factory(store)
    result = store.get_metric_history_bulk_interval(
        run_ids=[run.info.run_id],
        metric_key="nonexistent",
        max_results=10,
        start_step=None,
        end_step=None,
    )
    assert result == []


def test_get_metric_history_bulk_interval_multiple_runs(store: SqlAlchemyStore):
    exp_id = _create_experiments(store, "test_bulk_interval_multi")
    run1 = _run_factory(store, config=_get_run_configs(experiment_id=exp_id))
    run2 = _run_factory(store, config=_get_run_configs(experiment_id=exp_id))

    metric_key = "shared_metric"
    for i in range(50):
        store.log_metric(
            run1.info.run_id,
            models.SqlMetric(
                key=metric_key, value=float(i), timestamp=1000 + i, step=i
            ).to_mlflow_entity(),
        )
        store.log_metric(
            run2.info.run_id,
            models.SqlMetric(
                key=metric_key, value=float(i * 2), timestamp=2000 + i, step=i
            ).to_mlflow_entity(),
        )

    result = store.get_metric_history_bulk_interval(
        run_ids=[run1.info.run_id, run2.info.run_id],
        metric_key=metric_key,
        max_results=10,
        start_step=None,
        end_step=None,
    )

    # Should have results from both runs
    run_ids_in_result = {m.run_id for m in result}
    assert run1.info.run_id in run_ids_in_result
    assert run2.info.run_id in run_ids_in_result


def test_get_metric_history_bulk_interval_with_step_range(store: SqlAlchemyStore):
    run = _run_factory(store)
    run_id = run.info.run_id

    metric_key = "test_metric"
    for i in range(100):
        store.log_metric(
            run_id,
            models.SqlMetric(
                key=metric_key, value=float(i), timestamp=1000 + i, step=i
            ).to_mlflow_entity(),
        )

    result = store.get_metric_history_bulk_interval(
        run_ids=[run_id],
        metric_key=metric_key,
        max_results=320,
        start_step=20,
        end_step=30,
    )

    returned_steps = {m.step for m in result}
    # All returned steps should be within the requested range
    assert all(20 <= s <= 30 for s in returned_steps)
    # Should contain all steps in range since max_results > range size
    assert returned_steps == set(range(20, 31))


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

    dataset_digest = uuid.uuid4().hex
    dataset1 = entities.Dataset(
        name="name1",
        digest=dataset_digest,
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


def _assert_create_run_appends_to_artifact_uri_path_correctly(
    artifact_root_uri, expected_artifact_uri_format
):
    # Note: Previously this test patched `is_local_uri` to prevent directory creation,
    # but SqlAlchemyStore no longer creates the artifact root directory during initialization.
    # The directory is now created lazily when the first artifact is logged.
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

        # Dispose the engine to close all connections and allow the temp directory to be removed
        # on Windows, where open file handles prevent file deletion.
        store._dispose_engine()

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
        locations=[exp_id], filter_string=f"run_id = '{run.info.run_id}'"
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
