import json
import math
import os
import pathlib
import re
import shutil
import tempfile
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Union
from unittest import mock

import pytest
import sqlalchemy
from packaging.version import Version

import mlflow
import mlflow.db
import mlflow.store.db.base_sql_model
from mlflow import entities
from mlflow.entities import (
    Experiment,
    ExperimentTag,
    Metric,
    Param,
    RunStatus,
    RunTag,
    SourceType,
    ViewType,
    _DatasetSummary,
)
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.exceptions import MlflowException
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
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.dbmodels import models
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase
from mlflow.store.tracking.dbmodels.models import (
    SqlDataset,
    SqlExperiment,
    SqlExperimentTag,
    SqlInput,
    SqlInputTag,
    SqlLatestMetric,
    SqlMetric,
    SqlParam,
    SqlRun,
    SqlTag,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore, _get_orderby_clauses
from mlflow.utils import mlflow_tags
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT, MLFLOW_RUN_NAME
from mlflow.utils.name_utils import _GENERATOR_PREDICATES
from mlflow.utils.os import is_windows
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import extract_db_type_from_uri

from tests.integration.utils import invoke_cli_runner
from tests.store.tracking import AbstractStoreTest
from tests.store.tracking.test_file_store import assert_dataset_inputs_equal

DB_URI = "sqlite:///"
ARTIFACT_URI = "artifact_folder"

pytestmark = pytest.mark.notrackingurimock


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
def store(tmp_path: Path) -> SqlAlchemyStore:
    store = _get_store(tmp_path)
    yield store
    _cleanup_database(store)


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
            SqlParam,
            SqlMetric,
            SqlLatestMetric,
            SqlTag,
            SqlInputTag,
            SqlInput,
            SqlDataset,
            SqlRun,
            SqlExperimentTag,
            SqlExperiment,
        ):
            session.query(model).delete()

        # Reset experiment_id to start at 1
        if reset_experiment_id := _get_query_to_reset_experiment_id(store):
            session.execute(sqlalchemy.sql.text(reset_experiment_id))


def _create_experiments(store: SqlAlchemyStore, names) -> Union[str, List]:
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
        "start_time": start_time or get_current_time_millis(),
        "tags": tags,
        "run_name": "name",
    }


def _run_factory(store: SqlAlchemyStore, config=None):
    if not config:
        config = _get_run_configs()
    if "experiment_id" not in config:
        config["experiment_id"] = _create_experiments(store, "test exp")

    return store.create_run(**config)


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
    del store
    store = _get_store(tmp_path)

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


def test_raise_duplicate_experiments(store: SqlAlchemyStore):
    with pytest.raises(Exception, match=r"Experiment\(name=.+\) already exists"):
        _create_experiments(store, ["test", "test"])


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
    assert [e.experiment_id for e in experiments] == [exp_id2, store.DEFAULT_EXPERIMENT_ID]

    experiments = store.search_experiments(filter_string=f"creation_time >= {time_before_create1}")
    assert [e.experiment_id for e in experiments] == [exp_id2, exp_id1]

    experiments = store.search_experiments(filter_string=f"creation_time < {time_before_create2}")
    assert [e.experiment_id for e in experiments] == [exp_id1, store.DEFAULT_EXPERIMENT_ID]

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
    with pytest.raises(MlflowException, match=r"It must be a positive integer, but got None"):
        store.search_experiments(max_results=None)
    with pytest.raises(MlflowException, match=r"It must be a positive integer, but got 0"):
        store.search_experiments(max_results=0)
    with pytest.raises(MlflowException, match=r"It must be at most \d+, but got 1000000"):
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


# This unit test class is under refactoring. Please use pytest for new unit tests: #10042
class TestSqlAlchemyStore(unittest.TestCase, AbstractStoreTest):
    def _get_store(self, db_uri=""):
        return SqlAlchemyStore(db_uri, ARTIFACT_URI)

    def create_test_run(self):
        return self._run_factory()

    def _setup_db_uri(self):
        if uri := MLFLOW_TRACKING_URI.get():
            self.temp_dbfile = None
            self.db_url = uri
        else:
            fd, self.temp_dbfile = tempfile.mkstemp()
            # Close handle immediately so that we can remove the file later on in Windows
            os.close(fd)
            self.db_url = f"{DB_URI}{self.temp_dbfile}"

    def setUp(self):
        self._setup_db_uri()
        self.store = self._get_store(self.db_url)

    def get_store(self):
        return self.store

    def _get_query_to_reset_experiment_id(self):
        dialect = self.store._get_dialect()
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

    def tearDown(self):
        if self.temp_dbfile:
            os.remove(self.temp_dbfile)
        else:
            with self.store.ManagedSessionMaker() as session:
                # Delete all rows in all tables
                for model in (
                    SqlParam,
                    SqlMetric,
                    SqlLatestMetric,
                    SqlTag,
                    SqlInputTag,
                    SqlInput,
                    SqlDataset,
                    SqlRun,
                    SqlExperimentTag,
                    SqlExperiment,
                ):
                    session.query(model).delete()

                # Reset experiment_id to start at 1
                reset_experiment_id = self._get_query_to_reset_experiment_id()
                if reset_experiment_id:
                    session.execute(sqlalchemy.sql.text(reset_experiment_id))
        shutil.rmtree(ARTIFACT_URI)

    def _experiment_factory(self, names):
        if isinstance(names, (list, tuple)):
            ids = []
            for name in names:
                # Sleep to ensure each experiment has a unique creation_time for
                # deterministic experiment search results
                time.sleep(0.001)
                ids.append(self.store.create_experiment(name=name))
            return ids

        time.sleep(0.001)
        return self.store.create_experiment(name=names)

    def test_run_needs_uuid(self):
        regex = {
            SQLITE: r"NOT NULL constraint failed",
            POSTGRES: r"null value in column .+ of relation .+ violates not-null constrain",
            MYSQL: r"(Field .+ doesn't have a default value|Instance .+ has a NULL identity key)",
            MSSQL: r"Cannot insert the value NULL into column .+, table .+",
        }[self.store._get_dialect()]
        # Depending on the implementation, a NULL identity key may result in different
        # exceptions, including IntegrityError (sqlite) and FlushError (MysQL).
        # Therefore, we check for the more generic 'SQLAlchemyError'
        with pytest.raises(MlflowException, match=regex) as exception_context:
            with self.store.ManagedSessionMaker() as session:
                session.add(models.SqlRun())
        assert exception_context.value.error_code == ErrorCode.Name(BAD_REQUEST)

    def test_run_data_model(self):
        with self.store.ManagedSessionMaker() as session:
            run_id = uuid.uuid4().hex
            run_data = models.SqlRun(run_uuid=run_id)
            m1 = models.SqlMetric(run_uuid=run_id, key="accuracy", value=0.89)
            m2 = models.SqlMetric(run_uuid=run_id, key="recal", value=0.89)
            p1 = models.SqlParam(run_uuid=run_id, key="loss", value="test param")
            p2 = models.SqlParam(run_uuid=run_id, key="blue", value="test param")

            session.add_all([m1, m2, p1, p2])
            session.add(run_data)
            session.commit()

            run_datums = session.query(models.SqlRun).all()
            actual = run_datums[0]
            assert len(run_datums) == 1
            assert len(actual.params) == 2
            assert len(actual.metrics) == 2

    def test_run_info(self):
        experiment_id = self._experiment_factory("test exp")
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
            if k in ["source_name", "source_type", "source_version", "name", "entry_point_name"]:
                continue

            v2 = getattr(run.info, k)
            if k == "source_type":
                assert v == SourceType.to_string(v2)
            else:
                assert v == v2

    def _get_run_configs(self, experiment_id=None, tags=None, start_time=None):
        return {
            "experiment_id": experiment_id,
            "user_id": "Anderson",
            "start_time": start_time if start_time is not None else get_current_time_millis(),
            "tags": tags,
            "run_name": "name",
        }

    def _run_factory(self, config=None):
        if not config:
            config = self._get_run_configs()

        experiment_id = config.get("experiment_id", None)
        if not experiment_id:
            experiment_id = self._experiment_factory("test exp")
            config["experiment_id"] = experiment_id

        return self.store.create_run(**config)

    def test_create_run_with_tags(self):
        experiment_id = self._experiment_factory("test_create_run")
        tags = [RunTag("3", "4"), RunTag("1", "2")]
        expected = self._get_run_configs(experiment_id=experiment_id, tags=tags)

        actual = self.store.create_run(**expected)

        assert actual.info.experiment_id == experiment_id
        assert actual.info.user_id == expected["user_id"]
        assert actual.info.run_name == expected["run_name"]
        assert actual.info.start_time == expected["start_time"]

        assert len(actual.data.tags) == len(tags)
        expected_tags = {tag.key: tag.value for tag in tags}
        assert actual.data.tags == expected_tags

    def test_create_run_sets_name(self):
        experiment_id = self._experiment_factory("test_create_run_run_name")
        configs = self._get_run_configs(experiment_id=experiment_id)
        run_id = self.store.create_run(**configs).info.run_id
        run = self.store.get_run(run_id)
        assert run.info.run_name == configs["run_name"]
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == configs["run_name"]
        run_id = self.store.create_run(
            experiment_id=experiment_id,
            user_id="user",
            start_time=0,
            run_name=None,
            tags=[RunTag(mlflow_tags.MLFLOW_RUN_NAME, "test")],
        ).info.run_id
        run = self.store.get_run(run_id)
        assert run.info.run_name == "test"

        with pytest.raises(
            MlflowException,
            match=re.escape(
                "Both 'run_name' argument and 'mlflow.runName' tag are specified, but with "
                "different values (run_name='test', mlflow.runName='test_2').",
            ),
        ):
            self.store.create_run(
                experiment_id=experiment_id,
                user_id="user",
                start_time=0,
                run_name="test",
                tags=[RunTag(mlflow_tags.MLFLOW_RUN_NAME, "test_2")],
            )

    def test_get_run_with_name(self):
        experiment_id = self._experiment_factory("test_get_run")
        configs = self._get_run_configs(experiment_id=experiment_id)

        run_id = self.store.create_run(**configs).info.run_id

        run = self.store.get_run(run_id)

        assert run.info.experiment_id == experiment_id
        assert run.info.run_name == configs["run_name"]

        no_run_configs = {
            "experiment_id": experiment_id,
            "user_id": "Anderson",
            "start_time": get_current_time_millis(),
            "tags": [],
            "run_name": None,
        }
        run_id = self.store.create_run(**no_run_configs).info.run_id
        run = self.store.get_run(run_id)
        assert run.info.run_name.split("-")[0] in _GENERATOR_PREDICATES

        name_empty_str_run = self.store.create_run(**{**configs, **{"run_name": ""}})
        run_name = name_empty_str_run.info.run_name
        assert run_name.split("-")[0] in _GENERATOR_PREDICATES

    def test_to_mlflow_entity_and_proto(self):
        # Create a run and log metrics, params, tags to the run
        created_run = self._run_factory()
        run_id = created_run.info.run_id
        self.store.log_metric(
            run_id=run_id, metric=entities.Metric(key="my-metric", value=3.4, timestamp=0, step=0)
        )
        self.store.log_param(run_id=run_id, param=Param(key="my-param", value="param-val"))
        self.store.set_tag(run_id=run_id, tag=RunTag(key="my-tag", value="tag-val"))

        # Verify that we can fetch the run & convert it to proto - Python protobuf bindings
        # will perform type-checking to ensure all values have the right types
        run = self.store.get_run(run_id)
        run.to_proto()

        # Verify attributes of the Python run entity
        assert isinstance(run.info, entities.RunInfo)
        assert isinstance(run.data, entities.RunData)

        assert run.data.metrics == {"my-metric": 3.4}
        assert run.data.params == {"my-param": "param-val"}
        assert run.data.tags["my-tag"] == "tag-val"

        # Get the parent experiment of the run, verify it can be converted to protobuf
        exp = self.store.get_experiment(run.info.experiment_id)
        exp.to_proto()

    def test_delete_run(self):
        run = self._run_factory()

        self.store.delete_run(run.info.run_id)

        with self.store.ManagedSessionMaker() as session:
            actual = session.query(models.SqlRun).filter_by(run_uuid=run.info.run_id).first()
            assert actual.lifecycle_stage == entities.LifecycleStage.DELETED
            assert (
                actual.deleted_time is not None
            )  # deleted time should be updated and thus not None anymore

            deleted_run = self.store.get_run(run.info.run_id)
            assert actual.run_uuid == deleted_run.info.run_id

    def test_hard_delete_run(self):
        run = self._run_factory()
        metric = entities.Metric("blahmetric", 100.0, get_current_time_millis(), 0)
        self.store.log_metric(run.info.run_id, metric)
        param = entities.Param("blahparam", "100.0")
        self.store.log_param(run.info.run_id, param)
        tag = entities.RunTag("test tag", "a boogie")
        self.store.set_tag(run.info.run_id, tag)

        self.store._hard_delete_run(run.info.run_id)

        with self.store.ManagedSessionMaker() as session:
            actual_run = session.query(models.SqlRun).filter_by(run_uuid=run.info.run_id).first()
            assert actual_run is None
            actual_metric = (
                session.query(models.SqlMetric).filter_by(run_uuid=run.info.run_id).first()
            )
            assert actual_metric is None
            actual_param = (
                session.query(models.SqlParam).filter_by(run_uuid=run.info.run_id).first()
            )
            assert actual_param is None
            actual_tag = session.query(models.SqlTag).filter_by(run_uuid=run.info.run_id).first()
            assert actual_tag is None

    def test_get_deleted_runs(self):
        run = self._run_factory()
        deleted_run_ids = self.store._get_deleted_runs()
        assert deleted_run_ids == []

        self.store.delete_run(run.info.run_uuid)
        deleted_run_ids = self.store._get_deleted_runs()
        assert deleted_run_ids == [run.info.run_uuid]

    def test_log_metric(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = 100.0
        metric = entities.Metric(tkey, tval, get_current_time_millis(), 0)
        metric2 = entities.Metric(tkey, tval, get_current_time_millis() + 2, 0)
        nan_metric = entities.Metric("NaN", float("nan"), 0, 0)
        pos_inf_metric = entities.Metric("PosInf", float("inf"), 0, 0)
        neg_inf_metric = entities.Metric("NegInf", -float("inf"), 0, 0)
        self.store.log_metric(run.info.run_id, metric)
        self.store.log_metric(run.info.run_id, metric2)
        self.store.log_metric(run.info.run_id, nan_metric)
        self.store.log_metric(run.info.run_id, pos_inf_metric)
        self.store.log_metric(run.info.run_id, neg_inf_metric)

        run = self.store.get_run(run.info.run_id)
        assert tkey in run.data.metrics
        assert run.data.metrics[tkey] == tval

        # SQL store _get_run method returns full history of recorded metrics.
        # Should return duplicates as well
        # MLflow RunData contains only the last reported values for metrics.
        with self.store.ManagedSessionMaker() as session:
            sql_run_metrics = self.store._get_run(session, run.info.run_id).metrics
            assert len(sql_run_metrics) == 5
            assert len(run.data.metrics) == 4
            assert math.isnan(run.data.metrics["NaN"])
            assert run.data.metrics["PosInf"] == 1.7976931348623157e308
            assert run.data.metrics["NegInf"] == -1.7976931348623157e308

    def test_log_metric_concurrent_logging_succeeds(self):
        """
        Verifies that concurrent logging succeeds without deadlock, which has been an issue
        in previous MLflow releases
        """
        experiment_id = self._experiment_factory("concurrency_exp")
        run_config = self._get_run_configs(experiment_id=experiment_id)
        run1 = self._run_factory(run_config)
        run2 = self._run_factory(run_config)

        def log_metrics(run):
            for metric_val in range(100):
                self.store.log_metric(
                    run.info.run_id, Metric("metric_key", metric_val, get_current_time_millis(), 0)
                )
            for batch_idx in range(5):
                self.store.log_batch(
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
                self.store.log_metric(
                    run.info.run_id, Metric("metric_key", metric_val, get_current_time_millis(), 0)
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
            assert len(self.store.get_metric_history(run.info.run_id, "metric_key")) >= 100
            for batch_idx in range(5):
                assert (
                    len(self.store.get_metric_history(run.info.run_id, f"metric_batch_{batch_idx}"))
                    >= 100
                )

    def test_log_metric_allows_multiple_values_at_same_ts_and_run_data_uses_max_ts_value(self):
        run = self._run_factory()
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
            self.store.log_metric(run_id, Metric(metric_name, value, timestamp, step))

        metric_history = self.store.get_metric_history(run_id, metric_name)
        logged_tuples = [(m.step, m.timestamp, m.value) for m in metric_history]
        assert set(logged_tuples) == set(tuples_to_log)

        run_data = self.store.get_run(run_id).data
        run_metrics = run_data.metrics
        assert len(run_metrics) == 1
        assert run_metrics[metric_name] == 20
        metric_obj = run_data._metric_objs[0]
        assert metric_obj.key == metric_name
        assert metric_obj.step == 3
        assert metric_obj.timestamp == 50
        assert metric_obj.value == 20

    def test_get_metric_history_paginated_request_raises(self):
        with pytest.raises(
            MlflowException,
            match="The SQLAlchemyStore backend does not support pagination for the "
            "`get_metric_history` API.",
        ):
            self.store.get_metric_history(
                "fake_run", "fake_metric", max_results=50, page_token="42"
            )

    def test_log_null_metric(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = None
        metric = entities.Metric(tkey, tval, get_current_time_millis(), 0)

        with pytest.raises(
            MlflowException, match=r"Got invalid value None for metric"
        ) as exception_context:
            self.store.log_metric(run.info.run_id, metric)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_log_param(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = "100.0"
        param = entities.Param(tkey, tval)
        param2 = entities.Param("new param", "new key")
        self.store.log_param(run.info.run_id, param)
        self.store.log_param(run.info.run_id, param2)
        self.store.log_param(run.info.run_id, param2)

        run = self.store.get_run(run.info.run_id)
        assert len(run.data.params) == 2
        assert tkey in run.data.params
        assert run.data.params[tkey] == tval

    def test_log_param_uniqueness(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = "100.0"
        param = entities.Param(tkey, tval)
        param2 = entities.Param(tkey, "newval")
        self.store.log_param(run.info.run_id, param)

        with pytest.raises(MlflowException, match=r"Changing param values is not allowed"):
            self.store.log_param(run.info.run_id, param2)

    def test_log_empty_str(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = ""
        param = entities.Param(tkey, tval)
        param2 = entities.Param("new param", "new key")
        self.store.log_param(run.info.run_id, param)
        self.store.log_param(run.info.run_id, param2)

        run = self.store.get_run(run.info.run_id)
        assert len(run.data.params) == 2
        assert tkey in run.data.params
        assert run.data.params[tkey] == tval

    def test_log_null_param(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = None
        param = entities.Param(tkey, tval)

        dialect = self.store._get_dialect()
        regex = {
            SQLITE: r"NOT NULL constraint failed",
            POSTGRES: r"null value in column .+ of relation .+ violates not-null constrain",
            MYSQL: r"Column .+ cannot be null",
            MSSQL: r"Cannot insert the value NULL into column .+, table .+",
        }[dialect]
        with pytest.raises(MlflowException, match=regex) as exception_context:
            self.store.log_param(run.info.run_id, param)
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
        reason="large string parameters are sent as TEXT/NTEXT; "
        "see tests/db/compose.yml for details",
    )
    def test_log_param_max_length_value(self):
        run = self._run_factory()
        tkey = "blahmetric"
        tval = "x" * 6000
        param = entities.Param(tkey, tval)
        self.store.log_param(run.info.run_id, param)
        run = self.store.get_run(run.info.run_id)
        assert run.data.params[tkey] == str(tval)
        with pytest.raises(MlflowException, match="exceeded length"):
            self.store.log_param(run.info.run_id, entities.Param(tkey, "x" * 6001))

    def test_set_experiment_tag(self):
        exp_id = self._experiment_factory("setExperimentTagExp")
        tag = entities.ExperimentTag("tag0", "value0")
        new_tag = entities.RunTag("tag0", "value00000")
        self.store.set_experiment_tag(exp_id, tag)
        experiment = self.store.get_experiment(exp_id)
        assert experiment.tags["tag0"] == "value0"
        # test that updating a tag works
        self.store.set_experiment_tag(exp_id, new_tag)
        experiment = self.store.get_experiment(exp_id)
        assert experiment.tags["tag0"] == "value00000"
        # test that setting a tag on 1 experiment does not impact another experiment.
        exp_id_2 = self._experiment_factory("setExperimentTagExp2")
        experiment2 = self.store.get_experiment(exp_id_2)
        assert len(experiment2.tags) == 0
        # setting a tag on different experiments maintains different values across experiments
        different_tag = entities.RunTag("tag0", "differentValue")
        self.store.set_experiment_tag(exp_id_2, different_tag)
        experiment = self.store.get_experiment(exp_id)
        assert experiment.tags["tag0"] == "value00000"
        experiment2 = self.store.get_experiment(exp_id_2)
        assert experiment2.tags["tag0"] == "differentValue"
        # test can set multi-line tags
        multi_line_Tag = entities.ExperimentTag("multiline tag", "value2\nvalue2\nvalue2")
        self.store.set_experiment_tag(exp_id, multi_line_Tag)
        experiment = self.store.get_experiment(exp_id)
        assert experiment.tags["multiline tag"] == "value2\nvalue2\nvalue2"
        # test cannot set tags that are too long
        long_tag = entities.ExperimentTag("longTagKey", "a" * 5001)
        with pytest.raises(MlflowException, match="exceeded length limit of 5000"):
            self.store.set_experiment_tag(exp_id, long_tag)
        # test can set tags that are somewhat long
        long_tag = entities.ExperimentTag("longTagKey", "a" * 4999)
        self.store.set_experiment_tag(exp_id, long_tag)
        # test cannot set tags on deleted experiments
        self.store.delete_experiment(exp_id)
        with pytest.raises(MlflowException, match="must be in the 'active' state"):
            self.store.set_experiment_tag(exp_id, entities.ExperimentTag("should", "notset"))

    def test_set_tag(self):
        run = self._run_factory()

        tkey = "test tag"
        tval = "a boogie"
        new_val = "new val"
        tag = entities.RunTag(tkey, tval)
        new_tag = entities.RunTag(tkey, new_val)
        self.store.set_tag(run.info.run_id, tag)
        # Overwriting tags is allowed
        self.store.set_tag(run.info.run_id, new_tag)
        # test setting tags that are too long fails.
        with pytest.raises(MlflowException, match="exceeded length limit of 5000"):
            self.store.set_tag(run.info.run_id, entities.RunTag("longTagKey", "a" * 5001))
        # test can set tags that are somewhat long
        self.store.set_tag(run.info.run_id, entities.RunTag("longTagKey", "a" * 4999))
        run = self.store.get_run(run.info.run_id)
        assert tkey in run.data.tags
        assert run.data.tags[tkey] == new_val

    def test_delete_tag(self):
        run = self._run_factory()
        k0, v0 = "tag0", "val0"
        k1, v1 = "tag1", "val1"
        tag0 = entities.RunTag(k0, v0)
        tag1 = entities.RunTag(k1, v1)
        self.store.set_tag(run.info.run_id, tag0)
        self.store.set_tag(run.info.run_id, tag1)
        # delete a tag and check whether it is correctly deleted.
        self.store.delete_tag(run.info.run_id, k0)
        run = self.store.get_run(run.info.run_id)
        assert k0 not in run.data.tags
        assert k1 in run.data.tags
        assert run.data.tags[k1] == v1

        # test that deleting a tag works correctly with multiple runs having the same tag.
        run2 = self._run_factory(config=self._get_run_configs(run.info.experiment_id))
        self.store.set_tag(run.info.run_id, tag0)
        self.store.set_tag(run2.info.run_id, tag0)
        self.store.delete_tag(run.info.run_id, k0)
        run = self.store.get_run(run.info.run_id)
        run2 = self.store.get_run(run2.info.run_id)
        assert k0 not in run.data.tags
        assert k0 in run2.data.tags
        # test that you cannot delete tags that don't exist.
        with pytest.raises(MlflowException, match="No tag with name"):
            self.store.delete_tag(run.info.run_id, "fakeTag")
        # test that you cannot delete tags for nonexistent runs
        with pytest.raises(MlflowException, match="Run with id=randomRunId not found"):
            self.store.delete_tag("randomRunId", k0)
        # test that you cannot delete tags for deleted runs.
        self.store.delete_run(run.info.run_id)
        with pytest.raises(MlflowException, match="must be in the 'active' state"):
            self.store.delete_tag(run.info.run_id, k1)

    def test_get_metric_history(self):
        run = self._run_factory()

        key = "test"
        expected = [
            models.SqlMetric(key=key, value=0.6, timestamp=1, step=0).to_mlflow_entity(),
            models.SqlMetric(key=key, value=0.7, timestamp=2, step=0).to_mlflow_entity(),
        ]

        for metric in expected:
            self.store.log_metric(run.info.run_id, metric)

        actual = self.store.get_metric_history(run.info.run_id, key)

        assert sorted(
            [(m.key, m.value, m.timestamp) for m in expected],
        ) == sorted(
            [(m.key, m.value, m.timestamp) for m in actual],
        )

    def test_rename_experiment(self):
        new_name = "new name"
        experiment_id = self._experiment_factory("test name")
        experiment = self.store.get_experiment(experiment_id)
        time.sleep(0.01)
        self.store.rename_experiment(experiment_id, new_name)

        renamed_experiment = self.store.get_experiment(experiment_id)

        assert renamed_experiment.name == new_name
        assert renamed_experiment.last_update_time > experiment.last_update_time

    def test_update_run_info(self):
        experiment_id = self._experiment_factory("test_update_run_info")
        for new_status_string in models.RunStatusTypes:
            run = self._run_factory(config=self._get_run_configs(experiment_id=experiment_id))
            endtime = get_current_time_millis()
            actual = self.store.update_run_info(
                run.info.run_id, RunStatus.from_string(new_status_string), endtime, None
            )
            assert actual.status == new_status_string
            assert actual.end_time == endtime

        # test updating run name without changing other attributes.
        origin_run_info = self.store.get_run(run.info.run_id).info
        updated_info = self.store.update_run_info(run.info.run_id, None, None, "name_abc2")
        assert updated_info.run_name == "name_abc2"
        assert updated_info.status == origin_run_info.status
        assert updated_info.end_time == origin_run_info.end_time

    def test_update_run_name(self):
        experiment_id = self._experiment_factory("test_update_run_name")
        configs = self._get_run_configs(experiment_id=experiment_id)

        run_id = self.store.create_run(**configs).info.run_id
        run = self.store.get_run(run_id)
        assert run.info.run_name == configs["run_name"]

        self.store.update_run_info(run_id, RunStatus.FINISHED, 1000, "new name")
        run = self.store.get_run(run_id)
        assert run.info.run_name == "new name"
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "new name"

        self.store.update_run_info(run_id, RunStatus.FINISHED, 1000, None)
        run = self.store.get_run(run_id)
        assert run.info.run_name == "new name"
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "new name"

        self.store.update_run_info(run_id, RunStatus.FINISHED, 1000, "")
        run = self.store.get_run(run_id)
        assert run.info.run_name == "new name"
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "new name"

        self.store.delete_tag(run_id, mlflow_tags.MLFLOW_RUN_NAME)
        run = self.store.get_run(run_id)
        assert run.info.run_name == "new name"
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) is None

        self.store.update_run_info(run_id, RunStatus.FINISHED, 1000, "newer name")
        run = self.store.get_run(run_id)
        assert run.info.run_name == "newer name"
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "newer name"

        self.store.set_tag(run_id, entities.RunTag(mlflow_tags.MLFLOW_RUN_NAME, "newest name"))
        run = self.store.get_run(run_id)
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "newest name"
        assert run.info.run_name == "newest name"

        self.store.log_batch(
            run_id,
            metrics=[],
            params=[],
            tags=[entities.RunTag(mlflow_tags.MLFLOW_RUN_NAME, "batch name")],
        )
        run = self.store.get_run(run_id)
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "batch name"
        assert run.info.run_name == "batch name"

    def test_restore_experiment(self):
        experiment_id = self._experiment_factory("helloexp")
        exp = self.store.get_experiment(experiment_id)
        assert exp.lifecycle_stage == entities.LifecycleStage.ACTIVE

        experiment_id = exp.experiment_id
        self.store.delete_experiment(experiment_id)

        deleted = self.store.get_experiment(experiment_id)
        assert deleted.experiment_id == experiment_id
        assert deleted.lifecycle_stage == entities.LifecycleStage.DELETED
        time.sleep(0.01)
        self.store.restore_experiment(exp.experiment_id)
        restored = self.store.get_experiment(exp.experiment_id)
        assert restored.experiment_id == experiment_id
        assert restored.lifecycle_stage == entities.LifecycleStage.ACTIVE
        assert restored.last_update_time > deleted.last_update_time

    def test_delete_restore_run(self):
        run = self._run_factory()
        assert run.info.lifecycle_stage == entities.LifecycleStage.ACTIVE

        # Verify that active runs can be restored (run restoration is idempotent)
        self.store.restore_run(run.info.run_id)

        # Verify that run deletion is idempotent
        self.store.delete_run(run.info.run_id)
        self.store.delete_run(run.info.run_id)

        deleted = self.store.get_run(run.info.run_id)
        assert deleted.info.run_id == run.info.run_id
        assert deleted.info.lifecycle_stage == entities.LifecycleStage.DELETED
        with self.store.ManagedSessionMaker() as session:
            assert self.store._get_run(session, deleted.info.run_id).deleted_time is not None
        # Verify that restoration of a deleted run is idempotent
        self.store.restore_run(run.info.run_id)
        self.store.restore_run(run.info.run_id)
        restored = self.store.get_run(run.info.run_id)
        assert restored.info.run_id == run.info.run_id
        assert restored.info.lifecycle_stage == entities.LifecycleStage.ACTIVE
        with self.store.ManagedSessionMaker() as session:
            assert self.store._get_run(session, restored.info.run_id).deleted_time is None

    def test_error_logging_to_deleted_run(self):
        exp = self._experiment_factory("error_logging")
        run_id = self._run_factory(self._get_run_configs(experiment_id=exp)).info.run_id

        self.store.delete_run(run_id)
        assert self.store.get_run(run_id).info.lifecycle_stage == entities.LifecycleStage.DELETED
        with pytest.raises(MlflowException, match=r"The run .+ must be in the 'active' state"):
            self.store.log_param(run_id, entities.Param("p1345", "v1"))

        with pytest.raises(MlflowException, match=r"The run .+ must be in the 'active' state"):
            self.store.log_metric(run_id, entities.Metric("m1345", 1.0, 123, 0))

        with pytest.raises(MlflowException, match=r"The run .+ must be in the 'active' state"):
            self.store.set_tag(run_id, entities.RunTag("t1345", "tv1"))

        # restore this run and try again
        self.store.restore_run(run_id)
        assert self.store.get_run(run_id).info.lifecycle_stage == entities.LifecycleStage.ACTIVE
        self.store.log_param(run_id, entities.Param("p1345", "v22"))
        self.store.log_metric(run_id, entities.Metric("m1345", 34.0, 85, 1))  # earlier timestamp
        self.store.set_tag(run_id, entities.RunTag("t1345", "tv44"))

        run = self.store.get_run(run_id)
        assert run.data.params == {"p1345": "v22"}
        assert run.data.metrics == {"m1345": 34.0}
        metric_history = self.store.get_metric_history(run_id, "m1345")
        assert len(metric_history) == 1
        metric_obj = metric_history[0]
        assert metric_obj.key == "m1345"
        assert metric_obj.value == 34.0
        assert metric_obj.timestamp == 85
        assert metric_obj.step == 1
        assert {("t1345", "tv44")} <= set(run.data.tags.items())

    # Tests for Search API
    def _search(
        self,
        experiment_id,
        filter_string=None,
        run_view_type=ViewType.ALL,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
    ):
        exps = [experiment_id] if isinstance(experiment_id, str) else experiment_id
        return [
            r.info.run_id
            for r in self.store.search_runs(exps, filter_string, run_view_type, max_results)
        ]

    def get_ordered_runs(self, order_clauses, experiment_id):
        return [
            r.data.tags[mlflow_tags.MLFLOW_RUN_NAME]
            for r in self.store.search_runs(
                experiment_ids=[experiment_id],
                filter_string="",
                run_view_type=ViewType.ALL,
                order_by=order_clauses,
            )
        ]

    def test_order_by_metric_tag_param(self):
        experiment_id = self.store.create_experiment("order_by_metric")

        def create_and_log_run(names):
            name = str(names[0]) + "/" + names[1]
            run_id = self.store.create_run(
                experiment_id,
                user_id="MrDuck",
                start_time=123,
                tags=[entities.RunTag("metric", names[1])],
                run_name=name,
            ).info.run_id
            if names[0] is not None:
                self.store.log_metric(run_id, entities.Metric("x", float(names[0]), 1, 0))
                self.store.log_metric(run_id, entities.Metric("y", float(names[1]), 1, 0))
            self.store.log_param(run_id, entities.Param("metric", names[1]))
            return run_id

        # the expected order in ascending sort is :
        # inf > number > -inf > None > nan
        for names in zip(
            [None, "nan", "inf", "-inf", "-1000", "0", "0", "1000"],
            ["1", "2", "3", "4", "5", "6", "7", "8"],
        ):
            create_and_log_run(names)

        # asc/asc
        assert self.get_ordered_runs(["metrics.x asc", "metrics.y asc"], experiment_id) == [
            "-inf/4",
            "-1000/5",
            "0/6",
            "0/7",
            "1000/8",
            "inf/3",
            "nan/2",
            "None/1",
        ]

        assert self.get_ordered_runs(["metrics.x asc", "tag.metric asc"], experiment_id) == [
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
        assert self.get_ordered_runs(["metrics.x asc", "metrics.y desc"], experiment_id) == [
            "-inf/4",
            "-1000/5",
            "0/7",
            "0/6",
            "1000/8",
            "inf/3",
            "nan/2",
            "None/1",
        ]

        assert self.get_ordered_runs(["metrics.x asc", "tag.metric desc"], experiment_id) == [
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
        assert self.get_ordered_runs(["metrics.x desc", "metrics.y asc"], experiment_id) == [
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
        assert self.get_ordered_runs(["metrics.x desc", "param.metric desc"], experiment_id) == [
            "inf/3",
            "1000/8",
            "0/7",
            "0/6",
            "-1000/5",
            "-inf/4",
            "nan/2",
            "None/1",
        ]

    def test_order_by_attributes(self):
        experiment_id = self.store.create_experiment("order_by_attributes")

        def create_run(start_time, end):
            return self.store.create_run(
                experiment_id,
                user_id="MrDuck",
                start_time=start_time,
                tags=[],
                run_name=str(end),
            ).info.run_id

        start_time = 123
        for end in [234, None, 456, -123, 789, 123]:
            run_id = create_run(start_time, end)
            self.store.update_run_info(
                run_id, run_status=RunStatus.FINISHED, end_time=end, run_name=None
            )
            start_time += 1

        # asc
        assert self.get_ordered_runs(["attribute.end_time asc"], experiment_id) == [
            "-123",
            "123",
            "234",
            "456",
            "789",
            "None",
        ]

        # desc
        assert self.get_ordered_runs(["attribute.end_time desc"], experiment_id) == [
            "789",
            "456",
            "234",
            "123",
            "-123",
            "None",
        ]

        # Sort priority correctly handled
        assert self.get_ordered_runs(
            ["attribute.start_time asc", "attribute.end_time desc"], experiment_id
        ) == ["234", "None", "456", "-123", "789", "123"]

    def test_search_vanilla(self):
        exp = self._experiment_factory("search_vanilla")
        runs = [self._run_factory(self._get_run_configs(exp)).info.run_id for r in range(3)]

        assert sorted(
            runs,
        ) == sorted(self._search(exp, run_view_type=ViewType.ALL))
        assert sorted(
            runs,
        ) == sorted(self._search(exp, run_view_type=ViewType.ACTIVE_ONLY))
        assert self._search(exp, run_view_type=ViewType.DELETED_ONLY) == []

        first = runs[0]

        self.store.delete_run(first)
        assert sorted(
            runs,
        ) == sorted(self._search(exp, run_view_type=ViewType.ALL))
        assert sorted(
            runs[1:],
        ) == sorted(self._search(exp, run_view_type=ViewType.ACTIVE_ONLY))
        assert self._search(exp, run_view_type=ViewType.DELETED_ONLY) == [first]

        self.store.restore_run(first)
        assert sorted(
            runs,
        ) == sorted(self._search(exp, run_view_type=ViewType.ALL))
        assert sorted(
            runs,
        ) == sorted(self._search(exp, run_view_type=ViewType.ACTIVE_ONLY))
        assert self._search(exp, run_view_type=ViewType.DELETED_ONLY) == []

    def test_search_params(self):
        experiment_id = self._experiment_factory("search_params")
        r1 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        r2 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id

        self.store.log_param(r1, entities.Param("generic_param", "p_val"))
        self.store.log_param(r2, entities.Param("generic_param", "p_val"))

        self.store.log_param(r1, entities.Param("generic_2", "some value"))
        self.store.log_param(r2, entities.Param("generic_2", "another value"))

        self.store.log_param(r1, entities.Param("p_a", "abc"))
        self.store.log_param(r2, entities.Param("p_b", "ABC"))

        # test search returns both runs
        filter_string = "params.generic_param = 'p_val'"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        # test search returns appropriate run (same key different values per run)
        filter_string = "params.generic_2 = 'some value'"
        assert self._search(experiment_id, filter_string) == [r1]
        filter_string = "params.generic_2 = 'another value'"
        assert self._search(experiment_id, filter_string) == [r2]

        filter_string = "params.generic_param = 'wrong_val'"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "params.generic_param != 'p_val'"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "params.generic_param != 'wrong_val'"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))
        filter_string = "params.generic_2 != 'wrong_val'"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "params.p_a = 'abc'"
        assert self._search(experiment_id, filter_string) == [r1]

        filter_string = "params.p_a = 'ABC'"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "params.p_a != 'ABC'"
        assert self._search(experiment_id, filter_string) == [r1]

        filter_string = "params.p_b = 'ABC'"
        assert self._search(experiment_id, filter_string) == [r2]

        filter_string = "params.generic_2 LIKE '%other%'"
        assert self._search(experiment_id, filter_string) == [r2]

        filter_string = "params.generic_2 LIKE 'other%'"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "params.generic_2 LIKE '%other'"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "params.generic_2 LIKE 'other'"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "params.generic_2 LIKE '%Other%'"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "params.generic_2 ILIKE '%Other%'"
        assert self._search(experiment_id, filter_string) == [r2]

    def test_search_tags(self):
        experiment_id = self._experiment_factory("search_tags")
        r1 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        r2 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id

        self.store.set_tag(r1, entities.RunTag("generic_tag", "p_val"))
        self.store.set_tag(r2, entities.RunTag("generic_tag", "p_val"))

        self.store.set_tag(r1, entities.RunTag("generic_2", "some value"))
        self.store.set_tag(r2, entities.RunTag("generic_2", "another value"))

        self.store.set_tag(r1, entities.RunTag("p_a", "abc"))
        self.store.set_tag(r2, entities.RunTag("p_b", "ABC"))

        # test search returns both runs
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string="tags.generic_tag = 'p_val'"))
        assert self._search(experiment_id, filter_string="tags.generic_tag = 'P_VAL'") == []
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string="tags.generic_tag != 'P_VAL'"))
        # test search returns appropriate run (same key different values per run)
        assert self._search(experiment_id, filter_string="tags.generic_2 = 'some value'") == [r1]
        assert self._search(experiment_id, filter_string="tags.generic_2 = 'another value'") == [r2]
        assert self._search(experiment_id, filter_string="tags.generic_tag = 'wrong_val'") == []
        assert self._search(experiment_id, filter_string="tags.generic_tag != 'p_val'") == []
        assert sorted(
            [r1, r2],
        ) == sorted(
            self._search(experiment_id, filter_string="tags.generic_tag != 'wrong_val'"),
        )
        assert sorted(
            [r1, r2],
        ) == sorted(
            self._search(experiment_id, filter_string="tags.generic_2 != 'wrong_val'"),
        )
        assert self._search(experiment_id, filter_string="tags.p_a = 'abc'") == [r1]
        assert self._search(experiment_id, filter_string="tags.p_b = 'ABC'") == [r2]
        assert self._search(experiment_id, filter_string="tags.generic_2 LIKE '%other%'") == [r2]
        assert self._search(experiment_id, filter_string="tags.generic_2 LIKE '%Other%'") == []
        assert self._search(experiment_id, filter_string="tags.generic_2 LIKE 'other%'") == []
        assert self._search(experiment_id, filter_string="tags.generic_2 LIKE '%other'") == []
        assert self._search(experiment_id, filter_string="tags.generic_2 LIKE 'other'") == []
        assert self._search(experiment_id, filter_string="tags.generic_2 ILIKE '%Other%'") == [r2]
        assert self._search(
            experiment_id,
            filter_string="tags.generic_2 ILIKE '%Other%' and tags.generic_tag = 'p_val'",
        ) == [r2]
        assert self._search(
            experiment_id,
            filter_string="tags.generic_2 ILIKE '%Other%' and tags.generic_tag ILIKE 'p_val'",
        ) == [r2]

    def test_search_metrics(self):
        experiment_id = self._experiment_factory("search_metric")
        r1 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        r2 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id

        self.store.log_metric(r1, entities.Metric("common", 1.0, 1, 0))
        self.store.log_metric(r2, entities.Metric("common", 1.0, 1, 0))

        self.store.log_metric(r1, entities.Metric("measure_a", 1.0, 1, 0))
        self.store.log_metric(r2, entities.Metric("measure_a", 200.0, 2, 0))
        self.store.log_metric(r2, entities.Metric("measure_a", 400.0, 3, 0))

        self.store.log_metric(r1, entities.Metric("m_a", 2.0, 2, 0))
        self.store.log_metric(r2, entities.Metric("m_b", 3.0, 2, 0))
        self.store.log_metric(r2, entities.Metric("m_b", 4.0, 8, 0))  # this is last timestamp
        self.store.log_metric(r2, entities.Metric("m_b", 8.0, 3, 0))

        filter_string = "metrics.common = 1.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.common > 0.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.common >= 0.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.common < 4.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.common <= 4.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.common != 1.0"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "metrics.common >= 3.0"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "metrics.common <= 0.75"
        assert self._search(experiment_id, filter_string) == []

        # tests for same metric name across runs with different values and timestamps
        filter_string = "metrics.measure_a > 0.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a < 50.0"
        assert self._search(experiment_id, filter_string) == [r1]

        filter_string = "metrics.measure_a < 1000.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a != -12.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a > 50.0"
        assert self._search(experiment_id, filter_string) == [r2]

        filter_string = "metrics.measure_a = 1.0"
        assert self._search(experiment_id, filter_string) == [r1]

        filter_string = "metrics.measure_a = 400.0"
        assert self._search(experiment_id, filter_string) == [r2]

        # test search with unique metric keys
        filter_string = "metrics.m_a > 1.0"
        assert self._search(experiment_id, filter_string) == [r1]

        filter_string = "metrics.m_b > 1.0"
        assert self._search(experiment_id, filter_string) == [r2]

        # there is a recorded metric this threshold but not last timestamp
        filter_string = "metrics.m_b > 5.0"
        assert self._search(experiment_id, filter_string) == []

        # metrics matches last reported timestamp for 'm_b'
        filter_string = "metrics.m_b = 4.0"
        assert self._search(experiment_id, filter_string) == [r2]

    def test_search_attrs(self):
        e1 = self._experiment_factory("search_attributes_1")
        r1 = self._run_factory(self._get_run_configs(experiment_id=e1)).info.run_id

        e2 = self._experiment_factory("search_attrs_2")
        r2 = self._run_factory(self._get_run_configs(experiment_id=e2)).info.run_id

        filter_string = ""
        assert sorted(
            [r1, r2],
        ) == sorted(self._search([e1, e2], filter_string))

        filter_string = "attribute.status != 'blah'"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search([e1, e2], filter_string))

        filter_string = f"attribute.status = '{RunStatus.to_string(RunStatus.RUNNING)}'"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search([e1, e2], filter_string))

        # change status for one of the runs
        self.store.update_run_info(r2, RunStatus.FAILED, 300, None)

        filter_string = "attribute.status = 'RUNNING'"
        assert self._search([e1, e2], filter_string) == [r1]

        filter_string = "attribute.status = 'FAILED'"
        assert self._search([e1, e2], filter_string) == [r2]

        filter_string = "attribute.status != 'SCHEDULED'"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search([e1, e2], filter_string))

        filter_string = "attribute.status = 'SCHEDULED'"
        assert self._search([e1, e2], filter_string) == []

        filter_string = "attribute.status = 'KILLED'"
        assert self._search([e1, e2], filter_string) == []

        if is_windows():
            expected_artifact_uri = (
                pathlib.Path.cwd().joinpath(ARTIFACT_URI, e1, r1, "artifacts").as_uri()
            )
            filter_string = f"attr.artifact_uri = '{expected_artifact_uri}'"
        else:
            expected_artifact_uri = (
                pathlib.Path.cwd().joinpath(ARTIFACT_URI, e1, r1, "artifacts").as_posix()
            )
            filter_string = f"attr.artifact_uri = '{expected_artifact_uri}'"
        assert self._search([e1, e2], filter_string) == [r1]

        filter_string = f"attr.artifact_uri = '{ARTIFACT_URI}/{e1.upper()}/{r1.upper()}/artifacts'"
        assert self._search([e1, e2], filter_string) == []

        filter_string = f"attr.artifact_uri != '{ARTIFACT_URI}/{e1.upper()}/{r1.upper()}/artifacts'"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search([e1, e2], filter_string))

        filter_string = f"attr.artifact_uri = '{ARTIFACT_URI}/{e2}/{r1}/artifacts'"
        assert self._search([e1, e2], filter_string) == []

        filter_string = "attribute.artifact_uri = 'random_artifact_path'"
        assert self._search([e1, e2], filter_string) == []

        filter_string = "attribute.artifact_uri != 'random_artifact_path'"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search([e1, e2], filter_string))

        filter_string = f"attribute.artifact_uri LIKE '%{r1}%'"
        assert self._search([e1, e2], filter_string) == [r1]

        filter_string = f"attribute.artifact_uri LIKE '%{r1[:16]}%'"
        assert self._search([e1, e2], filter_string) == [r1]

        filter_string = f"attribute.artifact_uri LIKE '%{r1[-16:]}%'"
        assert self._search([e1, e2], filter_string) == [r1]

        filter_string = f"attribute.artifact_uri LIKE '%{r1.upper()}%'"
        assert self._search([e1, e2], filter_string) == []

        filter_string = f"attribute.artifact_uri ILIKE '%{r1.upper()}%'"
        assert self._search([e1, e2], filter_string) == [r1]

        filter_string = f"attribute.artifact_uri ILIKE '%{r1[:16].upper()}%'"
        assert self._search([e1, e2], filter_string) == [r1]

        filter_string = f"attribute.artifact_uri ILIKE '%{r1[-16:].upper()}%'"
        assert self._search([e1, e2], filter_string) == [r1]

        for k, v in {"experiment_id": e1, "lifecycle_stage": "ACTIVE"}.items():
            with pytest.raises(MlflowException, match=r"Invalid attribute key '.+' specified"):
                self._search([e1, e2], f"attribute.{k} = '{v}'")

    def test_search_full(self):
        experiment_id = self._experiment_factory("search_params")
        r1 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        r2 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id

        self.store.log_param(r1, entities.Param("generic_param", "p_val"))
        self.store.log_param(r2, entities.Param("generic_param", "p_val"))

        self.store.log_param(r1, entities.Param("p_a", "abc"))
        self.store.log_param(r2, entities.Param("p_b", "ABC"))

        self.store.log_metric(r1, entities.Metric("common", 1.0, 1, 0))
        self.store.log_metric(r2, entities.Metric("common", 1.0, 1, 0))

        self.store.log_metric(r1, entities.Metric("m_a", 2.0, 2, 0))
        self.store.log_metric(r2, entities.Metric("m_b", 3.0, 2, 0))
        self.store.log_metric(r2, entities.Metric("m_b", 4.0, 8, 0))
        self.store.log_metric(r2, entities.Metric("m_b", 8.0, 3, 0))

        filter_string = "params.generic_param = 'p_val' and metrics.common = 1.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        # all params and metrics match
        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 and metrics.m_a > 1.0"
        )
        assert self._search(experiment_id, filter_string) == [r1]

        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 "
            "and metrics.m_a > 1.0 and params.p_a LIKE 'a%'"
        )
        assert self._search(experiment_id, filter_string) == [r1]

        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 "
            "and metrics.m_a > 1.0 and params.p_a LIKE 'A%'"
        )
        assert self._search(experiment_id, filter_string) == []

        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 "
            "and metrics.m_a > 1.0 and params.p_a ILIKE 'A%'"
        )
        assert self._search(experiment_id, filter_string) == [r1]

        # test with mismatch param
        filter_string = (
            "params.random_bad_name = 'p_val' and metrics.common = 1.0 and metrics.m_a > 1.0"
        )
        assert self._search(experiment_id, filter_string) == []

        # test with mismatch metric
        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 and metrics.m_a > 100.0"
        )
        assert self._search(experiment_id, filter_string) == []

    def test_search_with_max_results(self):
        exp = self._experiment_factory("search_with_max_results")
        runs = [
            self._run_factory(self._get_run_configs(exp, start_time=r)).info.run_id
            for r in range(1200)
        ]
        # reverse the ordering, since we created in increasing order of start_time
        runs.reverse()

        assert runs[:1000] == self._search(exp)
        for n in [0, 1, 2, 4, 8, 10, 20, 50, 100, 500, 1000, 1200, 2000]:
            if n == 0 and self.store._get_dialect() == MSSQL:
                # In SQL server, `max_results = 0` results in the following error:
                # The number of rows provided for a FETCH clause must be greater then zero.
                continue
            assert runs[: min(1200, n)] == self._search(exp, max_results=n)

        with pytest.raises(
            MlflowException, match=r"Invalid value for request parameter max_results"
        ):
            self._search(exp, max_results=int(1e10))

    def test_search_with_deterministic_max_results(self):
        exp = self._experiment_factory("test_search_with_deterministic_max_results")
        # Create 10 runs with the same start_time.
        # Sort based on run_id
        runs = sorted(
            [
                self._run_factory(self._get_run_configs(exp, start_time=10)).info.run_id
                for r in range(10)
            ]
        )
        for n in [0, 1, 2, 4, 8, 10, 20]:
            if n == 0 and self.store._get_dialect() == MSSQL:
                # In SQL server, `max_results = 0` results in the following error:
                # The number of rows provided for a FETCH clause must be greater then zero.
                continue
            assert runs[: min(10, n)] == self._search(exp, max_results=n)

    def test_search_runs_pagination(self):
        exp = self._experiment_factory("test_search_runs_pagination")
        # test returned token behavior
        runs = sorted(
            [
                self._run_factory(self._get_run_configs(exp, start_time=10)).info.run_id
                for r in range(10)
            ]
        )
        result = self.store.search_runs([exp], None, ViewType.ALL, max_results=4)
        assert [r.info.run_id for r in result] == runs[0:4]
        assert result.token is not None
        result = self.store.search_runs(
            [exp], None, ViewType.ALL, max_results=4, page_token=result.token
        )
        assert [r.info.run_id for r in result] == runs[4:8]
        assert result.token is not None
        result = self.store.search_runs(
            [exp], None, ViewType.ALL, max_results=4, page_token=result.token
        )
        assert [r.info.run_id for r in result] == runs[8:]
        assert result.token is None

    def test_search_runs_run_name(self):
        exp_id = self._experiment_factory("test_search_runs_pagination")
        run1 = self._run_factory(dict(self._get_run_configs(exp_id), run_name="run_name1"))
        run2 = self._run_factory(dict(self._get_run_configs(exp_id), run_name="run_name2"))
        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.run_name = 'run_name1'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run1.info.run_id]
        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.`Run name` = 'run_name1'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run1.info.run_id]
        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.`run name` = 'run_name2'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run2.info.run_id]
        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.`Run Name` = 'run_name2'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run2.info.run_id]
        result = self.store.search_runs(
            [exp_id],
            filter_string="tags.`mlflow.runName` = 'run_name2'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run2.info.run_id]

        self.store.update_run_info(
            run1.info.run_id,
            RunStatus.FINISHED,
            end_time=run1.info.end_time,
            run_name="new_run_name1",
        )
        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.run_name = 'new_run_name1'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run1.info.run_id]

        # TODO: Test attribute-based search after set_tag

        # Test run name filter works for runs logged in MLflow <= 1.29.0
        with self.store.ManagedSessionMaker() as session:
            sql_run1 = session.query(SqlRun).filter(SqlRun.run_uuid == run1.info.run_id).one()
            sql_run1.name = ""

        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.run_name = 'new_run_name1'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run1.info.run_id]

        result = self.store.search_runs(
            [exp_id],
            filter_string="tags.`mlflow.runName` = 'new_run_name1'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run1.info.run_id]

    def test_search_runs_run_id(self):
        exp_id = self._experiment_factory("test_search_runs_run_id")
        # Set start_time to ensure the search result is deterministic
        run1 = self._run_factory(dict(self._get_run_configs(exp_id), start_time=1))
        run2 = self._run_factory(dict(self._get_run_configs(exp_id), start_time=2))
        run_id1 = run1.info.run_id
        run_id2 = run2.info.run_id

        result = self.store.search_runs(
            [exp_id],
            filter_string=f"attributes.run_id = '{run_id1}'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run_id1]

        result = self.store.search_runs(
            [exp_id],
            filter_string=f"attributes.run_id != '{run_id1}'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run_id2]

        result = self.store.search_runs(
            [exp_id],
            filter_string=f"attributes.run_id IN ('{run_id1}')",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run_id1]

        result = self.store.search_runs(
            [exp_id],
            filter_string=f"attributes.run_id NOT IN ('{run_id1}')",
            run_view_type=ViewType.ACTIVE_ONLY,
        )

        result = self.store.search_runs(
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
            result = self.store.search_runs(
                [exp_id], filter_string=filter_string, run_view_type=ViewType.ACTIVE_ONLY
            )
            assert [r.info.run_id for r in result] == [run_id2, run_id1]

        result = self.store.search_runs(
            [exp_id],
            filter_string=f"attributes.run_id NOT IN ('{run_id1}', '{run_id2}')",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert result == []

    def test_search_runs_start_time_alias(self):
        exp_id = self._experiment_factory("test_search_runs_start_time_alias")
        # Set start_time to ensure the search result is deterministic
        run1 = self._run_factory(dict(self._get_run_configs(exp_id), start_time=1))
        run2 = self._run_factory(dict(self._get_run_configs(exp_id), start_time=2))
        run_id1 = run1.info.run_id
        run_id2 = run2.info.run_id

        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.run_name = 'name'",
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attributes.start_time DESC"],
        )
        assert [r.info.run_id for r in result] == [run_id2, run_id1]

        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.run_name = 'name'",
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attributes.created ASC"],
        )
        assert [r.info.run_id for r in result] == [run_id1, run_id2]

        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.run_name = 'name'",
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attributes.Created DESC"],
        )
        assert [r.info.run_id for r in result] == [run_id2, run_id1]

        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.start_time > 0",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert {r.info.run_id for r in result} == {run_id1, run_id2}

        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.created > 1",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run_id2]

        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.Created > 2",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert result == []

    def test_search_runs_datasets(self):
        exp_id = self._experiment_factory("test_search_runs_datasets")
        # Set start_time to ensure the search result is deterministic
        run1 = self._run_factory(dict(self._get_run_configs(exp_id), start_time=1))
        run2 = self._run_factory(dict(self._get_run_configs(exp_id), start_time=3))
        run3 = self._run_factory(dict(self._get_run_configs(exp_id), start_time=2))

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

        self.store.log_inputs(run1.info.run_id, inputs_run1)
        self.store.log_inputs(run2.info.run_id, inputs_run2)
        self.store.log_inputs(run3.info.run_id, inputs_run3)
        run_id1 = run1.info.run_id
        run_id2 = run2.info.run_id
        run_id3 = run3.info.run_id

        result = self.store.search_runs(
            [exp_id],
            filter_string="dataset.name = 'name1'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert {r.info.run_id for r in result} == {run_id2, run_id1}

        result = self.store.search_runs(
            [exp_id],
            filter_string="dataset.digest = 'digest2'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert {r.info.run_id for r in result} == {run_id3, run_id1}

        result = self.store.search_runs(
            [exp_id],
            filter_string="dataset.name = 'name4'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert set(result) == set()

        result = self.store.search_runs(
            [exp_id],
            filter_string="dataset.context = 'train'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert {r.info.run_id for r in result} == {run_id2, run_id1}

        result = self.store.search_runs(
            [exp_id],
            filter_string="dataset.context = 'test'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert {r.info.run_id for r in result} == {run_id3}

        result = self.store.search_runs(
            [exp_id],
            filter_string="dataset.context = 'test' and dataset.name = 'name2'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert {r.info.run_id for r in result} == {run_id3}

        result = self.store.search_runs(
            [exp_id],
            filter_string="dataset.name = 'name2' and dataset.context = 'test'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert {r.info.run_id for r in result} == {run_id3}

        result = self.store.search_runs(
            [exp_id],
            filter_string="datasets.name IN ('name1', 'name2')",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert {r.info.run_id for r in result} == {run_id3, run_id1, run_id2}

        result = self.store.search_runs(
            [exp_id],
            filter_string="datasets.digest IN ('digest1', 'digest2')",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert {r.info.run_id for r in result} == {run_id3, run_id1, run_id2}

        result = self.store.search_runs(
            [exp_id],
            filter_string="datasets.name LIKE 'Name%'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert {r.info.run_id for r in result} == set()

        result = self.store.search_runs(
            [exp_id],
            filter_string="datasets.name ILIKE 'Name%'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert {r.info.run_id for r in result} == {run_id3, run_id1, run_id2}

        result = self.store.search_runs(
            [exp_id],
            filter_string="datasets.context ILIKE 'test%'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert {r.info.run_id for r in result} == {run_id3}

        result = self.store.search_runs(
            [exp_id],
            filter_string="datasets.context IN ('test', 'train')",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert {r.info.run_id for r in result} == {run_id3, run_id1, run_id2}

    def test_search_datasets(self):
        exp_id1 = self._experiment_factory("test_search_datasets_1")
        # Create an additional experiment to ensure we filter on specified experiment
        # and search works on multiple experiments.
        exp_id2 = self._experiment_factory("test_search_datasets_2")

        run1 = self._run_factory(dict(self._get_run_configs(exp_id1), start_time=1))
        run2 = self._run_factory(dict(self._get_run_configs(exp_id1), start_time=2))
        run3 = self._run_factory(dict(self._get_run_configs(exp_id2), start_time=3))

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

        self.store.log_inputs(run1.info.run_id, inputs_run1)
        self.store.log_inputs(run2.info.run_id, inputs_run2)
        self.store.log_inputs(run3.info.run_id, inputs_run3)

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
        results = self.store._search_datasets([exp_id1])
        expected_results = [
            _DatasetSummary(exp_id1, dataset1.name, dataset1.digest, "train"),
            _DatasetSummary(exp_id1, dataset2.name, dataset2.digest, "eval"),
            _DatasetSummary(exp_id1, dataset2.name, dataset2.digest, "test"),
            _DatasetSummary(exp_id1, dataset4.name, dataset4.digest, None),
        ]
        assert_has_same_elements(results, expected_results)

        # Verify results from both experiment are returned.
        results = self.store._search_datasets([exp_id1, exp_id2])
        expected_results.append(_DatasetSummary(exp_id2, dataset3.name, dataset3.digest, "train"))
        assert_has_same_elements(results, expected_results)

    def test_search_datasets_returns_no_more_than_max_results(self):
        exp_id = self.store.create_experiment("test_search_datasets")
        run = self._run_factory(dict(self._get_run_configs(exp_id), start_time=1))
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

        self.store.log_inputs(run.info.run_id, inputs)

        results = self.store._search_datasets([exp_id])
        assert len(results) == 1000

    def test_log_batch(self):
        experiment_id = self._experiment_factory("log_batch")
        run_id = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        metric_entities = [Metric("m1", 0.87, 12345, 0), Metric("m2", 0.49, 12345, 1)]
        param_entities = [Param("p1", "p1val"), Param("p2", "p2val")]
        tag_entities = [
            RunTag("t1", "t1val"),
            RunTag("t2", "t2val"),
            RunTag(MLFLOW_RUN_NAME, "my_run"),
        ]
        self.store.log_batch(
            run_id=run_id, metrics=metric_entities, params=param_entities, tags=tag_entities
        )
        run = self.store.get_run(run_id)
        assert run.data.tags == {"t1": "t1val", "t2": "t2val", MLFLOW_RUN_NAME: "my_run"}
        assert run.data.params == {"p1": "p1val", "p2": "p2val"}
        metric_histories = sum(
            [self.store.get_metric_history(run_id, key) for key in run.data.metrics], []
        )
        metrics = [(m.key, m.value, m.timestamp, m.step) for m in metric_histories]
        assert set(metrics) == {("m1", 0.87, 12345, 0), ("m2", 0.49, 12345, 1)}

    def test_log_batch_limits(self):
        # Test that log batch at the maximum allowed request size succeeds (i.e doesn't hit
        # SQL limitations, etc)
        experiment_id = self._experiment_factory("log_batch_limits")
        run_id = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        metric_tuples = [(f"m{i}", i, 12345, i * 2) for i in range(1000)]
        metric_entities = [Metric(*metric_tuple) for metric_tuple in metric_tuples]
        self.store.log_batch(run_id=run_id, metrics=metric_entities, params=[], tags=[])
        run = self.store.get_run(run_id)
        metric_histories = sum(
            [self.store.get_metric_history(run_id, key) for key in run.data.metrics], []
        )
        metrics = [(m.key, m.value, m.timestamp, m.step) for m in metric_histories]
        assert set(metrics) == set(metric_tuples)

    def test_log_batch_param_overwrite_disallowed(self):
        # Test that attempting to overwrite a param via log_batch results in an exception and that
        # no partial data is logged
        run = self._run_factory()
        tkey = "my-param"
        param = entities.Param(tkey, "orig-val")
        self.store.log_param(run.info.run_id, param)

        overwrite_param = entities.Param(tkey, "newval")
        tag = entities.RunTag("tag-key", "tag-val")
        metric = entities.Metric("metric-key", 3.0, 12345, 0)
        with pytest.raises(
            MlflowException, match=r"Changing param values is not allowed"
        ) as exception_context:
            self.store.log_batch(
                run.info.run_id, metrics=[metric], params=[overwrite_param], tags=[tag]
            )
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        self._verify_logged(self.store, run.info.run_id, metrics=[], params=[param], tags=[])

    def test_log_batch_with_unchanged_and_new_params(self):
        """
        Test case to ensure the following code works:
        ---------------------------------------------
        mlflow.log_params({"a": 0, "b": 1})
        mlflow.log_params({"a": 0, "c": 2})
        ---------------------------------------------
        """
        run = self._run_factory()
        self.store.log_batch(
            run.info.run_id,
            metrics=[],
            params=[entities.Param("a", "0"), entities.Param("b", "1")],
            tags=[],
        )
        self.store.log_batch(
            run.info.run_id,
            metrics=[],
            params=[entities.Param("a", "0"), entities.Param("c", "2")],
            tags=[],
        )
        self._verify_logged(
            self.store,
            run.info.run_id,
            metrics=[],
            params=[entities.Param("a", "0"), entities.Param("b", "1"), entities.Param("c", "2")],
            tags=[],
        )

    def test_log_batch_param_overwrite_disallowed_single_req(self):
        # Test that attempting to overwrite a param via log_batch results in an exception
        run = self._run_factory()
        pkey = "common-key"
        param0 = entities.Param(pkey, "orig-val")
        param1 = entities.Param(pkey, "newval")
        tag = entities.RunTag("tag-key", "tag-val")
        metric = entities.Metric("metric-key", 3.0, 12345, 0)
        with pytest.raises(
            MlflowException, match=r"Duplicate parameter keys have been submitted"
        ) as exception_context:
            self.store.log_batch(
                run.info.run_id, metrics=[metric], params=[param0, param1], tags=[tag]
            )
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        self._verify_logged(self.store, run.info.run_id, metrics=[], params=[], tags=[])

    def test_log_batch_accepts_empty_payload(self):
        run = self._run_factory()
        self.store.log_batch(run.info.run_id, metrics=[], params=[], tags=[])
        self._verify_logged(self.store, run.info.run_id, metrics=[], params=[], tags=[])

    def test_log_batch_internal_error(self):
        # Verify that internal errors during the DB save step for log_batch result in
        # MlflowExceptions
        run = self._run_factory()

        def _raise_exception_fn(*args, **kwargs):  # pylint: disable=unused-argument
            raise Exception("Some internal error")

        package = "mlflow.store.tracking.sqlalchemy_store.SqlAlchemyStore"
        with mock.patch(package + "._log_metrics") as metric_mock, mock.patch(
            package + "._log_params"
        ) as param_mock, mock.patch(package + "._set_tags") as tags_mock:
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
                    self.store.log_batch(run.info.run_id, **log_batch_kwargs)

    def test_log_batch_nonexistent_run(self):
        nonexistent_run_id = uuid.uuid4().hex
        with pytest.raises(
            MlflowException, match=rf"Run with id={nonexistent_run_id} not found"
        ) as exception_context:
            self.store.log_batch(nonexistent_run_id, [], [], [])
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    def test_log_batch_params_idempotency(self):
        run = self._run_factory()
        params = [Param("p-key", "p-val")]
        self.store.log_batch(run.info.run_id, metrics=[], params=params, tags=[])
        self.store.log_batch(run.info.run_id, metrics=[], params=params, tags=[])
        self._verify_logged(self.store, run.info.run_id, metrics=[], params=params, tags=[])

    def test_log_batch_tags_idempotency(self):
        run = self._run_factory()
        self.store.log_batch(
            run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")]
        )
        self.store.log_batch(
            run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")]
        )
        self._verify_logged(
            self.store, run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")]
        )

    def test_log_batch_allows_tag_overwrite(self):
        run = self._run_factory()
        self.store.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "val")])
        self.store.log_batch(
            run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "newval")]
        )
        self._verify_logged(
            self.store, run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "newval")]
        )

    def test_log_batch_allows_tag_overwrite_single_req(self):
        run = self._run_factory()
        tags = [RunTag("t-key", "val"), RunTag("t-key", "newval")]
        self.store.log_batch(run.info.run_id, metrics=[], params=[], tags=tags)
        self._verify_logged(self.store, run.info.run_id, metrics=[], params=[], tags=[tags[-1]])

    def test_log_batch_metrics(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = 100.0
        metric = entities.Metric(tkey, tval, get_current_time_millis(), 0)
        metric2 = entities.Metric(tkey, tval, get_current_time_millis() + 2, 0)
        nan_metric = entities.Metric("NaN", float("nan"), 0, 0)
        pos_inf_metric = entities.Metric("PosInf", float("inf"), 0, 0)
        neg_inf_metric = entities.Metric("NegInf", -float("inf"), 0, 0)

        # duplicate metric and metric2 values should be eliminated
        metrics = [metric, metric2, nan_metric, pos_inf_metric, neg_inf_metric, metric, metric2]
        self.store._log_metrics(run.info.run_id, metrics)

        run = self.store.get_run(run.info.run_id)
        assert tkey in run.data.metrics
        assert run.data.metrics[tkey] == tval

        # SQL store _get_run method returns full history of recorded metrics.
        # Should return duplicates as well
        # MLflow RunData contains only the last reported values for metrics.
        with self.store.ManagedSessionMaker() as session:
            sql_run_metrics = self.store._get_run(session, run.info.run_id).metrics
            assert len(sql_run_metrics) == 5
            assert len(run.data.metrics) == 4
            assert math.isnan(run.data.metrics["NaN"])
            assert run.data.metrics["PosInf"] == 1.7976931348623157e308
            assert run.data.metrics["NegInf"] == -1.7976931348623157e308

    def test_log_batch_same_metric_repeated_single_req(self):
        run = self._run_factory()
        metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
        metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
        self.store.log_batch(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])
        self._verify_logged(
            self.store, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[]
        )

    def test_log_batch_same_metric_repeated_multiple_reqs(self):
        run = self._run_factory()
        metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
        metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
        self.store.log_batch(run.info.run_id, params=[], metrics=[metric0], tags=[])
        self._verify_logged(self.store, run.info.run_id, params=[], metrics=[metric0], tags=[])
        self.store.log_batch(run.info.run_id, params=[], metrics=[metric1], tags=[])
        self._verify_logged(
            self.store, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[]
        )

    def test_log_batch_same_metrics_repeated_multiple_reqs(self):
        run = self._run_factory()
        metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
        metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
        self.store.log_batch(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])
        self._verify_logged(
            self.store, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[]
        )
        self.store.log_batch(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])
        self._verify_logged(
            self.store, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[]
        )

    def test_log_batch_null_metrics(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = None
        metric_1 = entities.Metric(tkey, tval, get_current_time_millis(), 0)

        tkey = "blahmetric2"
        tval = None
        metric_2 = entities.Metric(tkey, tval, get_current_time_millis(), 0)

        metrics = [metric_1, metric_2]

        with pytest.raises(
            MlflowException, match=r"Got invalid value None for metric"
        ) as exception_context:
            self.store.log_batch(run.info.run_id, metrics=metrics, params=[], tags=[])
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_log_batch_params_max_length_value(self):
        run = self._run_factory()
        param_entities = [Param("long param", "x" * 6000), Param("short param", "xyz")]
        expected_param_entities = [Param("long param", "x" * 6000), Param("short param", "xyz")]
        self.store.log_batch(run.info.run_id, [], param_entities, [])
        self._verify_logged(self.store, run.info.run_id, [], expected_param_entities, [])
        param_entities = [Param("long param", "x" * 6001)]
        with pytest.raises(MlflowException, match="exceeded length"):
            self.store.log_batch(run.info.run_id, [], param_entities, [])

    def test_upgrade_cli_idempotence(self):
        # Repeatedly run `mlflow db upgrade` against our database, verifying that the command
        # succeeds and that the DB has the latest schema
        engine = sqlalchemy.create_engine(self.db_url)
        assert _get_schema_version(engine) == _get_latest_schema_revision()
        for _ in range(3):
            invoke_cli_runner(mlflow.db.commands, ["upgrade", self.db_url])
            assert _get_schema_version(engine) == _get_latest_schema_revision()
        engine.dispose()

    def test_metrics_materialization_upgrade_succeeds_and_produces_expected_latest_metric_values(
        self,
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
        with TempDir() as tmp_db_dir:
            db_path = tmp_db_dir.path("tmp_db.sql")
            db_url = "sqlite:///" + db_path
            shutil.copy2(
                src=os.path.join(db_resources_path, "db_version_7ac759974ad8_with_metrics.sql"),
                dst=db_path,
            )

            invoke_cli_runner(mlflow.db.commands, ["upgrade", db_url])
            store = self._get_store(db_uri=db_url)
            with open(expected_metric_values_path) as f:
                expected_metric_values = json.load(f)

            for run_id, expected_metrics in expected_metric_values.items():
                fetched_run = store.get_run(run_id=run_id)
                assert fetched_run.data.metrics == expected_metrics

    def _generate_large_data(self, nb_runs=1000):
        experiment_id = self.store.create_experiment("test_experiment")

        current_run = 0

        run_ids = []
        metrics_list = []
        tags_list = []
        params_list = []
        latest_metrics_list = []

        for _ in range(nb_runs):
            run_id = self.store.create_run(
                experiment_id=experiment_id,
                start_time=current_run,
                tags=[],
                user_id="Anderson",
                run_name="name",
            ).info.run_uuid

            run_ids.append(run_id)

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

        with self.store.engine.begin() as conn:
            conn.execute(sqlalchemy.insert(SqlParam), params_list)
            conn.execute(sqlalchemy.insert(SqlMetric), metrics_list)
            conn.execute(sqlalchemy.insert(SqlLatestMetric), latest_metrics_list)
            conn.execute(sqlalchemy.insert(SqlTag), tags_list)

        return experiment_id, run_ids

    def test_search_runs_returns_expected_results_with_large_experiment(self):
        """
        This case tests the SQLAlchemyStore implementation of the SearchRuns API to ensure
        that search queries over an experiment containing many runs, each with a large number
        of metrics, parameters, and tags, are performant and return the expected results.
        """
        experiment_id, run_ids = self._generate_large_data()

        run_results = self.store.search_runs([experiment_id], None, ViewType.ALL, max_results=100)
        assert len(run_results) == 100
        # runs are sorted by desc start_time
        assert [run.info.run_id for run in run_results] == list(reversed(run_ids[900:]))

    def test_search_runs_correctly_filters_large_data(self):
        experiment_id, _ = self._generate_large_data(1000)

        run_results = self.store.search_runs(
            [experiment_id],
            "metrics.mkey_0 < 26 and metrics.mkey_0 > 5 ",
            ViewType.ALL,
            max_results=50,
        )
        assert len(run_results) == 20

        run_results = self.store.search_runs(
            [experiment_id],
            "metrics.mkey_0 < 26 and metrics.mkey_0 > 5 and tags.tkey_0 = 'tval_0' ",
            ViewType.ALL,
            max_results=10,
        )
        assert len(run_results) == 2  # 20 runs between 9 and 26, 2 of which have a 0 tkey_0 value

        run_results = self.store.search_runs(
            [experiment_id],
            "metrics.mkey_0 < 26 and metrics.mkey_0 > 5 "
            "and tags.tkey_0 = 'tval_0' "
            "and params.pkey_0 = 'pval_0'",
            ViewType.ALL,
            max_results=5,
        )
        assert len(run_results) == 1  # 2 runs on previous request, 1 of which has a 0 pkey_0 value

    def test_search_runs_keep_all_runs_when_sorting(self):
        experiment_id = self.store.create_experiment("test_experiment1")

        r1 = self.store.create_run(
            experiment_id=experiment_id, start_time=0, tags=[], user_id="Me", run_name="name"
        ).info.run_uuid
        r2 = self.store.create_run(
            experiment_id=experiment_id, start_time=0, tags=[], user_id="Me", run_name="name"
        ).info.run_uuid
        self.store.set_tag(r1, RunTag(key="t1", value="1"))
        self.store.set_tag(r1, RunTag(key="t2", value="1"))
        self.store.set_tag(r2, RunTag(key="t2", value="1"))

        run_results = self.store.search_runs(
            [experiment_id], None, ViewType.ALL, max_results=1000, order_by=["tag.t1"]
        )
        assert len(run_results) == 2

    def test_try_get_run_tag(self):
        run = self._run_factory()
        self.store.set_tag(run.info.run_id, entities.RunTag("k1", "v1"))
        self.store.set_tag(run.info.run_id, entities.RunTag("k2", "v2"))

        with self.store.ManagedSessionMaker() as session:
            tag = self.store._try_get_run_tag(session, run.info.run_id, "k0")
            assert tag is None

            tag = self.store._try_get_run_tag(session, run.info.run_id, "k1")
            assert tag.key == "k1"
            assert tag.value == "v1"

            tag = self.store._try_get_run_tag(session, run.info.run_id, "k2")
            assert tag.key == "k2"
            assert tag.value == "v2"

    def test_get_metric_history_on_non_existent_metric_key(self):
        experiment_id = self._experiment_factory("test_exp")[0]
        run = self.store.create_run(
            experiment_id=experiment_id, user_id="user", start_time=0, tags=[], run_name="name"
        )
        run_id = run.info.run_id
        metrics = self.store.get_metric_history(run_id, "test_metric")
        assert metrics == []

    def test_insert_large_text_in_dataset_table(self):
        with self.store.engine.begin() as conn:
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

    def test_log_inputs_and_retrieve_runs_behaves_as_expected(self):
        experiment_id = self._experiment_factory("test exp")
        run1 = self._run_factory(config=self._get_run_configs(experiment_id, start_time=1))
        run2 = self._run_factory(config=self._get_run_configs(experiment_id, start_time=3))
        run3 = self._run_factory(config=self._get_run_configs(experiment_id, start_time=2))

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

        self.store.log_inputs(run1.info.run_id, inputs_run1)
        self.store.log_inputs(run2.info.run_id, inputs_run2)
        self.store.log_inputs(run3.info.run_id, inputs_run3)

        run1 = self.store.get_run(run1.info.run_id)
        assert_dataset_inputs_equal(run1.inputs.dataset_inputs, inputs_run1)
        run2 = self.store.get_run(run2.info.run_id)
        assert_dataset_inputs_equal(run2.inputs.dataset_inputs, inputs_run2)
        run3 = self.store.get_run(run3.info.run_id)
        assert_dataset_inputs_equal(run3.inputs.dataset_inputs, inputs_run3)

        search_results_1 = self.store.search_runs(
            [experiment_id], None, ViewType.ALL, max_results=4, order_by=["start_time ASC"]
        )
        run1 = search_results_1[0]
        assert_dataset_inputs_equal(run1.inputs.dataset_inputs, inputs_run1)
        run2 = search_results_1[2]
        assert_dataset_inputs_equal(run2.inputs.dataset_inputs, inputs_run2)
        run3 = search_results_1[1]
        assert_dataset_inputs_equal(run3.inputs.dataset_inputs, inputs_run3)

        search_results_2 = self.store.search_runs(
            [experiment_id], None, ViewType.ALL, max_results=4, order_by=["start_time DESC"]
        )
        run1 = search_results_2[2]
        assert_dataset_inputs_equal(run1.inputs.dataset_inputs, inputs_run1)
        run2 = search_results_2[0]
        assert_dataset_inputs_equal(run2.inputs.dataset_inputs, inputs_run2)
        run3 = search_results_2[1]
        assert_dataset_inputs_equal(run3.inputs.dataset_inputs, inputs_run3)

    def test_log_input_multiple_times_does_not_overwrite_tags_or_dataset(self):
        experiment_id = self._experiment_factory("test exp")
        run = self._run_factory(config=self._get_run_configs(experiment_id))
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
        self.store.log_inputs(run.info.run_id, [entities.DatasetInput(dataset, tags)])

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
                entities.InputTag(key=f"key{i+1}", value=f"value{i+1}"),
            ]
            self.store.log_inputs(
                run.info.run_id, [entities.DatasetInput(overwrite_dataset, overwrite_tags)]
            )

        run = self.store.get_run(run.info.run_id)
        assert_dataset_inputs_equal(
            run.inputs.dataset_inputs, [entities.DatasetInput(dataset, tags)]
        )

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
        self.store.log_inputs(
            run.info.run_id, [entities.DatasetInput(other_name_dataset, other_name_input_tags)]
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
        self.store.log_inputs(
            run.info.run_id, [entities.DatasetInput(other_digest_dataset, other_digest_input_tags)]
        )

        run = self.store.get_run(run.info.run_id)
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
            new_run = self.store.create_run(
                experiment_id=experiment_id,
                user_id="user",
                start_time=0,
                tags=[],
                run_name=None,
            )
            new_tags = [
                entities.InputTag(key=f"key{i}", value=f"value{i}"),
                entities.InputTag(key=f"key{i+1}", value=f"value{i+1}"),
            ]
            self.store.log_inputs(new_run.info.run_id, [entities.DatasetInput(dataset, new_tags)])
            new_run = self.store.get_run(new_run.info.run_id)
            assert_dataset_inputs_equal(
                new_run.inputs.dataset_inputs, [entities.DatasetInput(dataset, new_tags)]
            )

    def test_log_inputs_handles_case_when_no_datasets_are_specified(self):
        experiment_id = self._experiment_factory("test exp")
        run = self._run_factory(config=self._get_run_configs(experiment_id))
        self.store.log_inputs(run.info.run_id)
        self.store.log_inputs(run.info.run_id, datasets=None)

    def test_log_inputs_fails_with_missing_inputs(self):
        experiment_id = self._experiment_factory("test exp")
        run = self._run_factory(config=self._get_run_configs(experiment_id))

        dataset = entities.Dataset(
            name="name1", digest="digest1", source_type="type", source="source"
        )

        tags = [entities.InputTag(key="key", value="train")]

        # Test input key missing
        with pytest.raises(MlflowException, match="InputTag key cannot be None"):
            self.store.log_inputs(
                run.info.run_id,
                [
                    entities.DatasetInput(
                        tags=[entities.InputTag(key=None, value="train")], dataset=dataset
                    )
                ],
            )

        # Test input value missing
        with pytest.raises(MlflowException, match="InputTag value cannot be None"):
            self.store.log_inputs(
                run.info.run_id,
                [
                    entities.DatasetInput(
                        tags=[entities.InputTag(key="key", value=None)], dataset=dataset
                    )
                ],
            )

        # Test dataset name missing
        with pytest.raises(MlflowException, match="Dataset name cannot be None"):
            self.store.log_inputs(
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
            self.store.log_inputs(
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
            self.store.log_inputs(
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
            self.store.log_inputs(
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

    def test_log_inputs_fails_with_too_large_inputs(self):
        experiment_id = self._experiment_factory("test exp")
        run = self._run_factory(config=self._get_run_configs(experiment_id))

        dataset = entities.Dataset(
            name="name1", digest="digest1", source_type="type", source="source"
        )

        tags = [entities.InputTag(key="key", value="train")]

        # Test input key too large (limit is 255)
        with pytest.raises(MlflowException, match="InputTag key exceeds the maximum length of 255"):
            self.store.log_inputs(
                run.info.run_id,
                [
                    entities.DatasetInput(
                        tags=[entities.InputTag(key="a" * 256, value="train")], dataset=dataset
                    )
                ],
            )

        # Test input value too large (limit is 500)
        with pytest.raises(
            MlflowException, match="InputTag value exceeds the maximum length of 500"
        ):
            self.store.log_inputs(
                run.info.run_id,
                [
                    entities.DatasetInput(
                        tags=[entities.InputTag(key="key", value="a" * 501)], dataset=dataset
                    )
                ],
            )

        # Test dataset name too large (limit is 500)
        with pytest.raises(MlflowException, match="Dataset name exceeds the maximum length of 500"):
            self.store.log_inputs(
                run.info.run_id,
                [
                    entities.DatasetInput(
                        tags=tags,
                        dataset=entities.Dataset(
                            name="a" * 501, digest="digest1", source_type="type", source="source"
                        ),
                    )
                ],
            )

        # Test dataset digest too large (limit is 36)
        with pytest.raises(
            MlflowException, match="Dataset digest exceeds the maximum length of 36"
        ):
            self.store.log_inputs(
                run.info.run_id,
                [
                    entities.DatasetInput(
                        tags=tags,
                        dataset=entities.Dataset(
                            name="name", digest="a" * 37, source_type="type", source="source"
                        ),
                    )
                ],
            )

        # Test dataset source too large (limit is 65535)
        with pytest.raises(
            MlflowException, match="Dataset source exceeds the maximum length of 65535"
        ):
            self.store.log_inputs(
                run.info.run_id,
                [
                    entities.DatasetInput(
                        tags=tags,
                        dataset=entities.Dataset(
                            name="name", digest="digest", source_type="type", source="a" * 65536
                        ),
                    )
                ],
            )

        # Test dataset schema too large (limit is 65535)
        with pytest.raises(
            MlflowException, match="Dataset schema exceeds the maximum length of 65535"
        ):
            self.store.log_inputs(
                run.info.run_id,
                [
                    entities.DatasetInput(
                        tags=tags,
                        dataset=entities.Dataset(
                            name="name",
                            digest="digest",
                            source_type="type",
                            source="source",
                            schema="a" * 65536,
                        ),
                    )
                ],
            )

        # Test dataset profile too large (limit is 16777215)
        with pytest.raises(
            MlflowException, match="Dataset profile exceeds the maximum length of 16777215"
        ):
            self.store.log_inputs(
                run.info.run_id,
                [
                    entities.DatasetInput(
                        tags=tags,
                        dataset=entities.Dataset(
                            name="name",
                            digest="digest",
                            source_type="type",
                            source="source",
                            profile="a" * 16777216,
                        ),
                    )
                ],
            )

    def test_log_inputs_with_duplicates_in_single_request(self):
        experiment_id = self._experiment_factory("test exp")
        run1 = self._run_factory(config=self._get_run_configs(experiment_id, start_time=1))

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

        self.store.log_inputs(run1.info.run_id, inputs_run1)
        run1 = self.store.get_run(run1.info.run_id)
        assert_dataset_inputs_equal(
            run1.inputs.dataset_inputs, [entities.DatasetInput(dataset1, tags1)]
        )


def test_sqlalchemy_store_behaves_as_expected_with_inmemory_sqlite_db(monkeypatch):
    monkeypatch.setenv("MLFLOW_SQLALCHEMYSTORE_POOLCLASS", "SingletonThreadPool")
    store = SqlAlchemyStore("sqlite:///:memory:", ARTIFACT_URI)
    experiment_id = store.create_experiment(name="exp1")
    run = store.create_run(
        experiment_id=experiment_id, user_id="user", start_time=0, tags=[], run_name="name"
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


class TestSqlAlchemyStoreMigratedDB(TestSqlAlchemyStore):
    """
    Test case where user has an existing DB with schema generated before MLflow 1.0,
    then migrates their DB.
    """

    def setUp(self):
        super()._setup_db_uri()
        engine = sqlalchemy.create_engine(self.db_url)
        InitialBase.metadata.create_all(engine)
        engine.dispose()
        invoke_cli_runner(mlflow.db.commands, ["upgrade", self.db_url])
        self.store = SqlAlchemyStore(self.db_url, ARTIFACT_URI)


class TextClauseMatcher:
    def __init__(self, text):
        self.text = text

    def __eq__(self, other):
        return self.text == other.text


@mock.patch("sqlalchemy.orm.session.Session", spec=True)
def test_set_zero_value_insertion_for_autoincrement_column_MYSQL(mock_session):
    mock_store = mock.Mock(SqlAlchemyStore)
    mock_store.db_type = MYSQL
    SqlAlchemyStore._set_zero_value_insertion_for_autoincrement_column(mock_store, mock_session)
    mock_session.execute.assert_called_with(
        TextClauseMatcher("SET @@SESSION.sql_mode='NO_AUTO_VALUE_ON_ZERO';")
    )


@mock.patch("sqlalchemy.orm.session.Session", spec=True)
def test_set_zero_value_insertion_for_autoincrement_column_MSSQL(mock_session):
    mock_store = mock.Mock(SqlAlchemyStore)
    mock_store.db_type = MSSQL
    SqlAlchemyStore._set_zero_value_insertion_for_autoincrement_column(mock_store, mock_session)
    mock_session.execute.assert_called_with(
        TextClauseMatcher("SET IDENTITY_INSERT experiments ON;")
    )


@mock.patch("sqlalchemy.orm.session.Session", spec=True)
def test_unset_zero_value_insertion_for_autoincrement_column_MYSQL(mock_session):
    mock_store = mock.Mock(SqlAlchemyStore)
    mock_store.db_type = MYSQL
    SqlAlchemyStore._unset_zero_value_insertion_for_autoincrement_column(mock_store, mock_session)
    mock_session.execute.assert_called_with(TextClauseMatcher("SET @@SESSION.sql_mode='';"))


@mock.patch("sqlalchemy.orm.session.Session", spec=True)
def test_unset_zero_value_insertion_for_autoincrement_column_MSSQL(mock_session):
    mock_store = mock.Mock(SqlAlchemyStore)
    mock_store.db_type = MSSQL
    SqlAlchemyStore._unset_zero_value_insertion_for_autoincrement_column(mock_store, mock_session)
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
                db_uri="sqlite:///" + dbfile_path, default_artifact_root=artifact_root_uri
            )
            exp_id = store.create_experiment(name="exp")
            exp = store.get_experiment(exp_id)
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
        ("file://my_server/my_path/my_sub_path", "file://my_server/my_path/my_sub_path/{e}"),
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
                db_uri="sqlite:///" + dbfile_path, default_artifact_root=artifact_root_uri
            )
            exp_id = store.create_experiment(name="exp")
            run = store.create_run(
                experiment_id=exp_id, user_id="user", start_time=0, tags=[], run_name="name"
            )
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
            "file://my_server/my_path/my_sub_path",
            "file://my_server/my_path/my_sub_path/{e}/{r}/artifacts",
        ),
        ("path/to/local/folder", "file://{cwd}/path/to/local/folder/{e}/{r}/artifacts"),
        ("/path/to/local/folder", "file:///{drive}path/to/local/folder/{e}/{r}/artifacts"),
        ("#path/to/local/folder?", "file://{cwd}/{e}/{r}/artifacts#path/to/local/folder?"),
        ("file:path/to/local/folder", "file://{cwd}/path/to/local/folder/{e}/{r}/artifacts"),
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
        ("file:path/to/local/folder", "file://{cwd}/path/to/local/folder/{e}/{r}/artifacts"),
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
