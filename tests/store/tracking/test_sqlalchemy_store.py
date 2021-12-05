import os
import shutil
import tempfile
import unittest
import warnings

import math
import random
import pytest
import sqlalchemy
import time
import mlflow
import uuid
import json
import pandas as pd
from unittest import mock

import mlflow.db
import mlflow.store.db.base_sql_model
from mlflow.entities import (
    ViewType,
    RunTag,
    SourceType,
    RunStatus,
    Experiment,
    Metric,
    Param,
    ExperimentTag,
)
from mlflow.protos.databricks_pb2 import (
    ErrorCode,
    RESOURCE_DOES_NOT_EXIST,
    INVALID_PARAMETER_VALUE,
    INTERNAL_ERROR,
)
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.db.utils import (
    _get_schema_version,
    _get_latest_schema_revision,
)
from mlflow.store.tracking.dbmodels import models
from mlflow.store.db.db_types import MYSQL, MSSQL
from mlflow import entities
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore, _get_orderby_clauses
from mlflow.utils import mlflow_tags
from mlflow.utils.file_utils import TempDir
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase
from tests.integration.utils import invoke_cli_runner
from tests.store.tracking import AbstractStoreTest

DB_URI = "sqlite:///"
ARTIFACT_URI = "artifact_folder"


class TestParseDbUri(unittest.TestCase):
    def test_correct_db_type_from_uri(self):
        # try each the main drivers per supported database type
        target_db_type_uris = {
            "sqlite": ("pysqlite", "pysqlcipher"),
            "postgresql": (
                "psycopg2",
                "pg8000",
                "psycopg2cffi",
                "pypostgresql",
                "pygresql",
                "zxjdbc",
            ),
            "mysql": (
                "mysqldb",
                "pymysql",
                "mysqlconnector",
                "cymysql",
                "oursql",
                "gaerdbms",
                "pyodbc",
                "zxjdbc",
            ),
            "mssql": ("pyodbc", "mxodbc", "pymssql", "zxjdbc", "adodbapi"),
        }
        for target_db_type, drivers in target_db_type_uris.items():
            # try the driver-less version, which will revert SQLAlchemy to the default driver
            uri = "%s://..." % target_db_type
            parsed_db_type = extract_db_type_from_uri(uri)
            self.assertEqual(target_db_type, parsed_db_type)
            # try each of the popular drivers (per SQLAlchemy's dialect pages)
            for driver in drivers:
                uri = "%s+%s://..." % (target_db_type, driver)
                parsed_db_type = extract_db_type_from_uri(uri)
                self.assertEqual(target_db_type, parsed_db_type)

    def _db_uri_error(self, db_uris, expected_message_part):
        for db_uri in db_uris:
            with self.assertRaises(MlflowException) as e:
                extract_db_type_from_uri(db_uri)
            self.assertIn(expected_message_part, e.exception.message)

    def test_fail_on_unsupported_db_type(self):
        bad_db_uri_strings = [
            "oracle://...",
            "oracle+cx_oracle://...",
            "snowflake://...",
            "://...",
            "abcdefg",
        ]
        self._db_uri_error(bad_db_uri_strings, "Supported database engines are ")

    def test_fail_on_multiple_drivers(self):
        bad_db_uri_strings = ["mysql+pymsql+pyodbc://..."]
        self._db_uri_error(
            bad_db_uri_strings,
            "mlflow.org/docs/latest/tracking.html#storage for format specifications",
        )


class TestSqlAlchemyStoreSqlite(unittest.TestCase, AbstractStoreTest):
    def _get_store(self, db_uri=""):
        return SqlAlchemyStore(db_uri, ARTIFACT_URI)

    def create_test_run(self):
        return self._run_factory()

    def setUp(self):
        self.maxDiff = None  # print all differences on assert failures
        fd, self.temp_dbfile = tempfile.mkstemp()
        # Close handle immediately so that we can remove the file later on in Windows
        os.close(fd)
        self.db_url = "%s%s" % (DB_URI, self.temp_dbfile)
        self.store = self._get_store(self.db_url)

    def get_store(self):
        return self.store

    def tearDown(self):
        mlflow.store.db.base_sql_model.Base.metadata.drop_all(self.store.engine)
        os.remove(self.temp_dbfile)
        shutil.rmtree(ARTIFACT_URI)

    def _experiment_factory(self, names):
        if type(names) is list:
            return [self.store.create_experiment(name=name) for name in names]

        return self.store.create_experiment(name=names)

    def test_default_experiment(self):
        experiments = self.store.list_experiments()
        self.assertEqual(len(experiments), 1)

        first = experiments[0]
        self.assertEqual(first.experiment_id, "0")
        self.assertEqual(first.name, "Default")

    def test_default_experiment_lifecycle(self):
        default_experiment = self.store.get_experiment(experiment_id=0)
        self.assertEqual(default_experiment.name, Experiment.DEFAULT_EXPERIMENT_NAME)
        self.assertEqual(default_experiment.lifecycle_stage, entities.LifecycleStage.ACTIVE)

        self._experiment_factory("aNothEr")
        all_experiments = [e.name for e in self.store.list_experiments()]
        self.assertCountEqual(set(["aNothEr", "Default"]), set(all_experiments))

        self.store.delete_experiment(0)

        self.assertCountEqual(["aNothEr"], [e.name for e in self.store.list_experiments()])
        another = self.store.get_experiment(1)
        self.assertEqual("aNothEr", another.name)

        default_experiment = self.store.get_experiment(experiment_id=0)
        self.assertEqual(default_experiment.name, Experiment.DEFAULT_EXPERIMENT_NAME)
        self.assertEqual(default_experiment.lifecycle_stage, entities.LifecycleStage.DELETED)

        # destroy SqlStore and make a new one
        del self.store
        self.store = self._get_store(self.db_url)

        # test that default experiment is not reactivated
        default_experiment = self.store.get_experiment(experiment_id=0)
        self.assertEqual(default_experiment.name, Experiment.DEFAULT_EXPERIMENT_NAME)
        self.assertEqual(default_experiment.lifecycle_stage, entities.LifecycleStage.DELETED)

        self.assertCountEqual(["aNothEr"], [e.name for e in self.store.list_experiments()])
        all_experiments = [e.name for e in self.store.list_experiments(ViewType.ALL)]
        self.assertCountEqual(set(["aNothEr", "Default"]), set(all_experiments))

        # ensure that experiment ID dor active experiment is unchanged
        another = self.store.get_experiment(1)
        self.assertEqual("aNothEr", another.name)

    def test_raise_duplicate_experiments(self):
        with self.assertRaises(Exception):
            self._experiment_factory(["test", "test"])

    def test_raise_experiment_dont_exist(self):
        with self.assertRaises(Exception):
            self.store.get_experiment(experiment_id=100)

    def test_delete_experiment(self):
        experiments = self._experiment_factory(["morty", "rick", "rick and morty"])

        all_experiments = self.store.list_experiments()
        self.assertEqual(len(all_experiments), len(experiments) + 1)  # default

        exp_id = experiments[0]
        self.store.delete_experiment(exp_id)

        updated_exp = self.store.get_experiment(exp_id)
        self.assertEqual(updated_exp.lifecycle_stage, entities.LifecycleStage.DELETED)

        self.assertEqual(len(self.store.list_experiments()), len(all_experiments) - 1)

    def test_get_experiment(self):
        name = "goku"
        experiment_id = self._experiment_factory(name)
        actual = self.store.get_experiment(experiment_id)
        self.assertEqual(actual.name, name)
        self.assertEqual(actual.experiment_id, experiment_id)

        actual_by_name = self.store.get_experiment_by_name(name)
        self.assertEqual(actual_by_name.name, name)
        self.assertEqual(actual_by_name.experiment_id, experiment_id)
        self.assertEqual(self.store.get_experiment_by_name("idontexist"), None)

    def test_list_experiments(self):
        testnames = ["blue", "red", "green"]

        experiments = self._experiment_factory(testnames)
        actual = self.store.list_experiments(
            max_results=SEARCH_MAX_RESULTS_DEFAULT, page_token=None
        )

        self.assertEqual(len(experiments) + 1, len(actual))  # default

        with self.store.ManagedSessionMaker() as session:
            for experiment_id in experiments:
                res = (
                    session.query(models.SqlExperiment)
                    .filter_by(experiment_id=experiment_id)
                    .first()
                )
                self.assertIn(res.name, testnames)
                self.assertEqual(str(res.experiment_id), experiment_id)

    def test_list_experiments_paginated_last_page(self):
        # 9 + 1 default experiment for 10 total
        testnames = ["randexp" + str(num) for num in random.sample(range(1, 100000), 9)]
        experiments = self._experiment_factory(testnames)
        max_results = 5
        returned_experiments = []
        result = self.store.list_experiments(max_results=max_results, page_token=None)
        self.assertEqual(len(result), max_results)
        returned_experiments.extend(result)
        while result.token:
            result = self.store.list_experiments(max_results=max_results, page_token=result.token)
            self.assertEqual(len(result), max_results)
            returned_experiments.extend(result)
        self.assertEqual(result.token, None)
        # make sure that at least all the experiments created in this test are found
        returned_exp_id_set = set([exp.experiment_id for exp in returned_experiments])
        self.assertEqual(set(experiments) - returned_exp_id_set, set())

    def test_list_experiments_paginated_returns_in_correct_order(self):
        testnames = ["randexp" + str(num) for num in random.sample(range(1, 100000), 20)]
        self._experiment_factory(testnames)

        # test that pagination will return all valid results in sorted order
        # by name ascending
        result = self.store.list_experiments(max_results=3, page_token=None)
        self.assertNotEqual(result.token, None)
        self.assertEqual([exp.name for exp in result[1:]], testnames[0:2])

        result = self.store.list_experiments(max_results=4, page_token=result.token)
        self.assertNotEqual(result.token, None)
        self.assertEqual([exp.name for exp in result], testnames[2:6])

        result = self.store.list_experiments(max_results=6, page_token=result.token)
        self.assertNotEqual(result.token, None)
        self.assertEqual([exp.name for exp in result], testnames[6:12])

        result = self.store.list_experiments(max_results=8, page_token=result.token)
        # this page token should be none
        self.assertEqual(result.token, None)
        self.assertEqual([exp.name for exp in result], testnames[12:])

    def test_list_experiments_paginated_errors(self):
        # test that providing a completely invalid page token throws
        with self.assertRaises(MlflowException) as exception_context:
            self.store.list_experiments(page_token="evilhax", max_results=20)
        assert exception_context.exception.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # test that providing too large of a max_results throws
        with self.assertRaises(MlflowException) as exception_context:
            self.store.list_experiments(page_token=None, max_results=int(1e15))
            assert exception_context.exception.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        self.assertIn(
            "Invalid value for request parameter max_results", exception_context.exception.message
        )

    def test_create_experiments(self):
        with self.store.ManagedSessionMaker() as session:
            result = session.query(models.SqlExperiment).all()
            self.assertEqual(len(result), 1)

        experiment_id = self.store.create_experiment(name="test exp")
        self.assertEqual(experiment_id, "1")
        with self.store.ManagedSessionMaker() as session:
            result = session.query(models.SqlExperiment).all()
            self.assertEqual(len(result), 2)

            test_exp = session.query(models.SqlExperiment).filter_by(name="test exp").first()
            self.assertEqual(str(test_exp.experiment_id), experiment_id)
            self.assertEqual(test_exp.name, "test exp")

        actual = self.store.get_experiment(experiment_id)
        self.assertEqual(actual.experiment_id, experiment_id)
        self.assertEqual(actual.name, "test exp")

    def test_create_experiment_appends_to_artifact_uri_path_correctly(self):
        cases = [
            ("path/to/local/folder", "path/to/local/folder/{e}"),
            ("/path/to/local/folder", "/path/to/local/folder/{e}"),
            ("#path/to/local/folder?", "#path/to/local/folder?/{e}"),
            ("file:path/to/local/folder", "file:path/to/local/folder/{e}"),
            ("file:///path/to/local/folder", "file:///path/to/local/folder/{e}"),
            ("file:path/to/local/folder?param=value", "file:path/to/local/folder/{e}?param=value"),
            ("file:///path/to/local/folder", "file:///path/to/local/folder/{e}"),
            (
                "file:///path/to/local/folder?param=value#fragment",
                "file:///path/to/local/folder/{e}?param=value#fragment",
            ),
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
        ]

        # Patch `is_local_uri` to prevent the SqlAlchemy store from attempting to create local
        # filesystem directories for file URI and POSIX path test cases
        with mock.patch("mlflow.store.tracking.sqlalchemy_store.is_local_uri", return_value=False):
            for i in range(len(cases)):
                artifact_root_uri, expected_artifact_uri_format = cases[i]
                with TempDir() as tmp:
                    dbfile_path = tmp.path("db")
                    store = SqlAlchemyStore(
                        db_uri="sqlite:///" + dbfile_path, default_artifact_root=artifact_root_uri
                    )
                    exp_id = store.create_experiment(name="exp")
                    exp = store.get_experiment(exp_id)
                    self.assertEqual(
                        exp.artifact_location, expected_artifact_uri_format.format(e=exp_id)
                    )

    def test_create_experiment_with_tags_works_correctly(self):
        experiment_id = self.store.create_experiment(
            name="test exp",
            artifact_location="some location",
            tags=[ExperimentTag("key1", "val1"), ExperimentTag("key2", "val2")],
        )
        experiment = self.store.get_experiment(experiment_id)
        assert len(experiment.tags) == 2
        assert experiment.tags["key1"] == "val1"
        assert experiment.tags["key2"] == "val2"

    def test_create_run_appends_to_artifact_uri_path_correctly(self):
        cases = [
            ("path/to/local/folder", "path/to/local/folder/{e}/{r}/artifacts"),
            ("/path/to/local/folder", "/path/to/local/folder/{e}/{r}/artifacts"),
            ("#path/to/local/folder?", "#path/to/local/folder?/{e}/{r}/artifacts"),
            ("file:path/to/local/folder", "file:path/to/local/folder/{e}/{r}/artifacts"),
            ("file:///path/to/local/folder", "file:///path/to/local/folder/{e}/{r}/artifacts"),
            (
                "file:path/to/local/folder?param=value",
                "file:path/to/local/folder/{e}/{r}/artifacts?param=value",
            ),
            ("file:///path/to/local/folder", "file:///path/to/local/folder/{e}/{r}/artifacts"),
            (
                "file:///path/to/local/folder?param=value#fragment",
                "file:///path/to/local/folder/{e}/{r}/artifacts?param=value#fragment",
            ),
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
        ]

        # Patch `is_local_uri` to prevent the SqlAlchemy store from attempting to create local
        # filesystem directories for file URI and POSIX path test cases
        with mock.patch("mlflow.store.tracking.sqlalchemy_store.is_local_uri", return_value=False):
            for i in range(len(cases)):
                artifact_root_uri, expected_artifact_uri_format = cases[i]
                with TempDir() as tmp:
                    dbfile_path = tmp.path("db")
                    store = SqlAlchemyStore(
                        db_uri="sqlite:///" + dbfile_path, default_artifact_root=artifact_root_uri
                    )
                    exp_id = store.create_experiment(name="exp")
                    run = store.create_run(
                        experiment_id=exp_id, user_id="user", start_time=0, tags=[]
                    )
                    self.assertEqual(
                        run.info.artifact_uri,
                        expected_artifact_uri_format.format(e=exp_id, r=run.info.run_id),
                    )

    def test_run_tag_model(self):
        # Create a run whose UUID we can reference when creating tag models.
        # `run_id` is a foreign key in the tags table; therefore, in order
        # to insert a tag with a given run UUID, the UUID must be present in
        # the runs table
        run = self._run_factory()
        with self.store.ManagedSessionMaker() as session:
            new_tag = models.SqlTag(run_uuid=run.info.run_id, key="test", value="val")
            session.add(new_tag)
            session.commit()
            added_tags = [
                tag for tag in session.query(models.SqlTag).all() if tag.key == new_tag.key
            ]
            self.assertEqual(len(added_tags), 1)
            added_tag = added_tags[0].to_mlflow_entity()
            self.assertEqual(added_tag.value, new_tag.value)

    def test_metric_model(self):
        # Create a run whose UUID we can reference when creating metric models.
        # `run_id` is a foreign key in the tags table; therefore, in order
        # to insert a metric with a given run UUID, the UUID must be present in
        # the runs table
        run = self._run_factory()
        with self.store.ManagedSessionMaker() as session:
            new_metric = models.SqlMetric(run_uuid=run.info.run_id, key="accuracy", value=0.89)
            session.add(new_metric)
            session.commit()
            metrics = session.query(models.SqlMetric).all()
            self.assertEqual(len(metrics), 1)

            added_metric = metrics[0].to_mlflow_entity()
            self.assertEqual(added_metric.value, new_metric.value)
            self.assertEqual(added_metric.key, new_metric.key)

    def test_param_model(self):
        # Create a run whose UUID we can reference when creating parameter models.
        # `run_id` is a foreign key in the tags table; therefore, in order
        # to insert a parameter with a given run UUID, the UUID must be present in
        # the runs table
        run = self._run_factory()
        with self.store.ManagedSessionMaker() as session:
            new_param = models.SqlParam(
                run_uuid=run.info.run_id, key="accuracy", value="test param"
            )
            session.add(new_param)
            session.commit()
            params = session.query(models.SqlParam).all()
            self.assertEqual(len(params), 1)

            added_param = params[0].to_mlflow_entity()
            self.assertEqual(added_param.value, new_param.value)
            self.assertEqual(added_param.key, new_param.key)

    def test_run_needs_uuid(self):
        # Depending on the implementation, a NULL identity key may result in different
        # exceptions, including IntegrityError (sqlite) and FlushError (MysQL).
        # Therefore, we check for the more generic 'SQLAlchemyError'
        with self.assertRaises(MlflowException) as exception_context:
            warnings.simplefilter("ignore")
            with self.store.ManagedSessionMaker() as session, warnings.catch_warnings():
                run = models.SqlRun()
                session.add(run)
                warnings.resetwarnings()
        assert exception_context.exception.error_code == ErrorCode.Name(INTERNAL_ERROR)

    def test_run_data_model(self):
        with self.store.ManagedSessionMaker() as session:
            m1 = models.SqlMetric(key="accuracy", value=0.89)
            m2 = models.SqlMetric(key="recal", value=0.89)
            p1 = models.SqlParam(key="loss", value="test param")
            p2 = models.SqlParam(key="blue", value="test param")

            session.add_all([m1, m2, p1, p2])

            run_data = models.SqlRun(run_uuid=uuid.uuid4().hex)
            run_data.params.append(p1)
            run_data.params.append(p2)
            run_data.metrics.append(m1)
            run_data.metrics.append(m2)

            session.add(run_data)
            session.commit()

            run_datums = session.query(models.SqlRun).all()
            actual = run_datums[0]
            self.assertEqual(len(run_datums), 1)
            self.assertEqual(len(actual.params), 2)
            self.assertEqual(len(actual.metrics), 2)

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
            "start_time": int(time.time()),
            "end_time": int(time.time()),
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
                self.assertEqual(v, SourceType.to_string(v2))
            else:
                self.assertEqual(v, v2)

    def _get_run_configs(self, experiment_id=None, tags=(), start_time=None):
        return {
            "experiment_id": experiment_id,
            "user_id": "Anderson",
            "start_time": start_time if start_time is not None else int(time.time()),
            "tags": tags,
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

        self.assertEqual(actual.info.experiment_id, experiment_id)
        self.assertEqual(actual.info.user_id, expected["user_id"])
        self.assertEqual(actual.info.start_time, expected["start_time"])

        self.assertEqual(len(actual.data.tags), len(tags))
        expected_tags = {tag.key: tag.value for tag in tags}
        self.assertEqual(actual.data.tags, expected_tags)

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
        self.assertIsInstance(run.info, entities.RunInfo)
        self.assertIsInstance(run.data, entities.RunData)

        self.assertEqual(run.data.metrics, {"my-metric": 3.4})
        self.assertEqual(run.data.params, {"my-param": "param-val"})
        self.assertEqual(run.data.tags["my-tag"], "tag-val")

        # Get the parent experiment of the run, verify it can be converted to protobuf
        exp = self.store.get_experiment(run.info.experiment_id)
        exp.to_proto()

    def test_delete_run(self):
        run = self._run_factory()

        self.store.delete_run(run.info.run_id)

        with self.store.ManagedSessionMaker() as session:
            actual = session.query(models.SqlRun).filter_by(run_uuid=run.info.run_id).first()
            self.assertEqual(actual.lifecycle_stage, entities.LifecycleStage.DELETED)

            deleted_run = self.store.get_run(run.info.run_id)
            self.assertEqual(actual.run_uuid, deleted_run.info.run_id)

    def test_hard_delete_run(self):
        run = self._run_factory()
        metric = entities.Metric("blahmetric", 100.0, int(1000 * time.time()), 0)
        self.store.log_metric(run.info.run_id, metric)
        param = entities.Param("blahparam", "100.0")
        self.store.log_param(run.info.run_id, param)
        tag = entities.RunTag("test tag", "a boogie")
        self.store.set_tag(run.info.run_id, tag)

        self.store._hard_delete_run(run.info.run_id)

        with self.store.ManagedSessionMaker() as session:
            actual_run = session.query(models.SqlRun).filter_by(run_uuid=run.info.run_id).first()
            self.assertEqual(None, actual_run)
            actual_metric = (
                session.query(models.SqlMetric).filter_by(run_uuid=run.info.run_id).first()
            )
            self.assertEqual(None, actual_metric)
            actual_param = (
                session.query(models.SqlParam).filter_by(run_uuid=run.info.run_id).first()
            )
            self.assertEqual(None, actual_param)
            actual_tag = session.query(models.SqlTag).filter_by(run_uuid=run.info.run_id).first()
            self.assertEqual(None, actual_tag)

    def test_get_deleted_runs(self):
        run = self._run_factory()
        deleted_run_ids = self.store._get_deleted_runs()
        self.assertEqual([], deleted_run_ids)

        self.store.delete_run(run.info.run_uuid)
        deleted_run_ids = self.store._get_deleted_runs()
        self.assertEqual([run.info.run_uuid], deleted_run_ids)

    def test_log_metric(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = 100.0
        metric = entities.Metric(tkey, tval, int(1000 * time.time()), 0)
        metric2 = entities.Metric(tkey, tval, int(1000 * time.time()) + 2, 0)
        nan_metric = entities.Metric("NaN", float("nan"), 0, 0)
        pos_inf_metric = entities.Metric("PosInf", float("inf"), 0, 0)
        neg_inf_metric = entities.Metric("NegInf", -float("inf"), 0, 0)
        self.store.log_metric(run.info.run_id, metric)
        self.store.log_metric(run.info.run_id, metric2)
        self.store.log_metric(run.info.run_id, nan_metric)
        self.store.log_metric(run.info.run_id, pos_inf_metric)
        self.store.log_metric(run.info.run_id, neg_inf_metric)

        run = self.store.get_run(run.info.run_id)
        self.assertTrue(tkey in run.data.metrics and run.data.metrics[tkey] == tval)

        # SQL store _get_run method returns full history of recorded metrics.
        # Should return duplicates as well
        # MLflow RunData contains only the last reported values for metrics.
        with self.store.ManagedSessionMaker() as session:
            sql_run_metrics = self.store._get_run(session, run.info.run_id).metrics
            self.assertEqual(5, len(sql_run_metrics))
            self.assertEqual(4, len(run.data.metrics))
            self.assertTrue(math.isnan(run.data.metrics["NaN"]))
            self.assertTrue(run.data.metrics["PosInf"] == 1.7976931348623157e308)
            self.assertTrue(run.data.metrics["NegInf"] == -1.7976931348623157e308)

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

    def test_log_null_metric(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = None
        metric = entities.Metric(tkey, tval, int(1000 * time.time()), 0)

        warnings.simplefilter("ignore")
        with self.assertRaises(MlflowException) as exception_context, warnings.catch_warnings():
            self.store.log_metric(run.info.run_id, metric)
            warnings.resetwarnings()
        assert exception_context.exception.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

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
        self.assertEqual(2, len(run.data.params))
        self.assertTrue(tkey in run.data.params and run.data.params[tkey] == tval)

    def test_log_param_uniqueness(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = "100.0"
        param = entities.Param(tkey, tval)
        param2 = entities.Param(tkey, "newval")
        self.store.log_param(run.info.run_id, param)

        with self.assertRaises(MlflowException) as e:
            self.store.log_param(run.info.run_id, param2)
        self.assertIn("Changing param values is not allowed. Param with key=", e.exception.message)

    def test_log_empty_str(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = ""
        param = entities.Param(tkey, tval)
        param2 = entities.Param("new param", "new key")
        self.store.log_param(run.info.run_id, param)
        self.store.log_param(run.info.run_id, param2)

        run = self.store.get_run(run.info.run_id)
        self.assertEqual(2, len(run.data.params))
        self.assertTrue(tkey in run.data.params and run.data.params[tkey] == tval)

    def test_log_null_param(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = None
        param = entities.Param(tkey, tval)

        with self.assertRaises(MlflowException) as exception_context:
            self.store.log_param(run.info.run_id, param)
        assert exception_context.exception.error_code == ErrorCode.Name(INTERNAL_ERROR)

    def test_set_experiment_tag(self):
        exp_id = self._experiment_factory("setExperimentTagExp")
        tag = entities.ExperimentTag("tag0", "value0")
        new_tag = entities.RunTag("tag0", "value00000")
        self.store.set_experiment_tag(exp_id, tag)
        experiment = self.store.get_experiment(exp_id)
        self.assertTrue(experiment.tags["tag0"] == "value0")
        # test that updating a tag works
        self.store.set_experiment_tag(exp_id, new_tag)
        experiment = self.store.get_experiment(exp_id)
        self.assertTrue(experiment.tags["tag0"] == "value00000")
        # test that setting a tag on 1 experiment does not impact another experiment.
        exp_id_2 = self._experiment_factory("setExperimentTagExp2")
        experiment2 = self.store.get_experiment(exp_id_2)
        self.assertTrue(len(experiment2.tags) == 0)
        # setting a tag on different experiments maintains different values across experiments
        different_tag = entities.RunTag("tag0", "differentValue")
        self.store.set_experiment_tag(exp_id_2, different_tag)
        experiment = self.store.get_experiment(exp_id)
        self.assertTrue(experiment.tags["tag0"] == "value00000")
        experiment2 = self.store.get_experiment(exp_id_2)
        self.assertTrue(experiment2.tags["tag0"] == "differentValue")
        # test can set multi-line tags
        multiLineTag = entities.ExperimentTag("multiline tag", "value2\nvalue2\nvalue2")
        self.store.set_experiment_tag(exp_id, multiLineTag)
        experiment = self.store.get_experiment(exp_id)
        self.assertTrue(experiment.tags["multiline tag"] == "value2\nvalue2\nvalue2")
        # test cannot set tags that are too long
        longTag = entities.ExperimentTag("longTagKey", "a" * 5001)
        with pytest.raises(MlflowException, match="exceeded length limit of 5000"):
            self.store.set_experiment_tag(exp_id, longTag)
        # test can set tags that are somewhat long
        longTag = entities.ExperimentTag("longTagKey", "a" * 4999)
        self.store.set_experiment_tag(exp_id, longTag)
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
        self.assertTrue(tkey in run.data.tags and run.data.tags[tkey] == new_val)

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
        self.assertTrue(k0 not in run.data.tags)
        self.assertTrue(k1 in run.data.tags and run.data.tags[k1] == v1)

        # test that deleting a tag works correctly with multiple runs having the same tag.
        run2 = self._run_factory(config=self._get_run_configs(run.info.experiment_id))
        self.store.set_tag(run.info.run_id, tag0)
        self.store.set_tag(run2.info.run_id, tag0)
        self.store.delete_tag(run.info.run_id, k0)
        run = self.store.get_run(run.info.run_id)
        run2 = self.store.get_run(run2.info.run_id)
        self.assertTrue(k0 not in run.data.tags)
        self.assertTrue(k0 in run2.data.tags)
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

        self.assertCountEqual(
            [(m.key, m.value, m.timestamp) for m in expected],
            [(m.key, m.value, m.timestamp) for m in actual],
        )

    def test_list_run_infos(self):
        experiment_id = self._experiment_factory("test_exp")
        r1 = self._run_factory(config=self._get_run_configs(experiment_id)).info.run_id
        r2 = self._run_factory(config=self._get_run_configs(experiment_id)).info.run_id

        def _runs(experiment_id, view_type):
            return [r.run_id for r in self.store.list_run_infos(experiment_id, view_type)]

        self.assertCountEqual([r1, r2], _runs(experiment_id, ViewType.ALL))
        self.assertCountEqual([r1, r2], _runs(experiment_id, ViewType.ACTIVE_ONLY))
        self.assertEqual(0, len(_runs(experiment_id, ViewType.DELETED_ONLY)))

        self.store.delete_run(r1)
        self.assertCountEqual([r1, r2], _runs(experiment_id, ViewType.ALL))
        self.assertCountEqual([r2], _runs(experiment_id, ViewType.ACTIVE_ONLY))
        self.assertCountEqual([r1], _runs(experiment_id, ViewType.DELETED_ONLY))

    def test_rename_experiment(self):
        new_name = "new name"
        experiment_id = self._experiment_factory("test name")
        self.store.rename_experiment(experiment_id, new_name)

        renamed_experiment = self.store.get_experiment(experiment_id)

        self.assertEqual(renamed_experiment.name, new_name)

    def test_update_run_info(self):
        experiment_id = self._experiment_factory("test_update_run_info")
        for new_status_string in models.RunStatusTypes:
            run = self._run_factory(config=self._get_run_configs(experiment_id=experiment_id))
            endtime = int(time.time())
            actual = self.store.update_run_info(
                run.info.run_id, RunStatus.from_string(new_status_string), endtime
            )
            self.assertEqual(actual.status, new_status_string)
            self.assertEqual(actual.end_time, endtime)

    def test_restore_experiment(self):
        experiment_id = self._experiment_factory("helloexp")
        exp = self.store.get_experiment(experiment_id)
        self.assertEqual(exp.lifecycle_stage, entities.LifecycleStage.ACTIVE)

        experiment_id = exp.experiment_id
        self.store.delete_experiment(experiment_id)

        deleted = self.store.get_experiment(experiment_id)
        self.assertEqual(deleted.experiment_id, experiment_id)
        self.assertEqual(deleted.lifecycle_stage, entities.LifecycleStage.DELETED)

        self.store.restore_experiment(exp.experiment_id)
        restored = self.store.get_experiment(exp.experiment_id)
        self.assertEqual(restored.experiment_id, experiment_id)
        self.assertEqual(restored.lifecycle_stage, entities.LifecycleStage.ACTIVE)

    def test_delete_restore_run(self):
        run = self._run_factory()
        self.assertEqual(run.info.lifecycle_stage, entities.LifecycleStage.ACTIVE)

        with self.assertRaises(MlflowException) as e:
            self.store.restore_run(run.info.run_id)
        self.assertIn("must be in the 'deleted' state", e.exception.message)

        self.store.delete_run(run.info.run_id)
        with self.assertRaises(MlflowException) as e:
            self.store.delete_run(run.info.run_id)
        self.assertIn("must be in the 'active' state", e.exception.message)

        deleted = self.store.get_run(run.info.run_id)
        self.assertEqual(deleted.info.run_id, run.info.run_id)
        self.assertEqual(deleted.info.lifecycle_stage, entities.LifecycleStage.DELETED)

        self.store.restore_run(run.info.run_id)
        with self.assertRaises(MlflowException) as e:
            self.store.restore_run(run.info.run_id)
            self.assertIn("must be in the 'deleted' state", e.exception.message)
        restored = self.store.get_run(run.info.run_id)
        self.assertEqual(restored.info.run_id, run.info.run_id)
        self.assertEqual(restored.info.lifecycle_stage, entities.LifecycleStage.ACTIVE)

    def test_error_logging_to_deleted_run(self):
        exp = self._experiment_factory("error_logging")
        run_id = self._run_factory(self._get_run_configs(experiment_id=exp)).info.run_id

        self.store.delete_run(run_id)
        self.assertEqual(
            self.store.get_run(run_id).info.lifecycle_stage, entities.LifecycleStage.DELETED
        )
        with self.assertRaises(MlflowException) as e:
            self.store.log_param(run_id, entities.Param("p1345", "v1"))
        self.assertIn("must be in the 'active' state", e.exception.message)

        with self.assertRaises(MlflowException) as e:
            self.store.log_metric(run_id, entities.Metric("m1345", 1.0, 123, 0))
        self.assertIn("must be in the 'active' state", e.exception.message)

        with self.assertRaises(MlflowException) as e:
            self.store.set_tag(run_id, entities.RunTag("t1345", "tv1"))
        self.assertIn("must be in the 'active' state", e.exception.message)

        # restore this run and try again
        self.store.restore_run(run_id)
        self.assertEqual(
            self.store.get_run(run_id).info.lifecycle_stage, entities.LifecycleStage.ACTIVE
        )
        self.store.log_param(run_id, entities.Param("p1345", "v22"))
        self.store.log_metric(run_id, entities.Metric("m1345", 34.0, 85, 1))  # earlier timestamp
        self.store.set_tag(run_id, entities.RunTag("t1345", "tv44"))

        run = self.store.get_run(run_id)
        self.assertEqual(run.data.params, {"p1345": "v22"})
        self.assertEqual(run.data.metrics, {"m1345": 34.0})
        metric_history = self.store.get_metric_history(run_id, "m1345")
        self.assertEqual(len(metric_history), 1)
        metric_obj = metric_history[0]
        self.assertEqual(metric_obj.key, "m1345")
        self.assertEqual(metric_obj.value, 34.0)
        self.assertEqual(metric_obj.timestamp, 85)
        self.assertEqual(metric_obj.step, 1)
        self.assertTrue(set([("t1345", "tv44")]) <= set(run.data.tags.items()))

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
                tags=[
                    entities.RunTag(mlflow_tags.MLFLOW_RUN_NAME, name),
                    entities.RunTag("metric", names[1]),
                ],
            ).info.run_id
            if names[0] is not None:
                self.store.log_metric(run_id, entities.Metric("x", float(names[0]), 1, 0))
                self.store.log_metric(run_id, entities.Metric("y", float(names[1]), 1, 0))
            self.store.log_param(run_id, entities.Param("metric", names[1]))
            return run_id

        # the expected order in ascending sort is :
        # inf > number > -inf > None > nan
        for names in zip(
            ["nan", None, "inf", "-inf", "-1000", "0", "0", "1000"],
            ["1", "2", "3", "4", "5", "6", "7", "8"],
        ):
            create_and_log_run(names)

        # asc/asc
        self.assertListEqual(
            ["-inf/4", "-1000/5", "0/6", "0/7", "1000/8", "inf/3", "None/2", "nan/1"],
            self.get_ordered_runs(["metrics.x asc", "metrics.y asc"], experiment_id),
        )

        self.assertListEqual(
            ["-inf/4", "-1000/5", "0/6", "0/7", "1000/8", "inf/3", "None/2", "nan/1"],
            self.get_ordered_runs(["metrics.x asc", "tag.metric asc"], experiment_id),
        )

        # asc/desc
        self.assertListEqual(
            ["-inf/4", "-1000/5", "0/7", "0/6", "1000/8", "inf/3", "None/2", "nan/1"],
            self.get_ordered_runs(["metrics.x asc", "metrics.y desc"], experiment_id),
        )

        self.assertListEqual(
            ["-inf/4", "-1000/5", "0/7", "0/6", "1000/8", "inf/3", "None/2", "nan/1"],
            self.get_ordered_runs(["metrics.x asc", "tag.metric desc"], experiment_id),
        )

        # desc / asc
        self.assertListEqual(
            ["inf/3", "1000/8", "0/6", "0/7", "-1000/5", "-inf/4", "nan/1", "None/2"],
            self.get_ordered_runs(["metrics.x desc", "metrics.y asc"], experiment_id),
        )

        # desc / desc
        self.assertListEqual(
            ["inf/3", "1000/8", "0/7", "0/6", "-1000/5", "-inf/4", "nan/1", "None/2"],
            self.get_ordered_runs(["metrics.x desc", "param.metric desc"], experiment_id),
        )

    def test_order_by_attributes(self):
        experiment_id = self.store.create_experiment("order_by_attributes")

        def create_run(start_time, end):
            return self.store.create_run(
                experiment_id,
                user_id="MrDuck",
                start_time=start_time,
                tags=[entities.RunTag(mlflow_tags.MLFLOW_RUN_NAME, end)],
            ).info.run_id

        start_time = 123
        for end in [234, None, 456, -123, 789, 123]:
            run_id = create_run(start_time, end)
            self.store.update_run_info(run_id, run_status=RunStatus.FINISHED, end_time=end)
            start_time += 1

        # asc
        self.assertListEqual(
            ["-123", "123", "234", "456", "789", None],
            self.get_ordered_runs(["attribute.end_time asc"], experiment_id),
        )

        # desc
        self.assertListEqual(
            ["789", "456", "234", "123", "-123", None],
            self.get_ordered_runs(["attribute.end_time desc"], experiment_id),
        )

        # Sort priority correctly handled
        self.assertListEqual(
            ["234", None, "456", "-123", "789", "123"],
            self.get_ordered_runs(
                ["attribute.start_time asc", "attribute.end_time desc"], experiment_id
            ),
        )

    def test_search_vanilla(self):
        exp = self._experiment_factory("search_vanilla")
        runs = [self._run_factory(self._get_run_configs(exp)).info.run_id for r in range(3)]

        self.assertCountEqual(runs, self._search(exp, run_view_type=ViewType.ALL))
        self.assertCountEqual(runs, self._search(exp, run_view_type=ViewType.ACTIVE_ONLY))
        self.assertCountEqual([], self._search(exp, run_view_type=ViewType.DELETED_ONLY))

        first = runs[0]

        self.store.delete_run(first)
        self.assertCountEqual(runs, self._search(exp, run_view_type=ViewType.ALL))
        self.assertCountEqual(runs[1:], self._search(exp, run_view_type=ViewType.ACTIVE_ONLY))
        self.assertCountEqual([first], self._search(exp, run_view_type=ViewType.DELETED_ONLY))

        self.store.restore_run(first)
        self.assertCountEqual(runs, self._search(exp, run_view_type=ViewType.ALL))
        self.assertCountEqual(runs, self._search(exp, run_view_type=ViewType.ACTIVE_ONLY))
        self.assertCountEqual([], self._search(exp, run_view_type=ViewType.DELETED_ONLY))

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
        self.assertCountEqual([r1, r2], self._search(experiment_id, filter_string))

        # test search returns appropriate run (same key different values per run)
        filter_string = "params.generic_2 = 'some value'"
        self.assertCountEqual([r1], self._search(experiment_id, filter_string))
        filter_string = "params.generic_2 = 'another value'"
        self.assertCountEqual([r2], self._search(experiment_id, filter_string))

        filter_string = "params.generic_param = 'wrong_val'"
        self.assertCountEqual([], self._search(experiment_id, filter_string))

        filter_string = "params.generic_param != 'p_val'"
        self.assertCountEqual([], self._search(experiment_id, filter_string))

        filter_string = "params.generic_param != 'wrong_val'"
        self.assertCountEqual([r1, r2], self._search(experiment_id, filter_string))
        filter_string = "params.generic_2 != 'wrong_val'"
        self.assertCountEqual([r1, r2], self._search(experiment_id, filter_string))

        filter_string = "params.p_a = 'abc'"
        self.assertCountEqual([r1], self._search(experiment_id, filter_string))

        filter_string = "params.p_b = 'ABC'"
        self.assertCountEqual([r2], self._search(experiment_id, filter_string))

        filter_string = "params.generic_2 LIKE '%other%'"
        self.assertCountEqual([r2], self._search(experiment_id, filter_string))

        filter_string = "params.generic_2 LIKE 'other%'"
        self.assertCountEqual([], self._search(experiment_id, filter_string))

        filter_string = "params.generic_2 LIKE '%other'"
        self.assertCountEqual([], self._search(experiment_id, filter_string))

        filter_string = "params.generic_2 LIKE 'other'"
        self.assertCountEqual([], self._search(experiment_id, filter_string))

        filter_string = "params.generic_2 LIKE '%Other%'"
        self.assertCountEqual([], self._search(experiment_id, filter_string))

        filter_string = "params.generic_2 ILIKE '%Other%'"
        self.assertCountEqual([r2], self._search(experiment_id, filter_string))

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
        self.assertCountEqual(
            [r1, r2], self._search(experiment_id, filter_string="tags.generic_tag = 'p_val'")
        )
        # test search returns appropriate run (same key different values per run)
        self.assertCountEqual(
            [r1], self._search(experiment_id, filter_string="tags.generic_2 = 'some value'")
        )
        self.assertCountEqual(
            [r2],
            self._search(experiment_id, filter_string="tags.generic_2 = 'another value'"),
        )
        self.assertCountEqual(
            [], self._search(experiment_id, filter_string="tags.generic_tag = 'wrong_val'")
        )
        self.assertCountEqual(
            [], self._search(experiment_id, filter_string="tags.generic_tag != 'p_val'")
        )
        self.assertCountEqual(
            [r1, r2],
            self._search(experiment_id, filter_string="tags.generic_tag != 'wrong_val'"),
        )
        self.assertCountEqual(
            [r1, r2],
            self._search(experiment_id, filter_string="tags.generic_2 != 'wrong_val'"),
        )
        self.assertCountEqual([r1], self._search(experiment_id, filter_string="tags.p_a = 'abc'"))
        self.assertCountEqual([r2], self._search(experiment_id, filter_string="tags.p_b = 'ABC'"))
        self.assertCountEqual(
            [r2], self._search(experiment_id, filter_string="tags.generic_2 LIKE '%other%'")
        )
        self.assertCountEqual(
            [], self._search(experiment_id, filter_string="tags.generic_2 LIKE '%Other%'")
        )
        self.assertCountEqual(
            [], self._search(experiment_id, filter_string="tags.generic_2 LIKE 'other%'")
        )
        self.assertCountEqual(
            [], self._search(experiment_id, filter_string="tags.generic_2 LIKE '%other'")
        )
        self.assertCountEqual(
            [], self._search(experiment_id, filter_string="tags.generic_2 LIKE 'other'")
        )
        self.assertCountEqual(
            [r2], self._search(experiment_id, filter_string="tags.generic_2 ILIKE '%Other%'")
        )
        self.assertCountEqual(
            [r2],
            self._search(
                experiment_id,
                filter_string="tags.generic_2 ILIKE '%Other%' " "and tags.generic_tag = 'p_val'",
            ),
        )
        self.assertCountEqual(
            [r2],
            self._search(
                experiment_id,
                filter_string="tags.generic_2 ILIKE '%Other%' and "
                "tags.generic_tag ILIKE 'p_val'",
            ),
        )

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
        self.assertCountEqual([r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.common > 0.0"
        self.assertCountEqual([r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.common >= 0.0"
        self.assertCountEqual([r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.common < 4.0"
        self.assertCountEqual([r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.common <= 4.0"
        self.assertCountEqual([r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.common != 1.0"
        self.assertCountEqual([], self._search(experiment_id, filter_string))

        filter_string = "metrics.common >= 3.0"
        self.assertCountEqual([], self._search(experiment_id, filter_string))

        filter_string = "metrics.common <= 0.75"
        self.assertCountEqual([], self._search(experiment_id, filter_string))

        # tests for same metric name across runs with different values and timestamps
        filter_string = "metrics.measure_a > 0.0"
        self.assertCountEqual([r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a < 50.0"
        self.assertCountEqual([r1], self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a < 1000.0"
        self.assertCountEqual([r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a != -12.0"
        self.assertCountEqual([r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a > 50.0"
        self.assertCountEqual([r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a = 1.0"
        self.assertCountEqual([r1], self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a = 400.0"
        self.assertCountEqual([r2], self._search(experiment_id, filter_string))

        # test search with unique metric keys
        filter_string = "metrics.m_a > 1.0"
        self.assertCountEqual([r1], self._search(experiment_id, filter_string))

        filter_string = "metrics.m_b > 1.0"
        self.assertCountEqual([r2], self._search(experiment_id, filter_string))

        # there is a recorded metric this threshold but not last timestamp
        filter_string = "metrics.m_b > 5.0"
        self.assertCountEqual([], self._search(experiment_id, filter_string))

        # metrics matches last reported timestamp for 'm_b'
        filter_string = "metrics.m_b = 4.0"
        self.assertCountEqual([r2], self._search(experiment_id, filter_string))

    def test_search_attrs(self):
        e1 = self._experiment_factory("search_attributes_1")
        r1 = self._run_factory(self._get_run_configs(experiment_id=e1)).info.run_id

        e2 = self._experiment_factory("search_attrs_2")
        r2 = self._run_factory(self._get_run_configs(experiment_id=e2)).info.run_id

        filter_string = ""
        self.assertCountEqual([r1, r2], self._search([e1, e2], filter_string))

        filter_string = "attribute.status != 'blah'"
        self.assertCountEqual([r1, r2], self._search([e1, e2], filter_string))

        filter_string = "attribute.status = '{}'".format(RunStatus.to_string(RunStatus.RUNNING))
        self.assertCountEqual([r1, r2], self._search([e1, e2], filter_string))

        # change status for one of the runs
        self.store.update_run_info(r2, RunStatus.FAILED, 300)

        filter_string = "attribute.status = 'RUNNING'"
        self.assertCountEqual([r1], self._search([e1, e2], filter_string))

        filter_string = "attribute.status = 'FAILED'"
        self.assertCountEqual([r2], self._search([e1, e2], filter_string))

        filter_string = "attribute.status != 'SCHEDULED'"
        self.assertCountEqual([r1, r2], self._search([e1, e2], filter_string))

        filter_string = "attribute.status = 'SCHEDULED'"
        self.assertCountEqual([], self._search([e1, e2], filter_string))

        filter_string = "attribute.status = 'KILLED'"
        self.assertCountEqual([], self._search([e1, e2], filter_string))

        filter_string = "attr.artifact_uri = '{}/{}/{}/artifacts'".format(ARTIFACT_URI, e1, r1)
        self.assertCountEqual([r1], self._search([e1, e2], filter_string))

        filter_string = "attr.artifact_uri = '{}/{}/{}/artifacts'".format(ARTIFACT_URI, e2, r1)
        self.assertCountEqual([], self._search([e1, e2], filter_string))

        filter_string = "attribute.artifact_uri = 'random_artifact_path'"
        self.assertCountEqual([], self._search([e1, e2], filter_string))

        filter_string = "attribute.artifact_uri != 'random_artifact_path'"
        self.assertCountEqual([r1, r2], self._search([e1, e2], filter_string))

        filter_string = "attribute.artifact_uri LIKE '%{}%'".format(r1)
        self.assertCountEqual([r1], self._search([e1, e2], filter_string))

        filter_string = "attribute.artifact_uri LIKE '%{}%'".format(r1[:16])
        self.assertCountEqual([r1], self._search([e1, e2], filter_string))

        filter_string = "attribute.artifact_uri LIKE '%{}%'".format(r1[-16:])
        self.assertCountEqual([r1], self._search([e1, e2], filter_string))

        filter_string = "attribute.artifact_uri LIKE '%{}%'".format(r1.upper())
        self.assertCountEqual([], self._search([e1, e2], filter_string))

        filter_string = "attribute.artifact_uri ILIKE '%{}%'".format(r1.upper())
        self.assertCountEqual([r1], self._search([e1, e2], filter_string))

        filter_string = "attribute.artifact_uri ILIKE '%{}%'".format(r1[:16].upper())
        self.assertCountEqual([r1], self._search([e1, e2], filter_string))

        filter_string = "attribute.artifact_uri ILIKE '%{}%'".format(r1[-16:].upper())
        self.assertCountEqual([r1], self._search([e1, e2], filter_string))

        for (k, v) in {
            "experiment_id": e1,
            "lifecycle_stage": "ACTIVE",
            "run_id": r1,
            "run_uuid": r2,
        }.items():
            with self.assertRaises(MlflowException) as e:
                self._search([e1, e2], "attribute.{} = '{}'".format(k, v))
            self.assertIn("Invalid attribute key", e.exception.message)

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
        self.assertCountEqual([r1, r2], self._search(experiment_id, filter_string))

        # all params and metrics match
        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 and metrics.m_a > 1.0"
        )
        self.assertCountEqual([r1], self._search(experiment_id, filter_string))

        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 "
            "and metrics.m_a > 1.0 and params.p_a LIKE 'a%'"
        )
        self.assertCountEqual([r1], self._search(experiment_id, filter_string))

        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 "
            "and metrics.m_a > 1.0 and params.p_a LIKE 'A%'"
        )
        self.assertCountEqual([], self._search(experiment_id, filter_string))

        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 "
            "and metrics.m_a > 1.0 and params.p_a ILIKE 'A%'"
        )
        self.assertCountEqual([r1], self._search(experiment_id, filter_string))

        # test with mismatch param
        filter_string = (
            "params.random_bad_name = 'p_val' and metrics.common = 1.0 and metrics.m_a > 1.0"
        )
        self.assertCountEqual([], self._search(experiment_id, filter_string))

        # test with mismatch metric
        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 and metrics.m_a > 100.0"
        )
        self.assertCountEqual([], self._search(experiment_id, filter_string))

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
            assert runs[: min(1200, n)] == self._search(exp, max_results=n)

        with self.assertRaises(MlflowException) as e:
            self._search(exp, max_results=int(1e10))
        self.assertIn("Invalid value for request parameter max_results. It ", e.exception.message)

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

    def test_log_batch(self):
        experiment_id = self._experiment_factory("log_batch")
        run_id = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        metric_entities = [Metric("m1", 0.87, 12345, 0), Metric("m2", 0.49, 12345, 1)]
        param_entities = [Param("p1", "p1val"), Param("p2", "p2val")]
        tag_entities = [RunTag("t1", "t1val"), RunTag("t2", "t2val")]
        self.store.log_batch(
            run_id=run_id, metrics=metric_entities, params=param_entities, tags=tag_entities
        )
        run = self.store.get_run(run_id)
        assert run.data.tags == {"t1": "t1val", "t2": "t2val"}
        assert run.data.params == {"p1": "p1val", "p2": "p2val"}
        metric_histories = sum(
            [self.store.get_metric_history(run_id, key) for key in run.data.metrics], []
        )
        metrics = [(m.key, m.value, m.timestamp, m.step) for m in metric_histories]
        assert set(metrics) == set([("m1", 0.87, 12345, 0), ("m2", 0.49, 12345, 1)])

    def test_log_batch_limits(self):
        # Test that log batch at the maximum allowed request size succeeds (i.e doesn't hit
        # SQL limitations, etc)
        experiment_id = self._experiment_factory("log_batch_limits")
        run_id = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        metric_tuples = [("m%s" % i, i, 12345, i * 2) for i in range(1000)]
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
        with self.assertRaises(MlflowException) as e:
            self.store.log_batch(
                run.info.run_id, metrics=[metric], params=[overwrite_param], tags=[tag]
            )
        self.assertIn("Changing param values is not allowed. Param with key=", e.exception.message)
        assert e.exception.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        self._verify_logged(self.store, run.info.run_id, metrics=[], params=[param], tags=[])

    def test_log_batch_param_overwrite_disallowed_single_req(self):
        # Test that attempting to overwrite a param via log_batch results in an exception
        run = self._run_factory()
        pkey = "common-key"
        param0 = entities.Param(pkey, "orig-val")
        param1 = entities.Param(pkey, "newval")
        tag = entities.RunTag("tag-key", "tag-val")
        metric = entities.Metric("metric-key", 3.0, 12345, 0)
        with self.assertRaises(MlflowException) as e:
            self.store.log_batch(
                run.info.run_id, metrics=[metric], params=[param0, param1], tags=[tag]
            )
        self.assertIn("Changing param values is not allowed. Param with key=", e.exception.message)
        assert e.exception.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        self._verify_logged(self.store, run.info.run_id, metrics=[], params=[param0], tags=[])

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
        with mock.patch(package + ".log_metric") as metric_mock, mock.patch(
            package + ".log_param"
        ) as param_mock, mock.patch(package + ".set_tag") as tags_mock:
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
                with self.assertRaises(MlflowException) as e:
                    self.store.log_batch(run.info.run_id, **log_batch_kwargs)
                self.assertIn(str(e.exception.message), "Some internal error")

    def test_log_batch_nonexistent_run(self):
        nonexistent_run_id = uuid.uuid4().hex
        with self.assertRaises(MlflowException) as e:
            self.store.log_batch(nonexistent_run_id, [], [], [])
        assert e.exception.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
        assert "Run with id=%s not found" % nonexistent_run_id in e.exception.message

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

    def test_upgrade_cli_idempotence(self):
        # Repeatedly run `mlflow db upgrade` against our database, verifying that the command
        # succeeds and that the DB has the latest schema
        engine = sqlalchemy.create_engine(self.db_url)
        assert _get_schema_version(engine) == _get_latest_schema_revision()
        for _ in range(3):
            invoke_cli_runner(mlflow.db.commands, ["upgrade", self.db_url])
            assert _get_schema_version(engine) == _get_latest_schema_revision()

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

        >>> import json
        >>> import mlflow
        >>> from mlflow.tracking.client import MlflowClient
        >>> mlflow.set_tracking_uri(
        ...     "sqlite:///../../resources/db/db_version_7ac759974ad8_with_metrics.sql")
        >>> client = MlflowClient()
        >>> summary_metrics = {
        ...     run.info.run_id: run.data.metrics for run
        ...     in client.search_runs(experiment_ids="0")
        ... }
        >>> with open("dump.json", "w") as dump_file:
        >>>     json.dump(summary_metrics, dump_file, indent=4)
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
            shutil.copyfile(
                src=os.path.join(db_resources_path, "db_version_7ac759974ad8_with_metrics.sql"),
                dst=db_path,
            )

            invoke_cli_runner(mlflow.db.commands, ["upgrade", db_url])
            store = self._get_store(db_uri=db_url)
            with open(expected_metric_values_path, "r") as f:
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
                experiment_id=experiment_id, start_time=current_run, tags=(), user_id="Anderson"
            ).info.run_uuid

            run_ids.append(run_id)

            for i in range(100):
                metric = {
                    "key": "mkey_%s" % i,
                    "value": i,
                    "timestamp": i * 2,
                    "step": i * 3,
                    "is_nan": 0,
                    "run_uuid": run_id,
                }
                metrics_list.append(metric)
                tag = {
                    "key": "tkey_%s" % i,
                    "value": "tval_%s" % (current_run % 10),
                    "run_uuid": run_id,
                }
                tags_list.append(tag)
                param = {
                    "key": "pkey_%s" % i,
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
                    "is_nan": 0,
                    "run_uuid": run_id,
                }
            )
            current_run += 1
        metrics = pd.DataFrame(metrics_list)
        metrics.to_sql("metrics", self.store.engine, if_exists="append", index=False)
        params = pd.DataFrame(params_list)
        params.to_sql("params", self.store.engine, if_exists="append", index=False)
        tags = pd.DataFrame(tags_list)
        tags.to_sql("tags", self.store.engine, if_exists="append", index=False)
        pd.DataFrame(latest_metrics_list).to_sql(
            "latest_metrics", self.store.engine, if_exists="append", index=False
        )
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
        self.assertListEqual(
            [run.info.run_id for run in run_results], list(reversed(run_ids[900:]))
        )

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
            "metrics.mkey_0 < 26 and metrics.mkey_0 > 5 " "and tags.tkey_0 = 'tval_0' ",
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
            experiment_id=experiment_id, start_time=0, tags=(), user_id="Me"
        ).info.run_uuid
        r2 = self.store.create_run(
            experiment_id=experiment_id, start_time=0, tags=(), user_id="Me"
        ).info.run_uuid
        self.store.set_tag(r1, RunTag(key="t1", value="1"))
        self.store.set_tag(r1, RunTag(key="t2", value="1"))
        self.store.set_tag(r2, RunTag(key="t2", value="1"))

        run_results = self.store.search_runs(
            [experiment_id], None, ViewType.ALL, max_results=1000, order_by=["tag.t1"]
        )
        assert len(run_results) == 2


def test_sqlalchemy_store_behaves_as_expected_with_inmemory_sqlite_db():
    store = SqlAlchemyStore("sqlite:///:memory:", ARTIFACT_URI)
    experiment_id = store.create_experiment(name="exp1")
    run = store.create_run(experiment_id=experiment_id, user_id="user", start_time=0, tags=[])
    run_id = run.info.run_id
    metric = entities.Metric("mymetric", 1, 0, 0)
    store.log_metric(run_id=run_id, metric=metric)
    param = entities.Param("myparam", "A")
    store.log_param(run_id=run_id, param=param)
    fetched_run = store.get_run(run_id=run_id)
    assert fetched_run.info.run_id == run_id
    assert metric.key in fetched_run.data.metrics
    assert param.key in fetched_run.data.params


def test_sqlalchemy_store_can_be_initialized_when_default_experiment_has_been_deleted(tmpdir):
    db_uri = "sqlite:///{}/mlflow.db".format(tmpdir.strpath)
    store = SqlAlchemyStore(db_uri, ARTIFACT_URI)
    store.delete_experiment("0")
    assert store.get_experiment("0").lifecycle_stage == entities.LifecycleStage.DELETED
    SqlAlchemyStore(db_uri, ARTIFACT_URI)


class TestSqlAlchemyStoreSqliteMigratedDB(TestSqlAlchemyStoreSqlite):
    """
    Test case where user has an existing DB with schema generated before MLflow 1.0,
    then migrates their DB.
    """

    def setUp(self):
        fd, self.temp_dbfile = tempfile.mkstemp()
        os.close(fd)
        self.db_url = "%s%s" % (DB_URI, self.temp_dbfile)
        engine = sqlalchemy.create_engine(self.db_url)
        InitialBase.metadata.create_all(engine)
        invoke_cli_runner(mlflow.db.commands, ["upgrade", self.db_url])
        self.store = SqlAlchemyStore(self.db_url, ARTIFACT_URI)

    def tearDown(self):
        os.remove(self.temp_dbfile)


@mock.patch("sqlalchemy.orm.session.Session", spec=True)
class TestZeroValueInsertion(unittest.TestCase):
    def test_set_zero_value_insertion_for_autoincrement_column_MYSQL(self, mock_session):
        mock_store = mock.Mock(SqlAlchemyStore)
        mock_store.db_type = MYSQL
        SqlAlchemyStore._set_zero_value_insertion_for_autoincrement_column(mock_store, mock_session)
        mock_session.execute.assert_called_with("SET @@SESSION.sql_mode='NO_AUTO_VALUE_ON_ZERO';")

    def test_set_zero_value_insertion_for_autoincrement_column_MSSQL(self, mock_session):
        mock_store = mock.Mock(SqlAlchemyStore)
        mock_store.db_type = MSSQL
        SqlAlchemyStore._set_zero_value_insertion_for_autoincrement_column(mock_store, mock_session)
        mock_session.execute.assert_called_with("SET IDENTITY_INSERT experiments ON;")

    def test_unset_zero_value_insertion_for_autoincrement_column_MYSQL(self, mock_session):
        mock_store = mock.Mock(SqlAlchemyStore)
        mock_store.db_type = MYSQL
        SqlAlchemyStore._unset_zero_value_insertion_for_autoincrement_column(
            mock_store, mock_session
        )
        mock_session.execute.assert_called_with("SET @@SESSION.sql_mode='';")

    def test_unset_zero_value_insertion_for_autoincrement_column_MSSQL(self, mock_session):
        mock_store = mock.Mock(SqlAlchemyStore)
        mock_store.db_type = MSSQL
        SqlAlchemyStore._unset_zero_value_insertion_for_autoincrement_column(
            mock_store, mock_session
        )
        mock_session.execute.assert_called_with("SET IDENTITY_INSERT experiments OFF;")


def test_get_attribute_name():
    assert models.SqlRun.get_attribute_name("artifact_uri") == "artifact_uri"
    assert models.SqlRun.get_attribute_name("status") == "status"
    assert models.SqlRun.get_attribute_name("start_time") == "start_time"
    assert models.SqlRun.get_attribute_name("end_time") == "end_time"

    # we want this to break if a searchable or orderable attribute has been added
    # and not referred to in this test
    # searchable attibutes are also orderable
    assert len(entities.RunInfo.get_orderable_attributes()) == 4


def test_get_orderby_clauses():
    store = SqlAlchemyStore("sqlite:///:memory:", ARTIFACT_URI)
    with store.ManagedSessionMaker() as session:
        # test that ['runs.start_time DESC', 'SqlRun.run_uuid'] is returned by default
        parsed = [str(x) for x in _get_orderby_clauses([], session)[0]]
        assert parsed == ["runs.start_time DESC", "SqlRun.run_uuid"]

        # test that the given 'start_time' replaces the default one ('runs.start_time DESC')
        parsed = [str(x) for x in _get_orderby_clauses(["attribute.start_time ASC"], session)[0]]
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

        # test that "=" is used rather than "is" when comparing to True
        parsed = [str(x) for x in _get_orderby_clauses(["metric.a"], session)[0]]
        assert "is_nan = true" in parsed[0]
        assert "value IS NULL" in parsed[0]
