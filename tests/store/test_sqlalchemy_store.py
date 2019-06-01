import os
import shutil
import six
import tempfile
import unittest
import warnings

import mock
import pytest
import sqlalchemy
import time
import mlflow
import uuid

import mlflow.db
from mlflow.entities import ViewType, RunTag, SourceType, RunStatus, Experiment, Metric, Param
from mlflow.protos.databricks_pb2 import ErrorCode, RESOURCE_DOES_NOT_EXIST,\
    INVALID_PARAMETER_VALUE, INTERNAL_ERROR
from mlflow.store import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.db.utils import _get_schema_version
from mlflow.store.dbmodels import models
from mlflow import entities
from mlflow.exceptions import MlflowException
from mlflow.store.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils import extract_db_type_from_uri
from mlflow.utils.search_utils import SearchFilter
from tests.resources.db.initial_models import Base as InitialBase
from tests.integration.utils import invoke_cli_runner


DB_URI = 'sqlite:///'
ARTIFACT_URI = 'artifact_folder'


class TestParseDbUri(unittest.TestCase):

    def test_correct_db_type_from_uri(self):
        # try each the main drivers per supported database type
        target_db_type_uris = {
            'sqlite': ('pysqlite', 'pysqlcipher'),
            'postgresql': ('psycopg2', 'pg8000', 'psycopg2cffi',
                           'pypostgresql', 'pygresql', 'zxjdbc'),
            'mysql': ('mysqldb', 'pymysql', 'mysqlconnector', 'cymysql',
                      'oursql', 'mysqldb', 'gaerdbms', 'pyodbc', 'zxjdbc'),
            'mssql': ('pyodbc', 'mxodbc', 'pymssql', 'zxjdbc', 'adodbapi')
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
        bad_db_uri_strings = ['oracle://...', 'oracle+cx_oracle://...',
                              'snowflake://...', '://...', 'abcdefg']
        self._db_uri_error(bad_db_uri_strings, "Supported database engines are ")

    def test_fail_on_multiple_drivers(self):
        bad_db_uri_strings = ['mysql+pymsql+pyodbc://...']
        self._db_uri_error(bad_db_uri_strings,
                           "mlflow.org/docs/latest/tracking.html#storage for format specifications")


class TestSqlAlchemyStoreSqlite(unittest.TestCase):

    def _get_store(self, db_uri=''):
        return SqlAlchemyStore(db_uri, ARTIFACT_URI)

    def setUp(self):
        self.maxDiff = None  # print all differences on assert failures
        fd, self.temp_dbfile = tempfile.mkstemp()
        # Close handle immediately so that we can remove the file later on in Windows
        os.close(fd)
        self.db_url = "%s%s" % (DB_URI, self.temp_dbfile)
        self.store = self._get_store(self.db_url)

    def tearDown(self):
        models.Base.metadata.drop_all(self.store.engine)
        os.remove(self.temp_dbfile)
        shutil.rmtree(ARTIFACT_URI)

    def _experiment_factory(self, names):
        if type(names) is list:
            return [self.store.create_experiment(name=name) for name in names]

        return self.store.create_experiment(name=names)

    def _verify_logged(self, run_id, metrics, params, tags):
        run = self.store.get_run(run_id)
        all_metrics = sum([self.store.get_metric_history(run_id, key)
                           for key in run.data.metrics], [])
        assert len(all_metrics) == len(metrics)
        logged_metrics = [(m.key, m.value, m.timestamp, m.step) for m in all_metrics]
        assert set(logged_metrics) == set([(m.key, m.value, m.timestamp, m.step) for m in metrics])
        logged_tags = set([(tag_key, tag_value) for tag_key, tag_value in run.data.tags.items()])
        assert set([(tag.key, tag.value) for tag in tags]) <= logged_tags
        assert len(run.data.params) == len(params)
        logged_params = [(param_key, param_val) for param_key, param_val in run.data.params.items()]
        assert set(logged_params) == set([(param.key, param.value) for param in params])

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

        self._experiment_factory('aNothEr')
        all_experiments = [e.name for e in self.store.list_experiments()]
        six.assertCountEqual(self, set(['aNothEr', 'Default']), set(all_experiments))

        self.store.delete_experiment(0)

        six.assertCountEqual(self, ['aNothEr'], [e.name for e in self.store.list_experiments()])
        another = self.store.get_experiment(1)
        self.assertEqual('aNothEr', another.name)

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

        six.assertCountEqual(self, ['aNothEr'], [e.name for e in self.store.list_experiments()])
        all_experiments = [e.name for e in self.store.list_experiments(ViewType.ALL)]
        six.assertCountEqual(self, set(['aNothEr', 'Default']), set(all_experiments))

        # ensure that experiment ID dor active experiment is unchanged
        another = self.store.get_experiment(1)
        self.assertEqual('aNothEr', another.name)

    def test_raise_duplicate_experiments(self):
        with self.assertRaises(Exception):
            self._experiment_factory(['test', 'test'])

    def test_raise_experiment_dont_exist(self):
        with self.assertRaises(Exception):
            self.store.get_experiment(experiment_id=100)

    def test_delete_experiment(self):
        experiments = self._experiment_factory(['morty', 'rick', 'rick and morty'])

        all_experiments = self.store.list_experiments()
        self.assertEqual(len(all_experiments), len(experiments) + 1)  # default

        exp_id = experiments[0]
        self.store.delete_experiment(exp_id)

        updated_exp = self.store.get_experiment(exp_id)
        self.assertEqual(updated_exp.lifecycle_stage, entities.LifecycleStage.DELETED)

        self.assertEqual(len(self.store.list_experiments()), len(all_experiments) - 1)

    def test_get_experiment(self):
        name = 'goku'
        experiment_id = self._experiment_factory(name)
        actual = self.store.get_experiment(experiment_id)
        self.assertEqual(actual.name, name)
        self.assertEqual(actual.experiment_id, experiment_id)

        actual_by_name = self.store.get_experiment_by_name(name)
        self.assertEqual(actual_by_name.name, name)
        self.assertEqual(actual_by_name.experiment_id, experiment_id)

    def test_list_experiments(self):
        testnames = ['blue', 'red', 'green']

        experiments = self._experiment_factory(testnames)
        actual = self.store.list_experiments()

        self.assertEqual(len(experiments) + 1, len(actual))  # default

        with self.store.ManagedSessionMaker() as session:
            for experiment_id in experiments:
                res = session.query(models.SqlExperiment).filter_by(
                    experiment_id=experiment_id).first()
                self.assertIn(res.name, testnames)
                self.assertEqual(str(res.experiment_id), experiment_id)

    def test_create_experiments(self):
        with self.store.ManagedSessionMaker() as session:
            result = session.query(models.SqlExperiment).all()
            self.assertEqual(len(result), 1)

        experiment_id = self.store.create_experiment(name='test exp')
        self.assertEqual(experiment_id, "1")
        with self.store.ManagedSessionMaker() as session:
            result = session.query(models.SqlExperiment).all()
            self.assertEqual(len(result), 2)

            test_exp = session.query(models.SqlExperiment).filter_by(name='test exp').first()
            self.assertEqual(str(test_exp.experiment_id), experiment_id)
            self.assertEqual(test_exp.name, 'test exp')

        actual = self.store.get_experiment(experiment_id)
        self.assertEqual(actual.experiment_id, experiment_id)
        self.assertEqual(actual.name, 'test exp')

    def test_run_tag_model(self):
        # Create a run whose UUID we can reference when creating tag models.
        # `run_id` is a foreign key in the tags table; therefore, in order
        # to insert a tag with a given run UUID, the UUID must be present in
        # the runs table
        run = self._run_factory()
        with self.store.ManagedSessionMaker() as session:
            new_tag = models.SqlTag(run_uuid=run.info.run_id, key='test', value='val')
            session.add(new_tag)
            session.commit()
            added_tags = [
                tag for tag in session.query(models.SqlTag).all()
                if tag.key == new_tag.key
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
            new_metric = models.SqlMetric(run_uuid=run.info.run_id, key='accuracy', value=0.89)
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
                run_uuid=run.info.run_id, key='accuracy', value='test param')
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
            m1 = models.SqlMetric(key='accuracy', value=0.89)
            m2 = models.SqlMetric(key='recal', value=0.89)
            p1 = models.SqlParam(key='loss', value='test param')
            p2 = models.SqlParam(key='blue', value='test param')

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
        experiment_id = self._experiment_factory('test exp')
        config = {
            'experiment_id': experiment_id,
            'name': 'test run',
            'user_id': 'Anderson',
            'run_uuid': 'test',
            'status': RunStatus.to_string(RunStatus.SCHEDULED),
            'source_type': SourceType.to_string(SourceType.LOCAL),
            'source_name': 'Python application',
            'entry_point_name': 'main.py',
            'start_time': int(time.time()),
            'end_time': int(time.time()),
            'source_version': mlflow.__version__,
            'lifecycle_stage': entities.LifecycleStage.ACTIVE,
            'artifact_uri': '//'
        }
        run = models.SqlRun(**config).to_mlflow_entity()

        for k, v in config.items():
            # These keys were removed from RunInfo.
            if k in ['source_name', 'source_type', 'source_version', 'name', 'entry_point_name']:
                continue

            v2 = getattr(run.info, k)
            if k == 'source_type':
                self.assertEqual(v, SourceType.to_string(v2))
            else:
                self.assertEqual(v, v2)

    def _get_run_configs(self, experiment_id=None, tags=(), start_time=None):
        return {
            'experiment_id': experiment_id,
            'user_id': 'Anderson',
            'start_time': start_time if start_time is not None else int(time.time()),
            'tags': tags
        }

    def _run_factory(self, config=None):
        if not config:
            config = self._get_run_configs()

        experiment_id = config.get("experiment_id", None)
        if not experiment_id:
            experiment_id = self._experiment_factory('test exp')
            config["experiment_id"] = experiment_id

        return self.store.create_run(**config)

    def test_create_run_with_tags(self):
        experiment_id = self._experiment_factory('test_create_run')
        tags = [RunTag('3', '4'), RunTag('1', '2')]
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
            run_id=run_id,
            metric=entities.Metric(key='my-metric', value=3.4, timestamp=0, step=0))
        self.store.log_param(run_id=run_id, param=Param(key='my-param', value='param-val'))
        self.store.set_tag(run_id=run_id, tag=RunTag(key='my-tag', value='tag-val'))

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

    def test_log_metric(self):
        run = self._run_factory()

        tkey = 'blahmetric'
        tval = 100.0
        metric = entities.Metric(tkey, tval, int(time.time()), 0)
        metric2 = entities.Metric(tkey, tval, int(time.time()) + 2, 0)
        self.store.log_metric(run.info.run_id, metric)
        self.store.log_metric(run.info.run_id, metric2)

        run = self.store.get_run(run.info.run_id)
        self.assertTrue(tkey in run.data.metrics and run.data.metrics[tkey] == tval)

        # SQL store _get_run method returns full history of recorded metrics.
        # Should return duplicates as well
        # MLflow RunData contains only the last reported values for metrics.
        with self.store.ManagedSessionMaker() as session:
            sql_run_metrics = self.store._get_run(session, run.info.run_id).metrics
            self.assertEqual(2, len(sql_run_metrics))
            self.assertEqual(1, len(run.data.metrics))

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

        tkey = 'blahmetric'
        tval = None
        metric = entities.Metric(tkey, tval, int(time.time()), 0)

        warnings.simplefilter("ignore")
        with self.assertRaises(MlflowException) as exception_context, warnings.catch_warnings():
            self.store.log_metric(run.info.run_id, metric)
            warnings.resetwarnings()
        assert exception_context.exception.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_log_param(self):
        run = self._run_factory()

        tkey = 'blahmetric'
        tval = '100.0'
        param = entities.Param(tkey, tval)
        param2 = entities.Param('new param', 'new key')
        self.store.log_param(run.info.run_id, param)
        self.store.log_param(run.info.run_id, param2)
        self.store.log_param(run.info.run_id, param2)

        run = self.store.get_run(run.info.run_id)
        self.assertEqual(2, len(run.data.params))
        self.assertTrue(tkey in run.data.params and run.data.params[tkey] == tval)

    def test_log_param_uniqueness(self):
        run = self._run_factory()

        tkey = 'blahmetric'
        tval = '100.0'
        param = entities.Param(tkey, tval)
        param2 = entities.Param(tkey, 'newval')
        self.store.log_param(run.info.run_id, param)

        with self.assertRaises(MlflowException) as e:
            self.store.log_param(run.info.run_id, param2)
        self.assertIn("Changing param value is not allowed. Param with key=", e.exception.message)

    def test_log_empty_str(self):
        run = self._run_factory()

        tkey = 'blahmetric'
        tval = ''
        param = entities.Param(tkey, tval)
        param2 = entities.Param('new param', 'new key')
        self.store.log_param(run.info.run_id, param)
        self.store.log_param(run.info.run_id, param2)

        run = self.store.get_run(run.info.run_id)
        self.assertEqual(2, len(run.data.params))
        self.assertTrue(tkey in run.data.params and run.data.params[tkey] == tval)

    def test_log_null_param(self):
        run = self._run_factory()

        tkey = 'blahmetric'
        tval = None
        param = entities.Param(tkey, tval)

        with self.assertRaises(MlflowException) as exception_context:
            self.store.log_param(run.info.run_id, param)
        assert exception_context.exception.error_code == ErrorCode.Name(INTERNAL_ERROR)

    def test_set_tag(self):
        run = self._run_factory()

        tkey = 'test tag'
        tval = 'a boogie'
        new_val = "new val"
        tag = entities.RunTag(tkey, tval)
        new_tag = entities.RunTag(tkey, new_val)
        self.store.set_tag(run.info.run_id, tag)
        # Overwriting tags is allowed
        self.store.set_tag(run.info.run_id, new_tag)

        run = self.store.get_run(run.info.run_id)
        self.assertTrue(tkey in run.data.tags and run.data.tags[tkey] == new_val)

    def test_get_metric_history(self):
        run = self._run_factory()

        key = 'test'
        expected = [
            models.SqlMetric(key=key, value=0.6, timestamp=1, step=0).to_mlflow_entity(),
            models.SqlMetric(key=key, value=0.7, timestamp=2, step=0).to_mlflow_entity()
        ]

        for metric in expected:
            self.store.log_metric(run.info.run_id, metric)

        actual = self.store.get_metric_history(run.info.run_id, key)

        six.assertCountEqual(self,
                             [(m.key, m.value, m.timestamp) for m in expected],
                             [(m.key, m.value, m.timestamp) for m in actual])

    def test_list_run_infos(self):
        experiment_id = self._experiment_factory('test_exp')
        r1 = self._run_factory(config=self._get_run_configs(experiment_id)).info.run_id
        r2 = self._run_factory(config=self._get_run_configs(experiment_id)).info.run_id

        def _runs(experiment_id, view_type):
            return [r.run_id for r in self.store.list_run_infos(experiment_id, view_type)]

        six.assertCountEqual(self, [r1, r2], _runs(experiment_id, ViewType.ALL))
        six.assertCountEqual(self, [r1, r2], _runs(experiment_id, ViewType.ACTIVE_ONLY))
        self.assertEqual(0, len(_runs(experiment_id, ViewType.DELETED_ONLY)))

        self.store.delete_run(r1)
        six.assertCountEqual(self, [r1, r2], _runs(experiment_id, ViewType.ALL))
        six.assertCountEqual(self, [r2], _runs(experiment_id, ViewType.ACTIVE_ONLY))
        six.assertCountEqual(self, [r1], _runs(experiment_id, ViewType.DELETED_ONLY))

    def test_rename_experiment(self):
        new_name = 'new name'
        experiment_id = self._experiment_factory('test name')
        self.store.rename_experiment(experiment_id, new_name)

        renamed_experiment = self.store.get_experiment(experiment_id)

        self.assertEqual(renamed_experiment.name, new_name)

    def test_update_run_info(self):
        run = self._run_factory()

        new_status = entities.RunStatus.FINISHED
        endtime = int(time.time())

        actual = self.store.update_run_info(run.info.run_id, new_status, endtime)

        self.assertEqual(actual.status, RunStatus.to_string(new_status))
        self.assertEqual(actual.end_time, endtime)

    def test_restore_experiment(self):
        experiment_id = self._experiment_factory('helloexp')
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
        self.assertIn("must be in 'deleted' state", e.exception.message)

        self.store.delete_run(run.info.run_id)
        with self.assertRaises(MlflowException) as e:
            self.store.delete_run(run.info.run_id)
        self.assertIn("must be in 'active' state", e.exception.message)

        deleted = self.store.get_run(run.info.run_id)
        self.assertEqual(deleted.info.run_id, run.info.run_id)
        self.assertEqual(deleted.info.lifecycle_stage, entities.LifecycleStage.DELETED)

        self.store.restore_run(run.info.run_id)
        with self.assertRaises(MlflowException) as e:
            self.store.restore_run(run.info.run_id)
            self.assertIn("must be in 'deleted' state", e.exception.message)
        restored = self.store.get_run(run.info.run_id)
        self.assertEqual(restored.info.run_id, run.info.run_id)
        self.assertEqual(restored.info.lifecycle_stage, entities.LifecycleStage.ACTIVE)

    def test_error_logging_to_deleted_run(self):
        exp = self._experiment_factory('error_logging')
        run_id = self._run_factory(self._get_run_configs(experiment_id=exp)).info.run_id

        self.store.delete_run(run_id)
        self.assertEqual(self.store.get_run(run_id).info.lifecycle_stage,
                         entities.LifecycleStage.DELETED)
        with self.assertRaises(MlflowException) as e:
            self.store.log_param(run_id, entities.Param("p1345", "v1"))
        self.assertIn("must be in 'active' state", e.exception.message)

        with self.assertRaises(MlflowException) as e:
            self.store.log_metric(run_id, entities.Metric("m1345", 1.0, 123, 0))
        self.assertIn("must be in 'active' state", e.exception.message)

        with self.assertRaises(MlflowException) as e:
            self.store.set_tag(run_id, entities.RunTag("t1345", "tv1"))
        self.assertIn("must be in 'active' state", e.exception.message)

        # restore this run and try again
        self.store.restore_run(run_id)
        self.assertEqual(self.store.get_run(run_id).info.lifecycle_stage,
                         entities.LifecycleStage.ACTIVE)
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
    def _search(self, experiment_id, filter_string=None,
                run_view_type=ViewType.ALL, max_results=SEARCH_MAX_RESULTS_DEFAULT):
        search_filter = SearchFilter(filter_string=filter_string)
        exps = [experiment_id] if isinstance(experiment_id, int) else experiment_id
        return [r.info.run_id
                for r in self.store.search_runs(exps, search_filter, run_view_type, max_results)]

    def test_search_vanilla(self):
        exp = self._experiment_factory('search_vanilla')
        runs = [self._run_factory(self._get_run_configs(exp)).info.run_id
                for r in range(3)]

        six.assertCountEqual(self, runs, self._search(exp, run_view_type=ViewType.ALL))
        six.assertCountEqual(self, runs, self._search(exp, run_view_type=ViewType.ACTIVE_ONLY))
        six.assertCountEqual(self, [], self._search(exp, run_view_type=ViewType.DELETED_ONLY))

        first = runs[0]

        self.store.delete_run(first)
        six.assertCountEqual(self, runs, self._search(exp, run_view_type=ViewType.ALL))
        six.assertCountEqual(self, runs[1:], self._search(exp, run_view_type=ViewType.ACTIVE_ONLY))
        six.assertCountEqual(self, [first], self._search(exp, run_view_type=ViewType.DELETED_ONLY))

        self.store.restore_run(first)
        six.assertCountEqual(self, runs, self._search(exp, run_view_type=ViewType.ALL))
        six.assertCountEqual(self, runs, self._search(exp, run_view_type=ViewType.ACTIVE_ONLY))
        six.assertCountEqual(self, [], self._search(exp, run_view_type=ViewType.DELETED_ONLY))

    def test_search_params(self):
        experiment_id = self._experiment_factory('search_params')
        r1 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        r2 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id

        self.store.log_param(r1, entities.Param('generic_param', 'p_val'))
        self.store.log_param(r2, entities.Param('generic_param', 'p_val'))

        self.store.log_param(r1, entities.Param('generic_2', 'some value'))
        self.store.log_param(r2, entities.Param('generic_2', 'another value'))

        self.store.log_param(r1, entities.Param('p_a', 'abc'))
        self.store.log_param(r2, entities.Param('p_b', 'ABC'))

        # test search returns both runs
        filter_string = "params.generic_param = 'p_val'"
        six.assertCountEqual(self, [r1, r2], self._search(experiment_id, filter_string))

        # test search returns appropriate run (same key different values per run)
        filter_string = "params.generic_2 = 'some value'"
        six.assertCountEqual(self, [r1], self._search(experiment_id, filter_string))
        filter_string = "params.generic_2 = 'another value'"
        six.assertCountEqual(self, [r2], self._search(experiment_id, filter_string))

        filter_string = "params.generic_param = 'wrong_val'"
        six.assertCountEqual(self, [], self._search(experiment_id, filter_string))

        filter_string = "params.generic_param != 'p_val'"
        six.assertCountEqual(self, [], self._search(experiment_id, filter_string))

        filter_string = "params.generic_param != 'wrong_val'"
        six.assertCountEqual(self, [r1, r2], self._search(experiment_id, filter_string))
        filter_string = "params.generic_2 != 'wrong_val'"
        six.assertCountEqual(self, [r1, r2], self._search(experiment_id, filter_string))

        filter_string = "params.p_a = 'abc'"
        six.assertCountEqual(self, [r1], self._search(experiment_id, filter_string))

        filter_string = "params.p_b = 'ABC'"
        six.assertCountEqual(self, [r2], self._search(experiment_id, filter_string))

    def test_search_tags(self):
        experiment_id = self._experiment_factory('search_tags')
        r1 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        r2 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id

        self.store.set_tag(r1, entities.RunTag('generic_tag', 'p_val'))
        self.store.set_tag(r2, entities.RunTag('generic_tag', 'p_val'))

        self.store.set_tag(r1, entities.RunTag('generic_2', 'some value'))
        self.store.set_tag(r2, entities.RunTag('generic_2', 'another value'))

        self.store.set_tag(r1, entities.RunTag('p_a', 'abc'))
        self.store.set_tag(r2, entities.RunTag('p_b', 'ABC'))

        # test search returns both runs
        six.assertCountEqual(self, [r1, r2],
                             self._search(experiment_id,
                                          filter_string="tags.generic_tag = 'p_val'"))
        # test search returns appropriate run (same key different values per run)
        six.assertCountEqual(self, [r1],
                             self._search(experiment_id,
                                          filter_string="tags.generic_2 = 'some value'"))
        six.assertCountEqual(self, [r2],
                             self._search(experiment_id,
                                          filter_string="tags.generic_2 = 'another value'"))
        six.assertCountEqual(self, [],
                             self._search(experiment_id,
                                          filter_string="tags.generic_tag = 'wrong_val'"))
        six.assertCountEqual(self, [],
                             self._search(experiment_id,
                                          filter_string="tags.generic_tag != 'p_val'"))
        six.assertCountEqual(self, [r1, r2],
                             self._search(experiment_id,
                                          filter_string="tags.generic_tag != 'wrong_val'"))
        six.assertCountEqual(self, [r1, r2],
                             self._search(experiment_id,
                                          filter_string="tags.generic_2 != 'wrong_val'"))
        six.assertCountEqual(self, [r1], self._search(experiment_id,
                                                      filter_string="tags.p_a = 'abc'"))
        six.assertCountEqual(self, [r2], self._search(experiment_id,
                                                      filter_string="tags.p_b = 'ABC'"))

    def test_search_metrics(self):
        experiment_id = self._experiment_factory('search_metric')
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
        six.assertCountEqual(self, [r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.common > 0.0"
        six.assertCountEqual(self, [r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.common >= 0.0"
        six.assertCountEqual(self, [r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.common < 4.0"
        six.assertCountEqual(self, [r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.common <= 4.0"
        six.assertCountEqual(self, [r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.common != 1.0"
        six.assertCountEqual(self, [], self._search(experiment_id, filter_string))

        filter_string = "metrics.common >= 3.0"
        six.assertCountEqual(self, [], self._search(experiment_id, filter_string))

        filter_string = "metrics.common <= 0.75"
        six.assertCountEqual(self, [], self._search(experiment_id, filter_string))

        # tests for same metric name across runs with different values and timestamps
        filter_string = "metrics.measure_a > 0.0"
        six.assertCountEqual(self, [r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a < 50.0"
        six.assertCountEqual(self, [r1], self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a < 1000.0"
        six.assertCountEqual(self, [r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a != -12.0"
        six.assertCountEqual(self, [r1, r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a > 50.0"
        six.assertCountEqual(self, [r2], self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a = 1.0"
        six.assertCountEqual(self, [r1], self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a = 400.0"
        six.assertCountEqual(self, [r2], self._search(experiment_id, filter_string))

        # test search with unique metric keys
        filter_string = "metrics.m_a > 1.0"
        six.assertCountEqual(self, [r1], self._search(experiment_id, filter_string))

        filter_string = "metrics.m_b > 1.0"
        six.assertCountEqual(self, [r2], self._search(experiment_id, filter_string))

        # there is a recorded metric this threshold but not last timestamp
        filter_string = "metrics.m_b > 5.0"
        six.assertCountEqual(self, [], self._search(experiment_id, filter_string))

        # metrics matches last reported timestamp for 'm_b'
        filter_string = "metrics.m_b = 4.0"
        six.assertCountEqual(self, [r2], self._search(experiment_id, filter_string))

    def test_search_attrs(self):
        e1 = self._experiment_factory('search_attributes_1')
        r1 = self._run_factory(self._get_run_configs(experiment_id=e1)).info.run_id

        e2 = self._experiment_factory('search_attrs_2')
        r2 = self._run_factory(self._get_run_configs(experiment_id=e2)).info.run_id

        filter_string = ""
        six.assertCountEqual(self, [r1, r2], self._search([e1, e2], filter_string))

        filter_string = "attribute.status != 'blah'"
        six.assertCountEqual(self, [r1, r2], self._search([e1, e2], filter_string))

        filter_string = "attribute.status = '{}'".format(RunStatus.to_string(RunStatus.RUNNING))
        six.assertCountEqual(self, [r1, r2], self._search([e1, e2], filter_string))

        # change status for one of the runs
        self.store.update_run_info(r2, RunStatus.FAILED, 300)

        filter_string = "attribute.status = 'RUNNING'"
        six.assertCountEqual(self, [r1], self._search([e1, e2], filter_string))

        filter_string = "attribute.status = 'FAILED'"
        six.assertCountEqual(self, [r2], self._search([e1, e2], filter_string))

        filter_string = "attribute.status != 'SCHEDULED'"
        six.assertCountEqual(self, [r1, r2], self._search([e1, e2], filter_string))

        filter_string = "attribute.status = 'SCHEDULED'"
        six.assertCountEqual(self, [], self._search([e1, e2], filter_string))

        filter_string = "attribute.status = 'KILLED'"
        six.assertCountEqual(self, [], self._search([e1, e2], filter_string))

        filter_string = "attr.artifact_uri = '{}/{}/{}/artifacts'".format(ARTIFACT_URI, e1, r1)
        six.assertCountEqual(self, [r1], self._search([e1, e2], filter_string))

        filter_string = "attr.artifact_uri = '{}/{}/{}/artifacts'".format(ARTIFACT_URI, e2, r1)
        six.assertCountEqual(self, [], self._search([e1, e2], filter_string))

        filter_string = "attribute.artifact_uri = 'random_artifact_path'"
        six.assertCountEqual(self, [], self._search([e1, e2], filter_string))

        filter_string = "attribute.artifact_uri != 'random_artifact_path'"
        six.assertCountEqual(self, [r1, r2], self._search([e1, e2], filter_string))

        for (k, v) in {"experiment_id": e1,
                       "lifecycle_stage": "ACTIVE",
                       "run_id": r1,
                       "run_uuid": r2}.items():
            with self.assertRaises(MlflowException) as e:
                self._search([e1, e2], "attribute.{} = '{}'".format(k, v))
            self.assertIn("Invalid attribute key", e.exception.message)

    def test_search_full(self):
        experiment_id = self._experiment_factory('search_params')
        r1 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        r2 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id

        self.store.log_param(r1, entities.Param('generic_param', 'p_val'))
        self.store.log_param(r2, entities.Param('generic_param', 'p_val'))

        self.store.log_param(r1, entities.Param('p_a', 'abc'))
        self.store.log_param(r2, entities.Param('p_b', 'ABC'))

        self.store.log_metric(r1, entities.Metric("common", 1.0, 1, 0))
        self.store.log_metric(r2, entities.Metric("common", 1.0, 1, 0))

        self.store.log_metric(r1, entities.Metric("m_a", 2.0, 2, 0))
        self.store.log_metric(r2, entities.Metric("m_b", 3.0, 2, 0))
        self.store.log_metric(r2, entities.Metric("m_b", 4.0, 8, 0))
        self.store.log_metric(r2, entities.Metric("m_b", 8.0, 3, 0))

        filter_string = "params.generic_param = 'p_val' and metrics.common = 1.0"
        six.assertCountEqual(self, [r1, r2], self._search(experiment_id, filter_string))

        # all params and metrics match
        filter_string = ("params.generic_param = 'p_val' and metrics.common = 1.0"
                         "and metrics.m_a > 1.0")
        six.assertCountEqual(self, [r1], self._search(experiment_id, filter_string))

        # test with mismatch param
        filter_string = ("params.random_bad_name = 'p_val' and metrics.common = 1.0"
                         "and metrics.m_a > 1.0")
        six.assertCountEqual(self, [], self._search(experiment_id, filter_string))

        # test with mismatch metric
        filter_string = ("params.generic_param = 'p_val' and metrics.common = 1.0"
                         "and metrics.m_a > 100.0")
        six.assertCountEqual(self, [], self._search(experiment_id, filter_string))

    def test_search_with_max_results(self):
        exp = self._experiment_factory('search_with_max_results')
        runs = [self._run_factory(self._get_run_configs(exp, start_time=r)).info.run_id
                for r in range(1200)]
        # reverse the ordering, since we created in increasing order of start_time
        runs.reverse()

        assert(runs[:1000] == self._search(exp))
        for n in [0, 1, 2, 4, 8, 10, 20, 50, 100, 500, 1000, 1200, 2000]:
            assert(runs[:min(1200, n)] == self._search(exp, max_results=n))

        with self.assertRaises(MlflowException) as e:
            self._search(exp, max_results=int(1e10))
        self.assertIn("Invalid value for request parameter max_results. It ", e.exception.message)

    def test_search_with_deterministic_max_results(self):
        exp = self._experiment_factory('test_search_with_deterministic_max_results')
        # Create 10 runs with the same start_time.
        # Sort based on run_id
        runs = sorted([self._run_factory(self._get_run_configs(exp, start_time=10)).info.run_id
                       for r in range(10)])
        for n in [0, 1, 2, 4, 8, 10, 20]:
            assert(runs[:min(10, n)] == self._search(exp, max_results=n))

    def test_log_batch(self):
        experiment_id = self._experiment_factory('log_batch')
        run_id = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        metric_entities = [Metric("m1", 0.87, 12345, 0), Metric("m2", 0.49, 12345, 1)]
        param_entities = [Param("p1", "p1val"), Param("p2", "p2val")]
        tag_entities = [RunTag("t1", "t1val"), RunTag("t2", "t2val")]
        self.store.log_batch(
            run_id=run_id, metrics=metric_entities, params=param_entities, tags=tag_entities)
        run = self.store.get_run(run_id)
        assert run.data.tags == {"t1": "t1val", "t2": "t2val"}
        assert run.data.params == {"p1": "p1val", "p2": "p2val"}
        metric_histories = sum(
            [self.store.get_metric_history(run_id, key) for key in run.data.metrics], [])
        metrics = [(m.key, m.value, m.timestamp, m.step) for m in metric_histories]
        assert set(metrics) == set([("m1", 0.87, 12345, 0), ("m2", 0.49, 12345, 1)])

    def test_log_batch_limits(self):
        # Test that log batch at the maximum allowed request size succeeds (i.e doesn't hit
        # SQL limitations, etc)
        experiment_id = self._experiment_factory('log_batch_limits')
        run_id = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        metric_tuples = [("m%s" % i, i, 12345, i * 2) for i in range(1000)]
        metric_entities = [Metric(*metric_tuple) for metric_tuple in metric_tuples]
        self.store.log_batch(run_id=run_id, metrics=metric_entities, params=[], tags=[])
        run = self.store.get_run(run_id)
        metric_histories = sum(
            [self.store.get_metric_history(run_id, key) for key in run.data.metrics], [])
        metrics = [(m.key, m.value, m.timestamp, m.step) for m in metric_histories]
        assert set(metrics) == set(metric_tuples)

    def test_log_batch_param_overwrite_disallowed(self):
        # Test that attempting to overwrite a param via log_batch results in an exception and that
        # no partial data is logged
        run = self._run_factory()
        tkey = 'my-param'
        param = entities.Param(tkey, 'orig-val')
        self.store.log_param(run.info.run_id, param)

        overwrite_param = entities.Param(tkey, 'newval')
        tag = entities.RunTag("tag-key", "tag-val")
        metric = entities.Metric("metric-key", 3.0, 12345, 0)
        with self.assertRaises(MlflowException) as e:
            self.store.log_batch(run.info.run_id, metrics=[metric], params=[overwrite_param],
                                 tags=[tag])
        self.assertIn("Changing param value is not allowed. Param with key=", e.exception.message)
        assert e.exception.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        self._verify_logged(run.info.run_id, metrics=[], params=[param], tags=[])

    def test_log_batch_param_overwrite_disallowed_single_req(self):
        # Test that attempting to overwrite a param via log_batch results in an exception
        run = self._run_factory()
        pkey = "common-key"
        param0 = entities.Param(pkey, "orig-val")
        param1 = entities.Param(pkey, 'newval')
        tag = entities.RunTag("tag-key", "tag-val")
        metric = entities.Metric("metric-key", 3.0, 12345, 0)
        with self.assertRaises(MlflowException) as e:
            self.store.log_batch(run.info.run_id, metrics=[metric], params=[param0, param1],
                                 tags=[tag])
        self.assertIn("Changing param value is not allowed. Param with key=", e.exception.message)
        assert e.exception.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        self._verify_logged(run.info.run_id, metrics=[], params=[param0], tags=[])

    def test_log_batch_accepts_empty_payload(self):
        run = self._run_factory()
        self.store.log_batch(run.info.run_id, metrics=[], params=[], tags=[])
        self._verify_logged(run.info.run_id, metrics=[], params=[], tags=[])

    def test_log_batch_internal_error(self):
        # Verify that internal errors during the DB save step for log_batch result in
        # MlflowExceptions
        run = self._run_factory()

        def _raise_exception_fn(*args, **kwargs):  # pylint: disable=unused-argument
            raise Exception("Some internal error")
        with mock.patch("mlflow.store.sqlalchemy_store.SqlAlchemyStore.log_metric") as metric_mock,\
                mock.patch(
                    "mlflow.store.sqlalchemy_store.SqlAlchemyStore.log_param") as param_mock,\
                mock.patch("mlflow.store.sqlalchemy_store.SqlAlchemyStore.set_tag") as tags_mock:
            metric_mock.side_effect = _raise_exception_fn
            param_mock.side_effect = _raise_exception_fn
            tags_mock.side_effect = _raise_exception_fn
            for kwargs in [{"metrics": [Metric("a", 3, 1, 0)]}, {"params": [Param("b", "c")]},
                           {"tags": [RunTag("c", "d")]}]:
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
        self._verify_logged(run.info.run_id, metrics=[], params=params, tags=[])

    def test_log_batch_tags_idempotency(self):
        run = self._run_factory()
        self.store.log_batch(
            run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")])
        self.store.log_batch(
            run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")])
        self._verify_logged(
            run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")])

    def test_log_batch_allows_tag_overwrite(self):
        run = self._run_factory()
        self.store.log_batch(
            run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "val")])
        self.store.log_batch(
            run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "newval")])
        self._verify_logged(
            run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "newval")])

    def test_log_batch_allows_tag_overwrite_single_req(self):
        run = self._run_factory()
        tags = [RunTag("t-key", "val"), RunTag("t-key", "newval")]
        self.store.log_batch(run.info.run_id, metrics=[], params=[], tags=tags)
        self._verify_logged(run.info.run_id, metrics=[], params=[], tags=[tags[-1]])

    def test_log_batch_same_metric_repeated_single_req(self):
        run = self._run_factory()
        metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
        metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
        self.store.log_batch(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])
        self._verify_logged(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])

    def test_log_batch_same_metric_repeated_multiple_reqs(self):
        run = self._run_factory()
        metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
        metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
        self.store.log_batch(run.info.run_id, params=[], metrics=[metric0], tags=[])
        self._verify_logged(run.info.run_id, params=[], metrics=[metric0], tags=[])
        self.store.log_batch(run.info.run_id, params=[], metrics=[metric1], tags=[])
        self._verify_logged(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])

    def test_upgrade_cli_idempotence(self):
        # Repeatedly run `mlflow db upgrade` against our database, verifying that the command
        # succeeds and that the DB has the latest schema
        engine = sqlalchemy.create_engine(self.db_url)
        assert _get_schema_version(engine) == SqlAlchemyStore._get_latest_schema_revision()
        for _ in range(3):
            invoke_cli_runner(mlflow.db.commands, ['upgrade', self.db_url])
            assert _get_schema_version(engine) == SqlAlchemyStore._get_latest_schema_revision()


class TestSqlAlchemyStoreSqliteMigratedDB(TestSqlAlchemyStoreSqlite):
    """
    Test case where user has an existing DB with schema generated before MLflow 1.0,
    then migrates their DB. TODO: update this test in MLflow 1.1 to use InitialBase from
    mlflow.store.db.initial_models.
    """
    def setUp(self):
        fd, self.temp_dbfile = tempfile.mkstemp()
        os.close(fd)
        self.db_url = "%s%s" % (DB_URI, self.temp_dbfile)
        engine = sqlalchemy.create_engine(self.db_url)
        InitialBase.metadata.create_all(engine)
        invoke_cli_runner(mlflow.db.commands, ['upgrade', self.db_url])
        self.store = SqlAlchemyStore(self.db_url, ARTIFACT_URI)

    def tearDown(self):
        os.remove(self.temp_dbfile)


@pytest.mark.release
class TestSqlAlchemyStoreMysqlDb(TestSqlAlchemyStoreSqlite):
    """
    Run tests against a MySQL database
    """
    DEFAULT_MYSQL_PORT = 3306

    def setUp(self):
        db_username = os.environ.get("MYSQL_TEST_USERNAME")
        db_password = os.environ.get("MYSQL_TEST_PASSWORD")
        db_port = int(os.environ["MYSQL_TEST_PORT"]) if "MYSQL_TEST_PORT" in os.environ \
            else TestSqlAlchemyStoreMysqlDb.DEFAULT_MYSQL_PORT
        if db_username is None or db_password is None:
            raise Exception(
                "Username and password for database tests must be specified via the "
                "MYSQL_TEST_USERNAME and MYSQL_TEST_PASSWORD environment variables. "
                "environment variable. In posix shells, you can rerun your test command "
                "with the environment variables set, e.g: MYSQL_TEST_USERNAME=your_username "
                "MYSQL_TEST_PASSWORD=your_password <your-test-command>. You may optionally "
                "specify a database port via MYSQL_TEST_PORT (default is 3306).")
        self._db_name = "test_sqlalchemy_store_%s" % uuid.uuid4().hex[:5]
        db_server_url = "mysql://%s:%s@localhost:%s" % (db_username, db_password, db_port)
        self._engine = sqlalchemy.create_engine(db_server_url)
        self._engine.execute("CREATE DATABASE %s" % self._db_name)
        self.db_url = "%s/%s" % (db_server_url, self._db_name)
        self.store = self._get_store(self.db_url)

    def tearDown(self):
        self._engine.execute("DROP DATABASE %s" % self._db_name)

    def test_log_many_entities(self):
        """
        Sanity check: verify that we can log a reasonable number of entities without failures due
        to connection leaks etc.
        """
        run = self._run_factory()
        for i in range(100):
            self.store.log_metric(run.info.run_id, entities.Metric("key", i, i * 2, i * 3))
            self.store.log_param(run.info.run_id, entities.Param("pkey-%s" % i,  "pval-%s" % i))
            self.store.set_tag(run.info.run_id, entities.RunTag("tkey-%s" % i,  "tval-%s" % i))
