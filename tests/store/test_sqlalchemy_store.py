import unittest
import warnings

import sqlalchemy
import time
import mlflow
import uuid
from mlflow.store.dbmodels import models
from mlflow import entities
from mlflow.exceptions import MlflowException
from mlflow.store.sqlalchemy_store import SqlAlchemyStore


DB_URI = 'sqlite://'


class TestSqlAlchemyStoreSqliteInMemory(unittest.TestCase):
    def setUp(self):
        self.store = SqlAlchemyStore(DB_URI)
        self.engine = sqlalchemy.create_engine(DB_URI)
        Session = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.session = Session()
        self.store.session = self.session
        self.store.engine = self.engine
        models.Base.metadata.create_all(self.engine)

    def tearDown(self):
        models.Base.metadata.drop_all(self.engine)

    def _experiment_factory(self, names):
        if type(names) is list:
            experiments = []
            for name in names:
                exp = self.store.create_experiment(name=name)
                experiments.append(exp)

            return experiments

        return self.store.create_experiment(name=names)

    def test_raise_duplicate_experiments(self):
        with self.assertRaises(Exception):
            self._experiment_factory(['test', 'test'])

    def test_raise_experiment_dont_exist(self):
        with self.assertRaises(Exception):
            self.store.get_experiment(experiment_id=100)

    def test_delete_experiment(self):
        experiments = self._experiment_factory(['morty', 'rick', 'rick and morty'])
        exp = experiments[0]
        self.store.delete_experiment(exp.experiment_id)

        actual = self.session.query(models.SqlExperiment).get(exp.experiment_id)
        self.assertEqual(len(self.store.list_experiments()), len(experiments) - 1)

        self.assertEqual(actual.lifecycle_stage, entities.Experiment.DELETED_LIFECYCLE)

    def test_get_experiment(self):
        name = 'goku'
        run_data = self._experiment_factory(name)
        actual = self.store.get_experiment(run_data.experiment_id)
        self.assertEqual(actual.name, run_data.name)
        self.assertEqual(actual.experiment_id, run_data.experiment_id)

    def test_list_experiments(self):
        testnames = ['blue', 'red', 'green']

        run_data = self._experiment_factory(testnames)
        actual = self.store.list_experiments()

        self.assertEqual(len(run_data), len(actual))

        for exp in run_data:
            res = self.session.query(models.SqlExperiment).filter_by(
                experiment_id=exp.experiment_id).first()
            self.assertEqual(res.name, exp.name)
            self.assertEqual(res.experiment_id, exp.experiment_id)

    def test_create_experiments(self):
        result = self.session.query(models.SqlExperiment).all()
        self.assertEqual(len(result), 0)

        run_data = self.store.create_experiment(name='test experiment')
        result = self.session.query(models.SqlExperiment).all()
        self.assertEqual(len(result), 1)

        actual = result[0]

        self.assertEqual(actual.experiment_id, run_data.experiment_id)
        self.assertEqual(actual.name, run_data.name)

    def test_run_tag_model(self):
        run_data = models.SqlTag(run_uuid='tuuid', key='test', value='val')
        self.session.add(run_data)
        self.session.commit()
        tags = self.session.query(models.SqlTag).all()
        self.assertEqual(len(tags), 1)

        actual = tags[0].to_mlflow_entity()

        self.assertEqual(actual.value, run_data.value)
        self.assertEqual(actual.key, run_data.key)

    def test_metric_model(self):
        run_data = models.SqlMetric(run_uuid='testuid', key='accuracy', value=0.89)
        self.session.add(run_data)
        self.session.commit()
        metrics = self.session.query(models.SqlMetric).all()
        self.assertEqual(len(metrics), 1)

        actual = metrics[0].to_mlflow_entity()

        self.assertEqual(actual.value, run_data.value)
        self.assertEqual(actual.key, run_data.key)

    def test_param_model(self):
        run_data = models.SqlParam(run_uuid='test', key='accuracy', value='test param')
        self.session.add(run_data)
        self.session.commit()
        params = self.session.query(models.SqlParam).all()
        self.assertEqual(len(params), 1)

        actual = params[0].to_mlflow_entity()

        self.assertEqual(actual.value, run_data.value)
        self.assertEqual(actual.key, run_data.key)

    def test_run_needs_uuid(self):
        run = models.SqlRun()
        self.session.add(run)

        with self.assertRaises(sqlalchemy.exc.IntegrityError):
            warnings.simplefilter("ignore")
            with warnings.catch_warnings():
                self.session.commit()
            warnings.resetwarnings()

    def test_run_data_model(self):
        m1 = models.SqlMetric(key='accuracy', value=0.89)
        m2 = models.SqlMetric(key='recal', value=0.89)
        p1 = models.SqlParam(key='loss', value='test param')
        p2 = models.SqlParam(key='blue', value='test param')

        self.session.add_all([m1, m2, p1, p2])

        run_data = models.SqlRun(run_uuid=uuid.uuid4().hex)
        run_data.params.append(p1)
        run_data.params.append(p2)
        run_data.metrics.append(m1)
        run_data.metrics.append(m2)

        self.session.add(run_data)
        self.session.commit()

        run_datums = self.session.query(models.SqlRun).all()
        actual = run_datums[0]
        self.assertEqual(len(run_datums), 1)
        self.assertEqual(len(actual.params), 2)
        self.assertEqual(len(actual.metrics), 2)

    def test_run_info(self):
        experiment = self._experiment_factory('test exp')
        config = {
            'experiment_id': experiment.experiment_id,
            'name': 'test run',
            'user_id': 'Anderson',
            'run_uuid': 'test',
            'status': entities.RunInfo.ACTIVE_LIFECYCLE,
            'source_type': entities.SourceType.LOCAL,
            'source_name': 'Python application',
            'entry_point_name': 'main.py',
            'start_time': int(time.time()),
            'end_time': int(time.time()),
            'source_version': mlflow.__version__,
            'lifecycle_stage': entities.RunInfo.ACTIVE_LIFECYCLE,
            'artifact_uri': '//'
        }
        run = models.SqlRun(**config).to_mlflow_entity()

        for k, v in config.items():
            self.assertEqual(v, getattr(run.info, k))

    def _run_factory(self, name='test', experiment_id=None, config=None):
        m1 = models.SqlMetric(key='accuracy', value=0.89)
        m2 = models.SqlMetric(key='recal', value=0.89)
        p1 = models.SqlParam(key='loss', value='test param')
        p2 = models.SqlParam(key='blue', value='test param')

        if not experiment_id:
            experiment = self._experiment_factory('test exp')
            experiment_id = experiment.experiment_id

        config = {
            'experiment_id': experiment_id,
            'name': name,
            'user_id': 'Anderson',
            'run_uuid': uuid.uuid4().hex,
            'status': entities.RunStatus.to_string(entities.RunStatus.SCHEDULED),
            'source_type': entities.SourceType.to_string(entities.SourceType.NOTEBOOK),
            'source_name': 'Python application',
            'entry_point_name': 'main.py',
            'start_time': int(time.time()),
            'end_time': int(time.time()),
            'source_version': mlflow.__version__,
            'lifecycle_stage': entities.RunInfo.ACTIVE_LIFECYCLE,
            'artifact_uri': '//'
        }

        run = models.SqlRun(**config)

        run.params.append(p1)
        run.params.append(p2)
        run.metrics.append(m1)
        run.metrics.append(m2)
        self.session.add(run)

        return run

    def test_create_run(self):
        expected = self._run_factory()
        name = 'booyya'
        expected.tags.append(models.SqlTag(key='3', value='4'))
        expected.tags.append(models.SqlTag(key='1', value='2'))

        tags = [t.to_mlflow_entity() for t in expected.tags]
        actual = self.store.create_run(expected.experiment_id, expected.user_id, name,
                                       entities.SourceType.from_string(expected.source_type),
                                       expected.source_name,
                                       expected.entry_point_name, expected.start_time,
                                       expected.source_version, tags, None)

        self.assertEqual(actual.info.experiment_id, expected.experiment_id)
        self.assertEqual(actual.info.user_id, expected.user_id)
        self.assertEqual(actual.info.name, name)
        self.assertEqual(actual.info.source_type, expected.source_type)
        self.assertEqual(actual.info.source_name, expected.source_name)
        self.assertEqual(actual.info.source_version, expected.source_version)
        self.assertEqual(actual.info.entry_point_name, expected.entry_point_name)
        self.assertEqual(actual.info.start_time, expected.start_time)
        self.assertEqual(len(actual.data.tags), 3)

        name_tag = models.SqlTag(key='mlflow.runName', value=name).to_mlflow_entity()
        self.assertListEqual(actual.data.tags, tags + [name_tag])

    def test_create_run_with_parent_id(self):
        expected = self._run_factory()
        name = 'booyya'
        expected.tags.append(models.SqlTag(key='3', value='4'))
        expected.tags.append(models.SqlTag(key='1', value='2'))

        tags = [t.to_mlflow_entity() for t in expected.tags]
        actual = self.store.create_run(expected.experiment_id, expected.user_id, name,
                                       entities.SourceType.from_string(expected.source_type),
                                       expected.source_name,
                                       expected.entry_point_name, expected.start_time,
                                       expected.source_version, tags, "parent_uuid_5")

        self.assertEqual(actual.info.experiment_id, expected.experiment_id)
        self.assertEqual(actual.info.user_id, expected.user_id)
        self.assertEqual(actual.info.name, name)
        self.assertEqual(actual.info.source_type, expected.source_type)
        self.assertEqual(actual.info.source_name, expected.source_name)
        self.assertEqual(actual.info.source_version, expected.source_version)
        self.assertEqual(actual.info.entry_point_name, expected.entry_point_name)
        self.assertEqual(actual.info.start_time, expected.start_time)
        self.assertEqual(len(actual.data.tags), 4)

        name_tag = models.SqlTag(key='mlflow.runName', value=name).to_mlflow_entity()
        parent_id_tag = models.SqlTag(key='mlflow.parentRunId',
                                      value='parent_uuid_5').to_mlflow_entity()
        self.assertListEqual(actual.data.tags, tags + [parent_id_tag, name_tag])

    def test_to_mlflow_entity(self):
        run = self._run_factory()
        run = run.to_mlflow_entity()

        self.assertIsInstance(run.info, entities.RunInfo)
        self.assertIsInstance(run.data, entities.RunData)

        for metric in run.data.metrics:
            self.assertIsInstance(metric, entities.Metric)

        for param in run.data.params:
            self.assertIsInstance(param, entities.Param)

        for tag in run.data.tags:
            self.assertIsInstance(tag, entities.RunTag)

    def test_delete_run(self):
        run = self._run_factory()
        self.session.commit()

        run_uuid = run.run_uuid
        self.store.delete_run(run_uuid)
        actual = self.session.query(models.SqlRun).filter_by(run_uuid=run_uuid).first()
        self.assertEqual(actual.lifecycle_stage, entities.RunInfo.DELETED_LIFECYCLE)

        deleted_run = self.store.get_run(run_uuid)
        self.assertEqual(actual.run_uuid, deleted_run.info.run_uuid)

    def test_log_metric(self):
        run = self._run_factory()

        self.session.commit()

        tkey = 'blahmetric'
        tval = 100.0
        metric = entities.Metric(tkey, tval, int(time.time()))
        metric2 = entities.Metric(tkey, tval, int(time.time()) + 2)
        self.store.log_metric(run.run_uuid, metric)
        self.store.log_metric(run.run_uuid, metric2)

        actual = self.session.query(models.SqlMetric).filter_by(key=tkey, value=tval)

        self.assertIsNotNone(actual)

        run = self.store.get_run(run.run_uuid)

        self.assertEqual(4, len(run.data.metrics))
        found = False
        for m in run.data.metrics:
            if m.key == tkey and m.value == tval:
                found = True

        self.assertTrue(found)

    def test_log_metric_uniqueness(self):
        run = self._run_factory()

        self.session.commit()

        tkey = 'blahmetric'
        tval = 100.0
        metric = entities.Metric(tkey, tval, int(time.time()))
        metric2 = entities.Metric(tkey, 1.02, int(time.time()))
        self.store.log_metric(run.run_uuid, metric)

        with self.assertRaises(MlflowException):
            self.store.log_metric(run.run_uuid, metric2)

    def test_log_param(self):
        run = self._run_factory('test')

        self.session.commit()

        tkey = 'blahmetric'
        tval = '100.0'
        param = entities.Param(tkey, tval)
        param2 = entities.Param('new param', 'new key')
        self.store.log_param(run.run_uuid, param)
        self.store.log_param(run.run_uuid, param2)

        actual = self.session.query(models.SqlParam).filter_by(key=tkey, value=tval)
        self.assertIsNotNone(actual)

        run = self.store.get_run(run.run_uuid)
        self.assertEqual(4, len(run.data.params))

        found = False
        for m in run.data.params:
            if m.key == tkey and m.value == tval:
                found = True

        self.assertTrue(found)

    def test_log_param_uniqueness(self):
        run = self._run_factory('test')

        self.session.commit()

        tkey = 'blahmetric'
        tval = '100.0'
        param = entities.Param(tkey, tval)
        param2 = entities.Param(tkey, 'newval')
        self.store.log_param(run.run_uuid, param)

        with self.assertRaises(MlflowException):
            self.store.log_param(run.run_uuid, param2)

    def test_set_tag(self):
        run = self._run_factory('test')

        self.session.commit()

        tkey = 'test tag'
        tval = 'a boogie'
        tag = entities.RunTag(tkey, tval)
        self.store.set_tag(run.run_uuid, tag)

        actual = self.session.query(models.SqlTag).filter_by(key=tkey, value=tval)

        self.assertIsNotNone(actual)

        run = self.store.get_run(run.run_uuid)

        found = False
        for m in run.data.tags:
            if m.key == tkey and m.value == tval:
                found = True

        self.assertTrue(found)

    def test_get_metric(self):
        run = self._run_factory('test')
        self.session.commit()

        for expected in run.metrics:
            actual = self.store.get_metric(run.run_uuid,
                                           expected.key)
            self.assertEqual(expected.value, actual)

    def test_get_param(self):
        run = self._run_factory('test')
        self.session.commit()

        for expected in run.params:
            actual = self.store.get_param(run.run_uuid,
                                          expected.key)
            self.assertEqual(expected.value, actual)

    def test_get_metric_history(self):
        run = self._run_factory('test')
        self.session.commit()
        key = 'test'
        expected = [
            models.SqlMetric(key=key, value=0.6, timestamp=1).to_mlflow_entity(),
            models.SqlMetric(key=key, value=0.7, timestamp=2).to_mlflow_entity()
        ]

        for metric in expected:
            self.store.log_metric(run.run_uuid, metric)

        actual = self.store.get_metric_history(run.run_uuid, key)

        self.assertEqual(len(expected), len(actual))

    def test_list_run_infos(self):
        exp1 = self._experiment_factory('test_exp')
        runs = [
            self._run_factory('t1', exp1.experiment_id).to_mlflow_entity(),
            self._run_factory('t2', exp1.experiment_id).to_mlflow_entity(),
        ]

        expected = [run.info for run in runs]

        actual = self.store.list_run_infos(exp1.experiment_id)

        self.assertEqual(len(expected), len(actual))

    def test_rename_experiment(self):
        new_name = 'new name'
        experiment = self._experiment_factory('test name')
        self.store.rename_experiment(experiment.experiment_id, new_name)

        renamed_experiment = self.store.get_experiment(experiment.experiment_id)

        self.assertEqual(renamed_experiment.name, new_name)

    def test_update_run_info(self):
        run = self._run_factory()
        new_status = entities.RunStatus.FINISHED
        endtime = int(time.time())

        actual = self.store.update_run_info(run.run_uuid, new_status, endtime)

        self.assertEqual(actual.status, entities.RunStatus.to_string(new_status))
        self.assertEqual(actual.end_time, endtime)

    def test_restore_experiment(self):
        exp = self._experiment_factory('helloexp')
        self.assertEqual(exp.lifecycle_stage, entities.Experiment.ACTIVE_LIFECYCLE)

        experiment_id = exp.experiment_id
        self.store.delete_experiment(experiment_id)

        deleted = self.store.get_experiment(experiment_id)
        self.assertEqual(deleted.experiment_id, experiment_id)
        self.assertEqual(deleted.lifecycle_stage, entities.Experiment.DELETED_LIFECYCLE)

        self.store.restore_experiment(exp.experiment_id)
        restored = self.store.get_experiment(exp.experiment_id)
        self.assertEqual(restored.experiment_id, experiment_id)
        self.assertEqual(restored.lifecycle_stage, entities.Experiment.ACTIVE_LIFECYCLE)

    def test_restore_run(self):
        run = self._run_factory()
        self.assertEqual(run.lifecycle_stage, entities.RunInfo.ACTIVE_LIFECYCLE)

        run_uuid = run.run_uuid
        self.store.delete_run(run_uuid)

        deleted = self.store.get_run(run_uuid)
        self.assertEqual(deleted.info.run_uuid, run_uuid)
        self.assertEqual(deleted.info.lifecycle_stage, entities.RunInfo.DELETED_LIFECYCLE)

        self.store.restore_run(run_uuid)
        restored = self.store.get_run(run_uuid)
        self.assertEqual(restored.info.run_uuid, run_uuid)
        self.assertEqual(restored.info.lifecycle_stage, entities.RunInfo.ACTIVE_LIFECYCLE)
