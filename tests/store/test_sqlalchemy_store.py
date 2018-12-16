import unittest
import sqlalchemy
import time
import mlflow
import pytest
import uuid
from mlflow.store.dbmodels import models
from mlflow import entities
from mlflow.exceptions import MlflowException
from mlflow.store.sqlalchemy_store import SqlAlchemyStore


class TestSqlAlchemyStoreSqliteInMemory(unittest.TestCase):
    def setUp(self):
        self.store = SqlAlchemyStore()
        self.engine = sqlalchemy.create_engine('sqlite:///:memory:')
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
        run_data = experiments[0]
        self.store.delete_experiment(run_data.experiment_id)

        all_experiments = self.store.list_experiments()
        self.assertEqual(len(all_experiments), len(experiments) - 1)

        # assert the deleted experiment is not is list
        for exp in all_experiments:
            self.assertNotEqual(exp.experiment_id, run_data.experiment_id)
            self.assertNotEqual(exp.name, run_data.name)

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
        run_data = models.SqlRunTag(key='test', value='val')
        self.session.add(run_data)
        self.session.commit()
        tags = self.session.query(models.SqlRunTag).all()
        self.assertEqual(len(tags), 1)

        actual = tags[0].to_mlflow_entity()

        self.assertEqual(actual.value, run_data.value)
        self.assertEqual(actual.key, run_data.key)

    def test_metric_model(self):
        run_data = models.SqlMetric(key='accuracy', value=0.89)
        self.session.add(run_data)
        self.session.commit()
        metrics = self.session.query(models.SqlMetric).all()
        self.assertEqual(len(metrics), 1)

        actual = metrics[0].to_mlflow_entity()

        self.assertEqual(actual.value, run_data.value)
        self.assertEqual(actual.key, run_data.key)

    def test_param_model(self):
        run_data = models.SqlParam(key='accuracy', value='test param')
        self.session.add(run_data)
        self.session.commit()
        params = self.session.query(models.SqlParam).all()
        self.assertEqual(len(params), 1)

        actual = params[0].to_mlflow_entity()

        self.assertEqual(actual.value, run_data.value)
        self.assertEqual(actual.key, run_data.key)

    def test_run_data_model(self):
        m1 = models.SqlMetric(key='accuracy', value=0.89)
        m2 = models.SqlMetric(key='recal', value=0.89)
        p1 = models.SqlParam(key='loss', value='test param')
        p2 = models.SqlParam(key='blue', value='test param')

        self.session.add_all([m1, m2, p1, p2])

        run_data = models.SqlRunData()
        run_data.params.append(p1)
        run_data.params.append(p2)
        run_data.metrics.append(m1)
        run_data.metrics.append(m2)

        self.session.add(run_data)
        self.session.commit()

        run_datums = self.session.query(models.SqlRunData).all()
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
        run_info = models.SqlRunInfo(**config).to_mlflow_entity()

        for k, v in config.items():
            self.assertEqual(v, getattr(run_info, k))

    def _run_factory(self, experiment_id=None):
        m1 = models.SqlMetric(key='accuracy', value=0.89)
        m2 = models.SqlMetric(key='recal', value=0.89)
        p1 = models.SqlParam(key='loss', value='test param')
        p2 = models.SqlParam(key='blue', value='test param')

        self.session.add_all([m1, m2, p1, p2])

        data = models.SqlRunData()
        data.params.append(p1)
        data.params.append(p2)
        data.metrics.append(m1)
        data.metrics.append(m2)
        self.session.add(data)

        if not experiment_id:
            experiment = self._experiment_factory('test exp')
            experiment_id = experiment.experiment_id

        config = {
            'experiment_id': experiment_id,
            'name': 'test run',
            'user_id': 'Anderson',
            'run_uuid': uuid.uuid4().hex,
            'status': entities.RunStatus.SCHEDULED,
            'source_type': entities.SourceType.NOTEBOOK,
            'source_name': 'Python application',
            'entry_point_name': 'main.py',
            'start_time': int(time.time()),
            'end_time': int(time.time()),
            'source_version': mlflow.__version__,
            'lifecycle_stage': entities.RunInfo.ACTIVE_LIFECYCLE,
            'artifact_uri': '//'
        }
        info = models.SqlRunInfo(**config)

        run = models.SqlRun(experiment_id=experiment_id,
                            info=info, data=data)
        self.session.add_all([run, info, data])
        return run, info, data

    def test_run_model(self):
        run, info, data = self._run_factory()

        self.assertEqual(run.info.run_uuid, info.run_uuid)
        self.assertListEqual(run.data.metrics, data.metrics)
        self.assertListEqual(run.data.params, data.params)
        self.assertListEqual(run.data.tags, data.tags)

    def test_to_mlflow_entity(self):
        run, _, _ = self._run_factory()
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
        run, _, _ = self._run_factory()
        self.session.commit()

        run_uuid = run.info.run_uuid
        self.store.delete_run(run_uuid)

        self.assertEqual(self.session.query(models.SqlRunInfo).count(), 0)
        self.assertEqual(self.session.query(models.SqlRunData).count(), 0)
        self.assertEqual(self.session.query(models.SqlMetric).count(), 0)
        self.assertEqual(self.session.query(models.SqlParam).count(), 0)

        with pytest.raises(MlflowException) as e:
            self.store.get_run(run_uuid)

        error = e.value
        self.assertEqual(error.error_code, 'RESOURCE_DOES_NOT_EXIST')

    def test_log_metric(self):
        run, info, _ = self._run_factory()

        self.session.commit()

        run_uuid = info.run_uuid
        tkey = 'blahmetric'
        tval = 100.0
        metric = entities.Metric(tkey, tval, int(time.time()))
        self.store.log_metric(run_uuid, metric)

        actual = self.session.query(models.SqlMetric).filter_by(key=tkey, value=tval)

        self.assertIsNotNone(actual)

        run = self.store.get_run(run_uuid)

        found = False
        for m in run.data.metrics:
            if m.key == tkey and m.value == tval:
                found = True

        self.assertTrue(found)

    def test_log_param(self):
        run, info, _ = self._run_factory()

        self.session.commit()

        run_uuid = info.run_uuid
        tkey = 'blahmetric'
        tval = '100.0'
        param = entities.Param(tkey, tval)
        self.store.log_param(run_uuid, param)

        actual = self.session.query(models.SqlParam).filter_by(key=tkey, value=tval)

        self.assertIsNotNone(actual)

        run = self.store.get_run(run_uuid)

        found = False
        for m in run.data.params:
            if m.key == tkey and m.value == tval:
                found = True

        self.assertTrue(found)

    def test_set_tag(self):
        run, info, _ = self._run_factory()

        self.session.commit()

        run_uuid = info.run_uuid
        tkey = 'test tag'
        tval = 'a boogie'
        tag = entities.RunTag(tkey, tval)
        self.store.set_tag(run_uuid, tag)

        actual = self.session.query(models.SqlRunTag).filter_by(key=tkey, value=tval)

        self.assertIsNotNone(actual)

        run = self.store.get_run(run_uuid)

        found = False
        for m in run.data.tags:
            if m.key == tkey and m.value == tval:
                found = True

        self.assertTrue(found)

    def test_get_metric(self):
        run, _, data = self._run_factory()
        self.session.commit()

        for expected in data.metrics:
            actual = self.store.get_metric(run.info.run_uuid,
                                           expected.key)
            self.assertEqual(expected.value, actual)

    def test_get_param(self):
        run, _, data = self._run_factory()
        self.session.commit()

        for expected in data.params:
            actual = self.store.get_param(run.info.run_uuid,
                                          expected.key)
            self.assertEqual(expected.value, actual)

    def test_get_metric_history(self):
        run, _, _ = self._run_factory()
        self.session.commit()
        key = 'test'
        expected = [
            models.SqlMetric(key=key, value=0.6).to_mlflow_entity(),
            models.SqlMetric(key=key, value=0.7).to_mlflow_entity()
        ]

        for metric in expected:
            self.store.log_metric(run.info.run_uuid, metric)

        actual = self.store.get_metric_history(run.info.run_uuid, key)

        self.assertEqual(len(expected), len(actual))

    def test_list_run_infos(self):
        exp1 = self._experiment_factory('test_exp')
        runs = [
            self._run_factory(exp1.experiment_id),
            self._run_factory(exp1.experiment_id),
        ]

        expected = [run[1] for run in runs]

        actual = self.store.list_run_infos(exp1.experiment_id)

        self.assertEqual(len(expected), len(actual))

    def test_rename_experiment(self):
        new_name = 'new name'
        experiment = self._experiment_factory('test name')
        self.store.rename_experiment(experiment.experiment_id, new_name)

        renamed_experiment = self.store.get_experiment(experiment.experiment_id)

        self.assertEqual(renamed_experiment.name, new_name)

    def test_update_run_info(self):
        run, _, _ = self._run_factory()
        new_status = entities.RunStatus.FINISHED
        endtime = int(time.time())

        actual = self.store.update_run_info(run.info.run_uuid, new_status, endtime)

        self.assertEqual(actual.status, new_status)
        self.assertEqual(actual.end_time, endtime)
