import unittest
import sqlalchemy
import shutil
import time
from mlflow.store.dbmodels import models
from mlflow import entities
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
        shutil.rmtree(self.store._root, ignore_errors=True)

    def _experiment_factory(self, names):
        if type(names) is list:
            experiments = []
            for name in names:
                exp = self.store.create_experiment(name=name)
                experiments.append(exp)

            return experiments

        return self.store.create_experiment(name=names)

    def test_delete_experiment(self):
        experiments = self._experiment_factory(['morty', 'rick', 'rick and morty'])
        expected = experiments[0]
        self.store.delete_experiment(expected.experiment_id)

        all_experiments = self.store.list_experiments()
        self.assertEqual(len(all_experiments), len(experiments) - 1)

        # assert the deleted experiment is not is list
        for exp in all_experiments:
            self.assertNotEqual(exp.experiment_id, expected.experiment_id)
            self.assertNotEqual(exp.name, expected.name)

    def test_get_experiment(self):
        name = 'goku'
        expected = self._experiment_factory(name)
        actual = self.store.get_experiment(expected.experiment_id)
        self.assertEqual(actual.name, expected.name)
        self.assertEqual(actual.experiment_id, expected.experiment_id)

    def test_list_experiments(self):
        testnames = ['blue', 'red', 'green']

        expected = self._experiment_factory(testnames)
        actual = self.store.list_experiments()

        self.assertEqual(len(expected), len(actual))

        for exp in expected:
            res = self.session.query(models.SqlExperiment).filter_by(
                experiment_id=exp.experiment_id).first()
            self.assertEqual(res.name, exp.name)
            self.assertEqual(res.experiment_id, exp.experiment_id)

    def test_create_experiments(self):
        result = self.session.query(models.SqlExperiment).all()
        self.assertEqual(len(result), 0)

        expected = self.store.create_experiment(name='test experiment')
        result = self.session.query(models.SqlExperiment).all()
        self.assertEqual(len(result), 1)

        actual = result[0]

        self.assertEqual(actual.experiment_id, expected.experiment_id)
        self.assertEqual(actual.name, expected.name)

    # def test_create_run_info(self):
    #     experiment = self._experiment_factory('test')

    #     config = {
    #         'run_uuid': 'abcder',
    #         'name': 'test run',
    #         'source_type': entities.source_type.SourceType.LOCAL,
    #         'source_name': 'Python Application',
    #         'entry_point_name': 'main.py',
    #         'start_time': int(time.time() * 1000),
    #         'source_version': '0.8.0',
    #         'tags': [entities.RunTag('key', 'val')],
    #         'parent_run_id': None
    #     }

    #     run_info = self.store._create_run_info(**config)

    #     self.session.query(models.RunInfo)
    #     for k, v in config.items():
    #         self.assertEqual(v, run_info[k])

    def test_run_tag_model(self):
        expected = models.SqlRunTag(key='test', value='val')
        self.session.add(expected)
        self.session.commit()
        tags = self.session.query(models.SqlRunTag).all()
        self.assertEqual(len(tags), 1)

        actual = tags[0].to_mlflow_entity()

        self.assertEqual(actual.value, expected.value)
        self.assertEqual(actual.key, expected.key)

    def test_metric_model(self):
        expected = models.SqlMetric(key='accuracy', value=0.89)
        self.session.add(expected)
        self.session.commit()
        metrics = self.session.query(models.SqlMetric).all()
        self.assertEqual(len(metrics), 1)

        actual = metrics[0].to_mlflow_entity()

        self.assertEqual(actual.value, expected.value)
        self.assertEqual(actual.key, expected.key)

    def test_param_model(self):
        expected = models.SqlParam(key='accuracy', value='test param')
        self.session.add(expected)
        self.session.commit()
        params = self.session.query(models.SqlParam).all()
        self.assertEqual(len(params), 1)

        actual = params[0].to_mlflow_entity()

        self.assertEqual(actual.value, expected.value)
        self.assertEqual(actual.key, expected.key)