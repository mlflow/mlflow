import unittest
import sqlalchemy
import shutil
from mlflow.store.dbmodels import models
from mlflow.store.sqlalchemy_store import SqlAlchemyStore


class TestSqlAlchemyStore(unittest.TestCase):

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
            res = self.session.query(models.Experiment).filter_by(id=exp.experiment_id).first()
            self.assertEqual(res.name, exp.name)
            self.assertEqual(res.id, exp.experiment_id)

    def test_create_experiments(self):
        result = self.session.query(models.Experiment).all()
        self.assertEqual(len(result), 0)

        expected = self.store.create_experiment(name='test experiment')
        result = self.session.query(models.Experiment).all()
        self.assertEqual(len(result), 1)

        actual = result[0]

        self.assertEqual(actual.id, expected.experiment_id)
        self.assertEqual(actual.name, expected.name)
