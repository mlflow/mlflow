import sqlalchemy
import os
from mlflow.store.abstract_store import AbstractStore
from mlflow.store.dbmodels import models
from mlflow import entities
from mlflow.utils.env import get_env
from mlflow.entities import ViewType


_TRACKING_DIR_ENV_VAR = "MLFLOW_TRACKING_DIR"


def _default_root_dir():
    return get_env(_TRACKING_DIR_ENV_VAR) or os.path.abspath("mlruns")


class SqlAlchemyStore(object):

    def __init__(self):
        self._dbfilename = 'mlflow-runs.db'
        self._root = _default_root_dir()

        if not os.path.exists(self._root):
            os.mkdir(self._root)

        self._dbpath = os.path.join(self._root, self._dbfilename)
        self.engine = sqlalchemy.create_engine('sqlite:///{}'.format(self._dbpath))
        models.Base.metadata.create_all(self.engine)
        models.Base.metadata.bind = self.engine
        DBSession = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.session = DBSession()

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        experiments = []
        for exp in self.session.query(models.SqlExperiment).all():
            experiments.append(exp.to_mlflow_entity())

        return experiments

    def create_experiment(self, name, artifact_store=None):
        experiment = models.SqlExperiment(name=name)
        self.session.add(experiment)
        self.session.commit()
        return experiment.to_mlflow_entity()

    def get_experiment(self, experiment_id):
        exp = self.session.query(models.SqlExperiment).filter_by(
            experiment_id=experiment_id).first()

        return exp.to_mlflow_entity()

    def delete_experiment(self, experiment_id):
        self.session.query(models.SqlExperiment).filter_by(
            experiment_id=experiment_id).delete()
