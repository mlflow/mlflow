import sqlalchemy
import os
import uuid
from sqlalchemy.exc import IntegrityError
from mlflow.store.abstract_store import AbstractStore
from mlflow.store.dbmodels import models
from mlflow import entities
from mlflow.utils.env import get_env
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
import mlflow.protos.databricks_pb2 as error_codes


_TRACKING_DIR_ENV_VAR = "MLFLOW_TRACKING_DIR"


def _default_root_dir():
    return get_env(_TRACKING_DIR_ENV_VAR) or os.path.abspath("mlruns")


class SqlAlchemyStore(object):

    def __init__(self, db_uri='sqlite://'):

        self.engine = sqlalchemy.create_engine(db_uri)
        models.Base.metadata.create_all(self.engine)
        models.Base.metadata.bind = self.engine
        DBSession = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.session = DBSession()

    def _save_to_db(self, objs):
        """
        Store in db
        """

        if type(objs) is list:
            self.session.add_all(objs)
        else:
            # single object
            self.session.add(objs)

        self.session.commit()

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        experiments = []
        for exp in self.session.query(models.SqlExperiment).all():
            experiments.append(exp.to_mlflow_entity())

        return experiments

    def create_experiment(self, name, artifact_store=None):
        if name is None or name == '':
            raise MlflowException('Invalid experiment name', error_codes.INVALID_PARAMETER_VALUE)
        try:
            experiment = models.SqlExperiment(
                name=name,  lifecycle_stage=entities.Experiment.ACTIVE_LIFECYCLE)
            self.session.add(experiment)
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            raise MlflowException('Experiment(name={}) already exists'.format(name),
                                  error_codes.RESOURCE_ALREADY_EXISTS)

        return experiment.to_mlflow_entity()

    def get_experiment(self, experiment_id):
        exp = self.session.query(models.SqlExperiment).filter_by(
            experiment_id=experiment_id).first()

        if not exp:
            raise MlflowException('No Experiment with id={} exists'.format(experiment_id),
                                  error_codes.RESOURCE_DOES_NOT_EXIST)

        return exp.to_mlflow_entity()

    def delete_experiment(self, experiment_id):
        self.session.query(models.SqlExperiment).filter_by(
            experiment_id=experiment_id).delete()
        self.session.commit()

    def restore_experiment(self, experiment_id):
        raise NotImplementedError()

    def rename_experiment(self, experiment_id, new_name):
        raise NotImplementedError()

    def create_run(self, experiment_id, user_id, run_name, source_type, source_name,
                   entry_point_name, start_time, source_version, tags, parent_run_id):
        experiment = self.get_experiment(experiment_id)

        if experiment.lifecycle_stage != entities.Experiment.ACTIVE_LIFECYCLE:
            raise MlflowException('Experiment with id={} must be active to create run',
                                  error_codes.INVALID_STATE)

        run_uuid = uuid.uuid4().hex
        run_info = models.SqlRunInfo(name=run_name, artifact_uri=None,
                                     experiment_id=experiment_id, source_type=source_type,
                                     source_name=source_name, entry_point_name=entry_point_name,
                                     user_id=user_id, status=entities.RunStatus.RUNNING,
                                     start_time=start_time, end_time=none,
                                     source_version=source_version,
                                     lifecycle_stage=entities.RunInfo.ACTIVE_LIFECYCLE)

        run_data = models.SqlRunData()
        run = models.SqlRun(run_info=run_info, run_data=run_data)
        self._save_to_db([run_info, run_data, run])

        run = run.to_mlflow_entity()

        return run, run.info, run.data

    def get_run(self, run_uuid):
        # TODO this won't always work need to fix how to subquery related models
        run_info = self.session.query(models.SqlRunInfo).filter_by(run_uuid=run_uuid).first()
        if run_info is None or run_info is None:
            raise MlflowException('Run(uuid={}) doesn\'t exist',
                                  error_codes.RESOURCE_DOES_NOT_EXIST)

        return run_info.run.to_mlflow_entity()

    def delete_run(self, run_uuid):
        run_info = self.session.query(models.SqlRunInfo).filter_by(run_uuid=run_uuid).first()
        run = run_info.run
        self.session.delete(run)

