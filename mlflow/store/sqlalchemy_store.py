import sqlalchemy
from sqlalchemy import orm
from sqlalchemy.exc import IntegrityError
from mlflow.store.dbmodels import models
from mlflow import entities
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
import mlflow.protos.databricks_pb2 as error_codes


class SqlAlchemyStore(object):

    def __init__(self, db_uri='sqlite://'):

        self.engine = sqlalchemy.create_engine(db_uri)
        models.Base.metadata.create_all(self.engine)
        models.Base.metadata.bind = self.engine
        db_session = orm.sessionmaker(bind=self.engine)
        self.session = db_session()

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
        mappings = {
            ViewType.ACTIVE_ONLY: entities.Experiment.ACTIVE_LIFECYCLE,
            ViewType.DELETED_ONLY: entities.Experiment.DELETED_LIFECYCLE
        }

        lifecycle_stage = mappings[view_type]
        experiments = []
        for exp in self.session.query(models.SqlExperiment).filter_by(
                lifecycle_stage=lifecycle_stage):
            experiments.append(exp.to_mlflow_entity())

        return experiments

    def create_experiment(self, name, artifact_store=None):
        if name is None or name == '':
            raise MlflowException('Invalid experiment name', error_codes.INVALID_PARAMETER_VALUE)
        try:
            experiment = models.SqlExperiment(
                name=name,  lifecycle_stage=entities.Experiment.ACTIVE_LIFECYCLE,
                artifact_location=artifact_store
            )
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
                   entry_point_name, start_time, source_version, tags, _parent_run_id):
        experiment = self.get_experiment(experiment_id)

        if experiment.lifecycle_stage != entities.Experiment.ACTIVE_LIFECYCLE:
            raise MlflowException('Experiment id={} must be active'.format(experiment_id),
                                  error_codes.INVALID_STATE)

        run_info = models.SqlRunInfo(name=run_name, artifact_uri=None,
                                     experiment_id=experiment_id, source_type=source_type,
                                     source_name=source_name, entry_point_name=entry_point_name,
                                     user_id=user_id, status=entities.RunStatus.RUNNING,
                                     start_time=start_time, end_time=None,
                                     source_version=source_version,
                                     lifecycle_stage=entities.RunInfo.ACTIVE_LIFECYCLE)

        run_data = models.SqlRunData()
        tag_models = [models.SqlRunTag(key=tag.key, value=tag.value) for tag in tags]
        run = models.SqlRun(run_info=run_info, run_data=run_data)
        self._save_to_db([run_info, run_data, run] + tag_models)

        run = run.to_mlflow_entity()

        return run, run.info, run.data

    def update_run_info(self, run_uuid, run_status, end_time):
        raise NotImplementedError()

    def restore_run(self, run_id):
        raise NotImplementedError()

    def get_run(self, run_uuid):
        # TODO this won't always work need to fix how to subquery related models
        run_info = self.session.query(models.SqlRunInfo).filter_by(run_uuid=run_uuid).first()
        if run_info is None or run_info is None:
            raise MlflowException('Run(uuid={}) doesn\'t exist'.format(run_uuid),
                                  error_codes.RESOURCE_DOES_NOT_EXIST)

        return run_info.run.to_mlflow_entity()

    def delete_run(self, run_uuid):
        run_info = self.session.query(models.SqlRunInfo).filter_by(run_uuid=run_uuid).first()
        run = run_info.run
        self.session.delete(run)

    def log_metric(self, run_uuid, metric):
        run_info = self.session.query(models.SqlRunInfo).filter_by(run_uuid=run_uuid).first()
        run = run_info.run

        new_metric = models.SqlMetric(key=metric.key, value=metric.value)
        run.data.metrics.append(new_metric)
        self._save_to_db([run, new_metric])

    def log_param(self, run_uuid, param):
        run_info = self.session.query(models.SqlRunInfo).filter_by(run_uuid=run_uuid).first()
        run = run_info.run

        new_param = models.SqlParam(key=param.key, value=param.value)
        run.data.params.append(new_param)
        self._save_to_db([run, new_param])

    def set_tag(self, run_uuid, tag):
        run_info = self.session.query(models.SqlRunInfo).filter_by(run_uuid=run_uuid).first()
        run = run_info.run

        new_tag = models.SqlRunTag(key=tag.key, value=tag.value)
        run.data.tags.append(new_tag)
        self._save_to_db([run, new_tag])

    def get_metric(self, run_uuid, key):
        run = self.get_run(run_uuid)

        for metric in run.data.metrics:
            if metric.key == key:
                return metric.value

        raise MlflowException('Metric={} does not exist'.format(key),
                              error_codes.RESOURCE_DOES_NOT_EXIST)
