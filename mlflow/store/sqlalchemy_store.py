import sqlalchemy
from sqlalchemy import orm
from sqlalchemy.exc import IntegrityError
from mlflow.store.dbmodels import models
from mlflow import entities
from mlflow.store.abstract_store import AbstractStore
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
import mlflow.protos.databricks_pb2 as error_codes


class SqlAlchemyStore(AbstractStore):

    def __init__(self, db_uri):
        super(SqlAlchemyStore, self).__init__()
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

    def _get_or_create(self, model, **kwargs):
        instance = self.session.query(model).filter_by(**kwargs).first()
        create = False

        if instance:
            return instance, create
        else:
            instance = model(**kwargs)
            self._save_to_db(instance)
            create = True

        return instance, create

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        mappings = {
            ViewType.ACTIVE_ONLY: entities.Experiment.ACTIVE_LIFECYCLE,
            ViewType.DELETED_ONLY: entities.Experiment.DELETED_LIFECYCLE
        }

        lifecycle_stage = mappings[view_type]
        experiments = []
        for exp in self.session.query(models.SqlExperiment).filter_by(
                lifecycle_stage=lifecycle_stage, is_deleted=False):
            experiments.append(exp.to_mlflow_entity())

        return experiments

    def create_experiment(self, name, artifact_location=None):
        if name is None or name == '':
            raise MlflowException('Invalid experiment name', error_codes.INVALID_PARAMETER_VALUE)
        try:
            experiment = models.SqlExperiment(
                name=name,  lifecycle_stage=entities.Experiment.ACTIVE_LIFECYCLE,
                artifact_location=artifact_location
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
            experiment_id=experiment_id, is_deleted=False).first()

        if not exp:
            raise MlflowException('No Experiment with id={} exists'.format(experiment_id),
                                  error_codes.RESOURCE_DOES_NOT_EXIST)

        return exp.to_mlflow_entity()

    def delete_experiment(self, experiment_id):
        exp = self.session.query(models.SqlExperiment).get(experiment_id)
        exp.is_deleted = True
        self._save_to_db(exp)

    def restore_experiment(self, experiment_id):
        exp = self.session.query(models.SqlExperiment).get(experiment_id)
        exp.is_deleted = False
        self._save_to_db(exp)

    def rename_experiment(self, experiment_id, new_name):
        experiment = self.session.query(models.SqlExperiment).get(experiment_id)
        experiment.name = new_name
        self._save_to_db(experiment)

    def create_run(self, experiment_id, user_id, run_name, source_type, source_name,
                   entry_point_name, start_time, source_version, tags, parent_run_id):
        _ = parent_run_id
        experiment = self.get_experiment(experiment_id)

        if experiment.lifecycle_stage != entities.Experiment.ACTIVE_LIFECYCLE:
            raise MlflowException('Experiment id={} must be active'.format(experiment_id),
                                  error_codes.INVALID_STATE)
        status = entities.RunStatus.to_string(entities.RunStatus.RUNNING)
        run = models.SqlRun(name=run_name, artifact_uri=None,
                            experiment_id=experiment_id, source_type=source_type,
                            source_name=source_name, entry_point_name=entry_point_name,
                            user_id=user_id, status=status,
                            start_time=start_time, end_time=None,
                            source_version=source_version,
                            lifecycle_stage=entities.RunInfo.ACTIVE_LIFECYCLE)

        for tag in tags:
            run.tags.append(models.SqlTag(key=tag.key, value=tag.value))
        self._save_to_db([run])

        run = run.to_mlflow_entity()

        return run

    def update_run_info(self, run_uuid, run_status, end_time):
        run = self.session.query(models.SqlRun).filter_by(run_uuid=run_uuid).first()
        run.status = run_status
        run.end_time = end_time

        self._save_to_db(run)
        run = run.to_mlflow_entity()

        return run.info

    def restore_run(self, run_id):
        run = self.session.query(models.SqlRun).filter_by(run_uuid=run_id).first()
        run.is_deleted = False
        self._save_to_db(run)

    def get_run(self, run_uuid):
        run = self.session.query(models.SqlRun).filter_by(run_uuid=run_uuid,
                                                          is_deleted=False).first()
        if run is None:
            raise MlflowException('Run(uuid={}) doesn\'t exist'.format(run_uuid),
                                  error_codes.RESOURCE_DOES_NOT_EXIST)

        return run.to_mlflow_entity()

    def delete_run(self, run_id):
        run = self.session.query(models.SqlRun).filter_by(run_uuid=run_id).first()
        run.is_deleted = True
        self._save_to_db(run)

    def log_metric(self, run_uuid, metric):
        try:
            self._get_or_create(models.SqlMetric, run_uuid=run_uuid, key=metric.key,
                                value=metric.value, timestamp=metric.timestamp)
        except IntegrityError:
            raise MlflowException('Metric={} must be unique'.format(metric),
                                  error_codes.INVALID_PARAMETER_VALUE)

    def log_param(self, run_uuid, param):
        # if we try to update the value of an existing param this will fail
        # because it will try to create it with same run_uuid, param key
        try:
            self._get_or_create(models.SqlParam, run_uuid=run_uuid, key=param.key,
                                value=param.value)
        except IntegrityError:
            raise MlflowException('changing parameter {} value is not allowed'.format((run_uuid,
                                                                                      param)),
                                  error_codes.INVALID_PARAMETER_VALUE)

    def set_tag(self, run_uuid, tag):
        run = self.session.query(models.SqlRun).filter_by(run_uuid=run_uuid).first()

        new_tag = models.SqlTag(key=tag.key, value=tag.value)
        run.tags.append(new_tag)
        self._save_to_db([run, new_tag])

    def get_metric(self, run_uuid, metric_key):
        run = self.get_run(run_uuid)

        for metric in run.data.metrics:
            if metric.key == metric_key:
                return metric.value

        raise MlflowException('Metric={} does not exist'.format(metric_key),
                              error_codes.RESOURCE_DOES_NOT_EXIST)

    def get_param(self, run_uuid, param_name):
        run = self.get_run(run_uuid)

        for param in run.data.params:
            if param.key == param_name:
                return param.value

        raise MlflowException('Param={} does not exist'.format(param_name),
                              error_codes.RESOURCE_DOES_NOT_EXIST)

    def get_metric_history(self, run_uuid, metric_key):
        run = self.get_run(run_uuid)
        metrics_values = []

        for metric in run.data.metrics:
            if metric.key == metric_key:
                metrics_values.append(metric.value)

        return metrics_values

    def search_runs(self, experiment_ids, search_expressions, run_view_type):
        raise NotImplementedError()

    def list_run_infos(self, experiment_id, _=None):
        exp = self.session.query(models.SqlExperiment).get(experiment_id)
        infos = []
        for run in exp.runs:
            infos.append(run.to_mlflow_entity().info)
        return infos
