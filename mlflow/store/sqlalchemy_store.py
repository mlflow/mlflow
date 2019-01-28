import sqlalchemy
from sqlalchemy.sql.expression import func, label, and_
import uuid

from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.store.dbmodels.models import Base, SqlExperiment, SqlRun, SqlMetric, SqlParam, SqlTag
from mlflow.entities import RunStatus, SourceType
from mlflow.store.abstract_store import AbstractStore
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS, \
    INVALID_STATE, RESOURCE_DOES_NOT_EXIST
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
from mlflow.utils.search_utils import does_run_match_clause


class SqlAlchemyStore(AbstractStore):

    def __init__(self, db_uri):
        super(SqlAlchemyStore, self).__init__()
        self.engine = sqlalchemy.create_engine(db_uri)
        Base.metadata.create_all(self.engine)
        Base.metadata.bind = self.engine
        db_session = sqlalchemy.orm.sessionmaker(bind=self.engine)
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
        created = False

        if instance:
            return instance, created
        else:
            instance = model(**kwargs)
            self._save_to_db(instance)
            created = True

        return instance, created

    def create_experiment(self, name, artifact_location=None):
        if name is None or name == '':
            raise MlflowException('Invalid experiment name', INVALID_PARAMETER_VALUE)
        try:
            experiment = SqlExperiment(
                name=name, lifecycle_stage=LifecycleStage.ACTIVE,
                artifact_location=artifact_location
            )
            self.session.add(experiment)
            self.session.commit()
        except sqlalchemy.exc.IntegrityError as e:
            self.session.rollback()
            print(e)
            raise MlflowException('Experiment(name={}) already exists'.format(name),
                                  RESOURCE_ALREADY_EXISTS, exc_info=e)

        return experiment.to_mlflow_entity()

    def _list_experiments(self, experiments, view_type=ViewType.ACTIVE_ONLY):
        stages = LifecycleStage.view_type_to_stages(view_type)
        conditions = [SqlExperiment.lifecycle_stage.in_(stages)]

        if len(experiments) > 0:
            conditions.append(SqlExperiment.experiment_id.in_(experiments))

        return self.session.query(SqlExperiment).filter(*conditions)

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        return [exp.to_mlflow_entity() for exp in self._list_experiments([], view_type)]

    def _get_experiment(self, experiment_id, view_type):
        experiments = self._list_experiments([experiment_id], view_type).all()
        if len(experiments) == 0:
            raise MlflowException('No Experiment with id={} exists'.format(experiment_id),
                                  RESOURCE_DOES_NOT_EXIST)
        if len(experiments) > 1:
            raise MlflowException('Expected only 1 experiment with id={}. Found {}.'.format(
                experiment_id, len(experiments)), INVALID_STATE)

        return experiments[0]

    def get_experiment(self, experiment_id):
        return self._get_experiment(experiment_id, ViewType.ALL).to_mlflow_entity()

    def delete_experiment(self, experiment_id):
        experiment = self._get_experiment(experiment_id, ViewType.ACTIVE_ONLY)
        experiment.lifecycle_stage = LifecycleStage.DELETED
        self._save_to_db(experiment)

    def restore_experiment(self, experiment_id):
        experiment = self._get_experiment(experiment_id, ViewType.DELETED_ONLY)
        experiment.lifecycle_stage = LifecycleStage.ACTIVE
        self._save_to_db(experiment)

    def rename_experiment(self, experiment_id, new_name):
        experiment = self._get_experiment(experiment_id, ViewType.ALL)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException('Cannot rename a non-active experiment.', INVALID_STATE)

        experiment.name = new_name
        self._save_to_db(experiment)

    def create_run(self, experiment_id, user_id, run_name, source_type, source_name,
                   entry_point_name, start_time, source_version, tags, parent_run_id):
        experiment = self.get_experiment(experiment_id)

        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException('Experiment id={} must be active'.format(experiment_id),
                                  INVALID_STATE)

        run_uuid = uuid.uuid4().hex
        run = SqlRun(name=run_name or "", artifact_uri=None, run_uuid=run_uuid,
                     experiment_id=experiment_id, source_type=SourceType.to_string(source_type),
                     source_name=source_name, entry_point_name=entry_point_name,
                     user_id=user_id, status=RunStatus.to_string(RunStatus.RUNNING),
                     start_time=start_time, end_time=None,
                     source_version=source_version, lifecycle_stage=LifecycleStage.ACTIVE)

        for tag in tags:
            run.tags.append(SqlTag(key=tag.key, value=tag.value))
        if parent_run_id:
            run.tags.append(SqlTag(key=MLFLOW_PARENT_RUN_ID, value=parent_run_id))
        if run_name:
            run.tags.append(SqlTag(key=MLFLOW_RUN_NAME, value=run_name))

        self._save_to_db([run])

        return run.to_mlflow_entity()

    def _get_run(self, run_uuid, view_type):
        stages = LifecycleStage.view_type_to_stages(view_type)
        runs = self.session.query(SqlRun).filter(SqlRun.run_uuid == run_uuid,
                                                 SqlRun.lifecycle_stage.in_(stages)).all()

        if len(runs) == 0:
            raise MlflowException('No runs with id={} exists'.format(run_uuid),
                                  RESOURCE_DOES_NOT_EXIST)
        if len(runs) > 1:
            raise MlflowException('Expected only 1 run with id={}. Found {}.'.format(run_uuid,
                                                                                     len(runs)),
                                  INVALID_STATE)

        return runs[0]

    def update_run_info(self, run_uuid, run_status, end_time):
        run = self._get_run(run_uuid, ViewType.ACTIVE_ONLY)
        run.status = RunStatus.to_string(run_status)
        run.end_time = end_time

        self._save_to_db(run)
        run = run.to_mlflow_entity()

        return run.info

    def get_run(self, run_uuid):
        run = self._get_run(run_uuid, ViewType.ALL)
        return run.to_mlflow_entity()

    def restore_run(self, run_id):
        run = self._get_run(run_id, ViewType.DELETED_ONLY)
        run.lifecycle_stage = LifecycleStage.ACTIVE
        self._save_to_db(run)

    def delete_run(self, run_id):
        run = self._get_run(run_id, ViewType.ACTIVE_ONLY)
        run.lifecycle_stage = LifecycleStage.DELETED
        self._save_to_db(run)

    def log_metric(self, run_uuid, metric):
        try:
            self._get_or_create(SqlMetric, run_uuid=run_uuid, key=metric.key,
                                value=metric.value, timestamp=metric.timestamp)
        except sqlalchemy.exc.IntegrityError:
            raise MlflowException('Metric={} must be unique'.format(metric),
                                  INVALID_PARAMETER_VALUE)

    def get_metric(self, run_uuid, metric_key):
        metric = self.session.query(SqlMetric).filter_by(
            run_uuid=run_uuid, key=metric_key
        ).order_by(sqlalchemy.desc(SqlMetric.timestamp)).first()

        if metric is None:
            raise MlflowException('Metric={} does not exist'.format(metric_key),
                                  RESOURCE_DOES_NOT_EXIST)

        return metric.to_mlflow_entity()

    def get_metric_history(self, run_uuid, metric_key):
        metrics = self.session.query(SqlMetric).filter_by(run_uuid=run_uuid, key=metric_key).all()
        return [metric.to_mlflow_entity() for metric in metrics]

    def log_param(self, run_uuid, param):
        # if we try to update the value of an existing param this will fail
        # because it will try to create it with same run_uuid, param key
        try:
            self._get_or_create(SqlParam, run_uuid=run_uuid, key=param.key,
                                value=param.value)
        except sqlalchemy.exc.IntegrityError:
            raise MlflowException('Changing param value is not allowed. Run={}'.format(run_uuid),
                                  INVALID_PARAMETER_VALUE)

    def get_param(self, run_uuid, param_name):
        param = self.session.query(SqlParam).filter_by(run_uuid=run_uuid, key=param_name).first()
        if param is None:
            raise MlflowException('Param={} does not exist'.format(param_name),
                                  RESOURCE_DOES_NOT_EXIST)
        return param.to_mlflow_entity()

    def set_tag(self, run_uuid, tag):
        new_tag = SqlTag(run_uuid=run_uuid, key=tag.key, value=tag.value)
        self._save_to_db(new_tag)

    def get_tag(self, run_uuid, tag_name):
        tag = self.session.query(SqlTag).filter_by(run_uuid=run_uuid, key=tag_name).first()
        if tag is None:
            raise MlflowException('Tag={} does not exist'.format(tag_name),
                                  RESOURCE_DOES_NOT_EXIST)
        return tag.to_mlflow_entity()

    def search_runs(self, experiment_ids, search_expressions, run_view_type):
        runs = [run.to_mlflow_entity()
                for exp in experiment_ids
                for run in self._list_runs(exp, run_view_type)]
        if len(search_expressions) == 0:
            return runs
        return [r for r in runs if all([does_run_match_clause(r, s) for s in search_expressions])]

    def _list_runs(self, experiment_id, run_view_type):
        exp = self._list_experiments(experiments=[experiment_id], view_type=ViewType.ALL).first()
        stages = set(LifecycleStage.view_type_to_stages(run_view_type))
        return [run for run in exp.runs if run.lifecycle_stage in stages]

    def list_run_infos(self, experiment_id, run_view_type):
        return [r.to_mlflow_entity().info for r in self._list_runs(experiment_id, run_view_type)]
