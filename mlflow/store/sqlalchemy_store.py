import sqlalchemy
import uuid

from six.moves import urllib

from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.store.dbmodels.db_types import MYSQL
from mlflow.store.dbmodels.models import Base, SqlExperiment, SqlRun, SqlMetric, SqlParam, SqlTag
from mlflow.entities import RunStatus, SourceType, Experiment
from mlflow.store.abstract_store import AbstractStore
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS, \
    INVALID_STATE, RESOURCE_DOES_NOT_EXIST
from mlflow.tracking.utils import _is_local_uri
from mlflow.utils.file_utils import build_path, mkdir
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
from mlflow.utils.search_utils import does_run_match_clause


class SqlAlchemyStore(AbstractStore):
    ARTIFACTS_FOLDER_NAME = "artifacts"

    def __init__(self, db_uri, default_artifact_root):
        super(SqlAlchemyStore, self).__init__()
        self.db_uri = db_uri
        self.db_type = urllib.parse.urlparse(db_uri).scheme
        self.artifact_root_uri = default_artifact_root
        self.engine = sqlalchemy.create_engine(db_uri)
        Base.metadata.create_all(self.engine)
        Base.metadata.bind = self.engine
        self.SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.session = self.SessionMaker()

        if _is_local_uri(default_artifact_root):
            mkdir(default_artifact_root)

        if len(self.list_experiments()) == 0:
            self._create_default_experiment()

    def _set_no_auto_for_zero_values(self):
        if self.db_type == MYSQL:
            self.session.execute("SET @@SESSION.sql_mode='NO_AUTO_VALUE_ON_ZERO';")

    # DB helper methods to allow zero values for columns with auto increments
    def _unset_no_auto_for_zero_values(self):
        if self.db_type == MYSQL:
            self.session.execute("SET @@SESSION.sql_mode='';")

    def _create_default_experiment(self):
        """
        MLflow UI and client code expects a default experiment with ID 0.
        This method uses SQL insert statement to create the default experiment as a hack, since
        experiment table uses 'experiment_id' column is a PK and is also set to auto increment.
        MySQL and other implementation do not allow value '0' for such cases.

        ToDo: Identify a less hack mechanism to create default experiment 0
        """
        table = SqlExperiment.__tablename__
        default_experiment = {
            SqlExperiment.experiment_id.name: Experiment.DEFAULT_EXPERIMENT_ID,
            SqlExperiment.name.name: Experiment.DEFAULT_EXPERIMENT_NAME,
            SqlExperiment.artifact_location.name: self._get_artifact_location(0),
            SqlExperiment.lifecycle_stage.name: LifecycleStage.ACTIVE
        }

        def decorate(s):
            if isinstance(s, str):
                return "'{}'".format(s)
            else:
                return "{}".format(s)

        # Get a list of keys to ensure we have a deterministic ordering
        columns = list(default_experiment.keys())
        values = ", ".join([decorate(default_experiment.get(c)) for c in columns])

        try:
            self._set_no_auto_for_zero_values()
            self.session.execute("INSERT INTO {} ({}) VALUES ({});".format(table,
                                                                           ", ".join(columns),
                                                                           values))
        finally:
            self._unset_no_auto_for_zero_values()
        self.session.commit()

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

    def _get_artifact_location(self, experiment_id):
        return build_path(self.artifact_root_uri, str(experiment_id))

    def create_experiment(self, name, artifact_location=None):
        if name is None or name == '':
            raise MlflowException('Invalid experiment name', INVALID_PARAMETER_VALUE)

        new_session = self.SessionMaker()
        try:
            experiment = SqlExperiment(
                name=name, lifecycle_stage=LifecycleStage.ACTIVE,
                artifact_location=artifact_location
            )
            new_session.add(experiment)
            if not artifact_location:
                # this requires a double write. The first one to generate an autoincrement-ed ID
                eid = new_session.query(SqlExperiment).filter_by(name=name).first().experiment_id
                experiment.artifact_location = self._get_artifact_location(eid)
            new_session.commit()
        except sqlalchemy.exc.IntegrityError as e:
            new_session.rollback()
            raise MlflowException('Experiment(name={}) already exists. '
                                  'Error: {}'.format(name, str(e)), RESOURCE_ALREADY_EXISTS)

        return experiment.experiment_id

    def _list_experiments(self, ids=None, names=None, view_type=ViewType.ACTIVE_ONLY):
        stages = LifecycleStage.view_type_to_stages(view_type)
        conditions = [SqlExperiment.lifecycle_stage.in_(stages)]

        if ids and len(ids) > 0:
            conditions.append(SqlExperiment.experiment_id.in_(ids))

        if names and len(names) > 0:
            conditions.append(SqlExperiment.name.in_(names))

        return self.session.query(SqlExperiment).filter(*conditions)

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        return [exp.to_mlflow_entity() for exp in self._list_experiments(view_type=view_type)]

    def _get_experiment(self, experiment_id, view_type):
        experiments = self._list_experiments(ids=[experiment_id], view_type=view_type).all()
        if len(experiments) == 0:
            raise MlflowException('No Experiment with id={} exists'.format(experiment_id),
                                  RESOURCE_DOES_NOT_EXIST)
        if len(experiments) > 1:
            raise MlflowException('Expected only 1 experiment with id={}. Found {}.'.format(
                experiment_id, len(experiments)), INVALID_STATE)

        return experiments[0]

    def get_experiment(self, experiment_id):
        return self._get_experiment(experiment_id, ViewType.ALL).to_mlflow_entity()

    def get_experiment_by_name(self, experiment_name):
        """
        Specialized implementation for SQL backed store.
        """
        experiments = self._list_experiments(names=[experiment_name], view_type=ViewType.ALL).all()
        if len(experiments) == 0:
            return None

        if len(experiments) > 1:
            raise MlflowException('Expected only 1 experiment with name={}. Found {}.'.format(
                experiment_name, len(experiments)), INVALID_STATE)

        return experiments[0]

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
        artifact_location = build_path(experiment.artifact_location, run_uuid,
                                       SqlAlchemyStore.ARTIFACTS_FOLDER_NAME)
        run = SqlRun(name=run_name or "", artifact_uri=artifact_location, run_uuid=run_uuid,
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

    def _get_run(self, run_uuid):
        runs = self.session.query(SqlRun).filter(SqlRun.run_uuid == run_uuid).all()

        if len(runs) == 0:
            raise MlflowException('Run with id={} not found'.format(run_uuid),
                                  RESOURCE_DOES_NOT_EXIST)
        if len(runs) > 1:
            raise MlflowException('Expected only 1 run with id={}. Found {}.'.format(run_uuid,
                                                                                     len(runs)),
                                  INVALID_STATE)

        return runs[0]

    def _check_run_is_active(self, run):
        if run.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException("The run {} must be in 'active' state. Current state is {}."
                                  .format(run.run_uuid, run.lifecycle_stage))

    def _check_run_is_deleted(self, run):
        if run.lifecycle_stage != LifecycleStage.DELETED:
            raise MlflowException("The run {} must be in 'deleted' state. Current state is {}."
                                  .format(run.run_uuid, run.lifecycle_stage))

    def update_run_info(self, run_uuid, run_status, end_time):
        run = self._get_run(run_uuid)
        self._check_run_is_active(run)
        run.status = RunStatus.to_string(run_status)
        run.end_time = end_time

        self._save_to_db(run)
        run = run.to_mlflow_entity()

        return run.info

    def get_run(self, run_uuid):
        run = self._get_run(run_uuid)
        return run.to_mlflow_entity()

    def restore_run(self, run_id):
        run = self._get_run(run_id)
        self._check_run_is_deleted(run)
        run.lifecycle_stage = LifecycleStage.ACTIVE
        self._save_to_db(run)

    def delete_run(self, run_id):
        run = self._get_run(run_id)
        self._check_run_is_active(run)
        run.lifecycle_stage = LifecycleStage.DELETED
        self._save_to_db(run)

    def log_metric(self, run_uuid, metric):
        run = self._get_run(run_uuid)
        self._check_run_is_active(run)
        try:
            # This will check for various integrity checks for metrics table.
            # ToDo: Consider prior checks for null, type, metric name validations, ... etc.
            self._get_or_create(SqlMetric, run_uuid=run_uuid, key=metric.key,
                                value=metric.value, timestamp=metric.timestamp)
        except sqlalchemy.exc.IntegrityError as ie:
            # Querying metrics from run entails pushing the query down to DB layer.
            # Hence the rollback.
            self.session.rollback()
            existing_metric = [m for m in run.metrics
                               if m.key == metric.key and m.timestamp == metric.timestamp]
            if len(existing_metric) == 0:
                raise MlflowException("Log metric request failed for run ID={}. Attempted to log"
                                      " metric={}. Error={}".format(run_uuid,
                                                                    (metric.key, metric.value),
                                                                    str(ie)))
            else:
                m = existing_metric[0]
                raise MlflowException('Metric={} must be unique. Metric already logged value {} '
                                      'at {}'.format(metric, m.value, m.timestamp),
                                      INVALID_PARAMETER_VALUE)

    def get_metric_history(self, run_uuid, metric_key):
        metrics = self.session.query(SqlMetric).filter_by(run_uuid=run_uuid, key=metric_key).all()
        return [metric.to_mlflow_entity() for metric in metrics]

    def log_param(self, run_uuid, param):
        run = self._get_run(run_uuid)
        self._check_run_is_active(run)
        # if we try to update the value of an existing param this will fail
        # because it will try to create it with same run_uuid, param key
        try:
            # This will check for various integrity checks for params table.
            # ToDo: Consider prior checks for null, type, param name validations, ... etc.
            self._get_or_create(SqlParam, run_uuid=run_uuid, key=param.key,
                                value=param.value)
        except sqlalchemy.exc.IntegrityError as ie:
            # Querying metrics from run entails pushing the query down to DB layer.
            # Hence the rollback.
            self.session.rollback()
            existing_params = [p.value for p in run.params if p.key == param.key]
            if len(existing_params) == 0:
                raise MlflowException("Log param request failed for run ID={}. Attempted to log"
                                      " param={}. Error={}".format(run_uuid,
                                                                   (param.key, param.value),
                                                                   str(ie)))
            else:
                old_value = existing_params[0]
                raise MlflowException("Changing param value is not allowed. Param with key='{}' was"
                                      " already logged with value='{}' for run ID='{}. Attempted "
                                      " logging new value '{}'.".format(param.key, old_value,
                                                                        run_uuid, param.value),
                                      INVALID_PARAMETER_VALUE)

    def set_tag(self, run_uuid, tag):
        run = self._get_run(run_uuid)
        self._check_run_is_active(run)
        new_tag = SqlTag(run_uuid=run_uuid, key=tag.key, value=tag.value)
        self._save_to_db(new_tag)

    def search_runs(self, experiment_ids, search_expressions, run_view_type):
        runs = [run.to_mlflow_entity()
                for exp in experiment_ids
                for run in self._list_runs(exp, run_view_type)]
        if len(search_expressions) == 0:
            return runs
        return [r for r in runs if all([does_run_match_clause(r, s) for s in search_expressions])]

    def _list_runs(self, experiment_id, run_view_type):
        exp = self._list_experiments(ids=[experiment_id], view_type=ViewType.ALL).first()
        stages = set(LifecycleStage.view_type_to_stages(run_view_type))
        return [run for run in exp.runs if run.lifecycle_stage in stages]
