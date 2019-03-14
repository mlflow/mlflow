import sqlalchemy
import uuid
from contextlib import contextmanager
from six.moves import urllib

from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.store.dbmodels.db_types import MYSQL
from mlflow.store.dbmodels.models import Base, SqlExperiment, SqlRun, SqlMetric, SqlParam, SqlTag
from mlflow.entities import RunStatus, SourceType, Experiment
from mlflow.store.abstract_store import AbstractStore
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS, \
    INVALID_STATE, RESOURCE_DOES_NOT_EXIST, INTERNAL_ERROR
from mlflow.tracking.utils import _is_local_uri
from mlflow.utils.file_utils import build_path, mkdir
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME


class SqlAlchemyStore(AbstractStore):
    """
    SQLAlchemy compliant backend store for tracking meta data for MLflow entities. Currently
    supported database types are ``mysql``, ``mssql``, ``sqlite``, and ``postgresql``. This store
    interacts with SQL store using SQLAlchemy abstractions defined for MLflow entities.
    :py:class:`mlflow.store.dbmodels.models.SqlExperiment`,
    :py:class:`mlflow.store.dbmodels.models.SqlRun`,
    :py:class:`mlflow.store.dbmodels.models.SqlTag`,
    :py:class:`mlflow.store.dbmodels.models.SqlMetric`, and
    :py:class:`mlflow.store.dbmodels.models.SqlParam`.

    Run artifacts are stored in a separate location using artifact stores conforming to
    :py:class:`mlflow.store.artifact_repo.ArtifactRepository`. Default artifact locations for
    user experiments are stored in the database along with metadata. Each run artifact location
    is recorded in :py:class:`mlflow.store.dbmodels.models.SqlRun` and stored in the backend DB.
    """
    ARTIFACTS_FOLDER_NAME = "artifacts"

    def __init__(self, db_uri, default_artifact_root):
        """
        Create a database backed store.

        :param db_uri: SQL connection string used by SQLAlchemy Engine to connect to the database.
                       Argument is expected to be in the format:
                       ``db_type://<user_name>:<password>@<host>:<port>/<database_name>`
                       Supported database types are ``mysql``, ``mssql``, ``sqlite``,
                       and ``postgresql``.
        :param default_artifact_root: Path/URI to location suitable for large data (such as a blob
                                      store object, DBFS path, or shared NFS file system).
        """
        super(SqlAlchemyStore, self).__init__()
        self.db_uri = db_uri
        self.db_type = urllib.parse.urlparse(db_uri).scheme
        self.artifact_root_uri = default_artifact_root
        self.engine = sqlalchemy.create_engine(db_uri)
        Base.metadata.create_all(self.engine)
        Base.metadata.bind = self.engine
        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = self._get_managed_session_maker(SessionMaker)

        if _is_local_uri(default_artifact_root):
            mkdir(default_artifact_root)

        if len(self.list_experiments()) == 0:
            with self.ManagedSessionMaker() as session:
                self._create_default_experiment(session)

    @staticmethod
    def _get_managed_session_maker(SessionMaker):
        """
        Creates a factory for producing exception-safe SQLAlchemy sessions that are made available
        using a context manager. Any session produced by this factory is automatically committed
        if no exceptions are encountered within its associated context. If an exception is
        encountered, the session is rolled back. Finally, any session produced by this factory is
        automatically closed when the session's associated context is exited.
        """

        @contextmanager
        def make_managed_session():
            """Provide a transactional scope around a series of operations."""
            session = SessionMaker()
            try:
                yield session
                session.commit()
            except MlflowException:
                session.rollback()
                raise
            except Exception as e:
                session.rollback()
                raise MlflowException(message=e, error_code=INTERNAL_ERROR)
            finally:
                session.close()

        return make_managed_session

    def _set_no_auto_for_zero_values(self, session):
        if self.db_type == MYSQL:
            session.execute("SET @@SESSION.sql_mode='NO_AUTO_VALUE_ON_ZERO';")

    # DB helper methods to allow zero values for columns with auto increments
    def _unset_no_auto_for_zero_values(self, session):
        if self.db_type == MYSQL:
            session.execute("SET @@SESSION.sql_mode='';")

    def _create_default_experiment(self, session):
        """
        MLflow UI and client code expects a default experiment with ID 0.
        This method uses SQL insert statement to create the default experiment as a hack, since
        experiment table uses 'experiment_id' column is a PK and is also set to auto increment.
        MySQL and other implementation do not allow value '0' for such cases.

        ToDo: Identify a less hacky mechanism to create default experiment 0
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
            self._set_no_auto_for_zero_values(session)
            session.execute("INSERT INTO {} ({}) VALUES ({});".format(
                table, ", ".join(columns), values))
        finally:
            self._unset_no_auto_for_zero_values(session)

    def _save_to_db(self, session, objs):
        """
        Store in db
        """
        if type(objs) is list:
            session.add_all(objs)
        else:
            # single object
            session.add(objs)

    def _get_or_create(self, session, model, **kwargs):
        instance = session.query(model).filter_by(**kwargs).first()
        created = False

        if instance:
            return instance, created
        else:
            instance = model(**kwargs)
            self._save_to_db(objs=instance, session=session)
            created = True

        return instance, created

    def _get_artifact_location(self, experiment_id):
        return build_path(self.artifact_root_uri, str(experiment_id))

    def create_experiment(self, name, artifact_location=None):
        if name is None or name == '':
            raise MlflowException('Invalid experiment name', INVALID_PARAMETER_VALUE)

        with self.ManagedSessionMaker() as session:
            try:
                experiment = SqlExperiment(
                    name=name, lifecycle_stage=LifecycleStage.ACTIVE,
                    artifact_location=artifact_location
                )
                session.add(experiment)
                if not artifact_location:
                    # this requires a double write. The first one to generate an autoincrement-ed ID
                    eid = session.query(SqlExperiment).filter_by(name=name).first().experiment_id
                    experiment.artifact_location = self._get_artifact_location(eid)
            except sqlalchemy.exc.IntegrityError as e:
                raise MlflowException('Experiment(name={}) already exists. '
                                      'Error: {}'.format(name, str(e)), RESOURCE_ALREADY_EXISTS)

            return experiment.experiment_id

    def _list_experiments(self, session, ids=None, names=None, view_type=ViewType.ACTIVE_ONLY):
        stages = LifecycleStage.view_type_to_stages(view_type)
        conditions = [SqlExperiment.lifecycle_stage.in_(stages)]

        if ids and len(ids) > 0:
            conditions.append(SqlExperiment.experiment_id.in_(ids))

        if names and len(names) > 0:
            conditions.append(SqlExperiment.name.in_(names))

        return session.query(SqlExperiment).filter(*conditions)

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        with self.ManagedSessionMaker() as session:
            return [exp.to_mlflow_entity() for exp in
                    self._list_experiments(session=session, view_type=view_type)]

    def _get_experiment(self, session, experiment_id, view_type):
        experiments = self._list_experiments(
            session=session, ids=[experiment_id], view_type=view_type).all()
        if len(experiments) == 0:
            raise MlflowException('No Experiment with id={} exists'.format(experiment_id),
                                  RESOURCE_DOES_NOT_EXIST)
        if len(experiments) > 1:
            raise MlflowException('Expected only 1 experiment with id={}. Found {}.'.format(
                experiment_id, len(experiments)), INVALID_STATE)

        return experiments[0]

    def get_experiment(self, experiment_id):
        with self.ManagedSessionMaker() as session:
            return self._get_experiment(session, experiment_id, ViewType.ALL).to_mlflow_entity()

    def get_experiment_by_name(self, experiment_name):
        """
        Specialized implementation for SQL backed store.
        """
        with self.ManagedSessionMaker() as session:
            experiments = self._list_experiments(
                names=[experiment_name], view_type=ViewType.ALL, session=session).all()
            if len(experiments) == 0:
                return None

            if len(experiments) > 1:
                raise MlflowException('Expected only 1 experiment with name={}. Found {}.'.format(
                    experiment_name, len(experiments)), INVALID_STATE)

            return experiments[0].to_mlflow_entity()

    def delete_experiment(self, experiment_id):
        with self.ManagedSessionMaker() as session:
            experiment = self._get_experiment(session, experiment_id, ViewType.ACTIVE_ONLY)
            experiment.lifecycle_stage = LifecycleStage.DELETED
            self._save_to_db(objs=experiment, session=session)

    def restore_experiment(self, experiment_id):
        with self.ManagedSessionMaker() as session:
            experiment = self._get_experiment(session, experiment_id, ViewType.DELETED_ONLY)
            experiment.lifecycle_stage = LifecycleStage.ACTIVE
            self._save_to_db(objs=experiment, session=session)

    def rename_experiment(self, experiment_id, new_name):
        with self.ManagedSessionMaker() as session:
            experiment = self._get_experiment(session, experiment_id, ViewType.ALL)
            if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
                raise MlflowException('Cannot rename a non-active experiment.', INVALID_STATE)

            experiment.name = new_name
            self._save_to_db(objs=experiment, session=session)

    def create_run(self, experiment_id, user_id, run_name, source_type, source_name,
                   entry_point_name, start_time, source_version, tags, parent_run_id):
        with self.ManagedSessionMaker() as session:
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

            self._save_to_db(objs=run, session=session)

            return run.to_mlflow_entity()

    def _get_run(self, session, run_uuid):
        runs = session.query(SqlRun).filter(SqlRun.run_uuid == run_uuid).all()

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
                                  .format(run.run_uuid, run.lifecycle_stage),
                                  INVALID_PARAMETER_VALUE)

    def _check_run_is_deleted(self, run):
        if run.lifecycle_stage != LifecycleStage.DELETED:
            raise MlflowException("The run {} must be in 'deleted' state. Current state is {}."
                                  .format(run.run_uuid, run.lifecycle_stage),
                                  INVALID_PARAMETER_VALUE)

    def update_run_info(self, run_uuid, run_status, end_time):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_uuid, session=session)
            self._check_run_is_active(run)
            run.status = RunStatus.to_string(run_status)
            run.end_time = end_time

            self._save_to_db(objs=run, session=session)
            run = run.to_mlflow_entity()

            return run.info

    def get_run(self, run_uuid):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_uuid, session=session)
            return run.to_mlflow_entity()

    def restore_run(self, run_id):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_deleted(run)
            run.lifecycle_stage = LifecycleStage.ACTIVE
            self._save_to_db(objs=run, session=session)

    def delete_run(self, run_id):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            run.lifecycle_stage = LifecycleStage.DELETED
            self._save_to_db(objs=run, session=session)

    def log_metric(self, run_uuid, metric):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_uuid, session=session)
            self._check_run_is_active(run)

            try:
                self._get_or_create(model=SqlMetric, run_uuid=run_uuid, key=metric.key,
                                    value=metric.value, timestamp=metric.timestamp, session=session)
                # Explicitly commit the session in order to catch potential integrity errors
                # while maintaining the current managed session scope ("commit" checks that
                # a transaction satisfies uniqueness constraints and throws integrity errors
                # when they are violated; "get_or_create()" does not perform these checks). It is
                # important that we maintain the same session scope because, in the case of
                # an integrity error, we want to examine the uniqueness of metric (timestamp, value)
                # tuples using the same database state that the session uses during "commit".
                # Creating a new session synchronizes the state with the database. As a result, if
                # the conflicting (timestamp, value) tuple were to be removed prior to the creation
                # of a new session, we would be unable to determine the cause of failure for the
                # first session's "commit" operation.
                session.commit()
            except sqlalchemy.exc.IntegrityError:
                # Roll back the current session to make it usable for further transactions. In the
                # event of an error during "commit", a rollback is required in order to continue
                # using the session. In this case, we re-use the session because the SqlRun, `run`,
                # is lazily evaluated during the invocation of `run.metrics`.
                session.rollback()
                existing_metric = [m for m in run.metrics
                                   if m.key == metric.key and m.timestamp == metric.timestamp]
                if len(existing_metric) > 0:
                    m = existing_metric[0]
                    raise MlflowException(
                        "Metric={} must be unique. Metric already logged value {}"
                        " at {}".format(metric, m.value, m.timestamp), INVALID_PARAMETER_VALUE)
                else:
                    raise

    def get_metric_history(self, run_uuid, metric_key):
        with self.ManagedSessionMaker() as session:
            metrics = session.query(SqlMetric).filter_by(run_uuid=run_uuid, key=metric_key).all()
            return [metric.to_mlflow_entity() for metric in metrics]

    def log_param(self, run_uuid, param):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_uuid, session=session)
            self._check_run_is_active(run)
            # if we try to update the value of an existing param this will fail
            # because it will try to create it with same run_uuid, param key
            try:
                # This will check for various integrity checks for params table.
                # ToDo: Consider prior checks for null, type, param name validations, ... etc.
                self._get_or_create(model=SqlParam, session=session, run_uuid=run_uuid,
                                    key=param.key, value=param.value)
                # Explicitly commit the session in order to catch potential integrity errors
                # while maintaining the current managed session scope ("commit" checks that
                # a transaction satisfies uniqueness constraints and throws integrity errors
                # when they are violated; "get_or_create()" does not perform these checks). It is
                # important that we maintain the same session scope because, in the case of
                # an integrity error, we want to examine the uniqueness of parameter values using
                # the same database state that the session uses during "commit". Creating a new
                # session synchronizes the state with the database. As a result, if the conflicting
                # parameter value were to be removed prior to the creation of a new session,
                # we would be unable to determine the cause of failure for the first session's
                # "commit" operation.
                session.commit()
            except sqlalchemy.exc.IntegrityError:
                # Roll back the current session to make it usable for further transactions. In the
                # event of an error during "commit", a rollback is required in order to continue
                # using the session. In this case, we re-use the session because the SqlRun, `run`,
                # is lazily evaluated during the invocation of `run.params`.
                session.rollback()
                existing_params = [p.value for p in run.params if p.key == param.key]
                if len(existing_params) > 0:
                    old_value = existing_params[0]
                    raise MlflowException(
                        "Changing param value is not allowed. Param with key='{}' was already"
                        " logged with value='{}' for run ID='{}. Attempted logging new value"
                        " '{}'.".format(
                            param.key, old_value, run_uuid, param.value), INVALID_PARAMETER_VALUE)
                else:
                    raise

    def set_tag(self, run_uuid, tag):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_uuid, session=session)
            self._check_run_is_active(run)
            session.merge(SqlTag(run_uuid=run_uuid, key=tag.key, value=tag.value))

    def search_runs(self, experiment_ids, search_filter, run_view_type):
        with self.ManagedSessionMaker() as session:
            runs = [run.to_mlflow_entity()
                    for exp in experiment_ids
                    for run in self._list_runs(session, exp, run_view_type)]
            return [run for run in runs if not search_filter or search_filter.filter(run)]

    def _list_runs(self, session, experiment_id, run_view_type):
        exp = self._list_experiments(
            ids=[experiment_id], view_type=ViewType.ALL, session=session).first()
        stages = set(LifecycleStage.view_type_to_stages(run_view_type))
        return [run for run in exp.runs if run.lifecycle_stage in stages]

    def log_batch(self, run_id, metrics, params, tags):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
        try:
            for param in params:
                self.log_param(run_id, param)
            for metric in metrics:
                self.log_metric(run_id, metric)
            for tag in tags:
                self.set_tag(run_id, tag)
        except MlflowException as e:
            raise e
        except Exception as e:
            raise MlflowException(e, INTERNAL_ERROR)
