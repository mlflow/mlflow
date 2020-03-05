import json
import logging
import uuid

import math
import sqlalchemy
import sqlalchemy.sql.expression as sql

from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.models import Model
from mlflow.store.tracking import SEARCH_MAX_RESULTS_THRESHOLD
from mlflow.store.db.db_types import MYSQL, MSSQL
import mlflow.store.db.utils
from mlflow.store.tracking.dbmodels.models import SqlExperiment, SqlRun, \
    SqlMetric, SqlParam, SqlTag, SqlExperimentTag, SqlLatestMetric
from mlflow.store.db.base_sql_model import Base
from mlflow.entities import RunStatus, SourceType, Experiment
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS, \
    INVALID_STATE, RESOURCE_DOES_NOT_EXIST, INTERNAL_ERROR
from mlflow.utils.uri import is_local_uri, extract_db_type_from_uri
from mlflow.utils.file_utils import mkdir, local_file_uri_to_path
from mlflow.utils.search_utils import SearchUtils
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.uri import append_to_uri_path
from mlflow.utils.validation import _validate_batch_log_limits, _validate_batch_log_data, \
    _validate_run_id, _validate_metric, _validate_experiment_tag, _validate_tag
from mlflow.utils.mlflow_tags import MLFLOW_LOGGED_MODELS

_logger = logging.getLogger(__name__)

# For each database table, fetch its columns and define an appropriate attribute for each column
# on the table's associated object representation (Mapper). This is necessary to ensure that
# columns defined via backreference are available as Mapper instance attributes (e.g.,
# ``SqlExperiment.tags`` and ``SqlRun.params``). For more information, see
# https://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.configure_mappers
# and https://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.mapper.Mapper
sqlalchemy.orm.configure_mappers()


class SqlAlchemyStore(AbstractStore):
    """
    SQLAlchemy compliant backend store for tracking meta data for MLflow entities. MLflow
    supports the database dialects ``mysql``, ``mssql``, ``sqlite``, and ``postgresql``.
    As specified in the
    `SQLAlchemy docs <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_ ,
    the database URI is expected in the format
    ``<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>``. If you do not
    specify a driver, SQLAlchemy uses a dialect's default driver.

    This store interacts with SQL store using SQLAlchemy abstractions defined for MLflow entities.
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
    DEFAULT_EXPERIMENT_ID = "0"

    def __init__(self, db_uri, default_artifact_root):
        """
        Create a database backed store.

        :param db_uri: The SQLAlchemy database URI string to connect to the database. See
                       the `SQLAlchemy docs
                       <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_
                       for format specifications. Mlflow supports the dialects ``mysql``,
                       ``mssql``, ``sqlite``, and ``postgresql``.
        :param default_artifact_root: Path/URI to location suitable for large data (such as a blob
                                      store object, DBFS path, or shared NFS file system).
        """
        super(SqlAlchemyStore, self).__init__()
        self.db_uri = db_uri
        self.db_type = extract_db_type_from_uri(db_uri)
        self.artifact_root_uri = default_artifact_root
        self.engine = mlflow.store.db.utils.create_sqlalchemy_engine(db_uri)
        # On a completely fresh MLflow installation against an empty database (verify database
        # emptiness by checking that 'experiments' etc aren't in the list of table names), run all
        # DB migrations
        expected_tables = [
            SqlExperiment.__tablename__,
            SqlRun.__tablename__,
            SqlMetric.__tablename__,
            SqlParam.__tablename__,
            SqlTag.__tablename__,
            SqlExperimentTag.__tablename__,
            SqlLatestMetric.__tablename__,
        ]
        inspected_tables = set(sqlalchemy.inspect(self.engine).get_table_names())
        if any([table not in inspected_tables for table in expected_tables]):
            mlflow.store.db.utils._initialize_tables(self.engine)
        Base.metadata.bind = self.engine
        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = mlflow.store.db.utils._get_managed_session_maker(SessionMaker)
        mlflow.store.db.utils._verify_schema(self.engine)

        if is_local_uri(default_artifact_root):
            mkdir(local_file_uri_to_path(default_artifact_root))

        if len(self.list_experiments()) == 0:
            with self.ManagedSessionMaker() as session:
                self._create_default_experiment(session)

    def _set_zero_value_insertion_for_autoincrement_column(self, session):
        if self.db_type == MYSQL:
            # config letting MySQL override default
            # to allow 0 value for experiment ID (auto increment column)
            session.execute("SET @@SESSION.sql_mode='NO_AUTO_VALUE_ON_ZERO';")
        if self.db_type == MSSQL:
            # config letting MSSQL override default
            # to allow any manual value inserted into IDENTITY column
            session.execute("SET IDENTITY_INSERT experiments ON;")

    # DB helper methods to allow zero values for columns with auto increments
    def _unset_zero_value_insertion_for_autoincrement_column(self, session):
        if self.db_type == MYSQL:
            session.execute("SET @@SESSION.sql_mode='';")
        if self.db_type == MSSQL:
            session.execute("SET IDENTITY_INSERT experiments OFF;")

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
            SqlExperiment.experiment_id.name: int(SqlAlchemyStore.DEFAULT_EXPERIMENT_ID),
            SqlExperiment.name.name: Experiment.DEFAULT_EXPERIMENT_NAME,
            SqlExperiment.artifact_location.name: str(self._get_artifact_location(0)),
            SqlExperiment.lifecycle_stage.name: LifecycleStage.ACTIVE
        }

        def decorate(s):
            if is_string_type(s):
                return "'{}'".format(s)
            else:
                return "{}".format(s)

        # Get a list of keys to ensure we have a deterministic ordering
        columns = list(default_experiment.keys())
        values = ", ".join([decorate(default_experiment.get(c)) for c in columns])

        try:
            self._set_zero_value_insertion_for_autoincrement_column(session)
            session.execute("INSERT INTO {} ({}) VALUES ({});".format(
                table, ", ".join(columns), values))
        finally:
            self._unset_zero_value_insertion_for_autoincrement_column(session)

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
        return append_to_uri_path(self.artifact_root_uri, str(experiment_id))

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

            session.flush()
            return str(experiment.experiment_id)

    def _list_experiments(self, session, ids=None, names=None, view_type=ViewType.ACTIVE_ONLY,
                          eager=False):
        """
        :param eager: If ``True``, eagerly loads each experiments's tags. If ``False``, these tags
                      are not eagerly loaded and will be loaded if/when their corresponding
                      object properties are accessed from a resulting ``SqlExperiment`` object.
        """
        stages = LifecycleStage.view_type_to_stages(view_type)
        conditions = [SqlExperiment.lifecycle_stage.in_(stages)]
        if ids and len(ids) > 0:
            int_ids = [int(eid) for eid in ids]
            conditions.append(SqlExperiment.experiment_id.in_(int_ids))
        if names and len(names) > 0:
            conditions.append(SqlExperiment.name.in_(names))

        query_options = self._get_eager_experiment_query_options() if eager else []
        return session \
            .query(SqlExperiment) \
            .options(*query_options) \
            .filter(*conditions) \
            .all()

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        with self.ManagedSessionMaker() as session:
            return [exp.to_mlflow_entity() for exp in
                    self._list_experiments(session=session, view_type=view_type, eager=True)]

    def _get_experiment(self, session, experiment_id, view_type, eager=False):
        """
        :param eager: If ``True``, eagerly loads the experiments's tags. If ``False``, these tags
                      are not eagerly loaded and will be loaded if/when their corresponding
                      object properties are accessed from the resulting ``SqlExperiment`` object.
        """
        experiment_id = experiment_id or SqlAlchemyStore.DEFAULT_EXPERIMENT_ID
        stages = LifecycleStage.view_type_to_stages(view_type)
        query_options = self._get_eager_experiment_query_options() if eager else []

        experiment = session \
            .query(SqlExperiment) \
            .options(*query_options) \
            .filter(
                SqlExperiment.experiment_id == experiment_id,
                SqlExperiment.lifecycle_stage.in_(stages)) \
            .one_or_none()

        if experiment is None:
            raise MlflowException('No Experiment with id={} exists'.format(experiment_id),
                                  RESOURCE_DOES_NOT_EXIST)

        return experiment

    @staticmethod
    def _get_eager_experiment_query_options():
        """
        :return: A list of SQLAlchemy query options that can be used to eagerly load the following
                 experiment attributes when fetching an experiment: ``tags``.
        """
        return [
            # Use a subquery load rather than a joined load in order to minimize the memory overhead
            # of the eager loading procedure. For more information about relationship loading
            # techniques, see https://docs.sqlalchemy.org/en/13/orm/
            # loading_relationships.html#relationship-loading-techniques
            sqlalchemy.orm.subqueryload(SqlExperiment.tags),
        ]

    def get_experiment(self, experiment_id):
        with self.ManagedSessionMaker() as session:
            return self._get_experiment(
                session, experiment_id, ViewType.ALL, eager=True).to_mlflow_entity()

    def get_experiment_by_name(self, experiment_name):
        """
        Specialized implementation for SQL backed store.
        """
        with self.ManagedSessionMaker() as session:
            stages = LifecycleStage.view_type_to_stages(ViewType.ALL)
            experiment = session \
                .query(SqlExperiment) \
                .options(*self._get_eager_experiment_query_options()) \
                .filter(
                    SqlExperiment.name == experiment_name,
                    SqlExperiment.lifecycle_stage.in_(stages)) \
                .one_or_none()
            return experiment.to_mlflow_entity() if experiment is not None else None

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

    def create_run(self, experiment_id, user_id, start_time, tags):
        with self.ManagedSessionMaker() as session:
            experiment = self.get_experiment(experiment_id)
            self._check_experiment_is_active(experiment)

            run_id = uuid.uuid4().hex
            artifact_location = append_to_uri_path(experiment.artifact_location, run_id,
                                                   SqlAlchemyStore.ARTIFACTS_FOLDER_NAME)
            run = SqlRun(name="", artifact_uri=artifact_location, run_uuid=run_id,
                         experiment_id=experiment_id,
                         source_type=SourceType.to_string(SourceType.UNKNOWN),
                         source_name="", entry_point_name="",
                         user_id=user_id, status=RunStatus.to_string(RunStatus.RUNNING),
                         start_time=start_time, end_time=None,
                         source_version="", lifecycle_stage=LifecycleStage.ACTIVE)

            tags_dict = {}
            for tag in tags:
                tags_dict[tag.key] = tag.value
            run.tags = [SqlTag(key=key, value=value) for key, value in tags_dict.items()]
            self._save_to_db(objs=run, session=session)

            return run.to_mlflow_entity()

    def _get_run(self, session, run_uuid, eager=False):
        """
        :param eager: If ``True``, eagerly loads the run's summary metrics (``latest_metrics``),
                      params, and tags when fetching the run. If ``False``, these attributes
                      are not eagerly loaded and will be loaded when their corresponding
                      object properties are accessed from the resulting ``SqlRun`` object.
        """
        query_options = self._get_eager_run_query_options() if eager else []
        runs = session \
            .query(SqlRun) \
            .options(*query_options) \
            .filter(SqlRun.run_uuid == run_uuid).all()

        if len(runs) == 0:
            raise MlflowException('Run with id={} not found'.format(run_uuid),
                                  RESOURCE_DOES_NOT_EXIST)
        if len(runs) > 1:
            raise MlflowException('Expected only 1 run with id={}. Found {}.'.format(run_uuid,
                                                                                     len(runs)),
                                  INVALID_STATE)

        return runs[0]

    @staticmethod
    def _get_eager_run_query_options():
        """
        :return: A list of SQLAlchemy query options that can be used to eagerly load the following
                 run attributes when fetching a run: ``latest_metrics``, ``params``, and ``tags``.
        """
        return [
            # Use a subquery load rather than a joined load in order to minimize the memory overhead
            # of the eager loading procedure. For more information about relationship loading
            # techniques, see https://docs.sqlalchemy.org/en/13/orm/
            # loading_relationships.html#relationship-loading-techniques
            sqlalchemy.orm.subqueryload(SqlRun.latest_metrics),
            sqlalchemy.orm.subqueryload(SqlRun.params),
            sqlalchemy.orm.subqueryload(SqlRun.tags),
        ]

    def _check_run_is_active(self, run):
        if run.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException("The run {} must be in the 'active' state. Current state is {}."
                                  .format(run.run_uuid, run.lifecycle_stage),
                                  INVALID_PARAMETER_VALUE)

    def _check_experiment_is_active(self, experiment):
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException("The experiment {} must be in the 'active' state. "
                                  "Current state is {}."
                                  .format(experiment.experiment_id, experiment.lifecycle_stage),
                                  INVALID_PARAMETER_VALUE)

    def _check_run_is_deleted(self, run):
        if run.lifecycle_stage != LifecycleStage.DELETED:
            raise MlflowException("The run {} must be in the 'deleted' state. Current state is {}."
                                  .format(run.run_uuid, run.lifecycle_stage),
                                  INVALID_PARAMETER_VALUE)

    def update_run_info(self, run_id, run_status, end_time):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            run.status = RunStatus.to_string(run_status)
            run.end_time = end_time

            self._save_to_db(objs=run, session=session)
            run = run.to_mlflow_entity()

            return run.info

    def _try_get_run_tag(self,  session, run_id, tagKey, eager=False):
        query_options = self._get_eager_run_query_options() if eager else []
        tags = session \
            .query(SqlTag) \
            .options(*query_options) \
            .filter(SqlTag.run_uuid == run_id and SqlTag.key == tagKey).all()
        return None if not tags else tags[0]

    def get_run(self, run_id):
        with self.ManagedSessionMaker() as session:
            # Load the run with the specified id and eagerly load its summary metrics, params, and
            # tags. These attributes are referenced during the invocation of
            # ``run.to_mlflow_entity()``, so eager loading helps avoid additional database queries
            # that are otherwise executed at attribute access time under a lazy loading model.
            run = self._get_run(run_uuid=run_id, session=session, eager=True)
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

    def _hard_delete_run(self, run_id):
        """
        Permanently delete a run (metadata and metrics, tags, parameters).
        This is used by the ``mlflow gc`` command line and is not intended to be used elsewhere.
        """
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            session.delete(run)

    def _get_deleted_runs(self):
        with self.ManagedSessionMaker() as session:
            run_ids = session\
                .query(SqlRun.run_uuid) \
                .filter(SqlRun.lifecycle_stage == LifecycleStage.DELETED) \
                .all()
            return [run_id[0] for run_id in run_ids]

    def log_metric(self, run_id, metric):
        _validate_metric(metric.key, metric.value, metric.timestamp, metric.step)
        is_nan = math.isnan(metric.value)
        if is_nan:
            value = 0
        elif math.isinf(metric.value):
            #  NB: Sql can not represent Infs = > We replace +/- Inf with max/min 64b float value
            value = 1.7976931348623157e308 if metric.value > 0 else -1.7976931348623157e308
        else:
            value = metric.value
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            # ToDo: Consider prior checks for null, type, metric name validations, ... etc.
            logged_metric, just_created = self._get_or_create(
                model=SqlMetric, run_uuid=run_id, key=metric.key, value=value,
                timestamp=metric.timestamp, step=metric.step, session=session, is_nan=is_nan)
            # Conditionally update the ``latest_metrics`` table if the logged metric  was not
            # already present in the ``metrics`` table. If the logged metric was already present,
            # we assume that the ``latest_metrics`` table already accounts for its presence
            if just_created:
                self._update_latest_metric_if_necessary(logged_metric, session)

    @staticmethod
    def _update_latest_metric_if_necessary(logged_metric, session):
        def _compare_metrics(metric_a, metric_b):
            """
            :return: True if ``metric_a`` is strictly more recent than ``metric_b``, as determined
                     by ``step``, ``timestamp``, and ``value``. False otherwise.
            """
            return (metric_a.step, metric_a.timestamp, metric_a.value) > \
                   (metric_b.step, metric_b.timestamp, metric_b.value)

        # Fetch the latest metric value corresponding to the specified run_id and metric key and
        # lock its associated row for the remainder of the transaction in order to ensure
        # isolation
        latest_metric = session \
            .query(SqlLatestMetric) \
            .filter(
                SqlLatestMetric.run_uuid == logged_metric.run_uuid,
                SqlLatestMetric.key == logged_metric.key) \
            .with_for_update() \
            .one_or_none()
        if latest_metric is None or _compare_metrics(logged_metric, latest_metric):
            session.merge(
                SqlLatestMetric(
                    run_uuid=logged_metric.run_uuid, key=logged_metric.key,
                    value=logged_metric.value, timestamp=logged_metric.timestamp,
                    step=logged_metric.step, is_nan=logged_metric.is_nan))

    def get_metric_history(self, run_id, metric_key):
        with self.ManagedSessionMaker() as session:
            metrics = session.query(SqlMetric).filter_by(run_uuid=run_id, key=metric_key).all()
            return [metric.to_mlflow_entity() for metric in metrics]

    def log_param(self, run_id, param):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            # if we try to update the value of an existing param this will fail
            # because it will try to create it with same run_uuid, param key
            try:
                # This will check for various integrity checks for params table.
                # ToDo: Consider prior checks for null, type, param name validations, ... etc.
                self._get_or_create(model=SqlParam, session=session, run_uuid=run_id,
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
                        "Changing param values is not allowed. Param with key='{}' was already"
                        " logged with value='{}' for run ID='{}'. Attempted logging new value"
                        " '{}'.".format(
                            param.key, old_value, run_id, param.value), INVALID_PARAMETER_VALUE)
                else:
                    raise

    def set_experiment_tag(self, experiment_id, tag):
        """
        Set a tag for the specified experiment

        :param experiment_id: String ID of the experiment
        :param tag: ExperimentRunTag instance to log
        """
        _validate_experiment_tag(tag.key, tag.value)
        with self.ManagedSessionMaker() as session:
            experiment = self._get_experiment(session,
                                              experiment_id,
                                              ViewType.ALL).to_mlflow_entity()
            self._check_experiment_is_active(experiment)
            session.merge(SqlExperimentTag(experiment_id=experiment_id,
                                           key=tag.key,
                                           value=tag.value))

    def set_tag(self, run_id, tag):
        """
        Set a tag on a run.

        :param run_id: String ID of the run
        :param tag: RunTag instance to log
        """
        with self.ManagedSessionMaker() as session:
            _validate_tag(tag.key, tag.value)
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            session.merge(SqlTag(run_uuid=run_id, key=tag.key, value=tag.value))

    def delete_tag(self, run_id, key):
        """
        Delete a tag from a run. This is irreversible.

        :param run_id: String ID of the run
        :param key: Name of the tag
        """
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            filtered_tags = session.query(SqlTag).filter_by(run_uuid=run_id, key=key).all()
            if len(filtered_tags) == 0:
                raise MlflowException(
                    "No tag with name: {} in run with id {}".format(key, run_id),
                    error_code=RESOURCE_DOES_NOT_EXIST)
            elif len(filtered_tags) > 1:
                raise MlflowException(
                    "Bad data in database - tags for a specific run must have "
                    "a single unique value."
                    "See https://mlflow.org/docs/latest/tracking.html#adding-tags-to-runs",
                    error_code=INVALID_STATE)
            session.delete(filtered_tags[0])

    def _search_runs(self, experiment_ids, filter_string, run_view_type, max_results, order_by,
                     page_token):

        def compute_next_token(current_size):
            next_token = None
            if max_results == current_size:
                final_offset = offset + max_results
                next_token = SearchUtils.create_page_token(final_offset)

            return next_token

        if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise MlflowException("Invalid value for request parameter max_results. It must be at "
                                  "most {}, but got value {}".format(SEARCH_MAX_RESULTS_THRESHOLD,
                                                                     max_results),
                                  INVALID_PARAMETER_VALUE)

        stages = set(LifecycleStage.view_type_to_stages(run_view_type))

        with self.ManagedSessionMaker() as session:
            # Fetch the appropriate runs and eagerly load their summary metrics, params, and
            # tags. These run attributes are referenced during the invocation of
            # ``run.to_mlflow_entity()``, so eager loading helps avoid additional database queries
            # that are otherwise executed at attribute access time under a lazy loading model.
            parsed_filters = SearchUtils.parse_search_filter(filter_string)
            parsed_orderby, sorting_joins = _get_orderby_clauses(order_by, session)

            query = session.query(SqlRun)
            for j in _get_sqlalchemy_filter_clauses(parsed_filters, session):
                query = query.join(j)
            # using an outer join is necessary here because we want to be able to sort
            # on a column (tag, metric or param) without removing the lines that
            # do not have a value for this column (which is what inner join would do)
            for j in sorting_joins:
                query = query.outerjoin(j)

            offset = SearchUtils.parse_start_offset_from_page_token(page_token)
            queried_runs = query.distinct() \
                .options(*self._get_eager_run_query_options()) \
                .filter(
                    SqlRun.experiment_id.in_(experiment_ids),
                    SqlRun.lifecycle_stage.in_(stages),
                    *_get_attributes_filtering_clauses(parsed_filters)) \
                .order_by(*parsed_orderby) \
                .offset(offset).limit(max_results).all()

            runs = [run.to_mlflow_entity() for run in queried_runs]
            next_page_token = compute_next_token(len(runs))

        return runs, next_page_token

    def log_batch(self, run_id, metrics, params, tags):
        _validate_run_id(run_id)
        _validate_batch_log_data(metrics, params, tags)
        _validate_batch_log_limits(metrics, params, tags)
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

    def record_logged_model(self, run_id, mlflow_model):
        if not isinstance(mlflow_model, Model):
            raise TypeError("Argument 'mlflow_model' should be mlflow.models.Model, got '{}'"
                            .format(type(mlflow_model)))
        model_dict = mlflow_model.to_dict()
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            previous_tag = [t for t in run.tags if t.key == MLFLOW_LOGGED_MODELS]
            if previous_tag:
                value = json.dumps(json.loads(previous_tag[0].value) + [model_dict])
            else:
                value = json.dumps([model_dict])
            _validate_tag(MLFLOW_LOGGED_MODELS, value)
            session.merge(SqlTag(key=MLFLOW_LOGGED_MODELS, value=value, run_uuid=run_id))


def _get_attributes_filtering_clauses(parsed):
    clauses = []
    for sql_statement in parsed:
        key_type = sql_statement.get('type')
        key_name = sql_statement.get('key')
        value = sql_statement.get('value')
        comparator = sql_statement.get('comparator')
        if SearchUtils.is_attribute(key_type, comparator):
            # validity of the comparator is checked in SearchUtils.parse_search_filter()
            op = SearchUtils.filter_ops.get(comparator)
            if op:
                # key_name is guaranteed to be a valid searchable attribute of entities.RunInfo
                # by the call to parse_search_filter
                attribute_name = SqlRun.get_attribute_name(key_name)
                clauses.append(op(getattr(SqlRun, attribute_name), value))
    return clauses


def _to_sqlalchemy_filtering_statement(sql_statement, session):
    key_type = sql_statement.get('type')
    key_name = sql_statement.get('key')
    value = sql_statement.get('value')
    comparator = sql_statement.get('comparator')

    if SearchUtils.is_metric(key_type, comparator):
        entity = SqlLatestMetric
        value = float(value)
    elif SearchUtils.is_param(key_type, comparator):
        entity = SqlParam
    elif SearchUtils.is_tag(key_type, comparator):
        entity = SqlTag
    elif SearchUtils.is_attribute(key_type, comparator):
        return None
    else:
        raise MlflowException("Invalid search expression type '%s'" % key_type,
                              error_code=INVALID_PARAMETER_VALUE)

    # validity of the comparator is checked in SearchUtils.parse_search_filter()
    op = SearchUtils.filter_ops.get(comparator)
    if op:
        return (
            session
            .query(entity)
            .filter(entity.key == key_name, op(entity.value, value))
            .subquery()
        )
    else:
        return None


def _get_sqlalchemy_filter_clauses(parsed, session):
    """creates SqlAlchemy subqueries
    that will be inner-joined to SQLRun to act as multi-clause filters."""
    filters = []
    for sql_statement in parsed:
        filter_query = _to_sqlalchemy_filtering_statement(sql_statement, session)
        if filter_query is not None:
            filters.append(filter_query)
    return filters


def _get_orderby_clauses(order_by_list, session):
    """Sorts a set of runs based on their natural ordering and an overriding set of order_bys.
    Runs are naturally ordered first by start time descending, then by run id for tie-breaking.
    """

    clauses = []
    ordering_joins = []
    clause_id = 0
    # contrary to filters, it is not easily feasible to separately handle sorting
    # on attributes and on joined tables as we must keep all clauses in the same order
    if order_by_list:
        for order_by_clause in order_by_list:
            clause_id += 1
            (key_type, key, ascending) = SearchUtils.parse_order_by(order_by_clause)
            if SearchUtils.is_attribute(key_type, '='):
                order_value = getattr(SqlRun, SqlRun.get_attribute_name(key))
            else:
                if SearchUtils.is_metric(key_type, '='):  # any valid comparator
                    entity = SqlLatestMetric
                elif SearchUtils.is_tag(key_type, '='):
                    entity = SqlTag
                elif SearchUtils.is_param(key_type, '='):
                    entity = SqlParam
                else:
                    raise MlflowException("Invalid identifier type '%s'" % key_type,
                                          error_code=INVALID_PARAMETER_VALUE)

                # build a subquery first because we will join it in the main request so that the
                # metric we want to sort on is available when we apply the sorting clause
                subquery = session \
                    .query(entity) \
                    .filter(entity.key == key) \
                    .subquery()

                ordering_joins.append(subquery)
                order_value = subquery.c.value

            # sqlite does not support NULLS LAST expression, so we sort first by
            # presence of the field (and is_nan for metrics), then by actual value
            # As the subqueries are created independently and used later in the
            # same main query, the CASE WHEN columns need to have unique names to
            # avoid ambiguity
            if SearchUtils.is_metric(key_type, '='):
                clauses.append(sql.case([
                    (subquery.c.is_nan.is_(True), 1),
                    (order_value.is_(None), 1)
                ], else_=0).label('clause_%s' % clause_id))
            else:  # other entities do not have an 'is_nan' field
                clauses.append(sql.case([(order_value.is_(None), 1)], else_=0)
                               .label('clause_%s' % clause_id))

            if ascending:
                clauses.append(order_value)
            else:
                clauses.append(order_value.desc())

    clauses.append(SqlRun.start_time.desc())
    clauses.append(SqlRun.run_uuid)
    return clauses, ordering_joins
