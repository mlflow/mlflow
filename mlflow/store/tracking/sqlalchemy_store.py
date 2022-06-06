import json
import logging
import random
import time
import uuid
import threading

import math
import sqlalchemy
import sqlalchemy.sql.expression as sql
from sqlalchemy.future import select

from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.store.tracking import SEARCH_MAX_RESULTS_THRESHOLD
from mlflow.store.db.db_types import MYSQL, MSSQL
import mlflow.store.db.utils
from mlflow.store.tracking.dbmodels.models import (
    SqlExperiment,
    SqlRun,
    SqlMetric,
    SqlParam,
    SqlTag,
    SqlExperimentTag,
    SqlLatestMetric,
)
from mlflow.store.db.base_sql_model import Base
from mlflow.entities import RunStatus, SourceType, Experiment
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.entities.paged_list import PagedList
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    INVALID_STATE,
    RESOURCE_DOES_NOT_EXIST,
    INTERNAL_ERROR,
)
from mlflow.utils.uri import is_local_uri, extract_db_type_from_uri
from mlflow.utils.file_utils import mkdir, local_file_uri_to_path
from mlflow.utils.search_utils import SearchUtils
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.uri import append_to_uri_path
from mlflow.utils.validation import (
    _validate_batch_log_limits,
    _validate_batch_log_data,
    _validate_run_id,
    _validate_metric,
    _validate_experiment_tag,
    _validate_tag,
    _validate_list_experiments_max_results,
    _validate_param_keys_unique,
    _validate_experiment_name,
)
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
    _db_uri_sql_alchemy_engine_map = {}
    _db_uri_sql_alchemy_engine_map_lock = threading.Lock()

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
        super().__init__()
        self.db_uri = db_uri
        self.db_type = extract_db_type_from_uri(db_uri)
        self.artifact_root_uri = default_artifact_root
        # Quick check to see if the respective SQLAlchemy database engine has already been created.
        if db_uri not in SqlAlchemyStore._db_uri_sql_alchemy_engine_map:
            with SqlAlchemyStore._db_uri_sql_alchemy_engine_map_lock:
                # Repeat check to prevent race conditions where one thread checks for an existing
                # engine while another is creating the respective one, resulting in multiple
                # engines being created. It isn't combined with the above check to prevent
                # inefficiency from multiple threads waiting for the lock to check for engine
                # existence if it has already been created.
                if db_uri not in SqlAlchemyStore._db_uri_sql_alchemy_engine_map:
                    SqlAlchemyStore._db_uri_sql_alchemy_engine_map[
                        db_uri
                    ] = mlflow.store.db.utils.create_sqlalchemy_engine_with_retry(db_uri)
        self.engine = SqlAlchemyStore._db_uri_sql_alchemy_engine_map[db_uri]
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
        if any(table not in inspected_tables for table in expected_tables):
            mlflow.store.db.utils._initialize_tables(self.engine)
        Base.metadata.bind = self.engine
        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = mlflow.store.db.utils._get_managed_session_maker(
            SessionMaker, self.db_type
        )
        mlflow.store.db.utils._verify_schema(self.engine)

        if is_local_uri(default_artifact_root):
            mkdir(local_file_uri_to_path(default_artifact_root))

        if len(self.list_experiments(view_type=ViewType.ALL)) == 0:
            with self.ManagedSessionMaker() as session:
                self._create_default_experiment(session)

    def _get_dialect(self):
        return self.engine.dialect.name

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
            SqlExperiment.lifecycle_stage.name: LifecycleStage.ACTIVE,
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
            session.execute(
                "INSERT INTO {} ({}) VALUES ({});".format(table, ", ".join(columns), values)
            )
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

    def create_experiment(self, name, artifact_location=None, tags=None):
        _validate_experiment_name(name)

        with self.ManagedSessionMaker() as session:
            try:
                experiment = SqlExperiment(
                    name=name,
                    lifecycle_stage=LifecycleStage.ACTIVE,
                    artifact_location=artifact_location,
                )
                experiment.tags = (
                    [SqlExperimentTag(key=tag.key, value=tag.value) for tag in tags] if tags else []
                )
                session.add(experiment)
                if not artifact_location:
                    # this requires a double write. The first one to generate an autoincrement-ed ID
                    eid = session.query(SqlExperiment).filter_by(name=name).first().experiment_id
                    experiment.artifact_location = self._get_artifact_location(eid)
            except sqlalchemy.exc.IntegrityError as e:
                raise MlflowException(
                    "Experiment(name={}) already exists. Error: {}".format(name, str(e)),
                    RESOURCE_ALREADY_EXISTS,
                )

            session.flush()
            return str(experiment.experiment_id)

    def _list_experiments(
        self,
        ids=None,
        names=None,
        view_type=ViewType.ACTIVE_ONLY,
        max_results=None,
        page_token=None,
        eager=False,
    ):
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

        max_results_for_query = None
        if max_results is not None:
            max_results_for_query = max_results + 1

            def compute_next_token(current_size):
                next_token = None
                if max_results_for_query == current_size:
                    final_offset = offset + max_results
                    next_token = SearchUtils.create_page_token(final_offset)

                return next_token

        with self.ManagedSessionMaker() as session:
            query_options = self._get_eager_experiment_query_options() if eager else []
            if max_results is not None:
                offset = SearchUtils.parse_start_offset_from_page_token(page_token)
                queried_experiments = (
                    session.query(SqlExperiment)
                    .options(*query_options)
                    .order_by(SqlExperiment.experiment_id)
                    .filter(*conditions)
                    .offset(offset)
                    .limit(max_results_for_query)
                    .all()
                )
            else:
                queried_experiments = (
                    session.query(SqlExperiment).options(*query_options).filter(*conditions).all()
                )

            experiments = [exp.to_mlflow_entity() for exp in queried_experiments]
        if max_results is not None:
            return PagedList(experiments[:max_results], compute_next_token(len(experiments)))
        else:
            return PagedList(experiments, None)

    def list_experiments(
        self,
        view_type=ViewType.ACTIVE_ONLY,
        max_results=None,
        page_token=None,
    ):
        """
        :param view_type: Qualify requested type of experiments.
        :param max_results: If passed, specifies the maximum number of experiments desired. If not
                            passed, all experiments will be returned.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``list_experiments`` call.
        :return: A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
                 :py:class:`Experiment <mlflow.entities.Experiment>` objects. The pagination token
                 for the next page can be obtained via the ``token`` attribute of the object.
        """
        _validate_list_experiments_max_results(max_results)
        return self._list_experiments(
            view_type=view_type, max_results=max_results, page_token=page_token, eager=True
        )

    def _get_experiment(self, session, experiment_id, view_type, eager=False):
        """
        :param eager: If ``True``, eagerly loads the experiments's tags. If ``False``, these tags
                      are not eagerly loaded and will be loaded if/when their corresponding
                      object properties are accessed from the resulting ``SqlExperiment`` object.
        """
        experiment_id = experiment_id or SqlAlchemyStore.DEFAULT_EXPERIMENT_ID
        stages = LifecycleStage.view_type_to_stages(view_type)
        query_options = self._get_eager_experiment_query_options() if eager else []

        experiment = (
            session.query(SqlExperiment)
            .options(*query_options)
            .filter(
                SqlExperiment.experiment_id == experiment_id,
                SqlExperiment.lifecycle_stage.in_(stages),
            )
            .one_or_none()
        )

        if experiment is None:
            raise MlflowException(
                "No Experiment with id={} exists".format(experiment_id), RESOURCE_DOES_NOT_EXIST
            )

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
                session, experiment_id, ViewType.ALL, eager=True
            ).to_mlflow_entity()

    def get_experiment_by_name(self, experiment_name):
        """
        Specialized implementation for SQL backed store.
        """
        with self.ManagedSessionMaker() as session:
            stages = LifecycleStage.view_type_to_stages(ViewType.ALL)
            experiment = (
                session.query(SqlExperiment)
                .options(*self._get_eager_experiment_query_options())
                .filter(
                    SqlExperiment.name == experiment_name, SqlExperiment.lifecycle_stage.in_(stages)
                )
                .one_or_none()
            )
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
                raise MlflowException("Cannot rename a non-active experiment.", INVALID_STATE)

            experiment.name = new_name
            self._save_to_db(objs=experiment, session=session)

    def create_run(self, experiment_id, user_id, start_time, tags):
        with self.ManagedSessionMaker() as session:
            experiment = self.get_experiment(experiment_id)
            self._check_experiment_is_active(experiment)

            run_id = uuid.uuid4().hex
            artifact_location = append_to_uri_path(
                experiment.artifact_location, run_id, SqlAlchemyStore.ARTIFACTS_FOLDER_NAME
            )
            run = SqlRun(
                name="",
                artifact_uri=artifact_location,
                run_uuid=run_id,
                experiment_id=experiment_id,
                source_type=SourceType.to_string(SourceType.UNKNOWN),
                source_name="",
                entry_point_name="",
                user_id=user_id,
                status=RunStatus.to_string(RunStatus.RUNNING),
                start_time=start_time,
                end_time=None,
                source_version="",
                lifecycle_stage=LifecycleStage.ACTIVE,
            )

            run.tags = [SqlTag(key=tag.key, value=tag.value) for tag in tags] if tags else []
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
        runs = (
            session.query(SqlRun).options(*query_options).filter(SqlRun.run_uuid == run_uuid).all()
        )

        if len(runs) == 0:
            raise MlflowException(
                "Run with id={} not found".format(run_uuid), RESOURCE_DOES_NOT_EXIST
            )
        if len(runs) > 1:
            raise MlflowException(
                "Expected only 1 run with id={}. Found {}.".format(run_uuid, len(runs)),
                INVALID_STATE,
            )

        return runs[0]

    @staticmethod
    def _get_eager_run_query_options():
        """
        :return: A list of SQLAlchemy query options that can be used to eagerly load the following
                 run attributes when fetching a run: ``latest_metrics``, ``params``, and ``tags``.
        """
        return [
            # Use a select in load rather than a joined load in order to minimize the memory
            # overhead of the eager loading procedure. For more information about relationship
            # loading techniques, see https://docs.sqlalchemy.org/en/13/orm/
            # loading_relationships.html#relationship-loading-techniques
            sqlalchemy.orm.selectinload(SqlRun.latest_metrics),
            sqlalchemy.orm.selectinload(SqlRun.params),
            sqlalchemy.orm.selectinload(SqlRun.tags),
        ]

    def _check_run_is_active(self, run):
        if run.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                "The run {} must be in the 'active' state. Current state is {}.".format(
                    run.run_uuid, run.lifecycle_stage
                ),
                INVALID_PARAMETER_VALUE,
            )

    def _check_experiment_is_active(self, experiment):
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                "The experiment {} must be in the 'active' state. "
                "Current state is {}.".format(experiment.experiment_id, experiment.lifecycle_stage),
                INVALID_PARAMETER_VALUE,
            )

    def _check_run_is_deleted(self, run):
        if run.lifecycle_stage != LifecycleStage.DELETED:
            raise MlflowException(
                "The run {} must be in the 'deleted' state. Current state is {}.".format(
                    run.run_uuid, run.lifecycle_stage
                ),
                INVALID_PARAMETER_VALUE,
            )

    def update_run_info(self, run_id, run_status, end_time):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            run.status = RunStatus.to_string(run_status)
            run.end_time = end_time

            self._save_to_db(objs=run, session=session)
            run = run.to_mlflow_entity()

            return run.info

    def _try_get_run_tag(self, session, run_id, tagKey, eager=False):
        query_options = self._get_eager_run_query_options() if eager else []
        tags = (
            session.query(SqlTag)
            .options(*query_options)
            .filter(SqlTag.run_uuid == run_id and SqlTag.key == tagKey)
            .all()
        )
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
            run_ids = (
                session.query(SqlRun.run_uuid)
                .filter(SqlRun.lifecycle_stage == LifecycleStage.DELETED)
                .all()
            )
            return [run_id[0] for run_id in run_ids]

    def _get_metric_value_details(self, metric):
        _validate_metric(metric.key, metric.value, metric.timestamp, metric.step)
        is_nan = math.isnan(metric.value)
        if is_nan:
            value = 0
        elif math.isinf(metric.value):
            #  NB: Sql can not represent Infs = > We replace +/- Inf with max/min 64b float value
            value = 1.7976931348623157e308 if metric.value > 0 else -1.7976931348623157e308
        else:
            value = metric.value
        return metric, value, is_nan

    def log_metric(self, run_id, metric):
        # simply call _log_metrics and let it handle the rest
        self._log_metrics(run_id, [metric])

    def _log_metrics(self, run_id, metrics):
        if not metrics:
            return

        # Duplicate metric values are eliminated here to maintain
        # the same behavior in log_metric
        metric_instances = []
        seen = set()
        for metric in metrics:
            metric, value, is_nan = self._get_metric_value_details(metric)
            if metric not in seen:
                metric_instances.append(
                    SqlMetric(
                        run_uuid=run_id,
                        key=metric.key,
                        value=value,
                        timestamp=metric.timestamp,
                        step=metric.step,
                        is_nan=is_nan,
                    )
                )
            seen.add(metric)

        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)

            def _insert_metrics(metric_instances):
                self._save_to_db(session=session, objs=metric_instances)
                self._update_latest_metrics_if_necessary(metric_instances, session)
                session.commit()

            try:
                _insert_metrics(metric_instances)
            except sqlalchemy.exc.IntegrityError:
                # Primary key can be violated if it is tried to log a metric with same value,
                # timestamp, step, and key within the same run.
                # Roll back the current session to make it usable for further transactions. In
                # the event of an error during "commit", a rollback is required in order to
                # continue using the session. In this case, we re-use the session to query
                # SqlMetric
                session.rollback()
                # Divide metric keys into batches of 100 to avoid loading too much metric
                # history data into memory at once
                metric_keys = [m.key for m in metric_instances]
                metric_key_batches = [
                    metric_keys[i : i + 100] for i in range(0, len(metric_keys), 100)
                ]
                for metric_key_batch in metric_key_batches:
                    # obtain the metric history corresponding to the given metrics
                    metric_history = (
                        session.query(SqlMetric)
                        .filter(
                            SqlMetric.run_uuid == run_id,
                            SqlMetric.key.in_(metric_key_batch),
                        )
                        .all()
                    )
                    # convert to a set of Metric instance to take advantage of its hashable
                    # and then obtain the metrics that were not logged earlier within this
                    # run_id
                    metric_history = {m.to_mlflow_entity() for m in metric_history}
                    non_existing_metrics = [
                        m for m in metric_instances if m.to_mlflow_entity() not in metric_history
                    ]
                    # if there exist metrics that were tried to be logged & rolled back even
                    # though they were not violating the PK, log them
                    _insert_metrics(non_existing_metrics)

    def _update_latest_metrics_if_necessary(self, logged_metrics, session):
        def _compare_metrics(metric_a, metric_b):
            """
            :return: True if ``metric_a`` is strictly more recent than ``metric_b``, as determined
                     by ``step``, ``timestamp``, and ``value``. False otherwise.
            """
            return (metric_a.step, metric_a.timestamp, metric_a.value) > (
                metric_b.step,
                metric_b.timestamp,
                metric_b.value,
            )

        def _overwrite_metric(new_metric, old_metric):
            """
            writes content of new_metric over old_metric. The content are
            `value`, `step`, `timestamp`, and `is_nan`.

            :return: old_metric with its content updated.
            """
            old_metric.value = new_metric.value
            old_metric.step = new_metric.step
            old_metric.timestamp = new_metric.timestamp
            old_metric.is_nan = new_metric.is_nan
            return old_metric

        if not logged_metrics:
            return

        # Fetch the latest metric value corresponding to the specified run_id and metric keys and
        # lock their associated rows for the remainder of the transaction in order to ensure
        # isolation
        latest_metrics = {}
        metric_keys = [m.key for m in logged_metrics]
        # Divide metric keys into batches of 500 to avoid binding too many parameters to the SQL
        # query, which may produce limit exceeded errors or poor performance on certain database
        # platforms
        metric_key_batches = [metric_keys[i : i + 500] for i in range(0, len(metric_keys), 500)]
        for metric_key_batch in metric_key_batches:
            # First, determine which metric keys are present in the database
            latest_metrics_key_records_from_db = (
                session.query(SqlLatestMetric.key)
                .filter(
                    SqlLatestMetric.run_uuid == logged_metrics[0].run_uuid,
                    SqlLatestMetric.key.in_(metric_key_batch),
                )
                .all()
            )
            # Then, take a write lock on the rows corresponding to metric keys that are present,
            # ensuring that they aren't modified by another transaction until they can be
            # compared to the metric values logged by this transaction while avoiding gap locking
            # and next-key locking which may otherwise occur when issuing a `SELECT FOR UPDATE`
            # against nonexistent rows
            if len(latest_metrics_key_records_from_db) > 0:
                latest_metric_keys_from_db = [
                    record[0] for record in latest_metrics_key_records_from_db
                ]
                latest_metrics_batch = (
                    session.query(SqlLatestMetric)
                    .filter(
                        SqlLatestMetric.run_uuid == logged_metrics[0].run_uuid,
                        SqlLatestMetric.key.in_(latest_metric_keys_from_db),
                    )
                    # Order by the metric run ID and key to ensure a consistent locking order
                    # across transactions, reducing deadlock likelihood
                    .order_by(SqlLatestMetric.run_uuid, SqlLatestMetric.key)
                    .with_for_update()
                    .all()
                )
                latest_metrics.update({m.key: m for m in latest_metrics_batch})

        # iterate over all logged metrics and compare them with corresponding
        # SqlLatestMetric entries
        # if there's no SqlLatestMetric entry for the current metric key,
        # create a new SqlLatestMetric instance and put it in
        # new_latest_metric_dict so that they can be saved later.
        new_latest_metric_dict = {}
        for logged_metric in logged_metrics:
            latest_metric = latest_metrics.get(logged_metric.key)
            # a metric key can be passed more then once within logged metrics
            # with different step/timestamp/value. However SqlLatestMetric
            # entries are inserted after this loop is completed.
            # so, retrieve the instances they were just created and use them
            # for comparison.
            new_latest_metric = new_latest_metric_dict.get(logged_metric.key)

            # just create a new SqlLatestMetric instance since both
            # latest_metric row or recently created instance does not exist
            if not latest_metric and not new_latest_metric:
                new_latest_metric = SqlLatestMetric(
                    run_uuid=logged_metric.run_uuid,
                    key=logged_metric.key,
                    value=logged_metric.value,
                    timestamp=logged_metric.timestamp,
                    step=logged_metric.step,
                    is_nan=logged_metric.is_nan,
                )
                new_latest_metric_dict[logged_metric.key] = new_latest_metric

            # there's no row but a new instance is recently created.
            # so, update the recent instance in new_latest_metric_dict if
            # metric comparison is successful.
            elif not latest_metric and new_latest_metric:
                if _compare_metrics(logged_metric, new_latest_metric):
                    new_latest_metric = _overwrite_metric(logged_metric, new_latest_metric)
                    new_latest_metric_dict[logged_metric.key] = new_latest_metric

            # compare with the row
            elif _compare_metrics(logged_metric, latest_metric):
                # editing the attributes of latest_metric, which is a
                # SqlLatestMetric instance will result in UPDATE in DB side.
                latest_metric = _overwrite_metric(logged_metric, latest_metric)

        if new_latest_metric_dict:
            self._save_to_db(session=session, objs=list(new_latest_metric_dict.values()))

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
                self._get_or_create(
                    model=SqlParam,
                    session=session,
                    run_uuid=run_id,
                    key=param.key,
                    value=param.value,
                )
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
                        " '{}'.".format(param.key, old_value, run_id, param.value),
                        INVALID_PARAMETER_VALUE,
                    )
                else:
                    raise

    def _log_params(self, run_id, params):
        if not params:
            return

        param_instances = [
            SqlParam(run_uuid=run_id, key=param.key, value=param.value) for param in params
        ]

        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            # commit the session to make sure that we catch any IntegrityError
            # and try to handle them.
            try:
                self._save_to_db(session=session, objs=param_instances)
                session.commit()
            except sqlalchemy.exc.IntegrityError:
                # Roll back the current session to make it usable for further transactions. In the
                # event of an error during "commit", a rollback is required in order to continue
                # using the session. In this case, we re-use the session because the SqlRun, `run`,
                # is lazily evaluated during the invocation of `run.params`.
                session.rollback()

                # in case of an integrity error, compare the parameters of the
                # run. If the parameters match the ones whom being saved,
                # ignore the exception since idempotency is reached.
                # Also, multiple params for the same key can still be passed within
                # the same batch. So, handle them by selecting the first param
                # for the given key
                run_params = {param.key: param.value for param in run.params}
                non_matching_params = []
                for param in param_instances:
                    existing_value = run_params.get(param.key)
                    if param.value != existing_value:
                        non_matching_params.append(
                            {
                                "key": param.key,
                                "old_value": existing_value,
                                "new_value": param.value,
                            }
                        )

                if non_matching_params:
                    raise MlflowException(
                        "Changing param values is not allowed. Params were already logged='{}'"
                        " for run ID='{}'.".format(non_matching_params, run_id),
                        INVALID_PARAMETER_VALUE,
                    )
                # if there's no mismatch, do not raise an Exception since
                # we are sure that idempotency is reached.

    def set_experiment_tag(self, experiment_id, tag):
        """
        Set a tag for the specified experiment

        :param experiment_id: String ID of the experiment
        :param tag: ExperimentRunTag instance to log
        """
        _validate_experiment_tag(tag.key, tag.value)
        with self.ManagedSessionMaker() as session:
            experiment = self._get_experiment(
                session, experiment_id, ViewType.ALL
            ).to_mlflow_entity()
            self._check_experiment_is_active(experiment)
            session.merge(
                SqlExperimentTag(experiment_id=experiment_id, key=tag.key, value=tag.value)
            )

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

    def _set_tags(self, run_id, tags):
        """
        Set multiple tags on a run

        :param run_id: String ID of the run
        :param tags: List of RunTag instances to log
        """
        if not tags:
            return

        for tag in tags:
            _validate_tag(tag.key, tag.value)

        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)

            def _try_insert_tags(attempt_number, max_retries):
                try:
                    current_tags = (
                        session.query(SqlTag)
                        .filter(SqlTag.run_uuid == run_id, SqlTag.key.in_([t.key for t in tags]))
                        .all()
                    )
                    current_tags = {t.key: t for t in current_tags}

                    new_tag_dict = {}
                    for tag in tags:
                        current_tag = current_tags.get(tag.key)
                        new_tag = new_tag_dict.get(tag.key)

                        # update the SqlTag if it is already present in DB
                        if current_tag:
                            current_tag.value = tag.value
                            continue

                        # if a SqlTag instance is already present in `new_tag_dict`,
                        # this means that multiple tags with the same key were passed to `set_tags`.
                        # In this case, we resolve potential conflicts by updating the value of the
                        # existing instance to the value of `tag`
                        if new_tag:
                            new_tag.value = tag.value
                        # otherwise, put it into the dict
                        else:
                            new_tag = SqlTag(run_uuid=run_id, key=tag.key, value=tag.value)

                        new_tag_dict[tag.key] = new_tag

                    # finally, save new entries to DB.
                    self._save_to_db(session=session, objs=list(new_tag_dict.values()))
                    session.commit()
                except sqlalchemy.exc.IntegrityError:
                    session.rollback()
                    # two concurrent operations may try to attempt to insert tags.
                    # apply retry here.
                    if attempt_number > max_retries:
                        raise MlflowException(
                            "Failed to set tags with given within {} retries. Keys: {}".format(
                                max_retries, [t.key for t in tags]
                            )
                        )
                    sleep_duration = (2**attempt_number) - 1
                    sleep_duration += random.uniform(0, 1)
                    time.sleep(sleep_duration)
                    _try_insert_tags(attempt_number + 1, max_retries=max_retries)

            _try_insert_tags(attempt_number=0, max_retries=3)

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
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            elif len(filtered_tags) > 1:
                raise MlflowException(
                    "Bad data in database - tags for a specific run must have "
                    "a single unique value."
                    "See https://mlflow.org/docs/latest/tracking.html#adding-tags-to-runs",
                    error_code=INVALID_STATE,
                )
            session.delete(filtered_tags[0])

    def _search_runs(
        self, experiment_ids, filter_string, run_view_type, max_results, order_by, page_token
    ):
        def compute_next_token(current_size):
            next_token = None
            if max_results == current_size:
                final_offset = offset + max_results
                next_token = SearchUtils.create_page_token(final_offset)

            return next_token

        if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at "
                "most {}, but got value {}".format(SEARCH_MAX_RESULTS_THRESHOLD, max_results),
                INVALID_PARAMETER_VALUE,
            )

        stages = set(LifecycleStage.view_type_to_stages(run_view_type))

        with self.ManagedSessionMaker() as session:
            # Fetch the appropriate runs and eagerly load their summary metrics, params, and
            # tags. These run attributes are referenced during the invocation of
            # ``run.to_mlflow_entity()``, so eager loading helps avoid additional database queries
            # that are otherwise executed at attribute access time under a lazy loading model.
            parsed_filters = SearchUtils.parse_search_filter(filter_string)
            cases_orderby, parsed_orderby, sorting_joins = _get_orderby_clauses(order_by, session)

            stmt = select(SqlRun, *cases_orderby)
            for j in _get_sqlalchemy_filter_clauses(parsed_filters, session, self._get_dialect()):
                stmt = stmt.join(j)
            # using an outer join is necessary here because we want to be able to sort
            # on a column (tag, metric or param) without removing the lines that
            # do not have a value for this column (which is what inner join would do)
            for j in sorting_joins:
                stmt = stmt.outerjoin(j)

            offset = SearchUtils.parse_start_offset_from_page_token(page_token)
            stmt = (
                stmt.distinct()
                .options(*self._get_eager_run_query_options())
                .filter(
                    SqlRun.experiment_id.in_(experiment_ids),
                    SqlRun.lifecycle_stage.in_(stages),
                    *_get_attributes_filtering_clauses(parsed_filters, self._get_dialect()),
                )
                .order_by(*parsed_orderby)
                .offset(offset)
                .limit(max_results)
            )
            queried_runs = session.execute(stmt).scalars(SqlRun).all()

            runs = [run.to_mlflow_entity() for run in queried_runs]
            next_page_token = compute_next_token(len(runs))

        return runs, next_page_token

    def log_batch(self, run_id, metrics, params, tags):
        _validate_run_id(run_id)
        _validate_batch_log_data(metrics, params, tags)
        _validate_batch_log_limits(metrics, params, tags)
        _validate_param_keys_unique(params)

        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            try:
                self._log_params(run_id, params)
                self._log_metrics(run_id, metrics)
                self._set_tags(run_id, tags)
            except MlflowException as e:
                raise e
            except Exception as e:
                raise MlflowException(e, INTERNAL_ERROR)

    def record_logged_model(self, run_id, mlflow_model):
        from mlflow.models import Model

        if not isinstance(mlflow_model, Model):
            raise TypeError(
                "Argument 'mlflow_model' should be mlflow.models.Model, got '{}'".format(
                    type(mlflow_model)
                )
            )
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


def _get_attributes_filtering_clauses(parsed, dialect):
    clauses = []
    for sql_statement in parsed:
        key_type = sql_statement.get("type")
        key_name = sql_statement.get("key")
        value = sql_statement.get("value")
        comparator = sql_statement.get("comparator").upper()
        if SearchUtils.is_string_attribute(
            key_type, key_name, comparator
        ) or SearchUtils.is_numeric_attribute(key_type, key_name, comparator):
            # key_name is guaranteed to be a valid searchable attribute of entities.RunInfo
            # by the call to parse_search_filter
            attribute = getattr(SqlRun, SqlRun.get_attribute_name(key_name))
            if comparator in SearchUtils.CASE_INSENSITIVE_STRING_COMPARISON_OPERATORS:
                op = SearchUtils.get_sql_filter_ops(attribute, comparator, dialect)
                clauses.append(op(value))
            elif comparator in SearchUtils.filter_ops:
                op = SearchUtils.filter_ops.get(comparator)
                clauses.append(op(attribute, value))
    return clauses


def _to_sqlalchemy_filtering_statement(sql_statement, session, dialect):
    key_type = sql_statement.get("type")
    key_name = sql_statement.get("key")
    value = sql_statement.get("value")
    comparator = sql_statement.get("comparator").upper()

    if SearchUtils.is_metric(key_type, comparator):
        entity = SqlLatestMetric
        value = float(value)
    elif SearchUtils.is_param(key_type, comparator):
        entity = SqlParam
    elif SearchUtils.is_tag(key_type, comparator):
        entity = SqlTag
    elif SearchUtils.is_string_attribute(
        key_type, key_name, comparator
    ) or SearchUtils.is_numeric_attribute(key_type, key_name, comparator):
        return None
    else:
        raise MlflowException(
            "Invalid search expression type '%s'" % key_type, error_code=INVALID_PARAMETER_VALUE
        )

    if comparator in SearchUtils.CASE_INSENSITIVE_STRING_COMPARISON_OPERATORS:
        op = SearchUtils.get_sql_filter_ops(entity.value, comparator, dialect)
        return session.query(entity).filter(entity.key == key_name, op(value)).subquery()
    elif comparator in SearchUtils.filter_ops:
        op = SearchUtils.filter_ops.get(comparator)
        return (
            session.query(entity).filter(entity.key == key_name, op(entity.value, value)).subquery()
        )
    else:
        return None


def _get_sqlalchemy_filter_clauses(parsed, session, dialect):
    """creates SqlAlchemy subqueries
    that will be inner-joined to SQLRun to act as multi-clause filters."""
    filters = []
    for sql_statement in parsed:
        filter_query = _to_sqlalchemy_filtering_statement(sql_statement, session, dialect)
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
    observed_order_by_clauses = set()
    select_clauses = []
    # contrary to filters, it is not easily feasible to separately handle sorting
    # on attributes and on joined tables as we must keep all clauses in the same order
    if order_by_list:
        for order_by_clause in order_by_list:
            clause_id += 1
            (key_type, key, ascending) = SearchUtils.parse_order_by_for_search_runs(order_by_clause)
            if SearchUtils.is_string_attribute(
                key_type, key, "="
            ) or SearchUtils.is_numeric_attribute(key_type, key, "="):
                order_value = getattr(SqlRun, SqlRun.get_attribute_name(key))
            else:
                if SearchUtils.is_metric(key_type, "="):  # any valid comparator
                    entity = SqlLatestMetric
                elif SearchUtils.is_tag(key_type, "="):
                    entity = SqlTag
                elif SearchUtils.is_param(key_type, "="):
                    entity = SqlParam
                else:
                    raise MlflowException(
                        "Invalid identifier type '%s'" % key_type,
                        error_code=INVALID_PARAMETER_VALUE,
                    )

                # build a subquery first because we will join it in the main request so that the
                # metric we want to sort on is available when we apply the sorting clause
                subquery = session.query(entity).filter(entity.key == key).subquery()

                ordering_joins.append(subquery)
                order_value = subquery.c.value

            # sqlite does not support NULLS LAST expression, so we sort first by
            # presence of the field (and is_nan for metrics), then by actual value
            # As the subqueries are created independently and used later in the
            # same main query, the CASE WHEN columns need to have unique names to
            # avoid ambiguity
            if SearchUtils.is_metric(key_type, "="):
                case = sql.case(
                    [
                        # Ideally the use of "IS" is preferred here but owing to sqlalchemy
                        # translation in MSSQL we are forced to use "=" instead.
                        # These 2 options are functionally identical / unchanged because
                        # the column (is_nan) is not nullable. However it could become an issue
                        # if this precondition changes in the future.
                        (subquery.c.is_nan == sqlalchemy.true(), 1),
                        (order_value.is_(None), 2),
                    ],
                    else_=0,
                ).label("clause_%s" % clause_id)

            else:  # other entities do not have an 'is_nan' field
                case = sql.case([(order_value.is_(None), 1)], else_=0).label(
                    "clause_%s" % clause_id
                )
            clauses.append(case.name)
            select_clauses.append(case)
            select_clauses.append(order_value)

            if (key_type, key) in observed_order_by_clauses:
                raise MlflowException(
                    "`order_by` contains duplicate fields: {}".format(order_by_list)
                )
            observed_order_by_clauses.add((key_type, key))

            if ascending:
                clauses.append(order_value)
            else:
                clauses.append(order_value.desc())

    if (SearchUtils._ATTRIBUTE_IDENTIFIER, SqlRun.start_time.key) not in observed_order_by_clauses:
        clauses.append(SqlRun.start_time.desc())
    clauses.append(SqlRun.run_uuid)
    return select_clauses, clauses, ordering_joins
