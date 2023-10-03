import json
import logging
import math
import random
import threading
import time
import uuid
from functools import reduce
from typing import List, Optional

import sqlalchemy
import sqlalchemy.sql.expression as sql
from sqlalchemy import and_, sql, text
from sqlalchemy.future import select

import mlflow.store.db.utils
from mlflow.entities import (
    DatasetInput,
    Experiment,
    Metric,
    Run,
    RunInputs,
    RunStatus,
    RunTag,
    SourceType,
    ViewType,
    _DatasetSummary,
)
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.db.db_types import MSSQL, MYSQL
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT, SEARCH_MAX_RESULTS_THRESHOLD
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.dbmodels.models import (
    SqlDataset,
    SqlExperiment,
    SqlExperimentTag,
    SqlInput,
    SqlInputTag,
    SqlLatestMetric,
    SqlMetric,
    SqlParam,
    SqlRun,
    SqlTag,
)
from mlflow.utils.file_utils import local_file_uri_to_path, mkdir
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATASET_CONTEXT,
    MLFLOW_LOGGED_MODELS,
    MLFLOW_RUN_NAME,
    _get_run_name_from_tags,
)
from mlflow.utils.name_utils import _generate_random_name
from mlflow.utils.search_utils import SearchExperimentsUtils, SearchUtils
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import (
    append_to_uri_path,
    extract_db_type_from_uri,
    is_local_uri,
    resolve_uri_if_local,
)
from mlflow.utils.validation import (
    _validate_batch_log_data,
    _validate_batch_log_limits,
    _validate_dataset_inputs,
    _validate_experiment_name,
    _validate_experiment_tag,
    _validate_metric,
    _validate_param,
    _validate_param_keys_unique,
    _validate_run_id,
    _validate_tag,
)

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
        self.artifact_root_uri = resolve_uri_if_local(default_artifact_root)
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
        if not mlflow.store.db.utils._all_tables_exist(self.engine):
            mlflow.store.db.utils._initialize_tables(self.engine)
        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = mlflow.store.db.utils._get_managed_session_maker(
            SessionMaker, self.db_type
        )
        mlflow.store.db.utils._verify_schema(self.engine)

        if is_local_uri(default_artifact_root):
            mkdir(local_file_uri_to_path(default_artifact_root))

        if len(self.search_experiments(view_type=ViewType.ALL)) == 0:
            with self.ManagedSessionMaker() as session:
                self._create_default_experiment(session)

    def _get_dialect(self):
        return self.engine.dialect.name

    def _dispose_engine(self):
        self.engine.dispose()

    def _set_zero_value_insertion_for_autoincrement_column(self, session):
        if self.db_type == MYSQL:
            # config letting MySQL override default
            # to allow 0 value for experiment ID (auto increment column)
            session.execute(sql.text("SET @@SESSION.sql_mode='NO_AUTO_VALUE_ON_ZERO';"))
        if self.db_type == MSSQL:
            # config letting MSSQL override default
            # to allow any manual value inserted into IDENTITY column
            session.execute(sql.text("SET IDENTITY_INSERT experiments ON;"))

    # DB helper methods to allow zero values for columns with auto increments
    def _unset_zero_value_insertion_for_autoincrement_column(self, session):
        if self.db_type == MYSQL:
            session.execute(sql.text("SET @@SESSION.sql_mode='';"))
        if self.db_type == MSSQL:
            session.execute(sql.text("SET IDENTITY_INSERT experiments OFF;"))

    def _create_default_experiment(self, session):
        """
        MLflow UI and client code expects a default experiment with ID 0.
        This method uses SQL insert statement to create the default experiment as a hack, since
        experiment table uses 'experiment_id' column is a PK and is also set to auto increment.
        MySQL and other implementation do not allow value '0' for such cases.

        ToDo: Identify a less hacky mechanism to create default experiment 0
        """
        table = SqlExperiment.__tablename__
        creation_time = get_current_time_millis()
        default_experiment = {
            SqlExperiment.experiment_id.name: int(SqlAlchemyStore.DEFAULT_EXPERIMENT_ID),
            SqlExperiment.name.name: Experiment.DEFAULT_EXPERIMENT_NAME,
            SqlExperiment.artifact_location.name: str(self._get_artifact_location(0)),
            SqlExperiment.lifecycle_stage.name: LifecycleStage.ACTIVE,
            SqlExperiment.creation_time.name: creation_time,
            SqlExperiment.last_update_time.name: creation_time,
        }

        def decorate(s):
            if is_string_type(s):
                return repr(s)
            else:
                return str(s)

        # Get a list of keys to ensure we have a deterministic ordering
        columns = list(default_experiment.keys())
        values = ", ".join([decorate(default_experiment.get(c)) for c in columns])

        try:
            self._set_zero_value_insertion_for_autoincrement_column(session)
            session.execute(
                sql.text(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({values});")
            )
        finally:
            self._unset_zero_value_insertion_for_autoincrement_column(session)

    def _get_or_create(self, session, model, **kwargs):
        instance = session.query(model).filter_by(**kwargs).first()
        created = False

        if instance:
            return instance, created
        else:
            instance = model(**kwargs)
            session.add(instance)
            created = True

        return instance, created

    def _get_artifact_location(self, experiment_id):
        return append_to_uri_path(self.artifact_root_uri, str(experiment_id))

    def create_experiment(self, name, artifact_location=None, tags=None):
        _validate_experiment_name(name)
        if artifact_location:
            artifact_location = resolve_uri_if_local(artifact_location)
        with self.ManagedSessionMaker() as session:
            try:
                creation_time = get_current_time_millis()
                experiment = SqlExperiment(
                    name=name,
                    lifecycle_stage=LifecycleStage.ACTIVE,
                    artifact_location=artifact_location,
                    creation_time=creation_time,
                    last_update_time=creation_time,
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
                    f"Experiment(name={name}) already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                )

            session.flush()
            return str(experiment.experiment_id)

    def _search_experiments(
        self,
        view_type,
        max_results,
        filter_string,
        order_by,
        page_token,
    ):
        def compute_next_token(current_size):
            next_token = None
            if max_results + 1 == current_size:
                final_offset = offset + max_results
                next_token = SearchExperimentsUtils.create_page_token(final_offset)

            return next_token

        if not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                "Invalid value for max_results. It must be a positive integer,"
                f" but got {max_results}",
                INVALID_PARAMETER_VALUE,
            )
        if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                f"Invalid value for max_results. It must be at most {SEARCH_MAX_RESULTS_THRESHOLD},"
                f" but got {max_results}",
                INVALID_PARAMETER_VALUE,
            )
        with self.ManagedSessionMaker() as session:
            parsed_filters = SearchExperimentsUtils.parse_search_filter(filter_string)
            attribute_filters, non_attribute_filters = _get_search_experiments_filter_clauses(
                parsed_filters, self._get_dialect()
            )

            order_by_clauses = _get_search_experiments_order_by_clauses(order_by)
            offset = SearchUtils.parse_start_offset_from_page_token(page_token)
            lifecycle_stags = set(LifecycleStage.view_type_to_stages(view_type))

            stmt = (
                reduce(lambda s, f: s.join(f), non_attribute_filters, select(SqlExperiment))
                .options(*self._get_eager_experiment_query_options())
                .filter(*attribute_filters, SqlExperiment.lifecycle_stage.in_(lifecycle_stags))
                .order_by(*order_by_clauses)
                .offset(offset)
                .limit(max_results + 1)
            )
            queried_experiments = session.execute(stmt).scalars(SqlExperiment).all()
            experiments = [e.to_mlflow_entity() for e in queried_experiments]
            next_page_token = compute_next_token(len(experiments))

        return experiments[:max_results], next_page_token

    def search_experiments(
        self,
        view_type=ViewType.ACTIVE_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        filter_string=None,
        order_by=None,
        page_token=None,
    ):
        experiments, next_page_token = self._search_experiments(
            view_type, max_results, filter_string, order_by, page_token
        )
        return PagedList(experiments, next_page_token)

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
                f"No Experiment with id={experiment_id} exists", RESOURCE_DOES_NOT_EXIST
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
            experiment.last_update_time = get_current_time_millis()
            runs = self._list_run_infos(session, experiment_id)
            for run in runs:
                self._mark_run_deleted(session, run)
            session.add(experiment)

    def _hard_delete_experiment(self, experiment_id):
        """
        Permanently delete a experiment (metadata and metrics, tags, parameters).
        This is used by the ``mlflow gc`` command line and is not intended to be used elsewhere.
        """
        with self.ManagedSessionMaker() as session:
            experiment = self._get_experiment(
                experiment_id=experiment_id, session=session, view_type=ViewType.DELETED_ONLY
            )
            session.delete(experiment)

    def _mark_run_deleted(self, session, run):
        run.lifecycle_stage = LifecycleStage.DELETED
        run.deleted_time = get_current_time_millis()
        session.add(run)

    def _mark_run_active(self, session, run):
        run.lifecycle_stage = LifecycleStage.ACTIVE
        run.deleted_time = None
        session.add(run)

    def _list_run_infos(self, session, experiment_id):
        return session.query(SqlRun).filter(SqlRun.experiment_id == experiment_id).all()

    def restore_experiment(self, experiment_id):
        with self.ManagedSessionMaker() as session:
            experiment = self._get_experiment(session, experiment_id, ViewType.DELETED_ONLY)
            experiment.lifecycle_stage = LifecycleStage.ACTIVE
            experiment.last_update_time = get_current_time_millis()
            runs = self._list_run_infos(session, experiment_id)
            for run in runs:
                self._mark_run_active(session, run)
            session.add(experiment)

    def rename_experiment(self, experiment_id, new_name):
        with self.ManagedSessionMaker() as session:
            experiment = self._get_experiment(session, experiment_id, ViewType.ALL)
            if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
                raise MlflowException("Cannot rename a non-active experiment.", INVALID_STATE)

            experiment.name = new_name
            experiment.last_update_time = get_current_time_millis()
            session.add(experiment)

    def create_run(self, experiment_id, user_id, start_time, tags, run_name):
        with self.ManagedSessionMaker() as session:
            experiment = self.get_experiment(experiment_id)
            self._check_experiment_is_active(experiment)

            # Note: we need to ensure the generated "run_id" only contains digits and lower
            # case letters, because some query filters contain "IN" clause, and in MYSQL the
            # "IN" clause is case-insensitive, we use a trick that filters out comparison values
            # containing upper case letters when parsing "IN" clause inside query filter.
            run_id = uuid.uuid4().hex
            artifact_location = append_to_uri_path(
                experiment.artifact_location, run_id, SqlAlchemyStore.ARTIFACTS_FOLDER_NAME
            )
            tags = tags or []
            run_name_tag = _get_run_name_from_tags(tags)
            if run_name and run_name_tag and (run_name != run_name_tag):
                raise MlflowException(
                    "Both 'run_name' argument and 'mlflow.runName' tag are specified, but with "
                    f"different values (run_name='{run_name}', mlflow.runName='{run_name_tag}').",
                    INVALID_PARAMETER_VALUE,
                )
            run_name = run_name or run_name_tag or _generate_random_name()
            if not run_name_tag:
                tags.append(RunTag(key=MLFLOW_RUN_NAME, value=run_name))
            run = SqlRun(
                name=run_name,
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
                deleted_time=None,
                source_version="",
                lifecycle_stage=LifecycleStage.ACTIVE,
            )

            run.tags = [SqlTag(key=tag.key, value=tag.value) for tag in tags]
            session.add(run)

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
            raise MlflowException(f"Run with id={run_uuid} not found", RESOURCE_DOES_NOT_EXIST)
        if len(runs) > 1:
            raise MlflowException(
                f"Expected only 1 run with id={run_uuid}. Found {len(runs)}.",
                INVALID_STATE,
            )

        return runs[0]

    def _get_run_inputs(self, session, run_uuids):
        datasets = (
            session.query(
                SqlInput.input_uuid, SqlInput.destination_id.label("run_uuid"), SqlDataset
            )
            .select_from(SqlDataset)
            .join(SqlInput, SqlInput.source_id == SqlDataset.dataset_uuid)
            .filter(SqlInput.destination_type == "RUN", SqlInput.destination_id.in_(run_uuids))
            .order_by("run_uuid")
        ).all()
        input_uuids = [dataset.input_uuid for dataset in datasets]
        input_tags = (
            session.query(
                SqlInput.input_uuid, SqlInput.destination_id.label("run_uuid"), SqlInputTag
            )
            .join(SqlInput, (SqlInput.input_uuid == SqlInputTag.input_uuid))
            .filter(SqlInput.input_uuid.in_(input_uuids))
            .order_by("run_uuid")
        ).all()

        all_dataset_inputs = []
        for run_uuid in run_uuids:
            dataset_inputs = []
            for input_uuid, dataset_run_uuid, dataset_sql in datasets:
                if run_uuid == dataset_run_uuid:
                    dataset_entity = dataset_sql.to_mlflow_entity()
                    tags = []
                    for tag_input_uuid, tag_run_uuid, tag_sql in input_tags:
                        if input_uuid == tag_input_uuid and run_uuid == tag_run_uuid:
                            tags.append(tag_sql.to_mlflow_entity())
                    dataset_input_entity = DatasetInput(dataset=dataset_entity, tags=tags)
                    dataset_inputs.append(dataset_input_entity)
            all_dataset_inputs.append(dataset_inputs)
        return all_dataset_inputs

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
                (
                    f"The run {run.run_uuid} must be in the 'active' state. "
                    f"Current state is {run.lifecycle_stage}."
                ),
                INVALID_PARAMETER_VALUE,
            )

    def _check_experiment_is_active(self, experiment):
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                (
                    f"The experiment {experiment.experiment_id} must be in the 'active' state. "
                    f"Current state is {experiment.lifecycle_stage}."
                ),
                INVALID_PARAMETER_VALUE,
            )

    def update_run_info(self, run_id, run_status, end_time, run_name):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            if run_status is not None:
                run.status = RunStatus.to_string(run_status)
            if end_time is not None:
                run.end_time = end_time
            if run_name:
                run.name = run_name
                run_name_tag = self._try_get_run_tag(session, run_id, MLFLOW_RUN_NAME)
                if run_name_tag is None:
                    run.tags.append(SqlTag(key=MLFLOW_RUN_NAME, value=run_name))
                else:
                    run_name_tag.value = run_name

            session.add(run)
            run = run.to_mlflow_entity()

            return run.info

    def _try_get_run_tag(self, session, run_id, tagKey, eager=False):
        query_options = self._get_eager_run_query_options() if eager else []
        return (
            session.query(SqlTag)
            .options(*query_options)
            .filter(SqlTag.run_uuid == run_id, SqlTag.key == tagKey)
            .one_or_none()
        )

    def get_run(self, run_id):
        with self.ManagedSessionMaker() as session:
            # Load the run with the specified id and eagerly load its summary metrics, params, and
            # tags. These attributes are referenced during the invocation of
            # ``run.to_mlflow_entity()``, so eager loading helps avoid additional database queries
            # that are otherwise executed at attribute access time under a lazy loading model.
            run = self._get_run(run_uuid=run_id, session=session, eager=True)
            mlflow_run = run.to_mlflow_entity()
            # Get the run inputs and add to the run
            inputs = self._get_run_inputs(run_uuids=[run_id], session=session)[0]
            return Run(mlflow_run.info, mlflow_run.data, RunInputs(dataset_inputs=inputs))

    def restore_run(self, run_id):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            run.lifecycle_stage = LifecycleStage.ACTIVE
            run.deleted_time = None
            session.add(run)

    def delete_run(self, run_id):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            run.lifecycle_stage = LifecycleStage.DELETED
            run.deleted_time = get_current_time_millis()
            session.add(run)

    def _hard_delete_run(self, run_id):
        """
        Permanently delete a run (metadata and metrics, tags, parameters).
        This is used by the ``mlflow gc`` command line and is not intended to be used elsewhere.
        """
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            session.delete(run)

    def _get_deleted_runs(self, older_than=0):
        """
        Get all deleted run ids.
        Args:
            older_than: get runs that is older than this variable in number of milliseconds.
                        defaults to 0 ms to get all deleted runs.
        """
        current_time = get_current_time_millis()
        with self.ManagedSessionMaker() as session:
            runs = (
                session.query(SqlRun)
                .filter(
                    SqlRun.lifecycle_stage == LifecycleStage.DELETED,
                    SqlRun.deleted_time <= (current_time - older_than),
                )
                .all()
            )
            return [run.run_uuid for run in runs]

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
                session.add_all(metric_instances)
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
            session.add_all(new_latest_metric_dict.values())

    def get_metric_history(self, run_id, metric_key, max_results=None, page_token=None):
        """
        Return all logged values for a given metric.

        :param run_id: Unique identifier for run
        :param metric_key: Metric name within the run
        :param max_results: An indicator for paginated results. This functionality is not
            implemented for SQLAlchemyStore and is unused in this store's implementation.
        :param page_token: An indicator for paginated results. This functionality is not
            implemented for SQLAlchemyStore and if the value is overridden with a value other than
            ``None``, an MlflowException will be thrown.

        :return: A List of :py:class:`mlflow.entities.Metric` entities if ``metric_key`` values
            have been logged to the ``run_id``, else an empty list.
        """
        # NB: The SQLAlchemyStore does not currently support pagination for this API.
        # Raise if `page_token` is specified, as the functionality to support paged queries
        # is not implemented.
        if page_token is not None:
            raise MlflowException(
                "The SQLAlchemyStore backend does not support pagination for the "
                f"`get_metric_history` API. Supplied argument `page_token` '{page_token}' must be "
                "`None`."
            )

        with self.ManagedSessionMaker() as session:
            metrics = session.query(SqlMetric).filter_by(run_uuid=run_id, key=metric_key).all()
            return PagedList([metric.to_mlflow_entity() for metric in metrics], None)

    class MetricWithRunId(Metric):
        def __init__(self, metric: Metric, run_id):
            super().__init__(
                key=metric.key,
                value=metric.value,
                timestamp=metric.timestamp,
                step=metric.step,
            )
            self._run_id = run_id

        @property
        def run_id(self):
            return self._run_id

        def to_dict(self):
            return {
                "key": self.key,
                "value": self.value,
                "timestamp": self.timestamp,
                "step": self.step,
                "run_id": self.run_id,
            }

    def get_metric_history_bulk(self, run_ids, metric_key, max_results):
        """
        Return all logged values for a given metric.

        :param run_ids: Unique identifiers of the runs from which to fetch the metric histories for
                        the specified key.
        :param metric_key: Metric name within the runs.
        :param max_results: The maximum number of results to return.

        :return: A List of :py:class:`SqlAlchemyStore.MetricWithRunId` objects if ``metric_key``
            values have been logged to one or more of the specified ``run_ids``, else an empty
            list. Results are sorted by run ID in lexicographically ascending order, followed by
            timestamp, step, and value in numerically ascending order.
        """
        # NB: The SQLAlchemyStore does not currently support pagination for this API.
        # Raise if `page_token` is specified, as the functionality to support paged queries
        # is not implemented.
        with self.ManagedSessionMaker() as session:
            metrics = (
                session.query(SqlMetric)
                .filter(
                    SqlMetric.key == metric_key,
                    SqlMetric.run_uuid.in_(run_ids),
                )
                .order_by(
                    SqlMetric.run_uuid,
                    SqlMetric.timestamp,
                    SqlMetric.step,
                    SqlMetric.value,
                )
                .limit(max_results)
                .all()
            )
            return [
                SqlAlchemyStore.MetricWithRunId(
                    run_id=metric.run_uuid,
                    metric=metric.to_mlflow_entity(),
                )
                for metric in metrics
            ]

    def _search_datasets(self, experiment_ids):
        """
        Return all dataset summaries associated to the given experiments.

        :param experiment_ids List of experiment ids to scope the search

        :return A List of :py:class:`SqlAlchemyStore.DatasetSummary` entities.
        """

        MAX_DATASET_SUMMARIES_RESULTS = 1000
        with self.ManagedSessionMaker() as session:
            # Note that the join with the input tag table is a left join. This is required so if an
            # input does not have the MLFLOW_DATASET_CONTEXT tag, we still return that entry as part
            # of the final result with the context set to None.
            summaries = (
                session.query(
                    SqlDataset.experiment_id, SqlDataset.name, SqlDataset.digest, SqlInputTag.value
                )
                .select_from(SqlDataset)
                .distinct()
                .join(SqlInput, SqlInput.source_id == SqlDataset.dataset_uuid)
                .join(
                    SqlInputTag,
                    and_(
                        SqlInput.input_uuid == SqlInputTag.input_uuid,
                        SqlInputTag.name == MLFLOW_DATASET_CONTEXT,
                    ),
                    isouter=True,
                )
                .filter(SqlDataset.experiment_id.in_(experiment_ids))
                .limit(MAX_DATASET_SUMMARIES_RESULTS)
                .all()
            )

            return [
                _DatasetSummary(
                    experiment_id=str(summary.experiment_id),
                    name=summary.name,
                    digest=summary.digest,
                    context=summary.value,
                )
                for summary in summaries
            ]

    def log_param(self, run_id, param):
        _validate_param(param.key, param.value)
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
                    if old_value != param.value:
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

        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            existing_params = {p.key: p.value for p in run.params}
            new_params = []
            non_matching_params = []
            for param in params:
                if param.key in existing_params:
                    if param.value != existing_params[param.key]:
                        non_matching_params.append(
                            {
                                "key": param.key,
                                "old_value": existing_params[param.key],
                                "new_value": param.value,
                            }
                        )
                    continue
                new_params.append(SqlParam(run_uuid=run_id, key=param.key, value=param.value))

            if non_matching_params:
                raise MlflowException(
                    "Changing param values is not allowed. Params were already"
                    f" logged='{non_matching_params}' for run ID='{run_id}'.",
                    INVALID_PARAMETER_VALUE,
                )

            if not new_params:
                return

            session.add_all(new_params)

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
            if tag.key == MLFLOW_RUN_NAME:
                run_status = RunStatus.from_string(run.status)
                self.update_run_info(run_id, run_status, run.end_time, tag.value)
            else:
                # NB: Updating the run_info will set the tag. No need to do it twice.
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
                        # NB: If the run name tag is explicitly set, update the run info attribute
                        # and do not resubmit the tag for overwrite as the tag will be set within
                        # `set_tag()` with a call to `update_run_info()`
                        if tag.key == MLFLOW_RUN_NAME:
                            self.set_tag(run_id, tag)
                        else:
                            current_tag = current_tags.get(tag.key)
                            new_tag = new_tag_dict.get(tag.key)

                            # update the SqlTag if it is already present in DB
                            if current_tag:
                                current_tag.value = tag.value
                                continue

                            # if a SqlTag instance is already present in `new_tag_dict`,
                            # this means that multiple tags with the same key were passed to
                            # `set_tags`.
                            # In this case, we resolve potential conflicts by updating the value
                            # of the existing instance to the value of `tag`
                            if new_tag:
                                new_tag.value = tag.value
                            # otherwise, put it into the dict
                            else:
                                new_tag = SqlTag(run_uuid=run_id, key=tag.key, value=tag.value)

                            new_tag_dict[tag.key] = new_tag

                    # finally, save new entries to DB.
                    session.add_all(new_tag_dict.values())
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
                    f"No tag with name: {key} in run with id {run_id}",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            elif len(filtered_tags) > 1:
                raise MlflowException(
                    "Bad data in database - tags for a specific run must have "
                    "a single unique value. "
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
                f"most {SEARCH_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
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
            (
                attribute_filters,
                non_attribute_filters,
                dataset_filters,
            ) = _get_sqlalchemy_filter_clauses(parsed_filters, session, self._get_dialect())
            for non_attr_filter in non_attribute_filters:
                stmt = stmt.join(non_attr_filter)
            for idx, dataset_filter in enumerate(dataset_filters):
                # need to reference the anon table in the join condition
                anon_table_name = f"anon_{idx+1}"
                stmt = stmt.join(
                    dataset_filter, text(f"runs.run_uuid = {anon_table_name}.destination_id")
                )
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
                    *attribute_filters,
                )
                .order_by(*parsed_orderby)
                .offset(offset)
                .limit(max_results)
            )
            queried_runs = session.execute(stmt).scalars(SqlRun).all()

            runs = [run.to_mlflow_entity() for run in queried_runs]
            run_ids = [run.info.run_id for run in runs]

            # add inputs to runs
            inputs = self._get_run_inputs(run_uuids=run_ids, session=session)
            runs_with_inputs = []
            for i, run in enumerate(runs):
                runs_with_inputs.append(
                    Run(run.info, run.data, RunInputs(dataset_inputs=inputs[i]))
                )

            next_page_token = compute_next_token(len(runs_with_inputs))

        return runs_with_inputs, next_page_token

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
                f"Argument 'mlflow_model' should be mlflow.models.Model, got '{type(mlflow_model)}'"
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

    def log_inputs(self, run_id: str, datasets: Optional[List[DatasetInput]] = None):
        """
        Log inputs, such as datasets, to the specified run.

        :param run_id: String id for the run
        :param datasets: List of :py:class:`mlflow.entities.DatasetInput` instances to log
                         as inputs to the run.

        :return: None.
        """
        _validate_run_id(run_id)
        if datasets is not None:
            if not isinstance(datasets, list):
                raise TypeError(f"Argument 'datasets' should be a list, got '{type(datasets)}'")
            _validate_dataset_inputs(datasets)

        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            experiment_id = run.experiment_id
            self._check_run_is_active(run)
            try:
                self._log_inputs_impl(experiment_id, run_id, datasets)
            except MlflowException as e:
                raise e
            except Exception as e:
                raise MlflowException(e, INTERNAL_ERROR)

    def _log_inputs_impl(
        self, experiment_id, run_id, dataset_inputs: Optional[List[DatasetInput]] = None
    ):
        if dataset_inputs is None or len(dataset_inputs) == 0:
            return
        for dataset_input in dataset_inputs:
            if dataset_input.dataset is None:
                raise MlflowException(
                    "Dataset input must have a dataset associated with it.", INTERNAL_ERROR
                )

        # dedup dataset_inputs list if two dataset inputs have the same name and digest
        # keeping the first occurrence
        name_digest_keys = {}
        for dataset_input in dataset_inputs:
            key = (dataset_input.dataset.name, dataset_input.dataset.digest)
            if key not in name_digest_keys:
                name_digest_keys[key] = dataset_input
        dataset_inputs = list(name_digest_keys.values())

        with self.ManagedSessionMaker() as session:
            dataset_names_to_check = [
                dataset_input.dataset.name for dataset_input in dataset_inputs
            ]
            dataset_digests_to_check = [
                dataset_input.dataset.digest for dataset_input in dataset_inputs
            ]
            # find all datasets with the same name and digest
            # if the dataset already exists, use the existing dataset uuid
            existing_datasets = (
                session.query(SqlDataset)
                .filter(SqlDataset.name.in_(dataset_names_to_check))
                .filter(SqlDataset.digest.in_(dataset_digests_to_check))
                .all()
            )
            dataset_uuids = {}
            for existing_dataset in existing_datasets:
                dataset_uuids[
                    (existing_dataset.name, existing_dataset.digest)
                ] = existing_dataset.dataset_uuid

            # collect all objects to write to DB in a single list
            objs_to_write = []

            # add datasets to objs_to_write
            for dataset_input in dataset_inputs:
                if (dataset_input.dataset.name, dataset_input.dataset.digest) not in dataset_uuids:
                    new_dataset_uuid = uuid.uuid4().hex
                    dataset_uuids[
                        (dataset_input.dataset.name, dataset_input.dataset.digest)
                    ] = new_dataset_uuid
                    objs_to_write.append(
                        SqlDataset(
                            dataset_uuid=new_dataset_uuid,
                            experiment_id=experiment_id,
                            name=dataset_input.dataset.name,
                            digest=dataset_input.dataset.digest,
                            dataset_source_type=dataset_input.dataset.source_type,
                            dataset_source=dataset_input.dataset.source,
                            dataset_schema=dataset_input.dataset.schema,
                            dataset_profile=dataset_input.dataset.profile,
                        )
                    )

            # find all inputs with the same source_id and destination_id
            # if the input already exists, use the existing input uuid
            existing_inputs = (
                session.query(SqlInput)
                .filter(SqlInput.source_type == "DATASET")
                .filter(SqlInput.source_id.in_(dataset_uuids.values()))
                .filter(SqlInput.destination_type == "RUN")
                .filter(SqlInput.destination_id == run_id)
                .all()
            )
            input_uuids = {}
            for existing_input in existing_inputs:
                input_uuids[
                    (existing_input.source_id, existing_input.destination_id)
                ] = existing_input.input_uuid

            # add input edges to objs_to_write
            for dataset_input in dataset_inputs:
                dataset_uuid = dataset_uuids[
                    (dataset_input.dataset.name, dataset_input.dataset.digest)
                ]
                if (dataset_uuid, run_id) not in input_uuids:
                    new_input_uuid = uuid.uuid4().hex
                    input_uuids[
                        (dataset_input.dataset.name, dataset_input.dataset.digest)
                    ] = new_input_uuid
                    objs_to_write.append(
                        SqlInput(
                            input_uuid=new_input_uuid,
                            source_type="DATASET",
                            source_id=dataset_uuid,
                            destination_type="RUN",
                            destination_id=run_id,
                        )
                    )
                    # add input tags to objs_to_write
                    for input_tag in dataset_input.tags:
                        objs_to_write.append(
                            SqlInputTag(
                                input_uuid=new_input_uuid,
                                name=input_tag.key,
                                value=input_tag.value,
                            )
                        )

            session.add_all(objs_to_write)


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
            clauses.append(
                SearchUtils.get_sql_comparison_func(comparator, dialect)(attribute, value)
            )
    return clauses


def _get_sqlalchemy_filter_clauses(parsed, session, dialect):
    """
    Creates run attribute filters and subqueries that will be inner-joined to SqlRun to act as
    multi-clause filters and return them as a tuple.
    """
    attribute_filters = []
    non_attribute_filters = []
    dataset_filters = []

    for sql_statement in parsed:
        key_type = sql_statement.get("type")
        key_name = sql_statement.get("key")
        value = sql_statement.get("value")
        comparator = sql_statement.get("comparator").upper()

        key_name = SearchUtils.translate_key_alias(key_name)

        if SearchUtils.is_string_attribute(
            key_type, key_name, comparator
        ) or SearchUtils.is_numeric_attribute(key_type, key_name, comparator):
            if key_name == "run_name":
                # Treat "attributes.run_name == <value>" as "tags.`mlflow.runName` == <value>".
                # The name column in the runs table is empty for runs logged in MLflow <= 1.29.0.
                key_filter = SearchUtils.get_sql_comparison_func("=", dialect)(
                    SqlTag.key, MLFLOW_RUN_NAME
                )
                val_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(
                    SqlTag.value, value
                )
                non_attribute_filters.append(
                    session.query(SqlTag).filter(key_filter, val_filter).subquery()
                )
            else:
                attribute = getattr(SqlRun, SqlRun.get_attribute_name(key_name))
                attr_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(
                    attribute, value
                )
                attribute_filters.append(attr_filter)
        else:
            if SearchUtils.is_metric(key_type, comparator):
                entity = SqlLatestMetric
                value = float(value)
            elif SearchUtils.is_param(key_type, comparator):
                entity = SqlParam
            elif SearchUtils.is_tag(key_type, comparator):
                entity = SqlTag
            elif SearchUtils.is_dataset(key_type, comparator):
                entity = SqlDataset
            else:
                raise MlflowException(
                    f"Invalid search expression type '{key_type}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            if entity == SqlDataset:
                if key_name == "context":
                    dataset_filters.append(
                        session.query(entity, SqlInput, SqlInputTag)
                        .join(SqlInput, SqlInput.source_id == SqlDataset.dataset_uuid)
                        .join(
                            SqlInputTag,
                            and_(
                                SqlInputTag.input_uuid == SqlInput.input_uuid,
                                SqlInputTag.name == MLFLOW_DATASET_CONTEXT,
                                SearchUtils.get_sql_comparison_func(comparator, dialect)(
                                    getattr(SqlInputTag, "value"), value
                                ),
                            ),
                        )
                        .subquery()
                    )
                else:
                    dataset_attr_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(
                        getattr(SqlDataset, key_name), value
                    )
                    dataset_filters.append(
                        session.query(entity, SqlInput)
                        .join(SqlInput, SqlInput.source_id == SqlDataset.dataset_uuid)
                        .filter(dataset_attr_filter)
                        .subquery()
                    )
            else:
                key_filter = SearchUtils.get_sql_comparison_func("=", dialect)(entity.key, key_name)
                val_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(
                    entity.value, value
                )
                non_attribute_filters.append(
                    session.query(entity).filter(key_filter, val_filter).subquery()
                )

    return attribute_filters, non_attribute_filters, dataset_filters


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
            key = SearchUtils.translate_key_alias(key)
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
                        f"Invalid identifier type '{key_type}'",
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
                    # Ideally the use of "IS" is preferred here but owing to sqlalchemy
                    # translation in MSSQL we are forced to use "=" instead.
                    # These 2 options are functionally identical / unchanged because
                    # the column (is_nan) is not nullable. However it could become an issue
                    # if this precondition changes in the future.
                    (subquery.c.is_nan == sqlalchemy.true(), 1),
                    (order_value.is_(None), 2),
                    else_=0,
                ).label(f"clause_{clause_id}")

            else:  # other entities do not have an 'is_nan' field
                case = sql.case((order_value.is_(None), 1), else_=0).label(f"clause_{clause_id}")
            clauses.append(case.name)
            select_clauses.append(case)
            select_clauses.append(order_value)

            if (key_type, key) in observed_order_by_clauses:
                raise MlflowException(f"`order_by` contains duplicate fields: {order_by_list}")
            observed_order_by_clauses.add((key_type, key))

            if ascending:
                clauses.append(order_value)
            else:
                clauses.append(order_value.desc())

    if (SearchUtils._ATTRIBUTE_IDENTIFIER, SqlRun.start_time.key) not in observed_order_by_clauses:
        clauses.append(SqlRun.start_time.desc())
    clauses.append(SqlRun.run_uuid)
    return select_clauses, clauses, ordering_joins


def _get_search_experiments_filter_clauses(parsed_filters, dialect):
    attribute_filters = []
    non_attribute_filters = []
    for f in parsed_filters:
        type_ = f["type"]
        key = f["key"]
        comparator = f["comparator"]
        value = f["value"]
        if type_ == "attribute":
            if SearchExperimentsUtils.is_string_attribute(
                type_, key, comparator
            ) and comparator not in ("=", "!=", "LIKE", "ILIKE"):
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparator for string attribute: {comparator}"
                )
            if SearchExperimentsUtils.is_numeric_attribute(
                type_, key, comparator
            ) and comparator not in ("=", "!=", "<", "<=", ">", ">="):
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparator for numeric attribute: {comparator}"
                )
            attr = getattr(SqlExperiment, key)
            attr_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(attr, value)
            attribute_filters.append(attr_filter)
        elif type_ == "tag":
            if comparator not in ("=", "!=", "LIKE", "ILIKE"):
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparator for tag: {comparator}"
                )
            val_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(
                SqlExperimentTag.value, value
            )
            key_filter = SearchUtils.get_sql_comparison_func("=", dialect)(
                SqlExperimentTag.key, key
            )
            non_attribute_filters.append(
                select(SqlExperimentTag).filter(key_filter, val_filter).subquery()
            )
        else:
            raise MlflowException.invalid_parameter_value(f"Invalid token type: {type_}")

    return attribute_filters, non_attribute_filters


def _get_search_experiments_order_by_clauses(order_by):
    order_by_clauses = []
    for type_, key, ascending in map(
        SearchExperimentsUtils.parse_order_by_for_search_experiments,
        order_by or ["creation_time DESC", "experiment_id ASC"],
    ):
        if type_ == "attribute":
            order_by_clauses.append((getattr(SqlExperiment, key), ascending))
        else:
            raise MlflowException.invalid_parameter_value(f"Invalid order_by entity: {type_}")

    # Add a tie-breaker
    if not any(col == SqlExperiment.experiment_id for col, _ in order_by_clauses):
        order_by_clauses.append((SqlExperiment.experiment_id, False))

    return [col.asc() if ascending else col.desc() for col, ascending in order_by_clauses]
