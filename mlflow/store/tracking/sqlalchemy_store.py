import json
import logging
import math
import random
import threading
import time
import uuid
from collections import defaultdict
from functools import reduce
from typing import Any, Optional, TypedDict

import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.sql.expression as sql
from sqlalchemy import and_, func, sql, text
from sqlalchemy.future import select

import mlflow.store.db.utils
from mlflow.entities import (
    DatasetInput,
    Experiment,
    Run,
    RunInputs,
    RunOutputs,
    RunStatus,
    RunTag,
    SourceType,
    TraceInfoV2,
    ViewType,
    _DatasetSummary,
)
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.logged_model import LoggedModel
from mlflow.entities.logged_model_input import LoggedModelInput
from mlflow.entities.logged_model_output import LoggedModelOutput
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.metric import Metric, MetricWithRunId
from mlflow.entities.trace_status import TraceStatus
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
from mlflow.store.tracking import (
    SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT,
    SEARCH_MAX_RESULTS_DEFAULT,
    SEARCH_MAX_RESULTS_THRESHOLD,
    SEARCH_TRACES_DEFAULT_MAX_RESULTS,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.dbmodels.models import (
    SqlDataset,
    SqlExperiment,
    SqlExperimentTag,
    SqlInput,
    SqlInputTag,
    SqlLatestMetric,
    SqlLoggedModel,
    SqlLoggedModelMetric,
    SqlLoggedModelParam,
    SqlLoggedModelTag,
    SqlMetric,
    SqlParam,
    SqlRun,
    SqlTag,
    SqlTraceInfo,
    SqlTraceRequestMetadata,
    SqlTraceTag,
)
from mlflow.tracing.utils import generate_request_id_v2
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.file_utils import local_file_uri_to_path, mkdir
from mlflow.utils.mlflow_tags import (
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_DATASET_CONTEXT,
    MLFLOW_LOGGED_MODELS,
    MLFLOW_RUN_NAME,
    _get_run_name_from_tags,
)
from mlflow.utils.name_utils import _generate_random_name
from mlflow.utils.search_utils import (
    SearchExperimentsUtils,
    SearchLoggedModelsPaginationToken,
    SearchTraceUtils,
    SearchUtils,
)
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
    _validate_experiment_artifact_location_length,
    _validate_experiment_name,
    _validate_experiment_tag,
    _validate_logged_model_name,
    _validate_metric,
    _validate_param,
    _validate_param_keys_unique,
    _validate_run_id,
    _validate_tag,
    _validate_trace_tag,
)

_logger = logging.getLogger(__name__)

# For each database table, fetch its columns and define an appropriate attribute for each column
# on the table's associated object representation (Mapper). This is necessary to ensure that
# columns defined via backreference are available as Mapper instance attributes (e.g.,
# ``SqlExperiment.tags`` and ``SqlRun.params``). For more information, see
# https://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.configure_mappers
# and https://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.mapper.Mapper
sqlalchemy.orm.configure_mappers()


class DatasetFilter(TypedDict, total=False):
    """
    Dataset filter used for search_logged_models.
    """

    dataset_name: str
    dataset_digest: str


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
    MODELS_FOLDER_NAME = "models"
    TRACE_FOLDER_NAME = "traces"
    DEFAULT_EXPERIMENT_ID = "0"
    _db_uri_sql_alchemy_engine_map = {}
    _db_uri_sql_alchemy_engine_map_lock = threading.Lock()

    def __init__(self, db_uri, default_artifact_root):
        """
        Create a database backed store.

        Args:
            db_uri: The SQLAlchemy database URI string to connect to the database. See
                the `SQLAlchemy docs
                <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_
                for format specifications. MLflow supports the dialects ``mysql``,
                ``mssql``, ``sqlite``, and ``postgresql``.
            default_artifact_root: Path/URI to location suitable for large data (such as a blob
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
                    SqlAlchemyStore._db_uri_sql_alchemy_engine_map[db_uri] = (
                        mlflow.store.db.utils.create_sqlalchemy_engine_with_retry(db_uri)
                    )
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
            _validate_experiment_artifact_location_length(artifact_location)
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

        self._validate_max_results_param(max_results)
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
                .filter(
                    *attribute_filters,
                    SqlExperiment.lifecycle_stage.in_(lifecycle_stags),
                )
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

    def _get_experiment(self, session, experiment_id, view_type, eager=False):  # noqa: D417
        """
        Args:
            eager: If ``True``, eagerly loads the experiments's tags. If ``False``, these tags
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
        A list of SQLAlchemy query options that can be used to eagerly load the following
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
                    SqlExperiment.name == experiment_name,
                    SqlExperiment.lifecycle_stage.in_(stages),
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
                experiment_id=experiment_id,
                session=session,
                view_type=ViewType.DELETED_ONLY,
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
                experiment.artifact_location,
                run_id,
                SqlAlchemyStore.ARTIFACTS_FOLDER_NAME,
            )
            tags = tags.copy() if tags else []
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

            run = run.to_mlflow_entity()
            inputs_list = self._get_run_inputs(session, [run_id])
            dataset_inputs = inputs_list[0] if inputs_list else []
            return Run(run.info, run.data, RunInputs(dataset_inputs=dataset_inputs))

    def _get_run(self, session, run_uuid, eager=False):  # noqa: D417
        """
        Args:
            eager: If ``True``, eagerly loads the run's summary metrics (``latest_metrics``),
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
        datasets_with_tags = (
            session.query(
                SqlInput.input_uuid,
                SqlInput.destination_id.label("run_uuid"),
                SqlDataset,
                SqlInputTag,
            )
            .select_from(SqlInput)
            .join(SqlDataset, SqlInput.source_id == SqlDataset.dataset_uuid)
            .outerjoin(SqlInputTag, SqlInputTag.input_uuid == SqlInput.input_uuid)
            .filter(SqlInput.destination_type == "RUN", SqlInput.destination_id.in_(run_uuids))
            .order_by("run_uuid")
        ).all()

        dataset_inputs_per_run = defaultdict(dict)
        for input_uuid, run_uuid, dataset_sql, tag_sql in datasets_with_tags:
            dataset_inputs = dataset_inputs_per_run[run_uuid]
            dataset_uuid = dataset_sql.dataset_uuid
            dataset_input = dataset_inputs.get(dataset_uuid)
            if dataset_input is None:
                dataset_entity = dataset_sql.to_mlflow_entity()
                dataset_input = DatasetInput(dataset=dataset_entity, tags=[])
                dataset_inputs[dataset_uuid] = dataset_input
            if tag_sql is not None:
                dataset_input.tags.append(tag_sql.to_mlflow_entity())
        return [list(dataset_inputs_per_run[run_uuid].values()) for run_uuid in run_uuids]

    @staticmethod
    def _get_eager_run_query_options():
        """
        A list of SQLAlchemy query options that can be used to eagerly load the following
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
            model_inputs = self._get_model_inputs(run_id, session)
            model_outputs = self._get_model_outputs(run_id, session)
            return Run(
                mlflow_run.info,
                mlflow_run.data,
                RunInputs(dataset_inputs=inputs, model_inputs=model_inputs),
                RunOutputs(model_outputs),
            )

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

    def log_metric(self, run_id, metric):
        # simply call _log_metrics and let it handle the rest
        self._log_metrics(run_id, [metric])
        self._log_model_metrics(run_id, [metric])

    def sanitize_metric_value(self, metric_value: float) -> tuple[bool, float]:
        """
        Returns a tuple of two values:
            - A boolean indicating whether the metric is NaN.
            - The metric value, which is set to 0 if the metric is NaN.
        """
        is_nan = math.isnan(metric_value)
        if is_nan:
            value = 0
        elif math.isinf(metric_value):
            #  NB: Sql can not represent Infs = > We replace +/- Inf with max/min 64b float
            # value
            value = 1.7976931348623157e308 if metric_value > 0 else -1.7976931348623157e308
        else:
            value = metric_value
        return is_nan, value

    def _log_metrics(self, run_id, metrics):
        # Duplicate metric values are eliminated here to maintain
        # the same behavior in log_metric
        metric_instances = []
        seen = set()
        is_single_metric = len(metrics) == 1
        for idx, metric in enumerate(metrics):
            _validate_metric(
                metric.key,
                metric.value,
                metric.timestamp,
                metric.step,
                path="" if is_single_metric else f"metrics[{idx}]",
            )
            if metric not in seen:
                is_nan, value = self.sanitize_metric_value(metric.value)
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

    def _log_model_metrics(
        self,
        run_id: str,
        metrics: list[Metric],
        dataset_uuid: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        if not metrics:
            return

        metric_instances: list[SqlLoggedModelMetric] = []
        is_single_metric = len(metrics) == 1
        seen: set[Metric] = set()
        for idx, metric in enumerate(metrics):
            if metric.model_id is None:
                continue

            if metric in seen:
                continue
            seen.add(metric)

            _validate_metric(
                metric.key,
                metric.value,
                metric.timestamp,
                metric.step,
                path="" if is_single_metric else f"metrics[{idx}]",
            )
            is_nan, value = self.sanitize_metric_value(metric.value)
            metric_instances.append(
                SqlLoggedModelMetric(
                    model_id=metric.model_id,
                    metric_name=metric.key,
                    metric_timestamp_ms=metric.timestamp,
                    metric_step=metric.step,
                    metric_value=value,
                    experiment_id=experiment_id or _get_experiment_id(),
                    run_id=run_id,
                    dataset_uuid=dataset_uuid,
                    dataset_name=metric.dataset_name,
                    dataset_digest=metric.dataset_digest,
                )
            )

        with self.ManagedSessionMaker() as session:
            try:
                session.add_all(metric_instances)
                session.commit()
            except sqlalchemy.exc.IntegrityError:
                # Primary key can be violated if it is tried to log a metric with same value,
                # timestamp, step, and key within the same run.
                session.rollback()
                metric_keys = [m.metric_name for m in metric_instances]
                metric_key_batches = (
                    metric_keys[i : i + 100] for i in range(0, len(metric_keys), 100)
                )
                for batch in metric_key_batches:
                    existing_metrics = (
                        session.query(SqlLoggedModelMetric)
                        .filter(
                            SqlLoggedModelMetric.run_id == run_id,
                            SqlLoggedModelMetric.metric_name.in_(batch),
                        )
                        .all()
                    )
                    existing_metrics = {m.to_mlflow_entity() for m in existing_metrics}
                    non_existing_metrics = [
                        m for m in metric_instances if m.to_mlflow_entity() not in existing_metrics
                    ]
                    session.add_all(non_existing_metrics)

    def _update_latest_metrics_if_necessary(self, logged_metrics, session):
        def _compare_metrics(metric_a, metric_b):
            """
            Returns:
                True if ``metric_a`` is strictly more recent than ``metric_b``, as determined
                by ``step``, ``timestamp``, and ``value``. False otherwise.
            """
            return (metric_a.step, metric_a.timestamp, metric_a.value) > (
                metric_b.step,
                metric_b.timestamp,
                metric_b.value,
            )

        def _overwrite_metric(new_metric, old_metric):
            """
            Writes content of new_metric over old_metric. The content are `value`, `step`,
            `timestamp`, and `is_nan`.

            Returns:
                old_metric with its content updated.
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

        Args:
            run_id: Unique identifier for run.
            metric_key: Metric name within the run.
            max_results: An indicator for paginated results.
            page_token: Token indicating the page of metric history to fetch.

        Returns:
            A :py:class:`mlflow.store.entities.paged_list.PagedList` of
            :py:class:`mlflow.entities.Metric` entities if ``metric_key`` values
            have been logged to the ``run_id``, else an empty list.

        """
        with self.ManagedSessionMaker() as session:
            query = session.query(SqlMetric).filter_by(run_uuid=run_id, key=metric_key)

            # Parse offset from page_token for pagination
            offset = SearchUtils.parse_start_offset_from_page_token(page_token)

            # Add ORDER BY clause to satisfy MSSQL requirement for OFFSET
            query = query.order_by(SqlMetric.timestamp, SqlMetric.step, SqlMetric.value)
            query = query.offset(offset)

            if max_results is not None:
                query = query.limit(max_results + 1)

            metrics = query.all()

            # Compute next token if more results are available
            next_token = None
            if max_results is not None and len(metrics) == max_results + 1:
                final_offset = offset + max_results
                next_token = SearchUtils.create_page_token(final_offset)
                metrics = metrics[:max_results]

            return PagedList([metric.to_mlflow_entity() for metric in metrics], next_token)

    def get_metric_history_bulk(self, run_ids, metric_key, max_results):
        """
        Return all logged values for a given metric.

        Args:
            run_ids: Unique identifiers of the runs from which to fetch the metric histories for
                the specified key.
            metric_key: Metric name within the runs.
            max_results: The maximum number of results to return.

        Returns:
            A List of SqlAlchemyStore.MetricWithRunId objects if metric_key values have been logged
            to one or more of the specified run_ids, else an empty list. Results are sorted by run
            ID in lexicographically ascending order, followed by timestamp, step, and value in
            numerically ascending order.
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
                MetricWithRunId(
                    run_id=metric.run_uuid,
                    metric=metric.to_mlflow_entity(),
                )
                for metric in metrics
            ]

    def get_max_step_for_metric(self, run_id, metric_key):
        with self.ManagedSessionMaker() as session:
            max_step = (
                session.query(func.max(SqlMetric.step))
                .filter(SqlMetric.run_uuid == run_id, SqlMetric.key == metric_key)
                .scalar()
            )
            return max_step or 0

    def get_metric_history_bulk_interval_from_steps(self, run_id, metric_key, steps, max_results):
        with self.ManagedSessionMaker() as session:
            metrics = (
                session.query(SqlMetric)
                .filter(
                    SqlMetric.key == metric_key,
                    SqlMetric.run_uuid == run_id,
                    SqlMetric.step.in_(steps),
                )
                .order_by(
                    SqlMetric.run_uuid,
                    SqlMetric.step,
                    SqlMetric.timestamp,
                    SqlMetric.value,
                )
                .limit(max_results)
                .all()
            )
            return [
                MetricWithRunId(
                    run_id=metric.run_uuid,
                    metric=metric.to_mlflow_entity(),
                )
                for metric in metrics
            ]

    def _search_datasets(self, experiment_ids):
        """
        Return all dataset summaries associated to the given experiments.

        Args:
            experiment_ids: List of experiment ids to scope the search

        Returns:
            A List of :py:class:`SqlAlchemyStore.DatasetSummary` entities.
        """

        MAX_DATASET_SUMMARIES_RESULTS = 1000
        with self.ManagedSessionMaker() as session:
            # Note that the join with the input tag table is a left join. This is required so if an
            # input does not have the MLFLOW_DATASET_CONTEXT tag, we still return that entry as part
            # of the final result with the context set to None.
            summaries = (
                session.query(
                    SqlDataset.experiment_id,
                    SqlDataset.name,
                    SqlDataset.digest,
                    SqlInputTag.value,
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
        param = _validate_param(param.key, param.value)
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

        Args:
            experiment_id: String ID of the experiment
            tag: ExperimentRunTag instance to log
        """
        _validate_experiment_tag(tag.key, tag.value)
        with self.ManagedSessionMaker() as session:
            tag = _validate_tag(tag.key, tag.value)
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

        Args:
            run_id: String ID of the run.
            tag: RunTag instance to log.
        """
        with self.ManagedSessionMaker() as session:
            tag = _validate_tag(tag.key, tag.value)
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

        Args:
            run_id: String ID of the run
            tags: List of RunTag instances to log
            path: current json path for error messages
        """
        if not tags:
            return

        tags = [_validate_tag(t.key, t.value, path=f"tags[{idx}]") for (idx, t) in enumerate(tags)]

        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)

            def _try_insert_tags(attempt_number, max_retries):
                try:
                    current_tags = (
                        session.query(SqlTag)
                        .filter(
                            SqlTag.run_uuid == run_id,
                            SqlTag.key.in_([t.key for t in tags]),
                        )
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

        Args:
            run_id: String ID of the run
            key: Name of the tag
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
        self,
        experiment_ids,
        filter_string,
        run_view_type,
        max_results,
        order_by,
        page_token,
    ):
        def compute_next_token(current_size):
            next_token = None
            if max_results == current_size:
                final_offset = offset + max_results
                next_token = SearchUtils.create_page_token(final_offset)

            return next_token

        self._validate_max_results_param(max_results, allow_null=True)

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
                anon_table_name = f"anon_{idx + 1}"
                stmt = stmt.join(
                    dataset_filter,
                    text(f"runs.run_uuid = {anon_table_name}.destination_id"),
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
        metrics, params, tags = _validate_batch_log_data(metrics, params, tags)
        _validate_batch_log_limits(metrics, params, tags)
        _validate_param_keys_unique(params)

        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            try:
                self._log_params(run_id, params)
                self._log_metrics(run_id, metrics)
                self._log_model_metrics(run_id, metrics)
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
        model_dict = mlflow_model.get_tags_dict()
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

    def log_inputs(
        self,
        run_id: str,
        datasets: Optional[list[DatasetInput]] = None,
        models: Optional[list[LoggedModelInput]] = None,
    ):
        """
        Log inputs, such as datasets, to the specified run.

        Args:
            run_id: String id for the run
            datasets: List of :py:class:`mlflow.entities.DatasetInput` instances to log
                as inputs to the run.
            models: List of :py:class:`mlflow.entities.LoggedModelInput` instances to log
                as inputs to the run.

        Returns:
            None.
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
                self._log_inputs_impl(experiment_id, run_id, datasets, models)
            except MlflowException as e:
                raise e
            except Exception as e:
                raise MlflowException(e, INTERNAL_ERROR)

    def _log_inputs_impl(
        self,
        experiment_id,
        run_id,
        dataset_inputs: Optional[list[DatasetInput]] = None,
        models: Optional[list[LoggedModelInput]] = None,
    ):
        dataset_inputs = dataset_inputs or []
        for dataset_input in dataset_inputs:
            if dataset_input.dataset is None:
                raise MlflowException(
                    "Dataset input must have a dataset associated with it.",
                    INTERNAL_ERROR,
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
                dataset_uuids[(existing_dataset.name, existing_dataset.digest)] = (
                    existing_dataset.dataset_uuid
                )

            # collect all objects to write to DB in a single list
            objs_to_write = []

            # add datasets to objs_to_write
            for dataset_input in dataset_inputs:
                if (
                    dataset_input.dataset.name,
                    dataset_input.dataset.digest,
                ) not in dataset_uuids:
                    new_dataset_uuid = uuid.uuid4().hex
                    dataset_uuids[(dataset_input.dataset.name, dataset_input.dataset.digest)] = (
                        new_dataset_uuid
                    )
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
                input_uuids[(existing_input.source_id, existing_input.destination_id)] = (
                    existing_input.input_uuid
                )

            # add input edges to objs_to_write
            for dataset_input in dataset_inputs:
                dataset_uuid = dataset_uuids[
                    (dataset_input.dataset.name, dataset_input.dataset.digest)
                ]
                if (dataset_uuid, run_id) not in input_uuids:
                    new_input_uuid = uuid.uuid4().hex
                    input_uuids[(dataset_input.dataset.name, dataset_input.dataset.digest)] = (
                        new_input_uuid
                    )
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

            if models:
                for model in models:
                    session.merge(
                        SqlInput(
                            input_uuid=uuid.uuid4().hex,
                            source_type="RUN_INPUT",
                            source_id=run_id,
                            destination_type="MODEL_INPUT",
                            destination_id=model.model_id,
                        )
                    )

            session.add_all(objs_to_write)

    def log_outputs(self, run_id: str, models: list[LoggedModelOutput]):
        with self.ManagedSessionMaker() as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            session.add_all(
                SqlInput(
                    input_uuid=uuid.uuid4().hex,
                    source_type="RUN_OUTPUT",
                    source_id=run_id,
                    destination_type="MODEL_OUTPUT",
                    destination_id=model.model_id,
                    step=model.step,
                )
                for model in models
            )

    def _get_model_inputs(
        self,
        run_id: str,
        session: Optional[sqlalchemy.orm.Session] = None,
    ) -> list[LoggedModelInput]:
        return [
            LoggedModelInput(model_id=input.destination_id)
            for input in (
                session.query(SqlInput)
                .filter(
                    SqlInput.source_type == "RUN_INPUT",
                    SqlInput.source_id == run_id,
                    SqlInput.destination_type == "MODEL_INPUT",
                )
                .all()
            )
        ]

    def _get_model_outputs(
        self,
        run_id: str,
        session: sqlalchemy.orm.Session,
    ) -> list[LoggedModelOutput]:
        return [
            LoggedModelOutput(model_id=output.destination_id, step=output.step)
            for output in session.query(SqlInput)
            .filter(
                SqlInput.source_type == "RUN_OUTPUT",
                SqlInput.source_id == run_id,
                SqlInput.destination_type == "MODEL_OUTPUT",
            )
            .all()
        ]

    #######################################################################################
    # Logged models
    #######################################################################################
    def create_logged_model(
        self,
        experiment_id: str,
        name: Optional[str] = None,
        source_run_id: Optional[str] = None,
        tags: Optional[list[LoggedModelTag]] = None,
        params: Optional[list[LoggedModelParameter]] = None,
        model_type: Optional[str] = None,
    ) -> LoggedModel:
        _validate_logged_model_name(name)
        with self.ManagedSessionMaker() as session:
            experiment = self.get_experiment(experiment_id)
            self._check_experiment_is_active(experiment)
            model_id = f"m-{str(uuid.uuid4()).replace('-', '')}"
            artifact_location = append_to_uri_path(
                experiment.artifact_location,
                SqlAlchemyStore.MODELS_FOLDER_NAME,
                model_id,
                SqlAlchemyStore.ARTIFACTS_FOLDER_NAME,
            )
            name = name or _generate_random_name()
            creation_timestamp = get_current_time_millis()
            logged_model = SqlLoggedModel(
                model_id=model_id,
                experiment_id=experiment_id,
                name=name,
                artifact_location=artifact_location,
                creation_timestamp_ms=creation_timestamp,
                last_updated_timestamp_ms=creation_timestamp,
                model_type=model_type,
                status=LoggedModelStatus.PENDING.to_int(),
                lifecycle_stage=LifecycleStage.ACTIVE,
                source_run_id=source_run_id,
            )
            session.add(logged_model)

            if params:
                session.add_all(
                    SqlLoggedModelParam(
                        model_id=logged_model.model_id,
                        experiment_id=experiment_id,
                        param_key=param.key,
                        param_value=param.value,
                    )
                    for param in params
                )

            if tags:
                session.add_all(
                    SqlLoggedModelTag(
                        model_id=logged_model.model_id,
                        experiment_id=experiment_id,
                        tag_key=tag.key,
                        tag_value=tag.value,
                    )
                    for tag in tags
                )

            session.commit()
            return logged_model.to_mlflow_entity()

    def log_logged_model_params(self, model_id: str, params: list[LoggedModelParameter]):
        with self.ManagedSessionMaker() as session:
            logged_model = session.query(SqlLoggedModel).get(model_id)
            if not logged_model:
                self._raise_model_not_found(model_id)

            session.add_all(
                SqlLoggedModelParam(
                    model_id=model_id,
                    experiment_id=logged_model.experiment_id,
                    param_key=param.key,
                    param_value=param.value,
                )
                for param in params
            )

    def _raise_model_not_found(self, model_id: str):
        raise MlflowException(
            f"Logged model with ID '{model_id}' not found.",
            RESOURCE_DOES_NOT_EXIST,
        )

    def get_logged_model(self, model_id: str) -> LoggedModel:
        with self.ManagedSessionMaker() as session:
            logged_model = (
                session.query(SqlLoggedModel)
                .filter(
                    SqlLoggedModel.model_id == model_id,
                    SqlLoggedModel.lifecycle_stage != LifecycleStage.DELETED,
                )
                .first()
            )
            if not logged_model:
                self._raise_model_not_found(model_id)

            return logged_model.to_mlflow_entity()

    def delete_logged_model(self, model_id):
        with self.ManagedSessionMaker() as session:
            logged_model = session.query(SqlLoggedModel).get(model_id)
            if not logged_model:
                self._raise_model_not_found(model_id)

            logged_model.lifecycle_stage = LifecycleStage.DELETED
            logged_model.last_updated_timestamp_ms = get_current_time_millis()
            session.commit()

    def finalize_logged_model(self, model_id: str, status: LoggedModelStatus) -> LoggedModel:
        with self.ManagedSessionMaker() as session:
            logged_model = session.query(SqlLoggedModel).get(model_id)
            if not logged_model:
                self._raise_model_not_found(model_id)

            logged_model.status = status.to_int()
            logged_model.last_updated_timestamp_ms = get_current_time_millis()
            session.commit()
            return logged_model.to_mlflow_entity()

    def set_logged_model_tags(self, model_id: str, tags: list[LoggedModelTag]) -> None:
        with self.ManagedSessionMaker() as session:
            logged_model = session.query(SqlLoggedModel).get(model_id)
            if not logged_model:
                self._raise_model_not_found(model_id)

            # TODO: Consider upserting tags in a single transaction for performance
            for tag in tags:
                session.merge(
                    SqlLoggedModelTag(
                        model_id=model_id,
                        experiment_id=logged_model.experiment_id,
                        tag_key=tag.key,
                        tag_value=tag.value,
                    )
                )

    def delete_logged_model_tag(self, model_id: str, key: str) -> None:
        with self.ManagedSessionMaker() as session:
            logged_model = session.query(SqlLoggedModel).get(model_id)
            if not logged_model:
                self._raise_model_not_found(model_id)

            count = (
                session.query(SqlLoggedModelTag)
                .filter(
                    SqlLoggedModelTag.model_id == model_id,
                    SqlLoggedModelTag.tag_key == key,
                )
                .delete()
            )
            if count == 0:
                raise MlflowException(
                    f"No tag with key {key!r} found for model with ID {model_id!r}.",
                    RESOURCE_DOES_NOT_EXIST,
                )

    def _apply_order_by_search_logged_models(
        self,
        models: sqlalchemy.orm.Query,
        session: sqlalchemy.orm.Session,
        order_by: Optional[list[dict[str, Any]]] = None,
    ) -> sqlalchemy.orm.Query:
        order_by_clauses = []
        has_creation_timestamp = False
        for ob in order_by or []:
            field_name = ob.get("field_name")
            ascending = ob.get("ascending", True)
            if "." not in field_name:
                name = SqlLoggedModel.ALIASES.get(field_name, field_name)
                if name == "creation_timestamp_ms":
                    has_creation_timestamp = True
                try:
                    col = getattr(SqlLoggedModel, name)
                except AttributeError:
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid order by field name: {field_name}"
                    )
                # Why not use `nulls_last`? Because it's not supported by all dialects (e.g., MySQL)
                order_by_clauses.extend(
                    [
                        # Sort nulls last
                        sqlalchemy.case((col.is_(None), 1), else_=0).asc(),
                        col.asc() if ascending else col.desc(),
                    ]
                )
                continue

            entity, name = field_name.split(".", 1)
            # TODO: Support filtering by other entities such as params if needed
            if entity != "metrics":
                raise MlflowException.invalid_parameter_value(
                    f"Invalid order by field name: {field_name}. Only metrics are supported."
                )

            # Sub query to get the latest metrics value for each (model_id, metric_name) pair
            dataset_filter = []
            if dataset_name := ob.get("dataset_name"):
                dataset_filter.append(SqlLoggedModelMetric.dataset_name == dataset_name)
            if dataset_digest := ob.get("dataset_digest"):
                dataset_filter.append(SqlLoggedModelMetric.dataset_digest == dataset_digest)

            subquery = (
                session.query(
                    SqlLoggedModelMetric.model_id,
                    SqlLoggedModelMetric.metric_value,
                    func.rank()
                    .over(
                        partition_by=[
                            SqlLoggedModelMetric.model_id,
                            SqlLoggedModelMetric.metric_name,
                        ],
                        order_by=[
                            SqlLoggedModelMetric.metric_timestamp_ms.desc(),
                            SqlLoggedModelMetric.metric_step.desc(),
                        ],
                    )
                    .label("rank"),
                )
                .filter(
                    SqlLoggedModelMetric.metric_name == name,
                    *dataset_filter,
                )
                .subquery()
            )
            subquery = select(subquery.c).where(subquery.c.rank == 1).subquery()

            models = models.outerjoin(subquery)
            # Why not use `nulls_last`? Because it's not supported by all dialects (e.g., MySQL)
            order_by_clauses.extend(
                [
                    # Sort nulls last
                    sqlalchemy.case((subquery.c.metric_value.is_(None), 1), else_=0).asc(),
                    subquery.c.metric_value.asc() if ascending else subquery.c.metric_value.desc(),
                ]
            )

        if not has_creation_timestamp:
            order_by_clauses.append(SqlLoggedModel.creation_timestamp_ms.desc())

        return models.order_by(*order_by_clauses)

    def _apply_filter_string_datasets_search_logged_models(
        self,
        models: sqlalchemy.orm.Query,
        session: sqlalchemy.orm.Session,
        experiment_ids: list[str],
        filter_string: Optional[str],
        datasets: Optional[list[dict[str, Any]]],
    ):
        from mlflow.utils.search_logged_model_utils import EntityType, parse_filter_string

        comparisons = parse_filter_string(filter_string)
        dialect = self._get_dialect()
        attr_filters: list[sqlalchemy.BinaryExpression] = []
        non_attr_filters: list[sqlalchemy.BinaryExpression] = []

        dataset_filters = []
        if datasets:
            for dataset in datasets:
                dataset_filter = SqlLoggedModelMetric.dataset_name == dataset["dataset_name"]
                if "dataset_digest" in dataset:
                    dataset_filter = dataset_filter & (
                        SqlLoggedModelMetric.dataset_digest == dataset["dataset_digest"]
                    )
                dataset_filters.append(dataset_filter)

        has_metric_filters = False
        for comp in comparisons:
            comp_func = SearchUtils.get_sql_comparison_func(comp.op, dialect)
            if comp.entity.type == EntityType.ATTRIBUTE:
                attr_filters.append(comp_func(getattr(SqlLoggedModel, comp.entity.key), comp.value))
            elif comp.entity.type == EntityType.METRIC:
                has_metric_filters = True
                metric_filters = [
                    SqlLoggedModelMetric.metric_name == comp.entity.key,
                    comp_func(SqlLoggedModelMetric.metric_value, comp.value),
                ]
                if dataset_filters:
                    metric_filters.append(sqlalchemy.or_(*dataset_filters))
                non_attr_filters.append(
                    session.query(SqlLoggedModelMetric).filter(*metric_filters).subquery()
                )
            elif comp.entity.type == EntityType.PARAM:
                non_attr_filters.append(
                    session.query(SqlLoggedModelParam)
                    .filter(
                        SqlLoggedModelParam.param_key == comp.entity.key,
                        comp_func(SqlLoggedModelParam.param_value, comp.value),
                    )
                    .subquery()
                )
            elif comp.entity.type == EntityType.TAG:
                non_attr_filters.append(
                    session.query(SqlLoggedModelTag)
                    .filter(
                        SqlLoggedModelTag.tag_key == comp.entity.key,
                        comp_func(SqlLoggedModelTag.tag_value, comp.value),
                    )
                    .subquery()
                )

        for f in non_attr_filters:
            models = models.join(f)

        # If there are dataset filters but no metric filters,
        # filter for models that have any metrics on the datasets
        if dataset_filters and not has_metric_filters:
            subquery = (
                session.query(SqlLoggedModelMetric.model_id)
                .filter(sqlalchemy.or_(*dataset_filters))
                .distinct()
                .subquery()
            )
            models = models.join(subquery)

        return models.filter(
            SqlLoggedModel.lifecycle_stage != LifecycleStage.DELETED,
            SqlLoggedModel.experiment_id.in_(experiment_ids),
            *attr_filters,
        )

    def search_logged_models(
        self,
        experiment_ids: list[str],
        filter_string: Optional[str] = None,
        datasets: Optional[list[DatasetFilter]] = None,
        max_results: Optional[int] = None,
        order_by: Optional[list[dict[str, Any]]] = None,
        page_token: Optional[str] = None,
    ) -> PagedList[LoggedModel]:
        if datasets and not all(d.get("dataset_name") for d in datasets):
            raise MlflowException(
                "`dataset_name` in the `datasets` clause must be specified.",
                INVALID_PARAMETER_VALUE,
            )
        if page_token:
            token = SearchLoggedModelsPaginationToken.decode(page_token)
            token.validate(experiment_ids, filter_string, order_by)
            offset = token.offset
        else:
            offset = 0

        max_results = max_results or SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT
        with self.ManagedSessionMaker() as session:
            models = session.query(SqlLoggedModel)
            models = self._apply_filter_string_datasets_search_logged_models(
                models, session, experiment_ids, filter_string, datasets
            )
            models = self._apply_order_by_search_logged_models(models, session, order_by)
            models = models.offset(offset).limit(max_results + 1).all()

            if len(models) > max_results:
                token = SearchLoggedModelsPaginationToken(
                    offset=offset + max_results,
                    experiment_ids=experiment_ids,
                    filter_string=filter_string,
                    order_by=order_by,
                ).encode()
            else:
                token = None

            return PagedList([lm.to_mlflow_entity() for lm in models[:max_results]], token=token)

    #######################################################################################
    # Below are Tracing APIs. We may refactor them to be in a separate class in the future.
    #######################################################################################
    def start_trace(
        self,
        experiment_id: str,
        timestamp_ms: int,
        request_metadata: dict[str, str],
        tags: dict[str, str],
    ) -> TraceInfoV2:
        """
        Create an initial TraceInfo object in the database.

        Args:
            experiment_id: String id of the experiment for this run.
            timestamp_ms: Start time of the trace, in milliseconds since the UNIX epoch.
            request_metadata: Metadata of the trace.
            tags: Tags of the trace.

        Returns:
            The created TraceInfo object.
        """
        with self.ManagedSessionMaker() as session:
            experiment = self.get_experiment(experiment_id)
            self._check_experiment_is_active(experiment)

            request_id = generate_request_id_v2()
            trace_info = SqlTraceInfo(
                request_id=request_id,
                experiment_id=experiment_id,
                timestamp_ms=timestamp_ms,
                execution_time_ms=None,
                status=TraceStatus.IN_PROGRESS,
            )

            trace_info.tags = [SqlTraceTag(key=k, value=v) for k, v in tags.items()]
            trace_info.tags.append(self._get_trace_artifact_location_tag(experiment, request_id))

            trace_info.request_metadata = [
                SqlTraceRequestMetadata(key=k, value=v) for k, v in request_metadata.items()
            ]
            session.add(trace_info)

            return trace_info.to_mlflow_entity()

    def _get_trace_artifact_location_tag(self, experiment, request_id: str) -> SqlTraceTag:
        # Trace data is stored as file artifacts regardless of the tracking backend choice.
        # We use subdirectory "/traces" under the experiment's artifact location to isolate
        # them from run artifacts.
        artifact_uri = append_to_uri_path(
            experiment.artifact_location,
            SqlAlchemyStore.TRACE_FOLDER_NAME,
            request_id,
            SqlAlchemyStore.ARTIFACTS_FOLDER_NAME,
        )
        return SqlTraceTag(key=MLFLOW_ARTIFACT_LOCATION, value=artifact_uri)

    def end_trace(
        self,
        request_id: str,
        timestamp_ms: int,
        status: TraceStatus,
        request_metadata: dict[str, str],
        tags: dict[str, str],
    ) -> TraceInfoV2:
        """
        Update the TraceInfo object in the database with the completed trace info.

        Args:
            request_id: Unique string identifier of the trace.
            timestamp_ms: End time of the trace, in milliseconds. The execution time field
                in the TraceInfo will be calculated by subtracting the start time from this.
            status: Status of the trace.
            request_metadata: Metadata of the trace. This will be merged with the existing
                metadata logged during the start_trace call.
            tags: Tags of the trace. This will be merged with the existing tags logged
                during the start_trace or set_trace_tag calls.

        Returns:
            The updated TraceInfo object.
        """
        with self.ManagedSessionMaker() as session:
            sql_trace_info = self._get_sql_trace_info(session, request_id)
            trace_start_time_ms = sql_trace_info.timestamp_ms
            execution_time_ms = timestamp_ms - trace_start_time_ms
            sql_trace_info.execution_time_ms = execution_time_ms
            sql_trace_info.status = status
            session.merge(sql_trace_info)
            for k, v in request_metadata.items():
                session.merge(SqlTraceRequestMetadata(request_id=request_id, key=k, value=v))
            for k, v in tags.items():
                session.merge(SqlTraceTag(request_id=request_id, key=k, value=v))
            return sql_trace_info.to_mlflow_entity()

    def get_trace_info(self, request_id, should_query_v3: bool = False) -> TraceInfoV2:
        """
        Fetch the trace info for the given request id.

        Args:
            request_id: Unique string identifier of the trace.
            should_query_v3: If True, the backend store will query the V3 API for the trace info.
                TODO: Remove this flag once the V3 API is the default in OSS.

        Returns:
            The TraceInfo object.
        """
        if should_query_v3:
            raise MlflowException.invalid_parameter_value(
                "GetTraceInfoV3 API is not supported in the FileStore backend.",
            )

        with self.ManagedSessionMaker() as session:
            sql_trace_info = self._get_sql_trace_info(session, request_id)
            return sql_trace_info.to_mlflow_entity()

    def _get_sql_trace_info(self, session, request_id) -> SqlTraceInfo:
        sql_trace_info = (
            session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == request_id).one_or_none()
        )
        if sql_trace_info is None:
            raise MlflowException(
                f"Trace with request_id '{request_id}' not found.",
                RESOURCE_DOES_NOT_EXIST,
            )
        return sql_trace_info

    def search_traces(
        self,
        experiment_ids: list[str],
        filter_string: Optional[str] = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
        model_id: Optional[str] = None,
        sql_warehouse_id: Optional[str] = None,
    ) -> tuple[list[TraceInfoV2], Optional[str]]:
        """
        Return traces that match the given list of search expressions within the experiments.

        Args:
            experiment_ids: List of experiment ids to scope the search.
            filter_string: A search filter string.
            max_results: Maximum number of traces desired.
            order_by: List of order_by clauses.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_traces`` call.
            model_id: If specified, search traces associated with the given model ID.
            sql_warehouse_id: Only used in Databricks. The ID of the SQL warehouse to use for
                searching traces in inference tables.

        Returns:
            A tuple of a list of :py:class:`TraceInfo <mlflow.entities.TraceInfo>` objects that
            satisfy the search expressions and a pagination token for the next page of results.
        """
        self._validate_max_results_param(max_results)

        with self.ManagedSessionMaker() as session:
            cases_orderby, parsed_orderby, sorting_joins = _get_orderby_clauses_for_search_traces(
                order_by or [], session
            )
            stmt = select(SqlTraceInfo, *cases_orderby)

            attribute_filters, non_attribute_filters = _get_filter_clauses_for_search_traces(
                filter_string, session, self._get_dialect()
            )
            for non_attr_filter in non_attribute_filters:
                stmt = stmt.join(non_attr_filter)

            # using an outer join is necessary here because we want to be able to sort
            # on a column (tag, metric or param) without removing the lines that
            # do not have a value for this column (which is what inner join would do)
            for j in sorting_joins:
                stmt = stmt.outerjoin(j)

            offset = SearchTraceUtils.parse_start_offset_from_page_token(page_token)
            stmt = (
                # NB: We don't need to distinct the results of joins because of the fact that
                #   the right tables of the joins are unique on the join key, request_id.
                #   This is because the subquery that is joined on the right side is conditioned
                #   by a key and value pair of tags/metadata, and the combination of key and
                #   request_id is unique in those tables.
                #   Be careful when changing the query building logic, as it may break this
                #   uniqueness property and require deduplication, which can be expensive.
                stmt.filter(
                    SqlTraceInfo.experiment_id.in_(experiment_ids),
                    *attribute_filters,
                )
                .order_by(*parsed_orderby)
                .offset(offset)
                .limit(max_results)
            )
            queried_traces = session.execute(stmt).scalars(SqlTraceInfo).all()
            trace_infos = [t.to_mlflow_entity() for t in queried_traces]

            # Compute next search token
            if max_results == len(trace_infos):
                final_offset = offset + max_results
                next_token = SearchTraceUtils.create_page_token(final_offset)
            else:
                next_token = None

            return trace_infos, next_token

    def _validate_max_results_param(self, max_results: int, allow_null=False):
        if (not allow_null and max_results is None) or max_results < 1:
            raise MlflowException(
                f"Invalid value {max_results} for parameter 'max_results' supplied. It must be "
                f"a positive integer",
                INVALID_PARAMETER_VALUE,
            )

        if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                f"Invalid value {max_results} for parameter 'max_results' supplied. It must be at "
                f"most {SEARCH_MAX_RESULTS_THRESHOLD}",
                INVALID_PARAMETER_VALUE,
            )

    def set_trace_tag(self, request_id: str, key: str, value: str):
        """
        Set a tag on the trace with the given request_id.

        Args:
            request_id: The ID of the trace.
            key: The string key of the tag.
            value: The string value of the tag.
        """
        with self.ManagedSessionMaker() as session:
            key, value = _validate_trace_tag(key, value)
            session.merge(SqlTraceTag(request_id=request_id, key=key, value=value))

    def delete_trace_tag(self, request_id: str, key: str):
        """
        Delete a tag on the trace with the given request_id.

        Args:
            request_id: The ID of the trace.
            key: The string key of the tag.
        """
        with self.ManagedSessionMaker() as session:
            tags = session.query(SqlTraceTag).filter_by(request_id=request_id, key=key)
            if tags.count() == 0:
                raise MlflowException(
                    f"No trace tag with key '{key}' for trace with request_id '{request_id}'",
                    RESOURCE_DOES_NOT_EXIST,
                )
            tags.delete()

    def _delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: Optional[int] = None,
        max_traces: Optional[int] = None,
        request_ids: Optional[list[str]] = None,
    ) -> int:
        """
        Delete traces based on the specified criteria.

        Args:
            experiment_id: ID of the associated experiment.
            max_timestamp_millis: The maximum timestamp in milliseconds since the UNIX epoch for
                deleting traces. Traces older than or equal to this timestamp will be deleted.
            max_traces: The maximum number of traces to delete.
            request_ids: A set of request IDs to delete.

        Returns:
            The number of traces deleted.
        """
        with self.ManagedSessionMaker() as session:
            filters = [SqlTraceInfo.experiment_id == experiment_id]
            if max_timestamp_millis:
                filters.append(SqlTraceInfo.timestamp_ms <= max_timestamp_millis)
            if request_ids:
                filters.append(SqlTraceInfo.request_id.in_(request_ids))
            if max_traces:
                filters.append(
                    SqlTraceInfo.request_id.in_(
                        session.query(SqlTraceInfo.request_id)
                        .filter(*filters)
                        # Delete the oldest traces first
                        .order_by(SqlTraceInfo.timestamp_ms)
                        .limit(max_traces)
                        .subquery()
                    )
                )

            return (
                session.query(SqlTraceInfo)
                .filter(and_(*filters))
                .delete(synchronize_session="fetch")
            )


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

            # MySQL does not support NULLS LAST expression, so we sort first by
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

    if (
        SearchUtils._ATTRIBUTE_IDENTIFIER,
        SqlRun.start_time.key,
    ) not in observed_order_by_clauses:
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


def _get_orderby_clauses_for_search_traces(order_by_list: list[str], session):
    """Sorts a set of traces based on their natural ordering and an overriding set of order_bys.
    Traces are ordered first by timestamp_ms descending, then by request_id for tie-breaking.
    """
    clauses = []
    ordering_joins = []
    observed_order_by_clauses = set()
    select_clauses = []

    for clause_id, order_by_clause in enumerate(order_by_list):
        (key_type, key, ascending) = SearchTraceUtils.parse_order_by_for_search_traces(
            order_by_clause
        )

        if SearchTraceUtils.is_attribute(key_type, key, "="):
            order_value = getattr(SqlTraceInfo, key)
        else:
            if SearchTraceUtils.is_tag(key_type, "="):
                entity = SqlTraceTag
            elif SearchTraceUtils.is_request_metadata(key_type, "="):
                entity = SqlTraceRequestMetadata
            else:
                raise MlflowException(
                    f"Invalid identifier type '{key_type}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            # Tags and request metadata requires a join to the main table (trace_info)
            subquery = session.query(entity).filter(entity.key == key).subquery()
            ordering_joins.append(subquery)
            order_value = subquery.c.value

        case = sql.case((order_value.is_(None), 1), else_=0).label(f"clause_{clause_id}")
        clauses.append(case.name)
        select_clauses.append(case)
        select_clauses.append(order_value)

        if (key_type, key) in observed_order_by_clauses:
            raise MlflowException(f"`order_by` contains duplicate fields: {order_by_list}")
        observed_order_by_clauses.add((key_type, key))
        clauses.append(order_value if ascending else order_value.desc())

    # Add descending trace start time as default ordering and a tie-breaker
    for attr, ascending in [
        (SqlTraceInfo.timestamp_ms, False),
        (SqlTraceInfo.request_id, True),
    ]:
        if (
            SearchTraceUtils._ATTRIBUTE_IDENTIFIER,
            attr.key,
        ) not in observed_order_by_clauses:
            clauses.append(attr if ascending else attr.desc())
    return select_clauses, clauses, ordering_joins


def _get_filter_clauses_for_search_traces(filter_string, session, dialect):
    """
    Creates trace attribute filters and subqueries that will be inner-joined
    to SqlTraceInfo to act as multi-clause filters and return them as a tuple.
    """
    attribute_filters = []
    non_attribute_filters = []

    parsed_filters = SearchTraceUtils.parse_search_filter_for_search_traces(filter_string)
    for sql_statement in parsed_filters:
        key_type = sql_statement.get("type")
        key_name = sql_statement.get("key")
        value = sql_statement.get("value")
        comparator = sql_statement.get("comparator").upper()

        if SearchTraceUtils.is_attribute(key_type, key_name, comparator):
            attribute = getattr(SqlTraceInfo, key_name)
            attr_filter = SearchTraceUtils.get_sql_comparison_func(comparator, dialect)(
                attribute, value
            )
            attribute_filters.append(attr_filter)
        else:
            if SearchTraceUtils.is_tag(key_type, comparator):
                entity = SqlTraceTag
            elif SearchTraceUtils.is_request_metadata(key_type, comparator):
                entity = SqlTraceRequestMetadata
            else:
                raise MlflowException(
                    f"Invalid search expression type '{key_type}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            key_filter = SearchTraceUtils.get_sql_comparison_func("=", dialect)(
                entity.key, key_name
            )
            val_filter = SearchTraceUtils.get_sql_comparison_func(comparator, dialect)(
                entity.value, value
            )
            non_attribute_filters.append(
                session.query(entity).filter(key_filter, val_filter).subquery()
            )

    return attribute_filters, non_attribute_filters
