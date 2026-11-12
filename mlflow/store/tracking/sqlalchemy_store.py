from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import random
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial, reduce, wraps
from pathlib import PurePath
from typing import Any, Concatenate, ParamSpec, TypedDict, TypeVar
from urllib.parse import urlparse

import sqlalchemy
import sqlalchemy.orm
from sqlalchemy import and_, func, or_, select, sql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

import mlflow.store.db.utils
from mlflow.entities import (
    DatasetInput,
    DatasetRecord,
    DatasetRecordSource,
    EvaluationDataset,
    Experiment,
    Run,
    RunInputs,
    RunOutputs,
    RunStatus,
    RunTag,
    ScorerVersion,
    SourceType,
    ViewType,
    _DatasetSummary,
)
from mlflow.entities.entity_type import EntityAssociationType
from mlflow.entities.gateway_endpoint import GatewayResourceType
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.logged_model import LoggedModel
from mlflow.entities.logged_model_input import LoggedModelInput
from mlflow.entities.logged_model_output import LoggedModelOutput
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.metric import Metric, MetricWithRunId
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.trace import Span
from mlflow.entities.workspace import WorkspaceDeletionMode
from mlflow.exceptions import (
    MlflowException,
)
from mlflow.genai.judges.instructions_judge import (
    EXPECTATIONS_FIELD,
    InstructionsJudge,
)
from mlflow.genai.scorers.base import (
    Scorer,
    SerializedScorer,
    _is_decorator_scorer,
    _serialized_scorer_is_custom_code,
)
from mlflow.genai.scorers.online.entities import (
    OnlineScorer,
    OnlineScoringConfig,
)
from mlflow.genai.scorers.scorer_utils import (
    _signature_param_names,
    build_gateway_model,
    container_scoring_enabled,
    custom_scorer_execution_blocked,
    custom_scorer_without_source_error_message,
    extract_endpoint_ref,
    extract_model_from_serialized_scorer,
    has_empty_call_source,
    is_gateway_model,
    update_model_in_serialized_scorer,
    validate_scorer_model,
    validate_scorer_name,
)
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    TEMPORARILY_UNAVAILABLE,
    ErrorCode,
)
from mlflow.store.db.db_types import MSSQL, MYSQL
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import (
    MAX_TRACE_LINKS_PER_REQUEST,
    SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT,
    SEARCH_MAX_RESULTS_DEFAULT,
    SEARCH_MAX_RESULTS_THRESHOLD,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.abstract_trace_store import AbstractTraceStore
from mlflow.store.tracking.dbmodels.models import (
    SqlDataset,
    SqlEntityAssociation,
    SqlEvaluationDataset,
    SqlEvaluationDatasetRecord,
    SqlEvaluationDatasetTag,
    SqlExperiment,
    SqlExperimentTag,
    SqlGatewayEndpoint,
    SqlGatewayEndpointBinding,
    SqlGatewaySecret,
    SqlInput,
    SqlInputTag,
    SqlLabelSchema,
    SqlLatestMetric,
    SqlLoggedModel,
    SqlLoggedModelMetric,
    SqlLoggedModelParam,
    SqlLoggedModelTag,
    SqlMetric,
    SqlOnlineScoringConfig,
    SqlParam,
    SqlReviewQueue,
    SqlReviewQueueItem,
    SqlReviewQueueLabelSchema,
    SqlReviewQueueUser,
    SqlRun,
    SqlScorer,
    SqlScorerVersion,
    SqlTag,
    SqlTraceInfo,
    _input_to_dict,
)
from mlflow.store.tracking.gateway.sqlalchemy_mixin import SqlAlchemyGatewayStoreMixin
from mlflow.store.tracking.mcp_server_registry.sqlalchemy_mixin import (
    SqlAlchemyMCPServerRegistryMixin,
)
from mlflow.store.tracking.skill_registry.sqlalchemy_mixin import (
    SqlAlchemySkillRegistryMixin,
)
from mlflow.store.tracking.sqlalchemy_store_upsert_utils import _bulk_upsert
from mlflow.store.tracking.sqlalchemy_trace_store import SqlAlchemyTraceStore
from mlflow.store.tracking.utils.trace_archival import (
    _resolve_effective_trace_archival_retention,
)
from mlflow.store.tracking.utils.trace_cleanup import (
    delete_experiment_trace_dependencies,
    delete_trace_associations,
)
from mlflow.tracing.trace_archival_config import get_trace_archival_server_config
from mlflow.utils.crypto import KEKManager, _decrypt_secret
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATASET_CONTEXT,
    MLFLOW_LOGGED_MODELS,
    MLFLOW_RUN_NAME,
    MLFLOW_USER,
    _get_run_name_from_tags,
)
from mlflow.utils.name_utils import _generate_random_name
from mlflow.utils.search_utils import (
    SearchEvaluationDatasetsUtils,
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
    _parse_experiment_id,
    _parse_experiment_ids,
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
    _validate_trace_archival_retention_string,
)
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

_T = TypeVar("_T")

_logger = logging.getLogger(__name__)

_RUN_DELETE_CASCADE_MODELS = (SqlMetric, SqlLatestMetric, SqlParam, SqlTag)

# Chunk size for exact-identity recovery lookups in _log_metrics.
# 100 keeps bound-parameter counts well under all supported dialect limits.
_METRIC_DEDUP_CHUNK_SIZE = 100

# Max deadlock retries for a DB-write path (start_trace/log_spans, log_metric/log_batch).
# Deterministic key ordering is the primary defense; this bounded retry is the safety net.
_DB_WRITE_MAX_DEADLOCK_RETRIES = 2

# For each database table, fetch its columns and define an appropriate attribute for each column
# on the table's associated object representation (Mapper). This is necessary to ensure that
# columns defined via backreference are available as Mapper instance attributes (e.g.,
# ``SqlExperiment.tags`` and ``SqlRun.params``). For more information, see
# https://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.configure_mappers
# and https://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.mapper.Mapper
sqlalchemy.orm.configure_mappers()

_TraceParams = ParamSpec("_TraceParams")
_TraceReturn = TypeVar("_TraceReturn")


def _delegate_trace_operation(
    method: Callable[Concatenate[AbstractTraceStore, _TraceParams], _TraceReturn],
) -> Callable[Concatenate["SqlAlchemyStore", _TraceParams], _TraceReturn]:
    @wraps(method)
    def wrapper(
        self: SqlAlchemyStore, *args: _TraceParams.args, **kwargs: _TraceParams.kwargs
    ) -> _TraceReturn:
        return getattr(self.trace_store, method.__name__)(*args, **kwargs)

    return wrapper


class DatasetFilter(TypedDict, total=False):
    """
    Dataset filter used for search_logged_models.
    """

    dataset_name: str
    dataset_digest: str


class SqlAlchemyStore(
    SqlAlchemySkillRegistryMixin,
    SqlAlchemyMCPServerRegistryMixin,
    SqlAlchemyGatewayStoreMixin,
    AbstractStore,
):
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
    EVALUATION_DATASET_ID_PREFIX = "d-"
    _engine_map: dict[str, sqlalchemy.engine.Engine] = {}
    _engine_map_lock = threading.Lock()

    @classmethod
    def _get_or_create_engine(cls, db_uri: str) -> sqlalchemy.engine.Engine:
        """Get a cached engine or create a new one for the given database URI."""
        if db_uri not in cls._engine_map:
            with cls._engine_map_lock:
                if db_uri not in cls._engine_map:
                    cls._engine_map[db_uri] = (
                        mlflow.store.db.utils.create_sqlalchemy_engine_with_retry(db_uri)
                    )
        return cls._engine_map[db_uri]

    def __init__(
        self,
        db_uri,
        default_artifact_root,
        read_db_uri=None,
        trace_store: AbstractTraceStore | None = None,
    ):
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
            read_db_uri: Optional SQLAlchemy database URI for a read replica. When provided,
                read operations (e.g. search_runs, get_experiment) are routed to this URI
                while write operations use ``db_uri``. If not provided, all operations
                use ``db_uri``.
            trace_store: Optional independently constructed trace store. Its tracking-store
                context is bound after database and store initialization succeeds.
        """
        self._trace_store_disabled = ContextVar("mlflow_trace_store_disabled", default=False)
        super().__init__(trace_store=trace_store)
        self.db_uri = db_uri
        self.db_type = extract_db_type_from_uri(db_uri)
        self.artifact_root_uri = resolve_uri_if_local(default_artifact_root)
        self.engine = self._get_or_create_engine(db_uri)
        # On a completely fresh MLflow installation against an empty database (verify database
        # emptiness by checking that 'experiments' etc aren't in the list of table names), run all
        # DB migrations
        if not mlflow.store.db.utils._all_tables_exist(self.engine):
            mlflow.store.db.utils._initialize_tables(self.engine)

        # Set up read replica engine if provided
        if read_db_uri and read_db_uri != db_uri:
            self.read_engine = self._get_or_create_engine(read_db_uri)
            WriteSessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
            ReadSessionMaker = sqlalchemy.orm.sessionmaker(bind=self.read_engine)
            self.ManagedSessionMaker = mlflow.store.db.utils._get_routing_session_maker(
                WriteSessionMaker, ReadSessionMaker, self.db_type
            )
        else:
            if read_db_uri and read_db_uri == db_uri:
                _logger.warning(
                    "read_db_uri is the same as the primary db_uri; "
                    "read replica routing will not be enabled. "
                    "This is likely a configuration mistake."
                )
            self.read_engine = None
            SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
            self.ManagedSessionMaker = mlflow.store.db.utils._get_managed_session_maker(
                SessionMaker, self.db_type
            )

        mlflow.store.db.utils._verify_schema(self.engine)
        if self.read_engine is not None:
            mlflow.store.db.utils._verify_schema(self.read_engine)

        # Note: We intentionally do NOT create the artifact root directory here.
        # The LocalArtifactRepository creates it lazily when the first artifact is logged.
        # This avoids permission errors in read-only environments (e.g., K8s containers)
        # when the artifact root is local but never actually used.

        self._default_trace_store = self._create_default_trace_store()
        self._trace_store = trace_store if trace_store is not None else self._default_trace_store
        self._initialize_store_state()
        if trace_store is not None:
            trace_store.set_tracking_store(self)

    @staticmethod
    def supports_trace_stores() -> bool:
        return True

    def _create_default_trace_store(self) -> SqlAlchemyTraceStore:
        return SqlAlchemyTraceStore(self)

    @property
    def trace_store(self) -> AbstractTraceStore:
        return self._default_trace_store if self._trace_store_disabled.get() else self._trace_store

    @property
    def _has_external_trace_store(self) -> bool:
        return self.trace_store is not self._default_trace_store

    @contextmanager
    def use_default_trace_store(self) -> Iterator[None]:
        """Use native store operations in this store's current context without changing scope.

        Nested blocks and exceptions restore the prior routing state. Async tasks and
        ``asyncio.to_thread`` inherit context; do not spawn unrelated background work here.
        """
        token = self._trace_store_disabled.set(True)
        try:
            yield
        finally:
            self._trace_store_disabled.reset(token)

    @property
    def supports_workspaces(self) -> bool:
        return False

    @property
    def supports_trace_archival(self) -> bool:
        return self.trace_store.supports_trace_archival

    async def log_spans_async(self, location: str, spans: list[Span]) -> list[Span]:
        return await self.trace_store.log_spans_async(location, spans)

    sum_gateway_trace_cost = _delegate_trace_operation(AbstractTraceStore.sum_gateway_trace_cost)

    get_online_trace_details = _delegate_trace_operation(
        AbstractTraceStore.get_online_trace_details
    )

    start_trace = _delegate_trace_operation(AbstractTraceStore.start_trace)

    get_trace_info = _delegate_trace_operation(AbstractTraceStore.get_trace_info)

    search_traces = _delegate_trace_operation(AbstractTraceStore.search_traces)

    find_completed_sessions = _delegate_trace_operation(AbstractTraceStore.find_completed_sessions)

    query_trace_metrics = _delegate_trace_operation(AbstractTraceStore.query_trace_metrics)

    set_trace_tag = _delegate_trace_operation(AbstractTraceStore.set_trace_tag)

    delete_trace_tag = _delegate_trace_operation(AbstractTraceStore.delete_trace_tag)

    delete_traces = _delegate_trace_operation(AbstractTraceStore.delete_traces)

    create_assessment = _delegate_trace_operation(AbstractTraceStore.create_assessment)

    get_assessment = _delegate_trace_operation(AbstractTraceStore.get_assessment)

    update_assessment = _delegate_trace_operation(AbstractTraceStore.update_assessment)

    delete_assessment = _delegate_trace_operation(AbstractTraceStore.delete_assessment)

    calculate_trace_filter_correlation = _delegate_trace_operation(
        AbstractTraceStore.calculate_trace_filter_correlation
    )

    log_spans = _delegate_trace_operation(AbstractTraceStore.log_spans)

    get_trace = _delegate_trace_operation(AbstractTraceStore.get_trace)

    batch_get_traces = _delegate_trace_operation(AbstractTraceStore.batch_get_traces)

    batch_get_trace_infos = _delegate_trace_operation(AbstractTraceStore.batch_get_trace_infos)

    archive_traces = _delegate_trace_operation(AbstractTraceStore.archive_traces)

    deprecated_start_trace_v2 = _delegate_trace_operation(
        AbstractTraceStore.deprecated_start_trace_v2
    )

    deprecated_end_trace_v2 = _delegate_trace_operation(AbstractTraceStore.deprecated_end_trace_v2)

    create_issue = _delegate_trace_operation(AbstractTraceStore.create_issue)

    get_issue = _delegate_trace_operation(AbstractTraceStore.get_issue)

    update_issue = _delegate_trace_operation(AbstractTraceStore.update_issue)

    search_issues = _delegate_trace_operation(AbstractTraceStore.search_issues)

    def _get_active_workspace(self) -> str:
        """
        Get the active workspace name.

        In single-tenant mode, always returns DEFAULT_WORKSPACE_NAME.
        Workspace-aware subclasses override this to enforce isolation.
        """
        return DEFAULT_WORKSPACE_NAME

    def _get_query(self, session, model):
        """
        Return a query for ``model``. Workspace-aware subclasses override this to enforce scoping.
        """
        return session.query(model)

    @staticmethod
    def _artifact_path_segments(uri: str | None) -> list[str]:
        if not uri:
            return []

        if is_local_uri(uri):
            # For filesystem artifact stores, use stdlib path handling to remain
            # platform-safe (e.g., Windows drive letters and separators). Drive/root
            # components are stripped for cross-platform segment comparison while
            # leaving the original URI untouched.
            path = local_file_uri_to_path(uri) if uri.startswith("file:") else uri
            pure_path = PurePath(path)
            drive_root = f"{pure_path.drive}{pure_path.root}"
            skip_segments = {drive_root, pure_path.drive, pure_path.root, "", "/", "\\"}
            return [segment for segment in pure_path.parts if segment not in skip_segments]

        parsed = urlparse(uri)
        path = parsed.path if parsed.scheme else uri
        return [segment for segment in path.split("/") if segment]

    def _with_workspace_field(self, instance):
        if hasattr(instance, "workspace") and getattr(instance, "workspace", None) is None:
            instance.workspace = DEFAULT_WORKSPACE_NAME
        return instance

    def _initialize_store_state(self):
        with self.ManagedSessionMaker() as session:
            workspace_scoped_experiment = (
                session
                .query(SqlExperiment.experiment_id)
                .filter(SqlExperiment.workspace.isnot(None))
                .filter(SqlExperiment.workspace != DEFAULT_WORKSPACE_NAME)
                .first()
            )
            if workspace_scoped_experiment:
                raise MlflowException(
                    "Cannot disable workspaces because experiments exist outside the default "
                    "workspace (i.e., assigned to non-default workspaces). Enable workspace "
                    "support (MLFLOW_ENABLE_WORKSPACES=true) or move those experiments back to the "
                    "default workspace before starting the tracking store in single-tenant mode.",
                    error_code=INVALID_STATE,
                )

        try:
            self.get_experiment(str(self.DEFAULT_EXPERIMENT_ID))
        except MlflowException as exc:
            if exc.error_code and exc.error_code != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                raise
            # Default experiment doesn't exist, create it
            with self.ManagedSessionMaker(read_only=False) as session:
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
            SqlExperiment.workspace.name: DEFAULT_WORKSPACE_NAME,
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
        except IntegrityError as e:
            # Tolerate experiment 0 already existing so this method is idempotent
            session.rollback()
            default_experiment_exists = (
                session
                .query(SqlExperiment.experiment_id)
                .filter(SqlExperiment.experiment_id == int(SqlAlchemyStore.DEFAULT_EXPERIMENT_ID))
                .first()
            )
            if default_experiment_exists is None:
                raise
            _logger.debug(
                "Default experiment (ID 0) already exists. Swallowing IntegrityError: %s",
                e,
            )
        finally:
            self._unset_zero_value_insertion_for_autoincrement_column(session)

    def _get_or_create(self, session, model, **kwargs):
        instance = self._get_query(session, model).filter_by(**kwargs).first()
        created = False

        if instance:
            return instance, created
        else:
            instance = self._with_workspace_field(model(**kwargs))
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
        if tags:
            for tag in tags:
                _validate_experiment_tag(tag.key, tag.value)
        with self.ManagedSessionMaker(read_only=False) as session:
            try:
                creation_time = get_current_time_millis()
                experiment = self._with_workspace_field(
                    SqlExperiment(
                        name=name,
                        lifecycle_stage=LifecycleStage.ACTIVE,
                        artifact_location=artifact_location,
                        creation_time=creation_time,
                        last_update_time=creation_time,
                    )
                )
                experiment.tags = (
                    [SqlExperimentTag(key=tag.key, value=tag.value) for tag in tags] if tags else []
                )
                session.add(experiment)
                session.flush()
                if not artifact_location:
                    # This requires a double flush: the first assigns the autoincremented ID so that
                    # we can derive the default artifact URI, and the second persists the update.
                    experiment.artifact_location = self._get_artifact_location(
                        experiment.experiment_id
                    )
                    session.flush()
            except sqlalchemy.exc.IntegrityError as e:
                raise MlflowException(
                    f"Experiment(name={name}) already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                )

            return str(experiment.experiment_id)

    def _search_experiments(
        self,
        view_type,
        max_results,
        filter_string,
        order_by,
        page_token,
    ):
        effective_retention_context = (
            self._get_effective_experiment_trace_archival_retention_context()
        )

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
            lifecycle_stages = set(LifecycleStage.view_type_to_stages(view_type))

            experiment_filters = [
                *attribute_filters,
                SqlExperiment.lifecycle_stage.in_(lifecycle_stages),
                *self._experiment_where_clauses(),
            ]
            stmt = (
                reduce(lambda s, f: s.join(f), non_attribute_filters, select(SqlExperiment))
                .options(*self._get_eager_experiment_query_options())
                .filter(*experiment_filters)
                .order_by(*order_by_clauses)
                .offset(offset)
                .limit(max_results + 1)
            )
            queried_experiments = session.execute(stmt).scalars(SqlExperiment).all()
            experiments = [
                self._to_experiment(e, effective_retention_context) for e in queried_experiments
            ]
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

    def _get_effective_experiment_trace_archival_retention_context(
        self,
    ) -> tuple[str, set[str]] | None:
        """
        Resolve the broader-scope retention context used to populate experiment entities.

        Returns:
            A tuple of ``(broader_retention, long_retention_allowlist)`` when broader-scope trace
            archival is configured for the current store context. Returns ``None`` when
            effective experiment retention should be left unset.
        """
        try:
            trace_archival_config = get_trace_archival_server_config()
            if trace_archival_config is None:
                return None
            if not trace_archival_config.enabled:
                return None

            broader_retention = trace_archival_config.retention
            resolved_trace_archival_config = self.resolve_trace_archival_config(
                default_trace_archival_location=trace_archival_config.location,
                default_retention=broader_retention,
            ).with_broader_defaults(
                default_location=trace_archival_config.location,
                default_retention=broader_retention,
            )
            broader_retention = resolved_trace_archival_config.config.retention

            return (
                _validate_trace_archival_retention_string(broader_retention),
                set(trace_archival_config.long_retention_allowlist),
            )
        except MlflowException:
            _logger.warning(
                "Ignoring invalid trace archival configuration while resolving effective "
                "experiment retention.",
                exc_info=True,
            )
            return None

    def _to_experiment(
        self,
        sql_experiment: SqlExperiment,
        effective_retention_context: tuple[str, set[str]] | None = None,
    ) -> Experiment:
        effective_trace_archival_retention = None
        if effective_retention_context is not None:
            broader_retention, long_retention_allowlist = effective_retention_context
            effective_trace_archival_retention = _resolve_effective_trace_archival_retention(
                experiment_id=str(sql_experiment.experiment_id),
                experiment_tags={tag.key: tag.value for tag in sql_experiment.tags},
                broader_retention=broader_retention,
                long_retention_allowlist=long_retention_allowlist,
            )

        return sql_experiment.to_mlflow_entity(
            effective_trace_archival_retention=effective_trace_archival_retention
        )

    def _get_experiment(self, session, experiment_id, view_type, eager=False):
        """
        Args:
            eager: If ``True``, eagerly loads the experiments's tags. If ``False``, these tags
                are not eagerly loaded and will be loaded if/when their corresponding
                object properties are accessed from the resulting ``SqlExperiment`` object.
        """
        experiment_id = experiment_id or SqlAlchemyStore.DEFAULT_EXPERIMENT_ID
        stages = LifecycleStage.view_type_to_stages(view_type)
        query_options = self._get_eager_experiment_query_options() if eager else []

        experiment_id_int = _parse_experiment_id(experiment_id)

        experiment = (
            self
            ._get_query(session, SqlExperiment)
            .options(*query_options)
            .filter(
                SqlExperiment.experiment_id == experiment_id_int,
                SqlExperiment.lifecycle_stage.in_(stages),
            )
            .one_or_none()
        )

        if experiment is None:
            raise MlflowException(
                f"No Experiment with id={experiment_id_int} exists", RESOURCE_DOES_NOT_EXIST
            )

        return experiment

    def _experiment_where_clauses(self):
        """
        Hook for subclasses to append additional filters to experiment queries.
        """
        return []

    def _filter_experiment_ids(self, session, experiment_ids, lifecycle_stage: str | None = None):
        """
        Hook for subclasses to filter experiment IDs (e.g., for workspaces).

        When ``lifecycle_stage`` is specified, only return IDs for experiments in that stage.
        """
        if lifecycle_stage is None:
            return experiment_ids

        return [
            row.experiment_id
            for row in session
            .query(SqlExperiment.experiment_id)
            .filter(
                SqlExperiment.experiment_id.in_(experiment_ids),
                SqlExperiment.lifecycle_stage == lifecycle_stage,
            )
            .all()
        ]

    def _filter_entity_ids(
        self, session, entity_type: EntityAssociationType, entity_ids: list[str]
    ):
        """
        Hook for subclasses to filter entity IDs (e.g., for workspaces).
        """
        if entity_type == EntityAssociationType.TRACE:
            return self.trace_store._filter_trace_ids(
                entity_ids, partial(self._filter_experiment_ids, session)
            )
        return entity_ids

    def _filter_association_query(self, session, query, target_type, id_column):
        """
        Hook for subclasses to add additional filters to entity association queries.
        Returns the query with any additional filters applied.
        """
        return query

    def _filter_endpoint_binding_query(self, session, query):
        """
        Hook for subclasses to add additional filters to endpoint binding queries.
        Returns the query with any additional filters applied.
        """
        return query

    def _validate_run_accessible(self, session, run_id: str) -> None:
        """
        Hook for subclasses to validate run access. No-op by default.

        In single-tenant mode, validation is not needed - the database will
        raise appropriate errors if the run doesn't exist (e.g., foreign key
        constraints or empty query results).
        """
        return

    def _validate_dataset_accessible(self, session, dataset_id: str) -> None:
        """
        Hook for subclasses to validate dataset access. No-op by default.

        In single-tenant mode, validation is not needed - the database will
        raise appropriate errors if the dataset doesn't exist.
        """
        return

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
        effective_retention_context = (
            self._get_effective_experiment_trace_archival_retention_context()
        )
        with self.ManagedSessionMaker() as session:
            experiment = self._get_experiment(session, experiment_id, ViewType.ALL, eager=True)
            return self._to_experiment(experiment, effective_retention_context)

    def get_experiment_by_name(self, experiment_name):
        """
        Specialized implementation for SQL backed store.
        """
        effective_retention_context = (
            self._get_effective_experiment_trace_archival_retention_context()
        )
        with self.ManagedSessionMaker() as session:
            stages = LifecycleStage.view_type_to_stages(ViewType.ALL)
            experiment = (
                self
                ._get_query(session, SqlExperiment)
                .options(*self._get_eager_experiment_query_options())
                .filter(
                    SqlExperiment.name == experiment_name,
                    SqlExperiment.lifecycle_stage.in_(stages),
                )
                .one_or_none()
            )
            if experiment is None:
                return None
            return self._to_experiment(experiment, effective_retention_context)

    def delete_experiment(self, experiment_id):
        with self.ManagedSessionMaker(read_only=False) as session:
            experiment = self._get_experiment(session, experiment_id, ViewType.ACTIVE_ONLY)
            experiment.lifecycle_stage = LifecycleStage.DELETED
            experiment.last_update_time = get_current_time_millis()
            runs = self._list_run_infos(session, experiment_id)
            for run in runs:
                self._mark_run_deleted(session, run)
            session.add(experiment)
        try:
            self.on_experiment_deleted(str(experiment_id))
        except Exception:
            _logger.exception(
                "Trace store hook failed after experiment deletion %s.", experiment_id
            )

    def _hard_delete_experiment(self, experiment_id):
        """
        Permanently delete a experiment (metadata and metrics, tags, parameters).
        This is used by the ``mlflow gc`` command line and is not intended to be used elsewhere.
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            experiment = self._get_experiment(
                experiment_id=experiment_id,
                session=session,
                view_type=ViewType.DELETED_ONLY,
            )
            delete_experiment_trace_dependencies(session, [_parse_experiment_id(experiment_id)])
            session.delete(experiment)
        try:
            self.on_experiment_permanently_deleted(str(experiment_id))
        except Exception:
            _logger.exception(
                "Trace store hook failed after permanent experiment deletion %s.", experiment_id
            )

    def on_run_deleted(self, run_id: str) -> None:
        self.trace_store.on_run_deleted(run_id)

    def on_run_permanently_deleted(self, run_id: str) -> None:
        self.trace_store.on_run_permanently_deleted(run_id)

    def on_experiment_deleted(self, experiment_id: str) -> None:
        self.trace_store.on_experiment_deleted(experiment_id)

    def on_experiment_permanently_deleted(self, experiment_id: str) -> None:
        self.trace_store.on_experiment_permanently_deleted(experiment_id)

    def before_workspace_deleted(self, workspace_name: str, mode: WorkspaceDeletionMode) -> None:
        if self._has_external_trace_store:
            if mode == WorkspaceDeletionMode.SET_DEFAULT:
                raise MlflowException.invalid_parameter_value(
                    "Workspace reassignment is not supported with a separate trace store."
                )
        self.trace_store.before_workspace_deleted(workspace_name, mode)

    def on_workspace_deleted(self, workspace_name: str) -> None:
        self.trace_store.on_workspace_deleted(workspace_name)

    def on_traces_deleted(self, trace_ids: list[str]) -> None:
        if trace_ids:
            with self.ManagedSessionMaker(read_only=False) as session:
                self._delete_review_queue_items_for_traces(session, trace_ids)
                delete_trace_associations(session, trace_ids)

    def _mark_run_deleted(self, session, run):
        self._default_trace_store.delete_run_assessments(session, run.run_uuid)

        run.lifecycle_stage = LifecycleStage.DELETED
        run.deleted_time = get_current_time_millis()
        session.add(run)

    def _mark_run_active(self, session, run):
        run.lifecycle_stage = LifecycleStage.ACTIVE
        run.deleted_time = None
        session.add(run)

    def _list_run_infos(self, session, experiment_id):
        return (
            self
            ._get_query(session, SqlRun)
            .filter(SqlRun.experiment_id == _parse_experiment_id(experiment_id))
            .all()
        )

    def restore_experiment(self, experiment_id):
        with self.ManagedSessionMaker(read_only=False) as session:
            experiment = self._get_experiment(session, experiment_id, ViewType.DELETED_ONLY)
            experiment.lifecycle_stage = LifecycleStage.ACTIVE
            experiment.last_update_time = get_current_time_millis()
            runs = self._list_run_infos(session, experiment_id)
            for run in runs:
                self._mark_run_active(session, run)
            session.add(experiment)

    def rename_experiment(self, experiment_id, new_name):
        _validate_experiment_name(new_name)
        with self.ManagedSessionMaker(read_only=False) as session:
            experiment = self._get_experiment(session, experiment_id, ViewType.ALL)
            if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
                raise MlflowException("Cannot rename a non-active experiment.", INVALID_STATE)

            experiment.name = new_name
            experiment.last_update_time = get_current_time_millis()
            session.add(experiment)

    def create_run(self, experiment_id, user_id, start_time, tags, run_name):
        with self.ManagedSessionMaker(read_only=False) as session:
            experiment = self.get_experiment(experiment_id)
            self._check_experiment_is_active(experiment)

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

    def _get_run(self, session, run_uuid, eager=False):
        """
        Args:
            eager: If ``True``, eagerly loads the run's summary metrics (``latest_metrics``),
                params, and tags when fetching the run. If ``False``, these attributes
                are not eagerly loaded and will be loaded when their corresponding
                object properties are accessed from the resulting ``SqlRun`` object.
        """
        query_options = self._get_eager_run_query_options() if eager else []
        runs = (
            self
            ._get_query(session, SqlRun)
            .options(*query_options)
            .filter(SqlRun.run_uuid == run_uuid)
            .all()
        )

        if len(runs) == 0:
            raise MlflowException(f"Run with id={run_uuid} not found", RESOURCE_DOES_NOT_EXIST)
        if len(runs) > 1:
            raise MlflowException(
                f"Expected only 1 run with id={run_uuid}. Found {len(runs)}.",
                INVALID_STATE,
            )

        return runs[0]

    def _dataset_query(self, session):
        return self._get_query(session, SqlEvaluationDataset)

    def _get_run_inputs(self, session, run_uuids):
        datasets_with_tags = (
            session
            .query(
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
        with self.ManagedSessionMaker(read_only=False) as session:
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
            session
            .query(SqlTag)
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
        with self.ManagedSessionMaker(read_only=False) as session:
            run = self._get_run(run_uuid=run_id, session=session)
            run.lifecycle_stage = LifecycleStage.ACTIVE
            run.deleted_time = None
            session.add(run)

    def delete_run(self, run_id):
        with self.ManagedSessionMaker(read_only=False) as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._mark_run_deleted(session, run)
        try:
            self.on_run_deleted(run_id)
        except Exception:
            _logger.exception("Trace store hook failed after run deletion %s.", run_id)

    def _hard_delete_run(self, run_id):
        """
        Permanently delete a run (metadata and metrics, tags, parameters).
        This is used by the ``mlflow gc`` command line and is not intended to be used elsewhere.
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            # ORM delete cascades load every child row into the session, which can exhaust memory
            # for runs with large metric histories. Bulk deletes keep memory usage independent of
            # history size.
            self._get_run(run_uuid=run_id, session=session)
            session.query(SqlEntityAssociation).filter(
                SqlEntityAssociation.source_type == EntityAssociationType.TRACE,
                SqlEntityAssociation.destination_type == EntityAssociationType.RUN,
                SqlEntityAssociation.destination_id == run_id,
            ).delete(synchronize_session=False)
            for model in _RUN_DELETE_CASCADE_MODELS:
                session.query(model).filter(model.run_uuid == run_id).delete(
                    synchronize_session=False
                )
            session.query(SqlRun).filter(SqlRun.run_uuid == run_id).delete(
                synchronize_session=False
            )
        try:
            self.on_run_permanently_deleted(run_id)
        except Exception:
            _logger.exception("Trace store hook failed after permanent run deletion %s.", run_id)

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
                self
                ._get_query(session, SqlRun)
                .filter(
                    SqlRun.lifecycle_stage == LifecycleStage.DELETED,
                    SqlRun.deleted_time <= (current_time - older_than),
                )
                .all()
            )
            return [run.run_uuid for run in runs]

    def log_metric(self, run_id, metric):
        # simply call _log_metrics and let it handle the rest

        if metric.model_id is not None:
            with self.ManagedSessionMaker() as session:
                run = self._get_run(run_uuid=run_id, session=session)
                self._check_run_is_active(run)
                experiment_id = run.experiment_id
            self._log_model_metrics(run_id, [metric], experiment_id=experiment_id)

        self._log_metrics(run_id, [metric])

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
        # Retry on DB deadlock: each attempt opens a fresh session in _log_metrics_once, so the
        # victim is redriven instead of surfacing an HTTP 503.
        return self._run_with_deadlock_retry(self._log_metrics_once, run_id, metrics)

    def _log_metrics_once(self, run_id, metrics):
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

        with self.ManagedSessionMaker(read_only=False) as session:
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
                # Determine which metrics from this batch already exist in the DB.
                # Uses chunked exact-PK lookups (OR-of-AND on all identity columns)
                # so recovery work is bounded by the batch size rather than the run's
                # history size. Each predicate matches at most one row, so recovery
                # returns at most one row per candidate identity.
                #
                # This replaces an earlier approach that scanned the full per-key
                # history and caused OOM / gateway timeouts on large runs
                # (see https://github.com/mlflow/mlflow/issues/24577).

                def _metric_id(key, timestamp, step, value, is_nan):
                    """Build the dedup identity for a metric within one run.

                    Covers metric_pk minus run_uuid: (key, timestamp, step, value, is_nan).
                    NaN metrics are stored as value=0, is_nan=True; keying on is_nan
                    (not float('nan')) avoids Python's nan != nan breaking set lookups.
                    """
                    return (key, timestamp, step, value, is_nan)

                # Build an ordered identity-to-instance mapping. Using a dict
                # preserves insertion order and deduplicates same-batch entries
                # (e.g. multiple NaN objects that sanitize to the same DB identity).
                metrics_by_id = {}
                for m in metric_instances:
                    metrics_by_id.setdefault(
                        _metric_id(m.key, m.timestamp, m.step, m.value, m.is_nan), m
                    )

                candidates = list(metrics_by_id)

                existing_ids = set()
                for i in range(0, len(candidates), _METRIC_DEDUP_CHUNK_SIZE):
                    chunk = candidates[i : i + _METRIC_DEDUP_CHUNK_SIZE]
                    row_predicates = [
                        and_(
                            SqlMetric.key == key,
                            SqlMetric.timestamp == timestamp,
                            SqlMetric.step == step,
                            SqlMetric.value == value,
                            SqlMetric.is_nan == is_nan,
                        )
                        for key, timestamp, step, value, is_nan in chunk
                    ]
                    rows = (
                        session
                        .query(
                            SqlMetric.key,
                            SqlMetric.timestamp,
                            SqlMetric.step,
                            SqlMetric.value,
                            SqlMetric.is_nan,
                        )
                        .filter(
                            SqlMetric.run_uuid == run_id,
                            or_(*row_predicates),
                        )
                        .all()
                    )
                    existing_ids.update(_metric_id(*row) for row in rows)

                metric_instances = [
                    m for metric_id, m in metrics_by_id.items() if metric_id not in existing_ids
                ]
                # if there exist metrics that were tried to be logged & rolled back even
                # though they were not violating the PK, log them
                if non_existing_metrics := metric_instances:
                    _insert_metrics(non_existing_metrics)

    def _log_model_metrics(
        self,
        run_id: str,
        metrics: list[Metric],
        experiment_id: str,
        dataset_uuid: str | None = None,
    ) -> None:
        if not metrics:
            return

        is_single_metric = len(metrics) == 1
        seen: set[Metric] = set()
        sanitized_metrics: list[tuple[Metric, float]] = []
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
            _, value = self.sanitize_metric_value(metric.value)
            sanitized_metrics.append((metric, value))

        if not sanitized_metrics:
            return

        with self.ManagedSessionMaker(read_only=False) as session:
            metric_instances = [
                SqlLoggedModelMetric(
                    model_id=metric.model_id,
                    metric_name=metric.key,
                    metric_timestamp_ms=metric.timestamp,
                    metric_step=metric.step,
                    metric_value=value,
                    experiment_id=experiment_id,
                    run_id=run_id,
                    dataset_uuid=dataset_uuid,
                    dataset_name=metric.dataset_name,
                    dataset_digest=metric.dataset_digest,
                )
                for metric, value in sanitized_metrics
            ]

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
                        session
                        .query(SqlLoggedModelMetric)
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
        # Globally sort + de-dup keys to prevent cross-batch deadlocks: every transaction locks
        # rows in the same order (the per-batch ORDER BY below only orders within one 500 batch).
        metric_keys = sorted({m.key for m in logged_metrics})
        # Divide metric keys into batches of 500 to avoid binding too many parameters to the SQL
        # query, which may produce limit exceeded errors or poor performance on certain database
        # platforms
        metric_key_batches = [metric_keys[i : i + 500] for i in range(0, len(metric_keys), 500)]
        for metric_key_batch in metric_key_batches:
            # First, determine which metric keys are present in the database
            latest_metrics_key_records_from_db = (
                session
                .query(SqlLatestMetric.key)
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
                    session
                    .query(SqlLatestMetric)
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
            self._validate_run_accessible(session, run_id)
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
            run_ids = self._filter_entity_ids(session, EntityAssociationType.RUN, list(run_ids))

            metrics = (
                session
                .query(SqlMetric)
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
            self._validate_run_accessible(session, run_id)
            max_step = (
                session
                .query(func.max(SqlMetric.step))
                .filter(SqlMetric.run_uuid == run_id, SqlMetric.key == metric_key)
                .scalar()
            )
            return max_step or 0

    def get_metric_history_bulk_interval_from_steps(self, run_id, metric_key, steps, max_results):
        with self.ManagedSessionMaker() as session:
            self._validate_run_accessible(session, run_id)
            metrics = (
                session
                .query(SqlMetric)
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

    def get_metric_history_bulk_interval(
        self,
        run_ids: list[str],
        metric_key: str,
        max_results: int,
        start_step: int | None,
        end_step: int | None,
    ) -> list[MetricWithRunId]:
        """Return up to ``max_results`` metrics per run, evenly sampled across the values logged
        for ``metric_key`` (optionally restricted to ``[start_step, end_step]``), always keeping
        the first and last points when ``max_results`` > 1 and returns only the last point when
        ``max_results`` == 1.

        The sampling is performed entirely in SQL using window functions, so the server
        materializes at most ~``max_results`` rows per run regardless of how many values were
        logged. The previous implementation downsampled by *distinct step* and then fetched
        every row for the sampled steps (capped at ``MAX_RESULTS_GET_METRIC_HISTORY``). That
        collapsed to a single step -- and thus returned tens of thousands of rows -- whenever
        many values shared one step (e.g. the default ``step=0`` used by ``log_metric`` without
        an explicit step), causing large memory spikes. Sampling by row instead of by step keeps
        the response bounded and representative of the whole range in that case.
        """
        if (start_step is None) != (end_step is None):
            raise MlflowException.invalid_parameter_value(
                "Both start_step and end_step must be specified together, "
                "or neither may be specified."
            )
        max_results = max(1, max_results)
        metrics_with_run_ids = []
        with self.ManagedSessionMaker() as session:
            for run_id in run_ids:
                self._validate_run_accessible(session, run_id)
            for run_id in run_ids:
                metrics_with_run_ids.extend(
                    self._sample_metric_history_single_run(
                        session, run_id, metric_key, max_results, start_step, end_step
                    )
                )
        return metrics_with_run_ids

    def _supports_window_functions(self, session) -> bool:
        # NTILE / ROW_NUMBER are available on every backend MLflow targets except MySQL < 8.0 and
        # MariaDB < 10.2, both of which MLflow still supports (e.g. MySQL 5.7).
        if self.db_type != MYSQL:
            return True
        dialect = session.get_bind().dialect
        version = getattr(dialect, "server_version_info", None)
        if not version:
            return True
        if getattr(dialect, "_is_mariadb", False):
            return version >= (10, 2)
        return version >= (8, 0)

    def _sample_metric_history_single_run(
        self, session, run_id, metric_key, max_results, start_step, end_step
    ) -> list[MetricWithRunId]:
        filters = [SqlMetric.run_uuid == run_id, SqlMetric.key == metric_key]
        if start_step is not None and end_step is not None:
            filters.append(SqlMetric.step >= start_step)
            filters.append(SqlMetric.step <= end_step)

        # ``is_nan`` is part of the metric primary key, so include it in every ordering to keep
        # sampling deterministic and preserve a NaN boundary point on ties.
        order_by = [SqlMetric.step, SqlMetric.timestamp, SqlMetric.value, SqlMetric.is_nan]

        if self._supports_window_functions(session):
            rows = self._sample_rows_with_window_functions(session, filters, order_by, max_results)
        else:
            rows = self._sample_rows_in_python(session, filters, order_by, max_results)

        return [
            MetricWithRunId(
                run_id=run_id,
                metric=Metric(
                    key=metric_key,
                    value=row.value if not row.is_nan else float("nan"),
                    timestamp=row.timestamp,
                    step=row.step,
                ),
            )
            for row in rows
        ]

    def _sample_rows_with_window_functions(self, session, filters, order_by, max_results):
        # Assign each row to one of ``max_results`` evenly sized buckets across the full ordered
        # result set, then keep the first row of each bucket. NTILE produces one bucket per row
        # when there are fewer than ``max_results`` rows, so all rows are returned in that case.
        bucketed = (
            session
            .query(
                SqlMetric.value.label("value"),
                SqlMetric.timestamp.label("timestamp"),
                SqlMetric.step.label("step"),
                SqlMetric.is_nan.label("is_nan"),
                func.ntile(max_results).over(order_by=order_by).label("bucket"),
            )
            .filter(*filters)
            .subquery()
        )
        bucket_order = [bucketed.c.step, bucketed.c.timestamp, bucketed.c.value, bucketed.c.is_nan]
        ranked = session.query(
            bucketed.c.value,
            bucketed.c.timestamp,
            bucketed.c.step,
            bucketed.c.is_nan,
            func
            .row_number()
            .over(partition_by=bucketed.c.bucket, order_by=bucket_order)
            .label("rn"),
        ).subquery()
        rows = (
            session
            .query(ranked.c.value, ranked.c.timestamp, ranked.c.step, ranked.c.is_nan)
            .filter(ranked.c.rn == 1)
            .order_by(ranked.c.step, ranked.c.timestamp, ranked.c.value, ranked.c.is_nan)
            .all()
        )

        # NTILE captures each bucket's first row but not necessarily the global last (maximum) one,
        # so include it to preserve the metric's end boundary. Replace the last sampled point when
        # already at ``max_results`` so the response never exceeds the requested size.
        last_row = (
            session
            .query(SqlMetric.value, SqlMetric.timestamp, SqlMetric.step, SqlMetric.is_nan)
            .filter(*filters)
            .order_by(
                SqlMetric.step.desc(),
                SqlMetric.timestamp.desc(),
                SqlMetric.value.desc(),
                SqlMetric.is_nan.desc(),
            )
            .first()
        )

        def _row_key(row):
            value = None if row.is_nan else row.value
            return (row.step, row.timestamp, value, row.is_nan)

        if last_row is not None and (not rows or _row_key(rows[-1]) != _row_key(last_row)):
            if len(rows) >= max_results:
                rows[-1] = last_row
            else:
                rows.append(last_row)
        return rows

    def _sample_rows_in_python(self, session, filters, order_by, max_results):
        # Fallback for backends without window functions (MySQL < 8.0 / MariaDB < 10.2). The rows
        # are streamed server-side and an evenly spaced sample is kept, so the server holds at most
        # ~``max_results`` rows in memory regardless of how many values were logged.
        total = session.query(func.count()).select_from(SqlMetric).filter(*filters).scalar()
        if not total:
            return []
        target_indices = set(self._evenly_spaced_indices(total, max_results))

        rows_iter = (
            session
            .query(SqlMetric.value, SqlMetric.timestamp, SqlMetric.step, SqlMetric.is_nan)
            .filter(*filters)
            .order_by(*order_by)
            .yield_per(1000)
        )
        return [row for idx, row in enumerate(rows_iter) if idx in target_indices]

    def _search_datasets(self, experiment_ids):
        """
        Return all dataset summaries associated to the given experiments.

        Args:
            experiment_ids: List of experiment ids to scope the search

        Returns:
            A List of :py:class:`SqlAlchemyStore.DatasetSummary` entities.
        """

        MAX_DATASET_SUMMARIES_RESULTS = 1000
        experiment_ids = _parse_experiment_ids(experiment_ids)
        with self.ManagedSessionMaker() as session:
            experiment_ids = self._filter_experiment_ids(session, experiment_ids)
            # Note that the join with the input tag table is a left join. This is required so if an
            # input does not have the MLFLOW_DATASET_CONTEXT tag, we still return that entry as part
            # of the final result with the context set to None.
            summaries = (
                session
                .query(
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
        with self.ManagedSessionMaker(read_only=False) as session:
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

        with self.ManagedSessionMaker(read_only=False) as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            existing_params = {p.key: p.value for p in run.params}
            new_params = []
            non_matching_params = []
            for param in params:
                if param.key in existing_params:
                    if param.value != existing_params[param.key]:
                        non_matching_params.append({
                            "key": param.key,
                            "old_value": existing_params[param.key],
                            "new_value": param.value,
                        })
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
        with self.ManagedSessionMaker(read_only=False) as session:
            experiment = self._get_experiment(
                session, experiment_id, ViewType.ALL
            ).to_mlflow_entity()
            self._check_experiment_is_active(experiment)
            session.merge(
                SqlExperimentTag(experiment_id=experiment_id, key=tag.key, value=tag.value)
            )

    def delete_experiment_tag(self, experiment_id, key):
        """
        Delete a tag from the specified experiment

        Args:
            experiment_id: String ID of the experiment
            key: String name of the tag to be deleted
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            experiment = self._get_experiment(
                session, experiment_id, ViewType.ALL
            ).to_mlflow_entity()
            self._check_experiment_is_active(experiment)
            filtered_tags = (
                session
                .query(SqlExperimentTag)
                .filter_by(experiment_id=_parse_experiment_id(experiment_id), key=key)
                .all()
            )
            if len(filtered_tags) == 0:
                raise MlflowException(
                    f"No tag with name: {key} in experiment with id {experiment_id}",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            elif len(filtered_tags) > 1:
                raise MlflowException(
                    "Bad data in database - tags for a specific experiment must have "
                    "a single unique value. "
                    "See https://mlflow.org/docs/latest/ml/getting-started/logging-first-model/step3-create-experiment/#notes-on-tags-vs-experiments",
                    error_code=INVALID_STATE,
                )
            session.delete(filtered_tags[0])

    def set_tag(self, run_id, tag):
        """
        Set a tag on a run.

        Args:
            run_id: String ID of the run.
            tag: RunTag instance to log.
        """
        with self.ManagedSessionMaker(read_only=False) as session:
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
        """
        if not tags:
            return

        tags = [_validate_tag(t.key, t.value, path=f"tags[{idx}]") for (idx, t) in enumerate(tags)]

        with self.ManagedSessionMaker(read_only=False) as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)

            def _try_insert_tags(attempt_number, max_retries):
                try:
                    current_tags = (
                        session
                        .query(SqlTag)
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
        with self.ManagedSessionMaker(read_only=False) as session:
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
            if max_results is not None and current_size == max_results + 1:
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
            for dataset_filter in dataset_filters:
                stmt = stmt.join(
                    dataset_filter,
                    SqlRun.run_uuid == dataset_filter.c.destination_id,
                )
            # using an outer join is necessary here because we want to be able to sort
            # on a column (tag, metric or param) without removing the lines that
            # do not have a value for this column (which is what inner join would do)
            for j in sorting_joins:
                stmt = stmt.outerjoin(j)

            offset = SearchUtils.parse_start_offset_from_page_token(page_token)
            experiment_ids = _parse_experiment_ids(experiment_ids)
            experiment_ids = self._filter_experiment_ids(session, experiment_ids)
            stmt = (
                stmt
                .distinct()
                .options(*self._get_eager_run_query_options())
                .filter(
                    SqlRun.experiment_id.in_(experiment_ids),
                    SqlRun.lifecycle_stage.in_(stages),
                    *attribute_filters,
                )
                .order_by(*parsed_orderby)
                .offset(offset)
            )
            if max_results is not None:
                stmt = stmt.limit(max_results + 1)
            queried_runs = session.execute(stmt).scalars(SqlRun).all()

            runs = [run.to_mlflow_entity() for run in queried_runs]
            run_ids = [run.info.run_id for run in runs]

            # add inputs and outputs to runs
            inputs = self._get_run_inputs(run_uuids=run_ids, session=session)
            model_outputs_map = self._get_model_outputs_bulk(run_ids=run_ids, session=session)
            runs_with_inputs_outputs = []
            for i, run in enumerate(runs):
                runs_with_inputs_outputs.append(
                    Run(
                        run.info,
                        run.data,
                        RunInputs(dataset_inputs=inputs[i]),
                        RunOutputs(model_outputs_map[run.info.run_id]),
                    )
                )

            next_page_token = compute_next_token(len(runs_with_inputs_outputs))

            # Trim results if we fetched an extra row to check for more pages
            if next_page_token and max_results is not None:
                runs_with_inputs_outputs = runs_with_inputs_outputs[:max_results]

        return runs_with_inputs_outputs, next_page_token

    def log_batch(self, run_id, metrics, params, tags):
        _validate_run_id(run_id)
        metrics, params, tags = _validate_batch_log_data(metrics, params, tags)
        _validate_batch_log_limits(metrics, params, tags)
        _validate_param_keys_unique(params)

        with self.ManagedSessionMaker(read_only=False) as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            try:
                self._log_params(run_id, params)
                self._log_metrics(run_id, metrics)
                self._log_model_metrics(run_id, metrics, experiment_id=run.experiment_id)
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
        with self.ManagedSessionMaker(read_only=False) as session:
            run = self._get_run(run_uuid=run_id, session=session)
            self._check_run_is_active(run)
            if previous_tag := [t for t in run.tags if t.key == MLFLOW_LOGGED_MODELS]:
                value = json.dumps(json.loads(previous_tag[0].value) + [model_dict])
            else:
                value = json.dumps([model_dict])
            _validate_tag(MLFLOW_LOGGED_MODELS, value)
            session.merge(SqlTag(key=MLFLOW_LOGGED_MODELS, value=value, run_uuid=run_id))

    def log_inputs(
        self,
        run_id: str,
        datasets: list[DatasetInput] | None = None,
        models: list[LoggedModelInput] | None = None,
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

        with self.ManagedSessionMaker(read_only=False) as session:
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
        dataset_inputs: list[DatasetInput] | None = None,
        models: list[LoggedModelInput] | None = None,
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

        with self.ManagedSessionMaker(read_only=False) as session:
            dataset_names_to_check = [
                dataset_input.dataset.name for dataset_input in dataset_inputs
            ]
            dataset_digests_to_check = [
                dataset_input.dataset.digest for dataset_input in dataset_inputs
            ]
            # find all datasets with the same name and digest
            # if the dataset already exists, use the existing dataset uuid
            existing_datasets = (
                session
                .query(SqlDataset)
                .filter(SqlDataset.experiment_id == experiment_id)
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
                session
                .query(SqlInput)
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
                    objs_to_write.extend(
                        SqlInputTag(
                            input_uuid=new_input_uuid,
                            name=input_tag.key,
                            value=input_tag.value,
                        )
                        for input_tag in dataset_input.tags
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
        with self.ManagedSessionMaker(read_only=False) as session:
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
        session: sqlalchemy.orm.Session | None = None,
    ) -> list[LoggedModelInput]:
        return [
            LoggedModelInput(model_id=input.destination_id)
            for input in (
                session
                .query(SqlInput)
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
            for output in session
            .query(SqlInput)
            .filter(
                SqlInput.source_type == "RUN_OUTPUT",
                SqlInput.source_id == run_id,
                SqlInput.destination_type == "MODEL_OUTPUT",
            )
            .all()
        ]

    def _get_model_outputs_bulk(
        self,
        run_ids: list[str],
        session: sqlalchemy.orm.Session,
    ) -> dict[str, list[LoggedModelOutput]]:
        """
        Fetch model outputs for multiple runs in a single query.
        Returns a dict mapping run_id to list of LoggedModelOutput.
        """
        outputs = (
            session
            .query(SqlInput)
            .filter(
                SqlInput.source_type == "RUN_OUTPUT",
                SqlInput.source_id.in_(run_ids),
                SqlInput.destination_type == "MODEL_OUTPUT",
            )
            .all()
        )

        outputs_per_run = defaultdict(list)
        for output in outputs:
            outputs_per_run[output.source_id].append(
                LoggedModelOutput(model_id=output.destination_id, step=output.step)
            )

        # Ensure all run_ids are present in the result, even if they have no outputs
        return {run_id: outputs_per_run.get(run_id, []) for run_id in run_ids}

    #######################################################################################
    # Logged models
    #######################################################################################
    def create_logged_model(
        self,
        experiment_id: str,
        name: str | None = None,
        source_run_id: str | None = None,
        tags: list[LoggedModelTag] | None = None,
        params: list[LoggedModelParameter] | None = None,
        model_type: str | None = None,
    ) -> LoggedModel:
        _validate_logged_model_name(name)
        with self.ManagedSessionMaker(read_only=False) as session:
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
        with self.ManagedSessionMaker(read_only=False) as session:
            logged_model = self._get_logged_model_record(session, model_id)
            session.add_all(
                SqlLoggedModelParam(
                    model_id=model_id,
                    experiment_id=logged_model.experiment_id,
                    param_key=param.key,
                    param_value=param.value,
                )
                for param in params
            )

    def _get_logged_model_record(self, session, model_id: str) -> SqlLoggedModel:
        logged_model = (
            self
            ._get_query(session, SqlLoggedModel)
            .filter(SqlLoggedModel.model_id == model_id)
            .one_or_none()
        )
        if not logged_model:
            self._raise_model_not_found(model_id)
        return logged_model

    def _raise_model_not_found(self, model_id: str):
        raise MlflowException(
            f"Logged model with ID '{model_id}' not found.",
            RESOURCE_DOES_NOT_EXIST,
        )

    def get_logged_model(self, model_id: str, allow_deleted: bool = False) -> LoggedModel:
        with self.ManagedSessionMaker() as session:
            query = self._get_query(session, SqlLoggedModel).filter(
                SqlLoggedModel.model_id == model_id,
            )
            if not allow_deleted:
                query = query.filter(SqlLoggedModel.lifecycle_stage != LifecycleStage.DELETED)

            logged_model = query.first()
            if not logged_model:
                self._raise_model_not_found(model_id)

            return logged_model.to_mlflow_entity()

    def delete_logged_model(self, model_id):
        with self.ManagedSessionMaker(read_only=False) as session:
            logged_model = self._get_logged_model_record(session, model_id)
            logged_model.lifecycle_stage = LifecycleStage.DELETED
            logged_model.last_updated_timestamp_ms = get_current_time_millis()
            session.commit()

    def _hard_delete_logged_model(self, model_id):
        with self.ManagedSessionMaker(read_only=False) as session:
            logged_model = session.get(SqlLoggedModel, model_id)
            if not logged_model:
                self._raise_model_not_found(model_id)
            session.delete(logged_model)

    def _get_deleted_logged_models(self, older_than=0):
        current_time = get_current_time_millis()
        with self.ManagedSessionMaker() as session:
            models = (
                self
                ._get_query(session, SqlLoggedModel)
                .filter(
                    SqlLoggedModel.lifecycle_stage == LifecycleStage.DELETED,
                    SqlLoggedModel.last_updated_timestamp_ms <= (current_time - older_than),
                )
                .all()
            )
            return [m.model_id for m in models]

    def finalize_logged_model(self, model_id: str, status: LoggedModelStatus) -> LoggedModel:
        with self.ManagedSessionMaker(read_only=False) as session:
            logged_model = self._get_logged_model_record(session, model_id)
            logged_model.status = status.to_int()
            logged_model.last_updated_timestamp_ms = get_current_time_millis()
            session.commit()
            return logged_model.to_mlflow_entity()

    def set_logged_model_tags(self, model_id: str, tags: list[LoggedModelTag]) -> None:
        with self.ManagedSessionMaker(read_only=False) as session:
            logged_model = self._get_logged_model_record(session, model_id)
            # Dedupe by key so a repeated key in one call keeps the last value (matching the
            # previous ``session.merge`` behavior); PostgreSQL's ``ON CONFLICT DO UPDATE``
            # rejects statements that touch the same row twice.
            deduped = {tag.key: tag.value for tag in tags}
            rows = [
                {
                    "model_id": model_id,
                    "experiment_id": logged_model.experiment_id,
                    "tag_key": key,
                    "tag_value": value,
                }
                for key, value in deduped.items()
            ]
            _bulk_upsert(session, SqlLoggedModelTag, rows)

    def delete_logged_model_tag(self, model_id: str, key: str) -> None:
        with self.ManagedSessionMaker(read_only=False) as session:
            self._get_logged_model_record(session, model_id)
            count = (
                session
                .query(SqlLoggedModelTag)
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

    def register_scorer(
        self, experiment_id: str, name: str, serialized_scorer: str
    ) -> ScorerVersion:
        """
        Register a scorer for an experiment.

        Args:
            experiment_id: The experiment ID.
            name: The scorer name.
            serialized_scorer: The serialized scorer string (JSON).

        Returns:
            mlflow.entities.ScorerVersion: The newly registered scorer version with scorer_id.

        Raises:
            MlflowException: If the scorer name is invalid, if the model is invalid,
                if the scorer references a gateway endpoint that does not exist, or if the
                experiment has a scorer whose name differs from ``name`` but that the database
                compares as equal (``RESOURCE_ALREADY_EXISTS``).
        """
        # Validate scorer name
        validate_scorer_name(name)

        with self.ManagedSessionMaker(read_only=False) as session:
            # Validate experiment exists and is active
            experiment = self.get_experiment(experiment_id)
            self._check_experiment_is_active(experiment)

            # Parse serialized_scorer and validate its contents
            serialized_data = json.loads(serialized_scorer)

            # Validate model if present
            model = extract_model_from_serialized_scorer(serialized_data)
            validate_scorer_model(model)

            endpoint_id = None
            if is_gateway_model(model):
                endpoint_name = extract_endpoint_ref(model)
                # Resolve name to ID - raises MlflowException if endpoint doesn't exist
                endpoint = self.get_gateway_endpoint(name=endpoint_name)
                endpoint_id = endpoint.endpoint_id
                # Update serialized scorer with endpoint ID instead of name
                serialized_data = update_model_in_serialized_scorer(
                    serialized_data, build_gateway_model(endpoint.endpoint_id)
                )
                serialized_scorer = json.dumps(serialized_data)

            scorer = self._find_sql_scorer_by_collation(
                session, int(experiment.experiment_id), name
            )
            if scorer is not None and scorer.scorer_name != name:
                # Only on a database that compares names without case or accents. The scorer
                # table allows one scorer per name there, so the only option would be to add a
                # version to the existing scorer, which would bypass the permission grants on its
                # real name. The message leaves out that name, which the caller may not be allowed
                # to see.
                raise MlflowException(
                    f"Experiment {experiment_id} already has a scorer whose name this database "
                    f"compares as equal to {name!r}, so a scorer named {name!r} cannot be "
                    "registered. Use a different name.",
                    RESOURCE_ALREADY_EXISTS,
                )

            if scorer is None:
                # Create the scorer record with a new UUID
                scorer_id = str(uuid.uuid4())
                scorer = SqlScorer(
                    experiment_id=int(experiment.experiment_id),
                    scorer_name=name,
                    scorer_id=scorer_id,
                )
                session.add(scorer)
                session.flush()  # Flush to get the scorer record

            # Find the maximum version for this scorer
            max_version = (
                session
                .query(func.max(SqlScorerVersion.scorer_version))
                .filter(SqlScorerVersion.scorer_id == scorer.scorer_id)
                .scalar()
            )

            # Set new version (1 if no existing scorer, otherwise max + 1)
            new_version = 1 if max_version is None else max_version + 1

            # Create and save the new scorer version record
            sql_scorer_version = SqlScorerVersion(
                scorer_id=scorer.scorer_id,
                scorer_version=new_version,
                serialized_scorer=serialized_scorer,
            )

            session.add(sql_scorer_version)

            # Create endpoint binding if scorer uses a gateway endpoint
            # Verify endpoint exists in DB (handles mocked tests and race conditions)
            if endpoint_id is not None:
                endpoint_exists = (
                    self
                    ._get_query(session, SqlGatewayEndpoint)
                    .filter(SqlGatewayEndpoint.endpoint_id == endpoint_id)
                    .first()
                )
                if endpoint_exists is not None:
                    # Delete any existing binding for this scorer (in case of re-registration)
                    # Use scorer_id for globally unique identification across experiments
                    session.query(SqlGatewayEndpointBinding).filter(
                        SqlGatewayEndpointBinding.resource_type == GatewayResourceType.SCORER.value,
                        SqlGatewayEndpointBinding.resource_id == scorer.scorer_id,
                    ).delete()

                    binding = SqlGatewayEndpointBinding(
                        endpoint_id=endpoint_id,
                        resource_type=GatewayResourceType.SCORER.value,
                        resource_id=scorer.scorer_id,
                        display_name=name,
                        created_at=get_current_time_millis(),
                        last_updated_at=get_current_time_millis(),
                    )
                    session.add(binding)

            session.flush()

            entity = sql_scorer_version.to_mlflow_entity()
            # Resolve gateway endpoint ID to name before returning
            return self.resolve_endpoint_in_scorer(entity)

    def list_scorers(self, experiment_id) -> list[ScorerVersion]:
        """
        List all scorers for an experiment.

        Args:
            experiment_id: The experiment ID.

        Returns:
            List of mlflow.entities.scorer.ScorerVersion objects
            (latest version for each scorer name) with gateway endpoint IDs resolved to names.
        """
        # Validate the experiment, then delegate to the more general
        # cross-experiment query. Single-experiment ``order_by`` collapses to
        # ``scorer_name`` because every row shares the same ``experiment_id``.
        experiment = self.get_experiment(experiment_id)
        self._check_experiment_is_active(experiment)
        return self.list_scorers_across_experiments([experiment.experiment_id])

    # SQLite caps bound parameters at 999 by default; pick a chunk size well
    # below that so callers passing large ID lists (e.g. the admin scorer
    # picker passing up to 1000 experiment ids per page, or an authorization
    # scope with hundreds of experiment ids) don't trip
    # ``too many SQL variables``. Shared by every batch query in this class
    # that chunks an ``IN (...)`` clause.
    _ID_CHUNK_SIZE = 500

    def list_scorers_across_experiments(self, experiment_ids: list[str]) -> list[ScorerVersion]:
        """
        Batched ``list_scorers``: returns the latest-version scorer for every
        ``(experiment_id, scorer_name)`` pair across the given experiments in
        a single query plan per chunk. Used by the RBAC admin scorer picker
        to avoid N+1 round trips. ``experiment_id`` validation and
        active-status checks are skipped — the caller is expected to have
        already filtered.
        """
        if not experiment_ids:
            return []
        # ``experiment_id`` is an INTEGER column but REST callers pass string
        # IDs. psycopg2 sent them as untyped literals PostgreSQL would
        # coerce; psycopg v3 binds them as typed VARCHAR and PostgreSQL
        # rejects the comparison ("operator does not exist:
        # integer = character varying"). Coerce with the same error contract
        # as ``_get_experiment``.
        experiment_ids = _parse_experiment_ids(experiment_ids)
        with self.ManagedSessionMaker() as session:
            scorer_ids: list[str] = []
            for chunk_start in range(0, len(experiment_ids), self._ID_CHUNK_SIZE):
                chunk = experiment_ids[chunk_start : chunk_start + self._ID_CHUNK_SIZE]
                scorer_ids.extend(
                    row.scorer_id
                    for row in session
                    .query(SqlScorer.scorer_id)
                    .filter(SqlScorer.experiment_id.in_(chunk))
                    .all()
                )
            if not scorer_ids:
                return []
            # ``scorer_ids`` is also chunked for the same reason; build the
            # latest-version subquery + final query per chunk and concat.
            sql_scorer_versions: list[SqlScorerVersion] = []
            for chunk_start in range(0, len(scorer_ids), self._ID_CHUNK_SIZE):
                chunk = scorer_ids[chunk_start : chunk_start + self._ID_CHUNK_SIZE]
                latest_versions = (
                    session
                    .query(
                        SqlScorerVersion.scorer_id,
                        func.max(SqlScorerVersion.scorer_version).label("max_version"),
                    )
                    .filter(SqlScorerVersion.scorer_id.in_(chunk))
                    .group_by(SqlScorerVersion.scorer_id)
                    .subquery()
                )
                sql_scorer_versions.extend(
                    session
                    .query(SqlScorerVersion)
                    .join(
                        latest_versions,
                        (SqlScorerVersion.scorer_id == latest_versions.c.scorer_id)
                        & (SqlScorerVersion.scorer_version == latest_versions.c.max_version),
                    )
                    .join(SqlScorer, SqlScorerVersion.scorer_id == SqlScorer.scorer_id)
                    .order_by(SqlScorer.experiment_id, SqlScorer.scorer_name)
                    .all()
                )
            # Per-chunk order_by isn't globally stable across chunks, so
            # restore deterministic (experiment_id, scorer_name) ordering.
            sql_scorer_versions.sort(
                key=lambda sv: (sv.scorer.experiment_id, sv.scorer.scorer_name)
            )
            resolved_scorers = self._batch_resolve_endpoint_in_serialized_scorers([
                sv.serialized_scorer for sv in sql_scorer_versions
            ])
            return [
                ScorerVersion(
                    experiment_id=str(sv.scorer.experiment_id),
                    scorer_id=sv.scorer_id,
                    scorer_version=sv.scorer_version,
                    scorer_name=sv.scorer.scorer_name,
                    serialized_scorer=resolved_scorers[i],
                    creation_time=sv.creation_time,
                )
                for i, sv in enumerate(sql_scorer_versions)
            ]

    def filter_active_experiment_ids(self, experiment_ids: list[str]) -> list[str]:
        if not experiment_ids:
            return []
        parsed_ids = _parse_experiment_ids(experiment_ids)
        with self.ManagedSessionMaker() as session:
            active_ids: list[str] = []
            for chunk_start in range(0, len(parsed_ids), self._ID_CHUNK_SIZE):
                active_ids.extend(
                    str(experiment_id)
                    for experiment_id in self._filter_experiment_ids(
                        session,
                        parsed_ids[chunk_start : chunk_start + self._ID_CHUNK_SIZE],
                        lifecycle_stage=LifecycleStage.ACTIVE,
                    )
                )
            return active_ids

    def get_scorer(self, experiment_id, name, version=None) -> ScorerVersion:
        """
        Get a specific scorer for an experiment.

        Args:
            experiment_id: The experiment ID.
            name: The scorer name.
            version: The scorer version. If None, returns the scorer with
                maximum version.

        Returns:
            A ScorerVersion entity object with gateway endpoint IDs resolved to names.

        Raises:
            MlflowException: If scorer is not found.
        """
        with self.ManagedSessionMaker() as session:
            # Validate experiment exists and is active
            experiment = self.get_experiment(experiment_id)
            self._check_experiment_is_active(experiment)

            scorer = self._get_sql_scorer_exact(session, int(experiment.experiment_id), name)

            if scorer is None:
                raise MlflowException(
                    f"Scorer with name '{name}' not found for experiment {experiment_id}.",
                    RESOURCE_DOES_NOT_EXIST,
                )

            try:
                sql_scorer_version = self._get_scorer_version(
                    session=session,
                    scorer_id=scorer.scorer_id,
                    version=version,
                )
            except MlflowException as e:
                if e.error_code != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                    raise
                if version is not None:
                    raise MlflowException(
                        f"Scorer with name '{name}' and version {version} not found for "
                        f"experiment {experiment_id}.",
                        RESOURCE_DOES_NOT_EXIST,
                    ) from e
                raise MlflowException(
                    f"Scorer with name '{name}' not found for experiment {experiment_id}.",
                    RESOURCE_DOES_NOT_EXIST,
                ) from e

            entity = sql_scorer_version.to_mlflow_entity()
            # Resolve gateway endpoint ID to name before returning
            return self.resolve_endpoint_in_scorer(entity)

    def _get_scorer_version(
        self,
        session: Session,
        scorer_id: str,
        version: int | None = None,
    ) -> SqlScorerVersion:
        scorer = (
            self._get_query(session, SqlScorer).filter(SqlScorer.scorer_id == scorer_id).first()
        )
        if scorer is None:
            raise MlflowException(
                f"Scorer with ID '{scorer_id}' not found.",
                RESOURCE_DOES_NOT_EXIST,
            )

        query = session.query(SqlScorerVersion).filter(SqlScorerVersion.scorer_id == scorer_id)
        if version is not None:
            sql_scorer_version = query.filter(SqlScorerVersion.scorer_version == version).first()
            if sql_scorer_version is None:
                raise MlflowException(
                    f"Scorer with ID '{scorer_id}' and version {version} not found.",
                    RESOURCE_DOES_NOT_EXIST,
                )
        else:
            sql_scorer_version = query.order_by(SqlScorerVersion.scorer_version.desc()).first()
            if sql_scorer_version is None:
                raise MlflowException(
                    f"Scorer with ID '{scorer_id}' not found.",
                    RESOURCE_DOES_NOT_EXIST,
                )
        return sql_scorer_version

    def delete_scorer(self, experiment_id, name, version=None) -> None:
        """
        Delete a scorer for an experiment.

        Args:
            experiment_id: The experiment ID.
            name: The scorer name.
            version: The scorer version to delete. If None, deletes all versions.

        Raises:
            MlflowException: If scorer is not found.
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            # Validate experiment exists and is active
            experiment = self.get_experiment(experiment_id)
            self._check_experiment_is_active(experiment)

            scorer = self._get_sql_scorer_exact(session, int(experiment.experiment_id), name)

            if scorer is None:
                raise MlflowException(
                    f"Scorer with name '{name}' not found for experiment {experiment_id}.",
                    RESOURCE_DOES_NOT_EXIST,
                )

            # Build the query for scorer versions
            query = session.query(SqlScorerVersion).filter(
                SqlScorerVersion.scorer_id == scorer.scorer_id
            )

            # If version is specified, filter by version
            if version is not None:
                query = query.filter(SqlScorerVersion.scorer_version == version)

            sql_scorer_versions = query.all()

            if not sql_scorer_versions:
                if version is not None:
                    raise MlflowException(
                        f"Scorer with name '{name}' and version {version} not found for"
                        f" experiment {experiment_id}.",
                        RESOURCE_DOES_NOT_EXIST,
                    )
                else:
                    raise MlflowException(
                        f"Scorer with name '{name}' not found for experiment {experiment_id}.",
                        RESOURCE_DOES_NOT_EXIST,
                    )

            # Delete the scorer versions
            for sql_scorer_version in sql_scorer_versions:
                session.delete(sql_scorer_version)

            # If we're deleting all versions, also delete the scorer record
            # and clean up associated endpoint bindings
            if version is None:
                # Delete endpoint bindings for this scorer (resource_id stores scorer_id)
                session.query(SqlGatewayEndpointBinding).filter(
                    SqlGatewayEndpointBinding.resource_type == GatewayResourceType.SCORER.value,
                    SqlGatewayEndpointBinding.resource_id == scorer.scorer_id,
                ).delete()

                session.delete(scorer)

    def list_scorer_versions(self, experiment_id, name) -> list[ScorerVersion]:
        """
        List all versions of a specific scorer for an experiment.

        Args:
            experiment_id: The experiment ID.
            name: The scorer name.

        Returns:
            List of mlflow.entities.scorer.ScorerVersion objects for all versions of the scorer,
            with gateway endpoint IDs resolved to names.

        Raises:
            MlflowException: If scorer is not found.
        """
        with self.ManagedSessionMaker() as session:
            # Validate experiment exists and is active
            experiment = self.get_experiment(experiment_id)
            self._check_experiment_is_active(experiment)

            scorer = self._get_sql_scorer_exact(session, int(experiment.experiment_id), name)

            if scorer is None:
                raise MlflowException(
                    f"Scorer with name '{name}' not found for experiment {experiment_id}.",
                    RESOURCE_DOES_NOT_EXIST,
                )

            # Query for all versions of the scorer
            sql_scorer_versions = (
                session
                .query(SqlScorerVersion)
                .filter(SqlScorerVersion.scorer_id == scorer.scorer_id)
                .order_by(SqlScorerVersion.scorer_version.asc())
                .all()
            )

            if not sql_scorer_versions:
                raise MlflowException(
                    f"Scorer with name '{name}' not found for experiment {experiment_id}.",
                    RESOURCE_DOES_NOT_EXIST,
                )

            # Batch resolve gateway endpoint IDs to names
            resolved_scorers = self._batch_resolve_endpoint_in_serialized_scorers([
                sv.serialized_scorer for sv in sql_scorer_versions
            ])
            return [
                ScorerVersion(
                    experiment_id=str(sv.scorer.experiment_id),
                    scorer_id=sv.scorer_id,
                    scorer_version=sv.scorer_version,
                    scorer_name=sv.scorer.scorer_name,
                    serialized_scorer=resolved_scorers[i],
                    creation_time=sv.creation_time,
                )
                for i, sv in enumerate(sql_scorer_versions)
            ]

    def get_online_scoring_configs(self, scorer_ids: list[str]) -> list[OnlineScoringConfig]:
        """
        Get online scoring configurations for multiple scorers by their IDs.

        A single scorer can have multiple configurations (e.g., running in different
        experiments or with different filter strings).

        Args:
            scorer_ids: List of scorer IDs to fetch configurations for.

        Returns:
            A list of OnlineScoringConfig objects for the specified scorers.
            Scorers without configurations are not included.
        """
        if not scorer_ids:
            return []

        with self.ManagedSessionMaker() as session:
            results = (
                self
                ._get_query(session, SqlOnlineScoringConfig)
                .filter(SqlOnlineScoringConfig.scorer_id.in_(scorer_ids))
                .all()
            )

            return [config.to_mlflow_entity() for config in results]

    @staticmethod
    def _find_sql_scorer_by_collation(session, experiment_id: int, name: str) -> SqlScorer | None:
        """Return the experiment's scorer whose name equals ``name`` in SQL.

        MySQL and SQL Server compare strings without case by default (MySQL also without accents,
        and SQL Server without trailing spaces), so the scorer's name can differ from ``name``.
        Use ``_get_sql_scorer_exact`` to act on the scorer that ``name`` names.
        """
        return (
            session
            .query(SqlScorer)
            .filter(
                # ``Experiment.experiment_id`` (the entity field) is a
                # string while the column is INTEGER; psycopg v3 binds
                # strings as typed VARCHAR, which PostgreSQL rejects
                # ("operator does not exist: integer = character
                # varying"), so callers pass int(). Safe: the id of a
                # validated experiment is always a stringified integer.
                SqlScorer.experiment_id == experiment_id,
                SqlScorer.scorer_name == name,
            )
            .first()
        )

    @classmethod
    def _get_sql_scorer_exact(cls, session, experiment_id: int, name: str) -> SqlScorer | None:
        """Return the experiment's scorer named exactly ``name``, or None.

        Permission grants on a scorer are keyed by the name in the request, so a scorer that
        matches ``name`` only under the database's collation is not found here. Otherwise a
        request for ``JUDGE`` would act on ``judge`` without the grants on ``judge``. Do not
        replace this check with a case-insensitive comparison: collations also ignore accents.
        """
        scorer = cls._find_sql_scorer_by_collation(session, experiment_id, name)
        return scorer if scorer is not None and scorer.scorer_name == name else None

    @staticmethod
    def _validate_online_judge(scorer_name: str, serialized_scorer: str) -> None:
        serialized_data = json.loads(serialized_scorer)
        model = extract_model_from_serialized_scorer(serialized_data)
        if not is_gateway_model(model):
            raise MlflowException(
                f"Scorer '{scorer_name}' does not use a gateway model. "
                "Automatic evaluation is only supported for scorers that use "
                "gateway models.",
                INVALID_PARAMETER_VALUE,
            )

        # Check if scorer requires expectations (ground truth data)
        try:
            scorer_obj = Scorer.model_validate_json(serialized_scorer)
        except Exception:
            # Deserialization should not fail for valid registered scorers. If it does,
            # fail open (skip validation) to avoid blocking users in case this is an
            # internal issue.
            scorer_obj = None

        if (
            isinstance(scorer_obj, InstructionsJudge)
            and EXPECTATIONS_FIELD in scorer_obj.get_input_fields()
        ):
            raise MlflowException(
                f"Scorer '{scorer_name}' requires expectations, but scorers with "
                "expectations are not currently supported for automatic evaluation.",
                INVALID_PARAMETER_VALUE,
            )

    @staticmethod
    def _validate_online_custom_scorer(scorer_name: str, serialized_data: dict[str, Any]) -> None:
        if not container_scoring_enabled():
            raise MlflowException(
                f"Scorer '{scorer_name}' is a custom scorer, and online scoring runs custom "
                "scorers only in job containers. To run them, an operator must set the "
                "environment variable 'MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND' to 'docker'.",
                INVALID_PARAMETER_VALUE,
            )
        if custom_scorer_execution_blocked(serialized_data):
            raise MlflowException(
                f"Scorer '{scorer_name}' is a custom scorer, and custom scorers are disabled on "
                "this server. To run them, an operator must set the environment variable "
                "'MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS' to 'true'.",
                INVALID_PARAMETER_VALUE,
            )
        if has_empty_call_source(serialized_data):
            raise MlflowException(
                custom_scorer_without_source_error_message(scorer_name), INVALID_PARAMETER_VALUE
            )
        # Online scoring jobs run custom scorers in job containers one at a time, so a custom
        # scorer in an ensemble cannot run there (the jobs skip it).
        serialized = SerializedScorer.from_dict(serialized_data)
        if not _is_decorator_scorer(serialized):
            raise MlflowException(
                f"Scorer '{scorer_name}' is not a single custom scorer (for example, it is an "
                "ensemble that includes custom scorers). Automatic evaluation supports custom "
                "scorers only on their own.",
                INVALID_PARAMETER_VALUE,
            )
        if EXPECTATIONS_FIELD.name in _signature_param_names(serialized.call_signature):
            raise MlflowException(
                f"Scorer '{scorer_name}' requires expectations, but scorers with "
                "expectations are not currently supported for automatic evaluation.",
                INVALID_PARAMETER_VALUE,
            )

    def upsert_online_scoring_config(
        self,
        experiment_id: str,
        scorer_name: str,
        sample_rate: float,
        filter_string: str | None = None,
    ) -> OnlineScoringConfig:
        """
        Create or update online scoring configuration for a scorer.

        Args:
            experiment_id: The ID of the Experiment containing the scorer.
            scorer_name: The scorer name.
            sample_rate: The sampling rate (0.0 to 1.0).
            filter_string: Optional filter expression for trace selection.

        Returns:
            The created or updated OnlineScoringConfig entity.

        Raises:
            MlflowException: If sample_rate is not a number, if sample_rate is outside
                the range [0.0, 1.0], if filter_string is not a string, if the
                filter_string syntax is invalid, if the scorer is not found, or if the
                scorer does not use a gateway model. When online scoring runs custom scorers
                in job containers, a custom scorer is accepted instead, unless custom scorers
                are disabled on the server, it has no source code, it is not a single custom
                scorer (for example, an ensemble that includes one), or it requires
                expectations.
        """
        if not isinstance(sample_rate, (int, float)):
            raise MlflowException.invalid_parameter_value(
                f"sample_rate must be a number, got {type(sample_rate).__name__}"
            )
        if not 0.0 <= sample_rate <= 1.0:
            raise MlflowException.invalid_parameter_value(
                f"sample_rate must be between 0.0 and 1.0, got {sample_rate}"
            )
        if filter_string:
            if not isinstance(filter_string, str):
                raise MlflowException.invalid_parameter_value(
                    f"filter_string must be a string, got {type(filter_string).__name__}"
                )
            # Validate the filter string syntax before storing
            SearchTraceUtils.parse_search_filter_for_search_traces(filter_string)

        with self.ManagedSessionMaker(read_only=False) as session:
            experiment = self.get_experiment(experiment_id)
            self._check_experiment_is_active(experiment)

            scorer = self._get_sql_scorer_exact(session, int(experiment.experiment_id), scorer_name)
            if scorer is None:
                raise MlflowException(
                    f"Scorer with name '{scorer_name}' not found for experiment {experiment_id}.",
                    RESOURCE_DOES_NOT_EXIST,
                )

            # Get the latest scorer version to validate online scoring compatibility
            latest_version = (
                session
                .query(SqlScorerVersion)
                .filter(SqlScorerVersion.scorer_id == scorer.scorer_id)
                .order_by(SqlScorerVersion.scorer_version.desc())
                .first()
            )
            if latest_version is not None and sample_rate > 0:
                serialized_data = json.loads(latest_version.serialized_scorer)
                # Without job containers, a custom scorer gets the judge checks, so it keeps the
                # "does not use a gateway model" error on servers that do not run custom scorers.
                if (
                    _serialized_scorer_is_custom_code(serialized_data)
                    and container_scoring_enabled()
                ):
                    self._validate_online_custom_scorer(scorer_name, serialized_data)
                else:
                    self._validate_online_judge(scorer_name, latest_version.serialized_scorer)

            # Delete existing online configs for this scorer
            session.query(SqlOnlineScoringConfig).filter(
                SqlOnlineScoringConfig.scorer_id == scorer.scorer_id
            ).delete()

            # Create new online config
            config = SqlOnlineScoringConfig(
                online_scoring_config_id=uuid.uuid4().hex,
                scorer_id=scorer.scorer_id,
                sample_rate=sample_rate,
                experiment_id=_parse_experiment_id(experiment_id),
                filter_string=filter_string,
            )
            session.add(config)
            session.flush()

            return config.to_mlflow_entity()

    def get_active_online_scorers(self) -> list["OnlineScorer"]:
        """
        Get all active online scorers across all experiments.

        Active online scorers are those with a sample_rate greater than zero whose latest version
        online scoring can run. Gateway endpoint IDs in the serialized scorers are resolved to
        endpoint names.

        Returns:
            List of OnlineScorer entities with name, experiment_id, serialized_scorer,
            sample_rate, and filter_string fields populated.
        """
        with self.ManagedSessionMaker() as session:
            # Subquery to get the max version for each scorer
            max_version_subquery = (
                session
                .query(
                    SqlScorerVersion.scorer_id,
                    func.max(SqlScorerVersion.scorer_version).label("max_version"),
                )
                .group_by(SqlScorerVersion.scorer_id)
                .subquery()
            )

            # Get all online configs with sample_rate > 0, joined with their latest version
            results = (
                self
                ._get_query(session, SqlOnlineScoringConfig)
                .filter(SqlOnlineScoringConfig.sample_rate > 0)
                .join(SqlScorer, SqlOnlineScoringConfig.scorer_id == SqlScorer.scorer_id)
                .join(
                    max_version_subquery,
                    SqlScorer.scorer_id == max_version_subquery.c.scorer_id,
                )
                .join(
                    SqlScorerVersion,
                    and_(
                        SqlScorerVersion.scorer_id == max_version_subquery.c.scorer_id,
                        SqlScorerVersion.scorer_version == max_version_subquery.c.max_version,
                    ),
                )
                .with_entities(SqlOnlineScoringConfig, SqlScorer, SqlScorerVersion)
                .all()
            )

            # Filter to only include scorers whose max version uses a gateway model, or that are
            # custom scorers when online scoring runs them in job containers
            gateway_results = []
            for config, scorer, version in results:
                serialized_data = json.loads(version.serialized_scorer)
                model = extract_model_from_serialized_scorer(serialized_data)
                if is_gateway_model(model):
                    gateway_results.append((config, scorer, version))
                elif _serialized_scorer_is_custom_code(serialized_data):
                    # The latest version can be registered, or the server's settings changed,
                    # after the config was enabled, so the checks made then are made again.
                    try:
                        self._validate_online_custom_scorer(scorer.scorer_name, serialized_data)
                    except MlflowException as e:
                        _logger.warning(
                            f"Skipping online scorer '{scorer.scorer_name}' in experiment "
                            f"{config.experiment_id}: {e.message}"
                        )
                    else:
                        gateway_results.append((config, scorer, version))

            # Resolve gateway endpoint IDs to names
            return [
                OnlineScorer(
                    name=scorer.scorer_name,
                    serialized_scorer=self._resolve_endpoint_in_serialized_scorer(
                        version.serialized_scorer
                    ),
                    online_config=config.to_mlflow_entity(),
                    scorer_version=version.scorer_version,
                )
                for config, scorer, version in gateway_results
            ]

    def _resolve_endpoint_in_serialized_scorer(self, serialized_scorer: str) -> str:
        """
        Resolve gateway endpoint ID to name in a serialized scorer string.

        Args:
            serialized_scorer: Serialized scorer JSON string.

        Returns:
            Serialized scorer JSON string with resolved endpoint name.
        """
        serialized_data = json.loads(serialized_scorer)
        model = extract_model_from_serialized_scorer(serialized_data)

        if is_gateway_model(model):
            endpoint_id = extract_endpoint_ref(model)
            try:
                endpoint = self.get_gateway_endpoint(endpoint_id)
                new_model = build_gateway_model(endpoint.name)
                serialized_data = update_model_in_serialized_scorer(serialized_data, new_model)
            except MlflowException:
                # Endpoint not found - keep original serialized scorer
                pass

        return json.dumps(serialized_data)

    def _apply_order_by_search_logged_models(
        self,
        models: sqlalchemy.orm.Query,
        session: sqlalchemy.orm.Session,
        order_by: list[dict[str, Any]] | None = None,
        experiment_ids: list[int] | None = None,
        datasets: list[dict[str, Any]] | None = None,
    ) -> sqlalchemy.orm.Query:
        order_by_clauses = []
        datasets_filter = _get_logged_model_metric_datasets_filter(datasets)
        has_creation_timestamp = False
        has_model_id = False
        for ob in order_by or []:
            field_name = ob.get("field_name")
            ascending = ob.get("ascending", True)
            if "." not in field_name:
                name = SqlLoggedModel.ALIASES.get(field_name, field_name)
                if name == "creation_timestamp_ms":
                    has_creation_timestamp = True
                if name == "model_id":
                    has_model_id = True
                try:
                    col = getattr(SqlLoggedModel, name)
                except AttributeError:
                    # error_code is INVALID_PARAMETER_VALUE but this is an attribute lookup failure
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid order by field name: {field_name}",
                        error_class="ATTRIBUTE_NOT_FOUND",
                    )
                # Why not use `nulls_last`? Because it's not supported by all dialects (e.g., MySQL)
                order_by_clauses.extend([
                    # Sort nulls last
                    sqlalchemy.case((col.is_(None), 1), else_=0).asc(),
                    col.asc() if ascending else col.desc(),
                ])
                continue

            entity, name = field_name.split(".", 1)
            # TODO: Support filtering by other entities such as params if needed
            if entity != "metrics":
                raise MlflowException.invalid_parameter_value(
                    f"Invalid order by field name: {field_name}. Only metrics are supported."
                )

            dataset_filter = [] if datasets_filter is None else [datasets_filter]
            if dataset_name := ob.get("dataset_name"):
                dataset_filter.append(SqlLoggedModelMetric.dataset_name == dataset_name)
            if dataset_digest := ob.get("dataset_digest"):
                dataset_filter.append(SqlLoggedModelMetric.dataset_digest == dataset_digest)

            subquery = _get_latest_logged_model_metrics_subquery(
                session, name, dataset_filter, experiment_ids=experiment_ids
            )
            models = models.outerjoin(subquery)
            # Why not use `nulls_last`? Because it's not supported by all dialects (e.g., MySQL)
            order_by_clauses.extend([
                # Sort nulls last
                sqlalchemy.case((subquery.c.metric_value.is_(None), 1), else_=0).asc(),
                subquery.c.metric_value.asc() if ascending else subquery.c.metric_value.desc(),
            ])

        if not has_creation_timestamp:
            order_by_clauses.append(SqlLoggedModel.creation_timestamp_ms.desc())
        if not has_model_id:
            order_by_clauses.append(SqlLoggedModel.model_id.asc())

        return models.order_by(*order_by_clauses)

    def _apply_filter_string_datasets_search_logged_models(
        self,
        models: sqlalchemy.orm.Query,
        session: sqlalchemy.orm.Session,
        experiment_ids: list[str],
        filter_string: str | None,
        datasets: list[dict[str, Any]] | None,
    ):
        from mlflow.utils.search_logged_model_utils import EntityType, parse_filter_string

        comparisons = parse_filter_string(filter_string)
        experiment_ids = _parse_experiment_ids(experiment_ids)
        dialect = self._get_dialect()
        attr_filters: list[sqlalchemy.BinaryExpression] = []
        non_attr_filters: list[sqlalchemy.BinaryExpression] = []

        datasets_filter = _get_logged_model_metric_datasets_filter(datasets)

        has_metric_filters = False
        for comp in comparisons:
            comp_func = SearchUtils.get_sql_comparison_func(comp.op, dialect)
            if comp.entity.type == EntityType.ATTRIBUTE:
                value = comp.value
                if comp.entity.key == "status":
                    if comp.op not in ("=", "!=", "IN", "NOT IN"):
                        raise MlflowException.invalid_parameter_value(
                            f"Invalid comparison operator for status: {comp.op}"
                        )
                    values = value if comp.op in ("IN", "NOT IN") else (value,)
                    try:
                        statuses = [LoggedModelStatus(status).to_int() for status in values]
                    except ValueError as e:
                        raise MlflowException.invalid_parameter_value(
                            f"Unknown model status in filter: {value!r}"
                        ) from e
                    value = statuses if comp.op in ("IN", "NOT IN") else statuses[0]
                attr_filters.append(comp_func(getattr(SqlLoggedModel, comp.entity.key), value))
            elif comp.entity.type == EntityType.METRIC:
                has_metric_filters = True
                # Compare against the same latest value that `order_by` uses rather than
                # against every logged step. The subquery has one row per model, so joining
                # it cannot duplicate a model before OFFSET/LIMIT is applied.
                latest_metrics = _get_latest_logged_model_metrics_subquery(
                    session,
                    comp.entity.key,
                    [] if datasets_filter is None else [datasets_filter],
                    experiment_ids=experiment_ids,
                )
                non_attr_filters.append(
                    select(latest_metrics.c.model_id)
                    .where(comp_func(latest_metrics.c.metric_value, comp.value))
                    .subquery()
                )
            elif comp.entity.type == EntityType.PARAM:
                non_attr_filters.append(
                    session
                    .query(SqlLoggedModelParam)
                    .filter(
                        SqlLoggedModelParam.param_key == comp.entity.key,
                        comp_func(SqlLoggedModelParam.param_value, comp.value),
                    )
                    .subquery()
                )
            elif comp.entity.type == EntityType.TAG:
                non_attr_filters.append(
                    session
                    .query(SqlLoggedModelTag)
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
        if datasets_filter is not None and not has_metric_filters:
            subquery = (
                session
                .query(SqlLoggedModelMetric.model_id)
                .filter(datasets_filter)
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
        filter_string: str | None = None,
        datasets: list[DatasetFilter] | None = None,
        max_results: int | None = None,
        order_by: list[dict[str, Any]] | None = None,
        page_token: str | None = None,
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
            models = self._get_query(session, SqlLoggedModel)
            models = self._apply_filter_string_datasets_search_logged_models(
                models, session, experiment_ids, filter_string, datasets
            )
            models = self._apply_order_by_search_logged_models(
                models,
                session,
                order_by,
                experiment_ids=_parse_experiment_ids(experiment_ids),
                datasets=datasets,
            )
            models = (
                models
                .options(
                    # Eagerly load the relationships read by `to_mlflow_entity` so that a page
                    # of results costs a fixed number of queries instead of three per model.
                    # Use a select in load rather than a joined load to limit memory overhead.
                    selectinload(SqlLoggedModel.tags),
                    selectinload(SqlLoggedModel.params),
                    selectinload(SqlLoggedModel.metrics),
                )
                .offset(offset)
                .limit(max_results + 1)
                .all()
            )

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

    def _run_with_deadlock_retry(self, fn, *args, **kwargs):
        """Run a DB-write operation, retrying on DB deadlocks.

        The managed session (see ``mlflow/store/db/utils.py``) surfaces a psycopg2
        DeadlockDetected / SQLAlchemy OperationalError as an ``MlflowException`` with
        ``error_code == TEMPORARILY_UNAVAILABLE``. We gate on that error code AND the
        ``deadlock`` substring so only deadlocks are retried and other transient errors are
        not masked. Each attempt re-invokes ``fn`` from scratch (opening a fresh managed
        session) rather than retrying on a rolled-back session.
        """
        temporarily_unavailable = ErrorCode.Name(TEMPORARILY_UNAVAILABLE)
        for attempt in range(_DB_WRITE_MAX_DEADLOCK_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except MlflowException as e:
                is_deadlock = (
                    e.error_code == temporarily_unavailable and "deadlock" in str(e).lower()
                )
                if not is_deadlock or attempt >= _DB_WRITE_MAX_DEADLOCK_RETRIES:
                    raise
                # Exponential backoff with jitter, matching `_try_insert_tags`.
                sleep_duration = (2**attempt) - 1
                sleep_duration += random.uniform(0, 1)
                time.sleep(sleep_duration)

    def _validate_max_results_param(self, max_results: int, allow_null=False):
        if (not allow_null and max_results is None) or (
            max_results is not None and max_results < 1
        ):
            raise MlflowException(
                f"Invalid value {max_results} for parameter 'max_results' supplied. It must be "
                f"a positive integer",
                INVALID_PARAMETER_VALUE,
            )

        if max_results is not None and max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                f"Invalid value {max_results} for parameter 'max_results' supplied. It must be at "
                f"most {SEARCH_MAX_RESULTS_THRESHOLD}",
                INVALID_PARAMETER_VALUE,
            )

    def _delete_review_queue_items_for_traces(self, session: Session, trace_ids: list[str]) -> None:
        """Remove review-queue items pointing at traces that are being deleted.

        ``review_queue_items.item_id`` has no foreign key into ``trace_info`` (an
        item can reference a trace, session, or span), so deleting a trace does
        not cascade to its queue items. Without this cleanup the item lingers in
        its queue and any review submitted for it fails on the assessments
        foreign key (surfacing the raw SQL error to the reviewer).
        """
        # Lazy import: importing `mlflow.genai.review_queues` triggers the `mlflow.genai`
        # package init, which can pull this module back in (see `dbmodels/models.py`).
        from mlflow.genai.review_queues import ReviewItemType

        if not trace_ids:
            return
        queues = self._get_query(session, SqlReviewQueue).with_entities(SqlReviewQueue.queue_id)
        (
            session
            .query(SqlReviewQueueItem)
            .filter(
                SqlReviewQueueItem.queue_id.in_(queues),
                SqlReviewQueueItem.item_type == ReviewItemType.TRACE.value,
                SqlReviewQueueItem.item_id.in_(trace_ids),
            )
            .delete(synchronize_session=False)
        )

    def link_traces_to_run(self, trace_ids: list[str], run_id: str) -> None:
        """
        Link multiple traces to a run by creating entity associations.

        Args:
            trace_ids: List of trace IDs to link to the run. Maximum 100 traces allowed.
            run_id: ID of the run to link traces to.

        Raises:
            MlflowException: If more than 100 traces are provided.
        """
        if not trace_ids:
            return

        if len(trace_ids) > MAX_TRACE_LINKS_PER_REQUEST:
            raise MlflowException(
                f"Cannot link more than {MAX_TRACE_LINKS_PER_REQUEST} traces to a run in "
                f"a single request. Provided {len(trace_ids)} traces.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        with self.ManagedSessionMaker(read_only=False) as session:
            if self._has_external_trace_store:
                self._get_run(session, run_id)
            else:
                self._validate_run_accessible(session, run_id)

            trace_ids = self._filter_entity_ids(
                session, EntityAssociationType.TRACE, list(trace_ids)
            )

            existing_associations = (
                session
                .query(SqlEntityAssociation)
                .filter(
                    SqlEntityAssociation.source_type == EntityAssociationType.TRACE,
                    SqlEntityAssociation.source_id.in_(trace_ids),
                    SqlEntityAssociation.destination_type == EntityAssociationType.RUN,
                    SqlEntityAssociation.destination_id == run_id,
                )
                .all()
            )
            existing_trace_ids = [association.source_id for association in existing_associations]

            trace_ids_to_add = [
                trace_id for trace_id in trace_ids if trace_id not in existing_trace_ids
            ]

            session.add_all(
                SqlEntityAssociation(
                    association_id=uuid.uuid4().hex,
                    source_type=EntityAssociationType.TRACE,
                    source_id=trace_id,
                    destination_type=EntityAssociationType.RUN,
                    destination_id=run_id,
                )
                for trace_id in trace_ids_to_add
            )

    def unlink_traces_from_run(self, trace_ids: list[str], run_id: str) -> None:
        if not trace_ids:
            return
        with self.ManagedSessionMaker(read_only=False) as session:
            self._get_run(session, run_id)
            trace_ids = self._filter_entity_ids(
                session, EntityAssociationType.TRACE, list(trace_ids)
            )
            (
                session
                .query(SqlEntityAssociation)
                .filter(
                    SqlEntityAssociation.source_type == EntityAssociationType.TRACE,
                    SqlEntityAssociation.source_id.in_(trace_ids),
                    SqlEntityAssociation.destination_type == EntityAssociationType.RUN,
                    SqlEntityAssociation.destination_id == run_id,
                )
                .delete(synchronize_session=False)
            )

    def link_prompts_to_trace(self, trace_id: str, prompt_versions: list[PromptVersion]) -> None:
        """
        Link multiple prompt versions to a trace by creating entity associations.

        Args:
            trace_id: ID of the trace to link prompt versions to.
            prompt_versions: List of PromptVersion objects to link.
        """
        if not prompt_versions:
            return

        with self.ManagedSessionMaker(read_only=False) as session:
            if self._has_external_trace_store:
                trace_exists = bool(
                    self._filter_entity_ids(session, EntityAssociationType.TRACE, [trace_id])
                )
            else:
                trace_exists = (
                    self._default_trace_store
                    ._trace_query(session)
                    .with_entities(SqlTraceInfo.request_id)
                    .filter(SqlTraceInfo.request_id == trace_id)
                    .first()
                )
            if not trace_exists:
                raise MlflowException(
                    f"Trace with ID '{trace_id}' not found.",
                    RESOURCE_DOES_NOT_EXIST,
                )

            # Build list of prompt version IDs (format: "name/version")
            prompt_ids = [f"{pv.name}/{pv.version}" for pv in prompt_versions]

            # Check for existing associations
            existing_associations = (
                session
                .query(SqlEntityAssociation)
                .filter(
                    SqlEntityAssociation.source_type == EntityAssociationType.TRACE,
                    SqlEntityAssociation.source_id == trace_id,
                    SqlEntityAssociation.destination_type == EntityAssociationType.PROMPT_VERSION,
                    SqlEntityAssociation.destination_id.in_(prompt_ids),
                )
                .all()
            )
            existing_prompt_ids = {
                association.destination_id for association in existing_associations
            }

            prompt_ids_to_add = [pid for pid in prompt_ids if pid not in existing_prompt_ids]

            session.add_all(
                SqlEntityAssociation(
                    association_id=uuid.uuid4().hex,
                    source_type=EntityAssociationType.TRACE,
                    source_id=trace_id,
                    destination_type=EntityAssociationType.PROMPT_VERSION,
                    destination_id=prompt_id,
                )
                for prompt_id in prompt_ids_to_add
            )

    #######################################################################################
    # Entity Association Methods
    #######################################################################################

    def _search_entity_associations(
        self,
        entity_ids: str | list[str],
        entity_type: EntityAssociationType,
        target_type: EntityAssociationType,
        search_direction: str,  # "forward" or "reverse"
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList[str]:
        """
        Common implementation for searching entity associations.

        Args:
            entity_ids: The ID(s) of the entity to search from. Can be a single ID or a list.
            entity_type: The type of the entity to search from.
            target_type: The type of the target entities to find.
            search_direction: "forward" to search source->destination, "reverse" for the opposite.
            max_results: Maximum number of results to return. If None, return all results.
            page_token: Token indicating the page of results to fetch.

        Returns:
            A :py:class:`mlflow.store.entities.paged_list.PagedList` of target entity IDs.
        """
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]

        with self.ManagedSessionMaker() as session:
            if (
                self._has_external_trace_store
                and entity_type == EntityAssociationType.PROMPT_VERSION
                and target_type == EntityAssociationType.TRACE
            ):
                pass  # Registry-owned prompt identities are scoped through visible trace results.
            else:
                entity_ids = self._filter_entity_ids(session, entity_type, entity_ids)
            query = session.query(SqlEntityAssociation)

            if search_direction == "forward":
                query = query.filter(
                    SqlEntityAssociation.source_type == entity_type,
                    SqlEntityAssociation.source_id.in_(entity_ids),
                    SqlEntityAssociation.destination_type == target_type,
                )
                order_field = SqlEntityAssociation.destination_id
                result_field = "destination_id"
            else:
                query = query.filter(
                    SqlEntityAssociation.destination_type == entity_type,
                    SqlEntityAssociation.destination_id.in_(entity_ids),
                    SqlEntityAssociation.source_type == target_type,
                )
                order_field = SqlEntityAssociation.source_id
                result_field = "source_id"

            if target_type == EntityAssociationType.TRACE and self._has_external_trace_store:
                first_association_time = func.min(SqlEntityAssociation.created_time)
                return self._search_backend_trace_associations(
                    session,
                    query
                    .with_entities(order_field.label(result_field))
                    .group_by(order_field)
                    .order_by(first_association_time, order_field),
                    result_field,
                    max_results,
                    page_token,
                )
            else:
                query = self._filter_association_query(session, query, target_type, order_field)

            offset = SearchUtils.parse_start_offset_from_page_token(page_token)

            query = query.order_by(SqlEntityAssociation.created_time, order_field)
            query = query.offset(offset)

            if max_results is not None:
                query = query.limit(max_results + 1)

            associations = query.all()

            next_token = None
            if max_results is not None and len(associations) == max_results + 1:
                final_offset = offset + max_results
                next_token = SearchUtils.create_page_token(final_offset)
                associations = associations[:max_results]

            results = list(dict.fromkeys([getattr(assoc, result_field) for assoc in associations]))
            return PagedList(results, next_token)

    def _search_backend_trace_associations(
        self, session, query, result_field: str, max_results: int | None, page_token: str | None
    ) -> PagedList[str]:
        self._validate_max_results_param(max_results, allow_null=True)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        results = []
        next_offset = offset
        while True:
            rows = query.offset(offset).limit(500).all()
            visible = set(
                self._filter_entity_ids(
                    session,
                    EntityAssociationType.TRACE,
                    list({getattr(row, result_field) for row in rows}),
                )
            )
            for index, row in enumerate(rows):
                trace_id = getattr(row, result_field)
                if trace_id in visible:
                    if max_results is not None and len(results) == max_results:
                        return PagedList(results, SearchUtils.create_page_token(next_offset))
                    results.append(trace_id)
                    next_offset = offset + index + 1
            if len(rows) < 500:
                return PagedList(results, None)
            offset += len(rows)

    def search_entities_by_source(
        self,
        source_ids: str | list[str],
        source_type: EntityAssociationType,
        destination_type: EntityAssociationType,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList[str]:
        """
        Get destination IDs associated with source entity/entities.

        Args:
            source_ids: The ID(s) of the source entity. Can be a single ID or a list.
            source_type: The type of the source entity.
            destination_type: The type of the destination entity.
            max_results: Maximum number of results to return. If None, return all results.
            page_token: Token indicating the page of results to fetch.

        Returns:
            A :py:class:`mlflow.store.entities.paged_list.PagedList` of destination IDs
            associated with the source entity/entities.
        """
        return self._search_entity_associations(
            source_ids, source_type, destination_type, "forward", max_results, page_token
        )

    def search_entities_by_destination(
        self,
        destination_ids: str | list[str],
        destination_type: EntityAssociationType,
        source_type: EntityAssociationType,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList[str]:
        """
        Get source IDs associated with destination entity/entities.

        Args:
            destination_ids: The ID(s) of the destination entity. Can be a single ID or a list.
            destination_type: The type of the destination entity.
            source_type: The type of the source entity.
            max_results: Maximum number of results to return. If None, return all results.
            page_token: Token indicating the page of results to fetch.

        Returns:
            A :py:class:`mlflow.store.entities.paged_list.PagedList` of source IDs
            associated with the destination entity/entities.
        """
        return self._search_entity_associations(
            destination_ids, destination_type, source_type, "reverse", max_results, page_token
        )

    #######################################################################################
    # Evaluation Dataset Methods
    #######################################################################################

    def _compute_dataset_digest(self, name: str, last_update_time: int) -> str:
        """
        Compute digest for an evaluation dataset.

        The digest includes the dataset name and last_update_time to ensure
        that any state change results in a different digest.

        Args:
            name: Dataset name
            last_update_time: Last update timestamp in milliseconds

        Returns:
            8-character digest string
        """
        digest_input = f"{name}:{last_update_time}".encode()
        return hashlib.sha256(digest_input).hexdigest()[:8]

    def create_dataset(
        self,
        name: str,
        tags: dict[str, str] | None = None,
        experiment_ids: list[str] | None = None,
    ) -> EvaluationDataset:
        """
        Create a new evaluation dataset in the database.

        Args:
            name: The name of the evaluation dataset.
            tags: Optional tags to associate with the dataset.
            experiment_ids: List of experiment IDs to associate with the dataset

        Returns:
            The created EvaluationDataset object with backend-generated metadata.
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            dataset_id = f"{self.EVALUATION_DATASET_ID_PREFIX}{uuid.uuid4().hex}"

            current_time = get_current_time_millis()
            digest = self._compute_dataset_digest(name, current_time)

            user_id = None
            if tags and MLFLOW_USER in tags:
                user_id = tags[MLFLOW_USER]

            created_dataset = EvaluationDataset(
                dataset_id=dataset_id,
                name=name,
                digest=digest,
                created_time=current_time,
                last_update_time=current_time,
                tags=tags or {},
                schema=None,  # Schema is computed when data is added
                profile=None,  # Profile is computed when data is added
                created_by=user_id,
                last_updated_by=user_id,
            )

            sql_dataset = self._with_workspace_field(
                SqlEvaluationDataset.from_mlflow_entity(created_dataset)
            )
            session.add(sql_dataset)

            if created_dataset.tags:
                for key, value in created_dataset.tags.items():
                    tag = SqlEvaluationDatasetTag(
                        dataset_id=dataset_id,
                        key=key,
                        value=value,
                    )
                    session.add(tag)

            if experiment_ids:
                for exp_id in experiment_ids:
                    association = SqlEntityAssociation(
                        source_type=EntityAssociationType.EVALUATION_DATASET,
                        source_id=dataset_id,
                        destination_type=EntityAssociationType.EXPERIMENT,
                        destination_id=str(exp_id),
                        created_time=current_time,
                    )
                    session.add(association)

            sql_dataset_with_tags = (
                self
                ._dataset_query(session)
                .filter(SqlEvaluationDataset.dataset_id == dataset_id)
                .one()
            )

            created_dataset = sql_dataset_with_tags.to_mlflow_entity()
            created_dataset.experiment_ids = experiment_ids or []
            created_dataset._tracking_store = self

            return created_dataset

    def get_dataset(self, dataset_id: str) -> EvaluationDataset:
        """
        Get an evaluation dataset by ID.

        Args:
            dataset_id: The ID of the dataset to retrieve.

        Returns:
            The EvaluationDataset object (without records or experiment_ids - lazy loading).
        """
        with self.ManagedSessionMaker() as session:
            sql_dataset = (
                self
                ._dataset_query(session)
                .filter(SqlEvaluationDataset.dataset_id == dataset_id)
                .one_or_none()
            )

            if sql_dataset is None:
                raise MlflowException(
                    f"Evaluation dataset with id '{dataset_id}' not found",
                    RESOURCE_DOES_NOT_EXIST,
                )

            return sql_dataset.to_mlflow_entity()

    def delete_dataset(self, dataset_id: str) -> None:
        """
        Delete an evaluation dataset and all its records.

        Args:
            dataset_id: The ID of the dataset to delete.
        """

        with self.ManagedSessionMaker(read_only=False) as session:
            sql_dataset = (
                self
                ._dataset_query(session)
                .filter(SqlEvaluationDataset.dataset_id == dataset_id)
                .one_or_none()
            )

            if sql_dataset is None:
                _logger.warning(f"Evaluation dataset with id '{dataset_id}' not found.")
                return

            session.query(SqlEntityAssociation).filter(
                or_(
                    and_(
                        SqlEntityAssociation.destination_type
                        == EntityAssociationType.EVALUATION_DATASET,
                        SqlEntityAssociation.destination_id == dataset_id,
                    ),
                    and_(
                        SqlEntityAssociation.source_type
                        == EntityAssociationType.EVALUATION_DATASET,
                        SqlEntityAssociation.source_id == dataset_id,
                    ),
                )
            ).delete()

            session.delete(sql_dataset)

    def search_datasets(
        self,
        experiment_ids: list[str] | None = None,
        filter_string: str | None = None,
        max_results: int = 1000,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[EvaluationDataset]:
        """
        Search for evaluation datasets.

        Args:
            experiment_ids: Filter by associated experiment IDs.
            filter_string: SQL-like filter string.
            max_results: Maximum number of results to return.
            order_by: List of fields to order by.
            page_token: Token for pagination.

        Returns:
            PagedList of EvaluationDataset objects (without records).
        """
        self._validate_max_results_param(max_results)

        offset = SearchUtils.parse_start_offset_from_page_token(page_token)

        with self.ManagedSessionMaker() as session:
            if filter_string:
                parsed_filters = SearchEvaluationDatasetsUtils.parse_search_filter(filter_string)
                attribute_filters, non_attribute_filters = _get_search_datasets_filter_clauses(
                    parsed_filters, self._get_dialect()
                )
            else:
                attribute_filters = []
                non_attribute_filters = []

            query = self._dataset_query(session)
            for f in non_attribute_filters:
                query = query.join(f)

            if experiment_ids:
                dataset_ids_result = self.search_entities_by_destination(
                    destination_ids=experiment_ids,
                    destination_type=EntityAssociationType.EXPERIMENT,
                    source_type=EntityAssociationType.EVALUATION_DATASET,
                )
                dataset_ids = dataset_ids_result.to_list()
                query = query.filter(SqlEvaluationDataset.dataset_id.in_(dataset_ids))

            query = query.filter(*attribute_filters)

            order_by_clauses = _get_search_datasets_order_by_clauses(order_by)
            query = query.order_by(*order_by_clauses)

            query = query.offset(offset).limit(max_results + 1)

            sql_datasets = query.all()

            next_page_token = None
            if len(sql_datasets) > max_results:
                sql_datasets = sql_datasets[:max_results]
                next_page_token = SearchUtils.create_page_token(offset + max_results)

            datasets = [sql_dataset.to_mlflow_entity() for sql_dataset in sql_datasets]

            return PagedList(datasets, next_page_token)

    def _update_dataset_schema(self, existing_schema_json, record_dicts):
        """
        Update dataset schema with new fields from records.
        This method combines schema computation and merging into a single operation
        for efficiency, since schemas are stored as JSON strings.

        Args:
            existing_schema_json: JSON string of existing schema or None
            record_dicts: List of record dictionaries being upserted

        Returns:
            Updated schema dictionary or None if no records and no existing schema
        """
        if not record_dicts and not existing_schema_json:
            return None

        schema = (
            json.loads(existing_schema_json)
            if existing_schema_json
            else {"inputs": {}, "outputs": {}, "expectations": {}, "version": "1.0"}
        )

        for record in record_dicts:
            if inputs := record.get("inputs"):
                for key, value in inputs.items():
                    if key not in schema["inputs"]:
                        schema["inputs"][key] = self._infer_field_type(value)

            if (outputs := record.get("outputs")) is not None:
                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        if key not in schema["outputs"]:
                            schema["outputs"][key] = self._infer_field_type(value)
                else:
                    if not schema["outputs"]:
                        schema["outputs"] = self._infer_field_type(outputs)

            if expectations := record.get("expectations"):
                for key, value in expectations.items():
                    if key not in schema["expectations"]:
                        schema["expectations"][key] = self._infer_field_type(value)

        return schema

    def _compute_dataset_profile(self, session, dataset_id):
        """
        Compute profile statistics for the dataset based on current state.

        Args:
            session: Database session
            dataset_id: ID of the dataset

        Returns:
            Profile dictionary with current statistics
        """
        total_records = (
            session
            .query(SqlEvaluationDatasetRecord)
            .filter(SqlEvaluationDatasetRecord.dataset_id == dataset_id)
            .count()
        )

        if total_records == 0:
            return None

        return {"num_records": total_records}

    def _infer_field_type(self, value):
        """
        Infer the type of a field value.

        Returns a string representation of the type.
        """
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "unknown"

    def _load_dataset_records(
        self,
        dataset_id: str,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> tuple[list[DatasetRecord], str | None]:
        """
        Load dataset records with cursor-based pagination support.

        Records are ordered by (created_time, dataset_record_id) to ensure deterministic
        pagination across all database backends.

        Args:
            dataset_id: The ID of the dataset.
            max_results: Maximum number of records to return. Defaults to
                LOAD_DATASET_RECORDS_MAX_RESULTS. If explicitly set to None, returns all records.
            page_token: Cursor token for pagination in format "created_time:record_id".
                If None, starts from the beginning.

        Returns:
            Tuple of (list of DatasetRecord objects, next_page_token).
            next_page_token is None if there are no more records.
        """
        from mlflow.store.tracking import LOAD_DATASET_RECORDS_MAX_RESULTS

        # Use default if not explicitly set
        if max_results is None:
            effective_max_results = None  # Return all records for internal use
        else:
            effective_max_results = max_results or LOAD_DATASET_RECORDS_MAX_RESULTS

        with self.ManagedSessionMaker() as session:
            self._validate_dataset_accessible(session, dataset_id)

            query = (
                session
                .query(SqlEvaluationDatasetRecord)
                .filter(SqlEvaluationDatasetRecord.dataset_id == dataset_id)
                .order_by(
                    SqlEvaluationDatasetRecord.created_time,
                    SqlEvaluationDatasetRecord.dataset_record_id,
                )
            )

            if page_token:
                try:
                    decoded = base64.b64decode(page_token.encode()).decode()
                    last_created_time, last_record_id = decoded.split(":", 1)
                    last_created_time = int(last_created_time)

                    query = query.filter(
                        (SqlEvaluationDatasetRecord.created_time > last_created_time)
                        | (
                            (SqlEvaluationDatasetRecord.created_time == last_created_time)
                            & (SqlEvaluationDatasetRecord.dataset_record_id > last_record_id)
                        )
                    )
                except (ValueError, AttributeError):
                    offset = int(page_token)
                    query = query.offset(offset)

            if effective_max_results is not None:
                sql_records = query.limit(effective_max_results + 1).all()

                if len(sql_records) > effective_max_results:
                    sql_records = sql_records[:effective_max_results]
                    last_record = sql_records[-1]
                    cursor = f"{last_record.created_time}:{last_record.dataset_record_id}"
                    next_page_token = base64.b64encode(cursor.encode()).decode()
                else:
                    next_page_token = None
            else:
                sql_records = query.all()
                next_page_token = None

            records = [record.to_mlflow_entity() for record in sql_records]
            return records, next_page_token

    def delete_dataset_tag(self, dataset_id: str, key: str) -> None:
        """
        Delete a tag from an evaluation dataset.

        This operation is idempotent - if the tag doesn't exist, it's a no-op.

        Args:
            dataset_id: The ID of the dataset.
            key: The tag key to delete.
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            dataset = self._dataset_query(session).filter_by(dataset_id=dataset_id).first()
            if not dataset:
                _logger.debug(
                    f"Dataset {dataset_id} not found. "
                    "It may have been deleted or is not accessible."
                )
                return

            deleted_count = (
                session
                .query(SqlEvaluationDatasetTag)
                .filter_by(dataset_id=dataset_id, key=key)
                .delete()
            )

            if deleted_count == 0:
                _logger.debug(
                    f"Tag '{key}' not found for evaluation dataset {dataset_id}. "
                    "It may have already been deleted or never existed."
                )

    def upsert_dataset_records(
        self, dataset_id: str, records: list[dict[str, Any]]
    ) -> dict[str, int]:
        """
        Bulk upsert records with input-based deduplication.

        Args:
            dataset_id: The ID of the dataset.
            records: List of record dictionaries.

        Returns:
            Dictionary with counts of inserted and updated records.
        """

        with self.ManagedSessionMaker(read_only=False) as session:
            self._validate_dataset_accessible(session, dataset_id)

            inserted_count = 0
            updated_count = 0
            current_time = get_current_time_millis()
            updated_by = None  # Track who last updated the dataset

            for record_dict in records:
                inputs_json = json.dumps(record_dict.get("inputs", {}), sort_keys=True)
                input_hash = hashlib.sha256(inputs_json.encode()).hexdigest()

                existing_record = (
                    session
                    .query(SqlEvaluationDatasetRecord)
                    .filter(
                        SqlEvaluationDatasetRecord.dataset_id == dataset_id,
                        SqlEvaluationDatasetRecord.input_hash == input_hash,
                    )
                    .one_or_none()
                )

                tags = record_dict.get("tags")
                if tags and MLFLOW_USER in tags:
                    updated_by = tags[MLFLOW_USER]

                if existing_record:
                    existing_record.merge(record_dict)
                    updated_count += 1
                else:
                    created_by = None
                    if tags and MLFLOW_USER in tags:
                        created_by = tags[MLFLOW_USER]

                    source = None
                    if source_data := record_dict.get("source"):
                        if isinstance(source_data, dict):
                            source = DatasetRecordSource.from_dict(source_data)
                        else:
                            source = source_data

                    record = DatasetRecord(
                        dataset_record_id=None,
                        dataset_id=dataset_id,
                        inputs=record_dict.get("inputs", {}),
                        outputs=record_dict.get("outputs", {}),
                        created_time=current_time,
                        last_update_time=current_time,
                        expectations=record_dict.get("expectations"),
                        tags=tags,
                        source=source,
                        created_by=created_by,
                        last_updated_by=created_by,
                    )

                    sql_record = SqlEvaluationDatasetRecord.from_mlflow_entity(record, input_hash)
                    session.add(sql_record)
                    inserted_count += 1

            dataset_info = (
                self
                ._dataset_query(session)
                .with_entities(SqlEvaluationDataset.schema, SqlEvaluationDataset.name)
                .filter(SqlEvaluationDataset.dataset_id == dataset_id)
                .first()
            )

            if dataset_info:
                existing_schema = dataset_info[0]
                dataset_name = dataset_info[1]
            else:
                existing_schema = None
                dataset_name = None

            updated_schema = self._update_dataset_schema(existing_schema, records)

            updated_profile = self._compute_dataset_profile(session, dataset_id)

            new_digest = self._compute_dataset_digest(dataset_name, current_time)

            update_fields = {
                "last_update_time": current_time,
                "last_updated_by": updated_by,
                "digest": new_digest,
            }

            if updated_schema:
                update_fields["schema"] = json.dumps(updated_schema)

            if updated_profile:
                update_fields["profile"] = json.dumps(updated_profile)

            self._dataset_query(session).filter(
                SqlEvaluationDataset.dataset_id == dataset_id
            ).update(update_fields)

            return {"inserted": inserted_count, "updated": updated_count}

    def delete_dataset_records(self, dataset_id: str, dataset_record_ids: list[str]) -> int:
        """
        Delete records from an evaluation dataset.

        Args:
            dataset_id: The ID of the dataset.
            dataset_record_ids: List of record IDs to delete.

        Returns:
            The number of records deleted.
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            self._validate_dataset_accessible(session, dataset_id)

            deleted_count = (
                session
                .query(SqlEvaluationDatasetRecord)
                .filter(
                    SqlEvaluationDatasetRecord.dataset_id == dataset_id,
                    SqlEvaluationDatasetRecord.dataset_record_id.in_(dataset_record_ids),
                )
                .delete(synchronize_session=False)
            )

            if deleted_count == 0:
                _logger.warning(
                    f"No records found to delete for dataset {dataset_id}. "
                    "Records may have already been deleted or never existed."
                )
                return 0

            dataset = (
                self
                ._get_query(session, SqlEvaluationDataset)
                .filter(SqlEvaluationDataset.dataset_id == dataset_id)
                .first()
            )
            if dataset:
                profile = json.loads(dataset.profile) if dataset.profile else {}
                new_count = max(0, profile.get("num_records", 0) - deleted_count)
                dataset.profile = json.dumps({"num_records": new_count} if new_count > 0 else None)

            return deleted_count

    def get_dataset_experiment_ids(self, dataset_id: str) -> list[str]:
        """
        Get experiment IDs associated with an evaluation dataset.
        This method is used for lazy loading of experiment_ids in the EvaluationDataset entity.

        Args:
            dataset_id: The ID of the dataset.

        Returns:
            List of experiment IDs associated with the dataset.
        """
        experiment_ids = self.search_entities_by_source(
            source_ids=dataset_id,
            source_type=EntityAssociationType.EVALUATION_DATASET,
            destination_type=EntityAssociationType.EXPERIMENT,
        )

        return experiment_ids.to_list()

    def set_dataset_tags(self, dataset_id: str, tags: dict[str, Any]) -> None:
        """
        Update tags for an evaluation dataset.
        This implements an upsert operation - existing tags are merged with new tags.

        Args:
            dataset_id: The ID of the dataset to update.
            tags: Dictionary of tags to update.

        Raises:
            MlflowException: If the dataset doesn't exist.
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            # NB: Checking that the dataset exists within this API avoids
            # very confusing error messages regarding foreign key constraint
            # violations that are different for various RDBMS backends and
            # a generic error message regarding existence of a dependent key.
            # Use .first() instead of .exists() for MSSQL compatibility
            dataset = self._dataset_query(session).filter_by(dataset_id=dataset_id).first()

            if not dataset:
                raise MlflowException(
                    f"Could not find evaluation dataset with ID {dataset_id}",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

            for key, value in tags.items():
                if value is not None:
                    existing_tag = (
                        session
                        .query(SqlEvaluationDatasetTag)
                        .filter_by(dataset_id=dataset_id, key=key)
                        .first()
                    )
                    if existing_tag:
                        existing_tag.value = str(value)
                    else:
                        new_tag = SqlEvaluationDatasetTag(
                            dataset_id=dataset_id,
                            key=key,
                            value=str(value),
                        )
                        session.add(new_tag)

    #######################################################################################
    # Below are legacy V2 Tracing APIs. DO NOT USE. Use the V3 APIs instead.
    #######################################################################################

    def add_dataset_to_experiments(
        self, dataset_id: str, experiment_ids: list[str]
    ) -> EvaluationDataset:
        """
        Add a dataset to additional experiments.
        """
        from mlflow.entities.entity_type import EntityAssociationType

        with self.ManagedSessionMaker(read_only=False) as session:
            dataset = self._dataset_query(session).filter_by(dataset_id=dataset_id).first()
            if not dataset:
                raise MlflowException(
                    f"Dataset '{dataset_id}' not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

            experiment_ids_str = [str(exp_id) for exp_id in experiment_ids]

            accessible_exp_ids = (
                {
                    str(row[0])
                    for row in self
                    ._get_query(session, SqlExperiment)
                    .filter(SqlExperiment.experiment_id.in_(experiment_ids_str))
                    .with_entities(SqlExperiment.experiment_id)
                    .all()
                }
                if experiment_ids_str
                else set()
            )

            for exp_id in experiment_ids_str:
                if exp_id not in accessible_exp_ids:
                    raise MlflowException(
                        f"No Experiment with id={exp_id}",
                        error_code=RESOURCE_DOES_NOT_EXIST,
                    )

            existing_associations = (
                session
                .query(SqlEntityAssociation)
                .filter(
                    SqlEntityAssociation.source_id == dataset_id,
                    SqlEntityAssociation.source_type == EntityAssociationType.EVALUATION_DATASET,
                    SqlEntityAssociation.destination_id.in_(experiment_ids_str),
                    SqlEntityAssociation.destination_type == EntityAssociationType.EXPERIMENT,
                )
                .all()
            )

            existing_exp_ids = {assoc.destination_id for assoc in existing_associations}

            new_associations = [
                SqlEntityAssociation(
                    association_id=uuid.uuid4().hex,
                    source_id=dataset_id,
                    source_type=EntityAssociationType.EVALUATION_DATASET,
                    destination_id=exp_id,
                    destination_type=EntityAssociationType.EXPERIMENT,
                )
                for exp_id in experiment_ids_str
                if exp_id not in existing_exp_ids
            ]

            if new_associations:
                session.bulk_save_objects(new_associations)

            dataset.last_update_time = get_current_time_millis()
            session.commit()

            return dataset.to_mlflow_entity()

    def remove_dataset_from_experiments(
        self, dataset_id: str, experiment_ids: list[str]
    ) -> EvaluationDataset:
        """
        Remove a dataset from experiments (idempotent).
        """
        from mlflow.entities.entity_type import EntityAssociationType

        with self.ManagedSessionMaker(read_only=False) as session:
            dataset = self._dataset_query(session).filter_by(dataset_id=dataset_id).first()
            if not dataset:
                raise MlflowException(
                    f"Dataset '{dataset_id}' not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

            existing_associations = (
                session
                .query(SqlEntityAssociation)
                .filter(
                    SqlEntityAssociation.source_id == dataset_id,
                    SqlEntityAssociation.source_type == EntityAssociationType.EVALUATION_DATASET,
                    SqlEntityAssociation.destination_id.in_([
                        str(exp_id) for exp_id in experiment_ids
                    ]),
                    SqlEntityAssociation.destination_type == EntityAssociationType.EXPERIMENT,
                )
                .all()
            )

            existing_exp_ids = {assoc.destination_id for assoc in existing_associations}

            for exp_id in experiment_ids:
                if str(exp_id) not in existing_exp_ids:
                    _logger.warning(
                        f"Dataset '{dataset_id}' was not associated with experiment '{exp_id}'"
                    )

            if existing_exp_ids:
                session.query(SqlEntityAssociation).filter(
                    SqlEntityAssociation.source_id == dataset_id,
                    SqlEntityAssociation.source_type == EntityAssociationType.EVALUATION_DATASET,
                    SqlEntityAssociation.destination_id.in_(list(existing_exp_ids)),
                    SqlEntityAssociation.destination_type == EntityAssociationType.EXPERIMENT,
                ).delete(synchronize_session=False)

                dataset.last_update_time = get_current_time_millis()

            session.commit()

            return dataset.to_mlflow_entity()

    # ===================================================================================
    # Issue Methods
    # ===================================================================================

    # ===================================================================================
    # Helper Methods for Secrets & Endpoints
    # ===================================================================================

    def _get_entity_or_raise(
        self,
        session: Session,
        model_class: type[_T],
        filters: dict[str, Any],
        entity_name: str,
    ) -> _T:
        """
        Get entity or raise RESOURCE_DOES_NOT_EXIST with descriptive message.

        Args:
            session: Database session.
            model_class: SQLAlchemy model class (e.g., SqlExperiment, SqlRun).
            filters: Dict of filter conditions (e.g., {"experiment_id": "123"}).
            entity_name: Human-readable entity name for error message (e.g., "Experiment").

        Returns:
            The entity object.

        Raises:
            MlflowException: If entity not found (RESOURCE_DOES_NOT_EXIST).
        """
        obj = self._get_query(session, model_class).filter_by(**filters).first()
        if not obj:
            filter_str = ", ".join(f"{k}='{v}'" for k, v in filters.items())
            raise MlflowException(
                f"{entity_name} not found ({filter_str})",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return obj

    def resolve_endpoint_in_scorer(self, scorer_version: ScorerVersion) -> ScorerVersion:
        """
        Resolve gateway endpoint ID to name in a scorer version.

        If the scorer's model field contains a gateway endpoint ID (gateway:/{id}),
        resolves it to the endpoint name. If the endpoint has been deleted,
        sets the model to None.

        Args:
            scorer_version: The scorer version to resolve.

        Returns:
            A new ScorerVersion with the resolved endpoint name, or the original
            if no gateway endpoint is used.
        """
        serialized_data = json.loads(scorer_version._serialized_scorer)
        model = extract_model_from_serialized_scorer(serialized_data)

        if not is_gateway_model(model):
            return scorer_version

        endpoint_ref = extract_endpoint_ref(model)

        # Try to resolve endpoint ID to name
        try:
            endpoint = self.get_gateway_endpoint(endpoint_id=endpoint_ref)
            new_model = build_gateway_model(endpoint.name)
        except MlflowException:
            # Endpoint was deleted or invalid, set model to null
            new_model = None

        serialized_data = update_model_in_serialized_scorer(serialized_data, new_model)

        return ScorerVersion(
            experiment_id=scorer_version.experiment_id,
            scorer_id=scorer_version.scorer_id,
            scorer_version=scorer_version.scorer_version,
            scorer_name=scorer_version.scorer_name,
            serialized_scorer=json.dumps(serialized_data),
            creation_time=scorer_version.creation_time,
        )

    def _batch_resolve_endpoint_in_serialized_scorers(
        self, serialized_scorers: list[str]
    ) -> list[str]:
        """
        Batch resolve gateway endpoint IDs to names in serialized scorer strings.

        Efficiently resolves endpoint IDs by fetching all endpoints once (lazily)
        and building an ID-to-name map, rather than making individual lookups.

        Args:
            serialized_scorers: List of serialized scorer JSON strings.

        Returns:
            List of serialized scorer JSON strings with resolved endpoint names.
        """
        resolved = []
        id_to_name: dict[str, str] = {}

        for serialized in serialized_scorers:
            serialized_data = json.loads(serialized)
            model = extract_model_from_serialized_scorer(serialized_data)

            if is_gateway_model(model):
                # Lazy load endpoints on first gateway model encountered
                if not id_to_name:
                    all_endpoints = self.list_gateway_endpoints()
                    id_to_name = {ep.endpoint_id: ep.name for ep in all_endpoints}

                endpoint_id = extract_endpoint_ref(model)
                endpoint_name = id_to_name.get(endpoint_id)
                new_model = build_gateway_model(endpoint_name) if endpoint_name else None
                serialized_data = update_model_in_serialized_scorer(serialized_data, new_model)

            resolved.append(json.dumps(serialized_data))

        return resolved

    def _cleanup_endpoint_bindings(self, session, resource_type: str, resource_id: str):
        """
        Delete all endpoint bindings for a resource.

        This should be called before deleting any resource that may have endpoint bindings
        to ensure orphaned bindings are cleaned up.

        Args:
            session: Database session.
            resource_type: Type of resource (e.g., "scorer", "experiment").
            resource_id: ID of the resource being deleted.
        """
        self._filter_endpoint_binding_query(
            session,
            session.query(SqlGatewayEndpointBinding).filter_by(
                resource_type=resource_type, resource_id=resource_id
            ),
        ).delete(synchronize_session=False)

    def _get_decrypted_secret(self, secret_id: str) -> dict[str, Any]:
        """
        Get decrypted secret value by ID.

        This is a privileged operation that decrypts a secret stored in the database.
        It should only be called server-side and never exposed to clients.

        Args:
            secret_id: ID of the secret to decrypt.

        Returns:
            Decrypted secret value as a dict.
        """
        with self.ManagedSessionMaker() as session:
            sql_secret = self._get_entity_or_raise(
                session,
                SqlGatewaySecret,
                {"secret_id": secret_id},
                "GatewaySecret",
            )

            kek_manager = KEKManager()
            return _decrypt_secret(
                encrypted_value=sql_secret.encrypted_value,
                wrapped_dek=sql_secret.wrapped_dek,
                kek_manager=kek_manager,
                secret_id=sql_secret.secret_id,
                secret_name=sql_secret.secret_name,
            )

    # ------------------------------------------------------------------
    # Label schemas: see mlflow/genai/label_schemas/ for the entity
    # dataclasses and validation rules.
    # ------------------------------------------------------------------

    def _label_schema_query(self, session):
        return self._get_query(session, SqlLabelSchema)

    def _validate_experiment_exists(self, session, experiment_id):
        # Use the canonical helper so we get lifecycle filtering (label
        # schemas can't be created against soft-deleted experiments) and
        # consistent INVALID_PARAMETER_VALUE on non-integer IDs.
        self._get_experiment(session, experiment_id, ViewType.ACTIVE_ONLY)

    def create_label_schema(
        self,
        experiment_id,
        *,
        name,
        type,
        input,
        instruction=None,
        enable_comment=False,
    ):
        from mlflow.genai.label_schemas.label_schemas import LabelSchema, LabelSchemaType
        from mlflow.genai.label_schemas.validation import validate_schema_for_create

        validate_schema_for_create(
            name=name,
            type=type,
            input=input,
            instruction=instruction,
            enable_comment=enable_comment,
        )

        with self.ManagedSessionMaker(read_only=False) as session:
            self._validate_experiment_exists(session, experiment_id)

            existing = (
                self
                ._label_schema_query(session)
                .filter(
                    SqlLabelSchema.experiment_id == int(experiment_id),
                    SqlLabelSchema.name == name,
                )
                .one_or_none()
            )
            if existing is not None:
                raise MlflowException(
                    f"Label schema with name '{name}' already exists for experiment "
                    f"'{experiment_id}'.",
                    error_code=RESOURCE_ALREADY_EXISTS,
                )

            schema_id = f"{SqlLabelSchema.LABEL_SCHEMA_ID_PREFIX}{uuid.uuid4().hex}"
            entity = LabelSchema(
                schema_id=schema_id,
                experiment_id=str(experiment_id),
                name=name,
                type=LabelSchemaType(str(type)),
                input=input,
                instruction=instruction,
                enable_comment=enable_comment,
            )
            sql_schema = SqlLabelSchema.from_mlflow_entity(entity)
            session.add(sql_schema)
            try:
                session.flush()
            except IntegrityError as e:
                # Race: another transaction inserted (experiment_id, name) between
                # the pre-check and the flush. Surface the expected MLflow error
                # code instead of the raw SQLAlchemy exception.
                raise MlflowException(
                    f"Label schema with name '{name}' already exists for experiment "
                    f"'{experiment_id}'.",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from e
            return sql_schema.to_mlflow_entity()

    def get_label_schema(self, schema_id):
        with self.ManagedSessionMaker() as session:
            sql_schema = (
                self
                ._label_schema_query(session)
                .filter(SqlLabelSchema.schema_id == schema_id)
                .one_or_none()
            )
            if sql_schema is None:
                raise MlflowException(
                    f"Label schema with id '{schema_id}' not found.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            return sql_schema.to_mlflow_entity()

    def get_label_schema_by_name(self, experiment_id, name):
        with self.ManagedSessionMaker() as session:
            # Validate via the canonical helper so a non-integer experiment ID
            # raises INVALID_PARAMETER_VALUE (rather than a raw ValueError from
            # int(...)), matching the other experiment-scoped store methods.
            self._validate_experiment_exists(session, experiment_id)
            sql_schema = (
                self
                ._label_schema_query(session)
                .filter(
                    SqlLabelSchema.experiment_id == int(experiment_id),
                    SqlLabelSchema.name == name,
                )
                .one_or_none()
            )
            if sql_schema is None:
                raise MlflowException(
                    f"Label schema with name '{name}' not found for experiment '{experiment_id}'.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            return sql_schema.to_mlflow_entity()

    def list_label_schemas(self, experiment_id, max_results=100, page_token=None):
        self._validate_max_results_param(max_results)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token) if page_token else 0
        # Writable session: the protected default question is seeded lazily on
        # first access (see `_ensure_default_label_schema`), so a list may create
        # it. This is the SDK + REST chokepoint, so every experiment always has
        # at least one question regardless of how it's first reached.
        with self.ManagedSessionMaker(read_only=False) as session:
            self._validate_experiment_exists(session, experiment_id)
            self._ensure_default_label_schema(session, experiment_id)

            results = (
                self
                ._label_schema_query(session)
                .filter(SqlLabelSchema.experiment_id == int(experiment_id))
                .order_by(SqlLabelSchema.created_time.desc(), SqlLabelSchema.schema_id.asc())
                .offset(offset)
                .limit(max_results + 1)
                .all()
            )

            next_token = None
            if len(results) > max_results:
                results = results[:max_results]
                next_token = SearchUtils.create_page_token(offset + max_results)

            entities = [r.to_mlflow_entity() for r in results]
            return PagedList(entities, next_token)

    def update_label_schema(
        self,
        schema_id,
        *,
        name=None,
        instruction=None,
        enable_comment=None,
        input=None,
    ):
        # Sparse update; ``type`` is immutable post-create and is not
        # accepted. Rename collisions are detected before the write.
        from mlflow.genai.label_schemas.validation import validate_schema_for_update

        with self.ManagedSessionMaker(read_only=False) as session:
            sql_schema = (
                self
                ._label_schema_query(session)
                .filter(SqlLabelSchema.schema_id == schema_id)
                .one_or_none()
            )
            if sql_schema is None:
                raise MlflowException(
                    f"Label schema with id '{schema_id}' not found.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

            existing_entity = sql_schema.to_mlflow_entity()
            validate_schema_for_update(
                existing=existing_entity,
                name=name,
                instruction=instruction,
                enable_comment=enable_comment,
                input=input,
            )

            if name is not None and name != sql_schema.name:
                collision = (
                    self
                    ._label_schema_query(session)
                    .filter(
                        SqlLabelSchema.experiment_id == sql_schema.experiment_id,
                        SqlLabelSchema.name == name,
                        SqlLabelSchema.schema_id != schema_id,
                    )
                    .one_or_none()
                )
                if collision is not None:
                    raise MlflowException(
                        f"Label schema with name '{name}' already exists for experiment "
                        f"'{sql_schema.experiment_id}'.",
                        error_code=RESOURCE_ALREADY_EXISTS,
                    )
                sql_schema.name = name
            if instruction is not None:
                sql_schema.instruction = instruction
            if enable_comment is not None:
                sql_schema.enable_comment = enable_comment
            if input is not None:
                input_type, input_config = _input_to_dict(input)
                sql_schema.input_type = input_type
                sql_schema.input_config = input_config

            sql_schema.last_update_time = get_current_time_millis()
            try:
                session.flush()
            except IntegrityError as e:
                # Race: a concurrent create/rename claimed (experiment_id, name)
                # between the rename pre-check above and this flush. Surface the
                # expected error code instead of a raw SQLAlchemy exception, as
                # the create path does.
                raise MlflowException(
                    f"Label schema with name '{sql_schema.name}' already exists for "
                    f"experiment '{sql_schema.experiment_id}'.",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from e
            return sql_schema.to_mlflow_entity()

    def delete_label_schema(self, schema_id):
        # No-op when the schema doesn't exist. Assessments whose ``name``
        # matches this schema retain their data and render as free-form
        # values in the UI after deletion (soft reference).
        with self.ManagedSessionMaker(read_only=False) as session:
            sql_schema = (
                self
                ._label_schema_query(session)
                .filter(SqlLabelSchema.schema_id == schema_id)
                .one_or_none()
            )
            if sql_schema is None:
                _logger.debug(f"Label schema with id '{schema_id}' not found; delete is a no-op.")
                return
            if sql_schema.is_default:
                raise MlflowException(
                    "The experiment's default question cannot be deleted.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            session.delete(sql_schema)

    def _ensure_default_label_schema(self, session, experiment_id):
        """Seed the experiment's protected default question if it's absent.

        The default question is a FEEDBACK free-text schema named
        ``DEFAULT_LABEL_SCHEMA_NAME``, marked ``is_default`` so it is undeletable
        and uneditable. It's created lazily from ``list_label_schemas`` (the SDK +
        REST chokepoint) so every experiment always presents at least one question
        on first access, without a dedicated endpoint. Idempotent: at most one row
        per experiment carries the reserved name.
        """
        from mlflow.genai.label_schemas.label_schemas import (
            InputText,
            LabelSchema,
            LabelSchemaType,
        )
        from mlflow.genai.label_schemas.validation import (
            DEFAULT_LABEL_SCHEMA_INSTRUCTION,
            DEFAULT_LABEL_SCHEMA_NAME,
        )

        # Key the idempotency check on the is_default flag (dialect-agnostic): a
        # name `==` is case-sensitive on some backends. A pre-existing user schema
        # literally named "Feedback" (only creatable before the reserved-name
        # rule) collides on the (experiment_id, name) unique constraint below, is
        # caught, and that experiment keeps its own "Feedback" with no separate
        # protected default — an accepted edge for already-created data.
        existing = (
            self
            ._label_schema_query(session)
            .filter(
                SqlLabelSchema.experiment_id == int(experiment_id),
                SqlLabelSchema.is_default == sqlalchemy.true(),
            )
            .first()
        )
        if existing is not None:
            return

        entity = LabelSchema(
            schema_id=f"{SqlLabelSchema.LABEL_SCHEMA_ID_PREFIX}{uuid.uuid4().hex}",
            experiment_id=str(experiment_id),
            name=DEFAULT_LABEL_SCHEMA_NAME,
            type=LabelSchemaType.FEEDBACK,
            input=InputText(),
            instruction=DEFAULT_LABEL_SCHEMA_INSTRUCTION,
            enable_comment=False,
            is_default=True,
        )
        try:
            # SAVEPOINT isolates this insert so a benign IntegrityError drops only
            # it, not the caller's transaction (same pattern as
            # add_items_to_review_queue). Two cases are safe to swallow: a
            # concurrent ensure won the (experiment_id, name) unique race (the
            # existing row is the default), or the experiment vanished concurrently
            # (a FK error, which the list's experiment validation then surfaces).
            with session.begin_nested():
                session.add(SqlLabelSchema.from_mlflow_entity(entity))
                session.flush()
        except IntegrityError:
            pass

    # ------------------------------------------------------------------
    # Review queues: see mlflow/genai/review_queues/ for the entity
    # dataclasses and validation rules. The parent `review_queues` table is
    # workspace-scoped via a join to `experiments` (`_review_queue_query`);
    # the three child tables are always reached through an already
    # workspace-validated `queue_id`, so they inherit that scope.
    # ------------------------------------------------------------------

    def _review_queue_query(self, session):
        return self._get_query(session, SqlReviewQueue)

    def _get_sql_review_queue(self, session, queue_id, *, for_update=False):
        """Fetch the workspace-scoped queue row or raise RESOURCE_DOES_NOT_EXIST.

        Pass ``for_update=True`` from mutating paths (attaching items, editing
        questions) to lock the queue row for the rest of the transaction. The
        question-freeze check reads the item count and then swaps the schema set;
        without the lock a concurrent attach could slip an item in between, leaving
        reviewers answering questions that were swapped out from under them. Taking
        the row lock serializes the editing and attaching paths against each other.
        """
        query = self._review_queue_query(session).filter(SqlReviewQueue.queue_id == queue_id)
        if for_update:
            query = query.with_for_update()
        sql_queue = query.one_or_none()
        if sql_queue is None:
            raise MlflowException(
                f"Review queue with id '{queue_id}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return sql_queue

    def _load_users_by_queue(self, session, queue_ids):
        """Map queue_id -> ordered list of assigned users for the given queues."""
        users_by_queue = {queue_id: [] for queue_id in queue_ids}
        if not queue_ids:
            return users_by_queue
        rows = (
            session
            .query(SqlReviewQueueUser)
            .filter(SqlReviewQueueUser.queue_id.in_(queue_ids))
            .order_by(SqlReviewQueueUser.user_id.asc())
            .all()
        )
        for row in rows:
            users_by_queue[row.queue_id].append(row.user_id)
        return users_by_queue

    def _load_schema_ids_by_queue(self, session, queue_ids):
        """Map queue_id -> ordered list of attached schema ids for the queues."""
        schemas_by_queue = {queue_id: [] for queue_id in queue_ids}
        if not queue_ids:
            return schemas_by_queue
        rows = (
            session
            .query(SqlReviewQueueLabelSchema)
            .filter(SqlReviewQueueLabelSchema.queue_id.in_(queue_ids))
            .order_by(SqlReviewQueueLabelSchema.schema_id.asc())
            .all()
        )
        for row in rows:
            schemas_by_queue[row.queue_id].append(row.schema_id)
        return schemas_by_queue

    def _hydrate_review_queues(self, session, sql_queues):
        """Convert queue rows to entities, batch-loading their association sets."""
        queue_ids = [q.queue_id for q in sql_queues]
        users_by_queue = self._load_users_by_queue(session, queue_ids)
        schemas_by_queue = self._load_schema_ids_by_queue(session, queue_ids)
        return [
            q.to_mlflow_entity(
                users=users_by_queue[q.queue_id],
                schema_ids=schemas_by_queue[q.queue_id],
            )
            for q in sql_queues
        ]

    def _validate_schema_ids_exist(self, session, experiment_id, schema_ids):
        """Raise INVALID_PARAMETER_VALUE if any schema id isn't in the experiment.

        ``review_queue_label_schemas.schema_id`` is a soft reference (no foreign
        key), so the store validates both existence and same-experiment
        membership here, yielding a clear error rather than leaving a bad id to
        surface (or silently persist) at write time.
        """
        if not schema_ids:
            return
        found = (
            self
            ._label_schema_query(session)
            .filter(
                SqlLabelSchema.experiment_id == int(experiment_id),
                SqlLabelSchema.schema_id.in_(schema_ids),
            )
            .with_entities(SqlLabelSchema.schema_id)
            .all()
        )
        found_ids = {row[0] for row in found}
        if missing := [schema_id for schema_id in schema_ids if schema_id not in found_ids]:
            raise MlflowException(
                f"Label schema id(s) {missing} not found for experiment '{experiment_id}'.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def create_review_queue(
        self,
        experiment_id,
        *,
        name,
        queue_type,
        created_by=None,
        users=None,
        schema_ids=None,
    ):
        from mlflow.genai.review_queues.validation import validate_queue_for_create

        validated = validate_queue_for_create(
            name=name,
            queue_type=queue_type,
            users=users,
            schema_ids=schema_ids,
        )
        with self.ManagedSessionMaker(read_only=False) as session:
            self._validate_experiment_exists(session, experiment_id)
            self._validate_schema_ids_exist(session, experiment_id, validated.schema_ids)

            now_ms = get_current_time_millis()
            sql_queue = SqlReviewQueue(
                queue_id=f"{SqlReviewQueue.QUEUE_ID_PREFIX}{uuid.uuid4().hex}",
                experiment_id=int(experiment_id),
                # Names are unique within an experiment case-insensitively via the
                # case-folded `name_key`, which the model validator derives from
                # `name` (the display casing).
                name=validated.name,
                queue_type=str(validated.queue_type),
                created_by=created_by,
                creation_time_ms=now_ms,
                last_update_time_ms=now_ms,
            )
            # Single source for the case-fold: the validator-derived key, reused by
            # the pre-check and the disambiguation re-query below (captured rather
            # than re-read off the object, which a savepoint rollback could expire).
            name_key = sql_queue.name_key

            existing = (
                self
                ._review_queue_query(session)
                .filter(
                    SqlReviewQueue.experiment_id == int(experiment_id),
                    SqlReviewQueue.name_key == name_key,
                )
                .one_or_none()
            )
            if existing is not None:
                raise MlflowException(
                    f"Review queue with name '{validated.name}' already exists "
                    "(names are case-insensitive).",
                    error_code=RESOURCE_ALREADY_EXISTS,
                )
            try:
                # SAVEPOINT around the add+flush so an IntegrityError rolls back
                # just this insert (not the whole transaction) and leaves the
                # session usable for the disambiguating re-query below.
                with session.begin_nested():
                    session.add(sql_queue)
                    session.flush()
            except IntegrityError as e:
                # The flush violated a constraint. Disambiguate by checking which
                # one now holds rather than assuming a cause. The duplicate check
                # is on `name_key` (matching the unique constraint), so a parallel
                # create of a case-variant name (e.g. `foo` vs an existing `Foo`)
                # is correctly classified as a duplicate and translated below,
                # rather than falling through and re-raising a raw IntegrityError.
                # It is intentionally unscoped: the unique constraint is global on
                # (experiment_id, name_key), and an experiment belongs to a single
                # workspace, so any row sharing this experiment_id is in the same
                # workspace as the queue being created. The unscoped lookup
                # therefore can't surface a foreign-workspace row; workspace scoping
                # on reads is irrelevant to this disambiguation.
                duplicate = (
                    session
                    .query(SqlReviewQueue)
                    .filter(
                        SqlReviewQueue.experiment_id == int(experiment_id),
                        SqlReviewQueue.name_key == name_key,
                    )
                    .first()
                )
                if duplicate is not None:
                    # A parallel transaction won the create race.
                    raise MlflowException(
                        f"Review queue with name '{validated.name}' already exists "
                        "(names are case-insensitive).",
                        error_code=RESOURCE_ALREADY_EXISTS,
                    ) from e
                experiment_present = (
                    session
                    .query(SqlExperiment.experiment_id)
                    .filter(SqlExperiment.experiment_id == int(experiment_id))
                    .first()
                )
                if experiment_present is None:
                    # The experiment FK failed because the experiment was deleted
                    # between the pre-check and the flush.
                    raise MlflowException(
                        f"Experiment '{experiment_id}' does not exist.",
                        error_code=RESOURCE_DOES_NOT_EXIST,
                    ) from e
                # Neither known cause holds; surface the real error rather than
                # mislabeling it.
                raise

            for user in validated.users:
                session.add(SqlReviewQueueUser(queue_id=sql_queue.queue_id, user_id=user))
            for schema_id in validated.schema_ids:
                session.add(
                    SqlReviewQueueLabelSchema(queue_id=sql_queue.queue_id, schema_id=schema_id)
                )
            session.flush()
            return self._hydrate_review_queues(session, [sql_queue])[0]

    def get_or_create_user_queue(self, experiment_id, *, user):
        from mlflow.genai.review_queues import ReviewQueueType
        from mlflow.genai.review_queues.validation import normalize_user

        name = normalize_user(user)
        try:
            return self.create_review_queue(
                experiment_id,
                name=name,
                queue_type="user",
                # A user queue is owned by its user (attribution only).
                created_by=name,
            )
        except MlflowException as e:
            if e.error_code != ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                raise
            # Lost the create race (or the queue already existed): return the
            # single existing user queue, keeping the call idempotent.
            existing = self.get_review_queue_by_name(experiment_id, name=name)
            if ReviewQueueType(existing.queue_type) != ReviewQueueType.USER:
                # A custom queue squatting on this user's name — don't hand it
                # back as if it were the user's personal queue.
                raise MlflowException(
                    f"A non-user queue named '{name}' already exists; cannot get-or-create "
                    f"a user queue with that name.",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from e
            return existing

    def get_review_queue(self, queue_id):
        with self.ManagedSessionMaker() as session:
            sql_queue = self._get_sql_review_queue(session, queue_id)
            return self._hydrate_review_queues(session, [sql_queue])[0]

    def get_review_queue_by_name(self, experiment_id, *, name):
        with self.ManagedSessionMaker() as session:
            self._validate_experiment_exists(session, experiment_id)
            sql_queue = (
                self
                ._review_queue_query(session)
                .filter(
                    SqlReviewQueue.experiment_id == int(experiment_id),
                    # Look up case-insensitively (matching the uniqueness key).
                    SqlReviewQueue.name_key == name.lower(),
                )
                .one_or_none()
            )
            if sql_queue is None:
                raise MlflowException(
                    f"Review queue with name '{name}' not found for experiment '{experiment_id}'.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            return self._hydrate_review_queues(session, [sql_queue])[0]

    def list_review_queues(
        self, experiment_id, *, user=None, item_id=None, max_results=None, page_token=None
    ):
        from mlflow.genai.review_queues.validation import normalize_user

        if max_results is None:
            max_results = SEARCH_MAX_RESULTS_DEFAULT
        else:
            self._validate_max_results_param(max_results)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token) if page_token else 0

        with self.ManagedSessionMaker() as session:
            self._validate_experiment_exists(session, experiment_id)
            query = self._review_queue_query(session).filter(
                SqlReviewQueue.experiment_id == int(experiment_id)
            )
            if user is not None:
                # Scope to queues the user is assigned to (their own user queue
                # plus any custom queue they belong to).
                assigned_queue_ids = session.query(SqlReviewQueueUser.queue_id).filter(
                    SqlReviewQueueUser.user_id == normalize_user(user)
                )
                query = query.filter(SqlReviewQueue.queue_id.in_(assigned_queue_ids))
            if item_id is not None:
                # Scope to queues that already contain this item, via the per-item
                # index on review_queue_items, so callers can see which queues a
                # trace is already a member of.
                containing_queue_ids = session.query(SqlReviewQueueItem.queue_id).filter(
                    SqlReviewQueueItem.item_id == item_id
                )
                query = query.filter(SqlReviewQueue.queue_id.in_(containing_queue_ids))

            results = (
                query
                .order_by(
                    SqlReviewQueue.creation_time_ms.desc(),
                    SqlReviewQueue.queue_id.asc(),
                )
                .offset(offset)
                .limit(max_results + 1)
                .all()
            )

            next_token = None
            if len(results) > max_results:
                results = results[:max_results]
                next_token = SearchUtils.create_page_token(offset + max_results)
            return PagedList(self._hydrate_review_queues(session, results), next_token)

    def update_review_queue(
        self, queue_id, *, users=None, schema_ids=None, name=None, new_owner=None
    ):
        from mlflow.genai.review_queues import ReviewQueueType
        from mlflow.genai.review_queues.validation import (
            normalize_schema_ids,
            normalize_users,
            validate_custom_queue_name,
            validate_queue_owner,
        )

        with self.ManagedSessionMaker(read_only=False) as session:
            sql_queue = self._get_sql_review_queue(session, queue_id, for_update=True)
            if ReviewQueueType(sql_queue.queue_type) == ReviewQueueType.USER:
                raise MlflowException(
                    "A user queue's name, assigned user, schemas, and owner are fixed "
                    "and cannot be updated.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            if users is None and schema_ids is None and name is None and new_owner is None:
                return self._hydrate_review_queues(session, [sql_queue])[0]

            if new_owner is not None:
                # Owner reassignment; authorization (MANAGE-only) is enforced at
                # the handler layer. Stored case-preserved (matching is
                # case-insensitive).
                sql_queue.created_by = validate_queue_owner(new_owner)

            renamed_to = None
            if name is not None:
                new_name = validate_custom_queue_name(name)
                if new_name != sql_queue.name:
                    # Assigning `name` re-derives `name_key` via the validator.
                    # Only a name_key change can violate the unique
                    # (experiment_id, name_key) constraint, so only then arm
                    # `renamed_to`, which translates a flush IntegrityError into a
                    # name collision (no upfront SELECT). A pure display-case change
                    # keeps the same name_key, so it can't collide; leaving
                    # `renamed_to` None there means an unrelated IntegrityError is
                    # surfaced untranslated, not mislabeled.
                    previous_name_key = sql_queue.name_key
                    sql_queue.name = new_name
                    if sql_queue.name_key != previous_name_key:
                        renamed_to = new_name

            if users is not None:
                normalized_users = normalize_users(users)
                session.query(SqlReviewQueueUser).filter(
                    SqlReviewQueueUser.queue_id == sql_queue.queue_id
                ).delete(synchronize_session=False)
                for user in normalized_users:
                    session.add(SqlReviewQueueUser(queue_id=sql_queue.queue_id, user_id=user))

            if schema_ids is not None:
                # A queue's questions are frozen once it has items to review:
                # changing the schema set after reviewers have started would
                # strand their answers or leave completed items with
                # never-seen questions. Editing is allowed only while the queue
                # is still empty (in setup). Assigned users stay editable.
                attached_item_count = (
                    session
                    .query(SqlReviewQueueItem)
                    .filter(SqlReviewQueueItem.queue_id == sql_queue.queue_id)
                    .count()
                )
                if attached_item_count > 0:
                    raise MlflowException(
                        "A review queue's questions are locked once items are assigned to it. "
                        "Remove the items before changing its questions.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                normalized_schema_ids = normalize_schema_ids(schema_ids)
                self._validate_schema_ids_exist(
                    session, str(sql_queue.experiment_id), normalized_schema_ids
                )
                session.query(SqlReviewQueueLabelSchema).filter(
                    SqlReviewQueueLabelSchema.queue_id == sql_queue.queue_id
                ).delete(synchronize_session=False)
                for schema_id in normalized_schema_ids:
                    session.add(
                        SqlReviewQueueLabelSchema(queue_id=sql_queue.queue_id, schema_id=schema_id)
                    )

            sql_queue.last_update_time_ms = get_current_time_millis()
            try:
                session.flush()
            except IntegrityError as e:
                # The only unique constraint here is (experiment_id, name_key): a
                # rename to a name already taken (case-insensitively) in the
                # experiment violates it. Surface that as a clean
                # RESOURCE_ALREADY_EXISTS. If no rename was applied the
                # violation is unrelated, so re-raise it untranslated rather than
                # blaming the name.
                if renamed_to is None:
                    raise
                raise MlflowException(
                    f"Review queue with name '{renamed_to}' already exists "
                    "(names are case-insensitive).",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from e
            return self._hydrate_review_queues(session, [sql_queue])[0]

    def delete_review_queue(self, queue_id):
        # No-op when the queue doesn't exist. Child rows (users, items,
        # schemas) are deleted explicitly so the behaviour doesn't depend on
        # DB-level ON DELETE CASCADE being honoured by the active dialect.
        # Reviewer assessments on the queue's items are untouched.
        with self.ManagedSessionMaker(read_only=False) as session:
            sql_queue = (
                self
                ._review_queue_query(session)
                .filter(SqlReviewQueue.queue_id == queue_id)
                .one_or_none()
            )
            if sql_queue is None:
                _logger.debug(f"Review queue with id '{queue_id}' not found; delete is a no-op.")
                return
            for child_model in (
                SqlReviewQueueUser,
                SqlReviewQueueItem,
                SqlReviewQueueLabelSchema,
            ):
                session.query(child_model).filter(
                    child_model.queue_id == sql_queue.queue_id
                ).delete(synchronize_session=False)
            session.delete(sql_queue)

    def add_items_to_review_queue(self, queue_id, *, item_ids, item_type="trace"):
        from mlflow.genai.review_queues import ReviewStatus
        from mlflow.genai.review_queues.validation import (
            coerce_item_type,
            validate_item_ids_for_attach,
        )

        coerced_item_type = coerce_item_type(item_type)
        normalized_item_ids = validate_item_ids_for_attach(item_ids)

        with self.ManagedSessionMaker(read_only=False) as session:
            sql_queue = self._get_sql_review_queue(session, queue_id, for_update=True)

            existing_item_ids = {
                row.item_id
                for row in session
                .query(SqlReviewQueueItem.item_id)
                .filter(
                    SqlReviewQueueItem.queue_id == sql_queue.queue_id,
                    SqlReviewQueueItem.item_id.in_(normalized_item_ids),
                )
                .all()
            }

            now_ms = get_current_time_millis()
            for item_id in normalized_item_ids:
                if item_id in existing_item_ids:
                    # Idempotent: keep the existing row and its status.
                    continue
                try:
                    # SAVEPOINT around the add+flush so a concurrent attach of
                    # the same item (the only possible IntegrityError here —
                    # the queue FK is validated and item_id has no FK) drops
                    # just that row rather than the whole batch. The row is
                    # then picked up by the re-read below. The add lives inside
                    # the savepoint so its pending state is rolled back cleanly.
                    with session.begin_nested():
                        session.add(
                            SqlReviewQueueItem(
                                queue_id=sql_queue.queue_id,
                                item_type=str(coerced_item_type),
                                item_id=item_id,
                                status=str(ReviewStatus.PENDING),
                                creation_time_ms=now_ms,
                                last_update_time_ms=now_ms,
                            )
                        )
                        session.flush()
                except IntegrityError:
                    pass

            final_rows = (
                session
                .query(SqlReviewQueueItem)
                .filter(
                    SqlReviewQueueItem.queue_id == sql_queue.queue_id,
                    SqlReviewQueueItem.item_id.in_(normalized_item_ids),
                )
                .all()
            )
            # Usually every requested id is present now (pre-existing,
            # just-inserted, or inserted by a racing writer), but a concurrent
            # remove/delete can drop a row between the inserts above and this
            # read, so skip any id that is no longer present rather than raising
            # a KeyError.
            rows_by_item = {row.item_id: row for row in final_rows}
            return [
                rows_by_item[item_id].to_mlflow_entity()
                for item_id in normalized_item_ids
                if item_id in rows_by_item
            ]

    def remove_items_from_review_queue(self, queue_id, *, item_ids):
        from mlflow.genai.review_queues.validation import validate_item_ids_for_attach

        normalized_item_ids = validate_item_ids_for_attach(item_ids)
        with self.ManagedSessionMaker(read_only=False) as session:
            sql_queue = self._get_sql_review_queue(session, queue_id)
            session.query(SqlReviewQueueItem).filter(
                SqlReviewQueueItem.queue_id == sql_queue.queue_id,
                SqlReviewQueueItem.item_id.in_(normalized_item_ids),
            ).delete(synchronize_session=False)

    def list_review_queue_items(self, queue_id, *, status=None, max_results=None, page_token=None):
        from mlflow.genai.review_queues.validation import coerce_status

        if max_results is None:
            max_results = SEARCH_MAX_RESULTS_DEFAULT
        else:
            self._validate_max_results_param(max_results)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token) if page_token else 0
        coerced_status = coerce_status(status) if status is not None else None

        with self.ManagedSessionMaker() as session:
            sql_queue = self._get_sql_review_queue(session, queue_id)
            query = session.query(SqlReviewQueueItem).filter(
                SqlReviewQueueItem.queue_id == sql_queue.queue_id
            )
            if coerced_status is not None:
                query = query.filter(SqlReviewQueueItem.status == str(coerced_status))

            results = (
                query
                .order_by(
                    SqlReviewQueueItem.creation_time_ms.desc(),
                    SqlReviewQueueItem.item_id.asc(),
                )
                .offset(offset)
                .limit(max_results + 1)
                .all()
            )

            next_token = None
            if len(results) > max_results:
                results = results[:max_results]
                next_token = SearchUtils.create_page_token(offset + max_results)
            return PagedList([row.to_mlflow_entity() for row in results], next_token)

    def set_review_queue_item_status(self, queue_id, *, item_id, status, completed_by=None):
        from mlflow.genai.review_queues import ReviewStatus
        from mlflow.genai.review_queues.validation import (
            USER_MAX_LENGTH,
            coerce_status,
            normalize_item_id,
            normalize_user,
        )

        new_status = coerce_status(status)
        item_id = normalize_item_id(item_id)
        normalized_completed_by = normalize_user(completed_by) if completed_by is not None else None

        if new_status == ReviewStatus.PENDING:
            if normalized_completed_by is not None:
                raise MlflowException(
                    "`completed_by` must not be set when reopening an item to `pending`.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        elif not normalized_completed_by:
            raise MlflowException(
                f"`completed_by` is required when setting status to `{new_status}`.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if normalized_completed_by is not None and len(normalized_completed_by) > USER_MAX_LENGTH:
            # Match the cap the assigned-user list enforces, so an over-long value
            # raises a clear error rather than failing at the VARCHAR write.
            raise MlflowException(
                f"`completed_by` must be at most {USER_MAX_LENGTH} characters; "
                f"got {len(normalized_completed_by)}.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        with self.ManagedSessionMaker(read_only=False) as session:
            sql_queue = self._get_sql_review_queue(session, queue_id)
            row = (
                session
                .query(SqlReviewQueueItem)
                .filter(
                    SqlReviewQueueItem.queue_id == sql_queue.queue_id,
                    SqlReviewQueueItem.item_id == item_id,
                )
                .one_or_none()
            )
            if row is None:
                raise MlflowException(
                    f"Item '{item_id}' is not attached to review queue '{queue_id}'.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

            # No-op (no timestamp churn) when the status doesn't change. Making a
            # same-status write idempotent regardless of `completed_by` is what
            # protects attribution: a repeat or concurrent re-completion by a
            # different reviewer can no longer overwrite the original completer
            # (the prior `status AND completed_by` guard let it through). A genuine
            # status transition (e.g. complete -> declined, or a reopen) is a real
            # change and still re-records the actor below.
            if row.status == str(new_status):
                return row.to_mlflow_entity()

            now_ms = get_current_time_millis()
            row.status = str(new_status)
            row.last_update_time_ms = now_ms
            if new_status == ReviewStatus.PENDING:
                # Reopening clears attribution; the next completion re-records it.
                row.completed_by = None
                row.completed_time_ms = None
            else:
                # The reviewer who set the current terminal status owns attribution.
                row.completed_by = normalized_completed_by
                row.completed_time_ms = now_ms
            session.flush()
            return row.to_mlflow_entity()


def _get_logged_model_metric_datasets_filter(
    datasets: list[dict[str, Any]] | None,
) -> sqlalchemy.ColumnElement[bool] | None:
    """
    Returns a filter matching logged model metrics on any of ``datasets``, or None if no
    datasets are given.
    """
    if not datasets:
        return None
    dataset_filters = []
    for dataset in datasets:
        dataset_filter = SqlLoggedModelMetric.dataset_name == dataset["dataset_name"]
        if "dataset_digest" in dataset:
            dataset_filter = dataset_filter & (
                SqlLoggedModelMetric.dataset_digest == dataset["dataset_digest"]
            )
        dataset_filters.append(dataset_filter)
    return sqlalchemy.or_(*dataset_filters)


def _get_latest_logged_model_metrics_subquery(
    session: sqlalchemy.orm.Session,
    metric_name: str,
    filters: list[sqlalchemy.ColumnElement[bool]],
    experiment_ids: list[int] | None = None,
) -> sqlalchemy.Subquery:
    """
    Returns a subquery with the latest ``(model_id, metric_value)`` row of ``metric_name`` per
    logged model. ``filters`` (e.g. dataset filters) are applied before the latest row is
    picked.

    The latest row is the one with the highest step, then timestamp, then value, which is how
    run metrics and ``FileStore`` pick the latest value. If ``experiment_ids`` is given, only
    metrics of logged models in those experiments are ranked.
    """
    if experiment_ids is not None:
        # Scope by the model's experiment rather than `SqlLoggedModelMetric.experiment_id`,
        # which is the experiment of the run that logged the metric and can differ from the
        # model's (e.g. an evaluation run in another experiment).
        filters = [
            *filters,
            SqlLoggedModelMetric.model_id.in_(
                select(SqlLoggedModel.model_id).where(
                    SqlLoggedModel.experiment_id.in_(experiment_ids)
                )
            ),
        ]
    ranked = (
        session
        .query(
            SqlLoggedModelMetric.model_id,
            SqlLoggedModelMetric.metric_value,
            func
            .row_number()
            .over(
                partition_by=[SqlLoggedModelMetric.model_id, SqlLoggedModelMetric.metric_name],
                order_by=[
                    SqlLoggedModelMetric.metric_step.desc(),
                    SqlLoggedModelMetric.metric_timestamp_ms.desc(),
                    SqlLoggedModelMetric.metric_value.desc(),
                ],
            )
            .label("row_num"),
        )
        .filter(SqlLoggedModelMetric.metric_name == metric_name, *filters)
        .subquery()
    )
    return select(ranked.c).where(ranked.c.row_num == 1).subquery()


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

            if comparator in ("IS NULL", "IS NOT NULL"):
                exists_subquery = select(entity.run_uuid).where(
                    entity.run_uuid == SqlRun.run_uuid,
                    entity.key == key_name,
                )
                if comparator == "IS NULL":
                    attribute_filters.append(~exists_subquery.exists())
                else:
                    attribute_filters.append(exists_subquery.exists())
            elif entity == SqlDataset:
                if key_name == "context":
                    dataset_filters.append(
                        session
                        .query(entity, SqlInput, SqlInputTag)
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
                        session
                        .query(entity, SqlInput)
                        .join(SqlInput, SqlInput.source_id == SqlDataset.dataset_uuid)
                        .filter(dataset_attr_filter)
                        .subquery()
                    )
            else:
                key_filter = SearchUtils.get_sql_comparison_func("=", dialect)(entity.key, key_name)
                val_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(
                    entity.value, value
                )
                if entity is SqlLatestMetric:
                    # NaN metrics are stored as value=0 with is_nan=True. Every comparison
                    # against NaN is false except "!=", so the placeholder 0 must never
                    # satisfy a numeric filter and a NaN metric always satisfies "!=".
                    if comparator == "!=":
                        val_filter = sqlalchemy.or_(val_filter, entity.is_nan == sqlalchemy.true())
                    else:
                        val_filter = sqlalchemy.and_(
                            val_filter, entity.is_nan == sqlalchemy.false()
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
        comparator = f["comparator"].upper()
        value = f["value"]
        if type_ == "attribute":
            if key == "experiment_id":
                # ``experiment_id`` is an INTEGER column but filter values are always
                # parsed as strings; only allow comparators that make sense for a
                # nominal identifier and coerce the value(s) to int before binding,
                # matching ``_parse_experiment_id``'s error contract.
                if comparator not in ("=", "!=", "IN", "NOT IN"):
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid comparator for experiment_id: {comparator}"
                    )
                value = (
                    tuple(_parse_experiment_id(v) for v in value)
                    if isinstance(value, tuple)
                    else _parse_experiment_id(value)
                )
            else:
                if SearchExperimentsUtils.is_string_attribute(
                    type_, key, comparator
                ) and comparator not in ("=", "!=", "LIKE", "ILIKE"):
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid comparator for string attribute: {comparator}"
                    )
                if SearchExperimentsUtils.is_numeric_attribute(type_, key, comparator):
                    if comparator not in ("=", "!=", "<", "<=", ">", ">="):
                        raise MlflowException.invalid_parameter_value(
                            f"Invalid comparator for numeric attribute: {comparator}"
                        )
                    if isinstance(value, float):
                        raise MlflowException.invalid_parameter_value(
                            f"Invalid value for numeric attribute '{key}': {str(value)!r}"
                        )
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        raise MlflowException.invalid_parameter_value(
                            f"Invalid value for numeric attribute '{key}': {value!r}"
                        )
            attr = getattr(SqlExperiment, key)
            attr_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(attr, value)
            attribute_filters.append(attr_filter)
        elif type_ == "tag":
            if comparator in ("IS NULL", "IS NOT NULL"):
                tag_exists_subquery = select(SqlExperimentTag.experiment_id).where(
                    SqlExperimentTag.experiment_id == SqlExperiment.experiment_id,
                    SqlExperimentTag.key == key,
                )
                if comparator == "IS NULL":
                    attribute_filters.append(~tag_exists_subquery.exists())
                else:
                    attribute_filters.append(tag_exists_subquery.exists())
            elif comparator not in ("=", "!=", "LIKE", "ILIKE"):
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparator for tag: {comparator}"
                )
            else:
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


# Keep child-query IN clauses below SQLite and MSSQL bound-parameter limits.


def _get_search_datasets_filter_clauses(parsed_filters, dialect):
    """
    Creates evaluation dataset attribute filters and non-attribute filters for tags.
    """
    attribute_filters = []
    non_attribute_filters = []

    for f in parsed_filters:
        type_ = f["type"]
        key = f["key"]
        comparator = f["comparator"]
        value = f["value"]

        if type_ == "attribute":
            if (
                SearchEvaluationDatasetsUtils.is_string_attribute(type_, key, comparator)
                and comparator not in ("=", "!=", "LIKE", "ILIKE")
                and not (key == "name" and comparator == "IN")
            ):
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparator for string attribute: {comparator}"
                )
            if key in SearchEvaluationDatasetsUtils.NUMERIC_ATTRIBUTES:
                if (
                    comparator
                    not in SearchEvaluationDatasetsUtils.VALID_NUMERIC_ATTRIBUTE_COMPARATORS
                ):
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid comparator for numeric attribute: {comparator}"
                    )
                if isinstance(value, float):
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid value for numeric attribute '{key}': {str(value)!r}"
                    )
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid value for numeric attribute '{key}': {value!r}"
                    )
            attr = getattr(SqlEvaluationDataset, key)
            attr_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(attr, value)
            attribute_filters.append(attr_filter)
        elif type_ == "tag":
            if comparator not in ("=", "!=", "LIKE", "ILIKE"):
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparator for tag: {comparator}"
                )
            val_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(
                SqlEvaluationDatasetTag.value, value
            )
            key_filter = SearchUtils.get_sql_comparison_func("=", dialect)(
                SqlEvaluationDatasetTag.key, key
            )
            non_attribute_filters.append(
                select(SqlEvaluationDatasetTag).filter(key_filter, val_filter).subquery()
            )
        else:
            raise MlflowException.invalid_parameter_value(f"Invalid token type: {type_}")

    return attribute_filters, non_attribute_filters


def _get_search_datasets_order_by_clauses(order_by):
    """
    Creates order by clauses for searching evaluation datasets.
    """
    if not order_by:
        order_by = ["created_time DESC"]

    order_by_clauses = []

    for order in order_by:
        type_, key, ascending = (
            SearchEvaluationDatasetsUtils.parse_order_by_for_search_evaluation_datasets(order)
        )
        if type_ == "attribute":
            field = key
        else:
            raise MlflowException.invalid_parameter_value(f"Invalid order_by entity: {type_}")

        order_by_clauses.append((getattr(SqlEvaluationDataset, field), ascending))

    # Add a tie-breaker
    if not any(col == SqlEvaluationDataset.dataset_id for col, _ in order_by_clauses):
        order_by_clauses.append((SqlEvaluationDataset.dataset_id, False))

    return [col.asc() if ascending else col.desc() for col, ascending in order_by_clauses]


# Maximum number of attempts to create trace_info rows in log_spans() Phase 2.
# Each IntegrityError means start_trace() raced ahead of us; we roll back and
# re-fetch before retrying. 10 attempts reduces span drops in high-concurrency
# scenarios without significant backend load increase (log_spans runs async).
