from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Literal, TypeVar

import sqlalchemy
from sqlalchemy import and_, case, exists, func, or_, select, sql
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Query, Session, aliased, defer, joinedload, selectinload
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import Select, Subquery

_SqlAlchemyStatement = TypeVar("_SqlAlchemyStatement", Select, Query)

from mlflow.entities import (
    Assessment,
    Expectation,
    Feedback,
    Issue,
    IssueSeverity,
    IssueStatus,
    Trace,
    TraceData,
    TraceInfo,
    ViewType,
)
from mlflow.entities.assessment import (
    ExpectationValue,
    FeedbackValue,
)
from mlflow.entities.entity_type import EntityAssociationType
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.span import LazySpan
from mlflow.entities.span_status import SpanStatusCode
from mlflow.entities.trace import Span
from mlflow.entities.trace_info_v2 import TraceInfoV2
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_metrics import (
    MetricAggregation,
    MetricDataPoint,
    MetricViewType,
)
from mlflow.entities.trace_state import TraceState
from mlflow.entities.trace_status import TraceStatus
from mlflow.entities.workspace import WorkspaceDeletionMode
from mlflow.environment_variables import MLFLOW_TRACE_DELETE_MAX_WORKERS
from mlflow.exceptions import (
    MlflowException,
    MlflowTraceArchivalMalformedTrace,
    MlflowTracingException,
)
from mlflow.genai.scorers.online.entities import (
    CompletedSession,
)
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.analytics import trace_correlation
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.db.db_types import MSSQL, MYSQL, SQLITE
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import (
    MAX_RESULTS_QUERY_TRACE_METRICS,
    SEARCH_ISSUES_DEFAULT_MAX_RESULTS,
    SEARCH_TRACES_DEFAULT_MAX_RESULTS,
)
from mlflow.store.tracking.abstract_trace_backend import AbstractTraceBackend
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessments,
    SqlEntityAssociation,
    SqlExperiment,
    SqlExperimentTag,
    SqlIssue,
    SqlIssueCategory,
    SqlSpan,
    SqlTraceInfo,
    SqlTraceMetadata,
    SqlTraceTag,
)
from mlflow.store.tracking.file_store import _validate_new_trace_id
from mlflow.store.tracking.sqlalchemy_store_upsert_utils import _bulk_upsert
from mlflow.store.tracking.utils.sql_trace_metrics_utils import (
    query_metrics,
    validate_query_trace_metrics_params,
)
from mlflow.store.tracking.utils.sql_trace_rollups import (
    MS_PER_DAY,
    ROLLUP_ELIGIBILITY_LAG_MS,
    RollupFamily,
    configure_rollup_read_snapshot,
    enqueue_rollup_rebuild_partitions,
    enqueue_rollup_rebuilds,
    merge_unbucketed_data_points,
    order_and_limit_data_points,
    raw_aggregations_for_plan,
    requires_sql_grouped_merge,
    resolve_rollup_read,
    rollup_read_is_current,
    rollups_enabled,
    serve_rollup_read,
    serve_sql_grouped_span_cost_read,
)
from mlflow.store.tracking.utils.trace_analytics import (
    COST_COLUMN_BY_KEY,
    PROMOTED_TRACE_METADATA_KEYS,
    TOKEN_COLUMN_BY_KEY,
    TRACE_ANALYTICS_COLUMNS_BY_METADATA_KEY,
    analytics_columns_from_metadata,
    assessment_aggregate,
    bounded_model_dimension,
    compatibility_metadata_from_columns,
    finite_float_or_none,
    get_trace_analytics_metadata_filter,
    token_count_or_none,
    validate_session_id,
    validate_trace_name,
)
from mlflow.store.tracking.utils.trace_archival import (
    _TRACE_ARCHIVAL_EXPERIMENT_ID_CHUNK_SIZE,
    _ArchiveNowCleanupRequest,
    _ArchiveNowRemainingState,
    _ArchiveNowRequest,
    _format_trace_archival_duration_millis,
    _parse_trace_archival_duration_millis,
    _resolve_effective_trace_archival_retention,
    _TraceArchiveCandidate,
    _TraceDeleteSelection,
    _TraceReadSnapshot,
    _TraceSpanSnapshot,
)
from mlflow.store.tracking.utils.trace_cleanup import (
    delete_trace_associations,
)
from mlflow.telemetry.events import UpdateIssueEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.tracing.analysis import TraceFilterCorrelationResult
from mlflow.tracing.constant import (
    AssessmentMetadataKey,
    GenAiSemconvKey,
    SpanAttributeKey,
    SpansLocation,
    TraceArchivalFailureReason,
    TraceExperimentTagKey,
    TraceMetadataKey,
    TraceSizeStatsKey,
    TraceTagKey,
)
from mlflow.tracing.otel.otel_archival import TRACE_ARCHIVAL_FILENAME, spans_to_traces_data_pb
from mlflow.tracing.otel.translation import (
    translate_loaded_span,
    translate_span_when_storing,
)
from mlflow.tracing.utils import (
    SpanAggregationNode,
    TraceJSONEncoder,
    aggregate_cost_from_span_nodes,
    aggregate_usage_from_span_nodes,
    generate_request_id_v2,
    try_json_loads,
)
from mlflow.tracing.utils.artifact_utils import (
    get_archive_uri_for_trace,
)
from mlflow.tracing.utils.truncation import _get_truncated_preview, _replace_nul
from mlflow.utils import chunk_list
from mlflow.utils.issue_utils import (
    issue_category_hash,
    issue_keyset_token,
    issue_page_position,
    issue_page_token,
    issue_query_fingerprint,
    validate_issue_fields,
    validate_issue_search,
)
from mlflow.utils.mlflow_tags import (
    MLFLOW_ARTIFACT_LOCATION,
)
from mlflow.utils.search_utils import (
    SearchIssuesUtils,
    SearchTraceUtils,
    SearchUtils,
)
from mlflow.utils.thread_utils import map_with_context
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import (
    append_to_uri_path,
)
from mlflow.utils.validation import (
    _parse_experiment_id,
    _parse_experiment_ids,
    _resolve_experiment_ids_and_locations,
    _validate_trace_archival_location,
    _validate_trace_archival_retention_string,
    _validate_trace_tag,
    validate_delete_traces_parameters,
)
from mlflow.utils.workspace_context import get_request_workspace

_T = TypeVar("_T")

_logger = logging.getLogger(__name__)

_SPAN_SEARCH_TIMESTAMP_TOLERANCE_MS = 10_000


class SqlAlchemyTraceStore(AbstractTraceBackend):
    ARTIFACTS_FOLDER_NAME = "artifacts"
    TRACE_FOLDER_NAME = "traces"
    _TRACE_BATCH_QUERY_ID_CHUNK_SIZE = 400

    def __init__(self, tracking_store):
        self.set_tracking_store(tracking_store)

    @property
    def supports_trace_archival(self) -> bool:
        return True

    def _filter_trace_ids(self, trace_ids, filter_experiment_ids):
        if not self._tracking_store.supports_workspaces:
            return trace_ids
        return super()._filter_trace_ids(trace_ids, filter_experiment_ids)

    # Native SQL lifecycle cleanup runs in the tracking transaction or through foreign keys.
    def on_run_deleted(self, run_id: str) -> None:
        pass

    def on_run_permanently_deleted(self, run_id: str) -> None:
        pass

    def on_experiment_deleted(self, experiment_id: str) -> None:
        pass

    def on_experiment_permanently_deleted(self, experiment_id: str) -> None:
        pass

    def before_workspace_deleted(self, workspace_name: str, mode: WorkspaceDeletionMode) -> None:
        pass

    def on_workspace_deleted(self, workspace_name: str) -> None:
        pass

    def delete_run_assessments(self, session, run_id: str) -> None:
        source_run_id_pattern = f'"{AssessmentMetadataKey.SOURCE_RUN_ID}": "{run_id}"'
        assessment_filter = SqlAssessments.assessment_metadata.contains(source_run_id_pattern)
        self._enqueue_assessment_rebuilds_for_filter(session, assessment_filter)
        session.query(SqlAssessments).filter(assessment_filter).delete(synchronize_session=False)

    def _validate_trace_accessible(self, session, trace_id: str) -> None:
        """
        Hook for subclasses to validate trace access. No-op by default.

        In single-tenant mode, validation is not needed - the database will
        raise appropriate errors if the trace doesn't exist.
        """
        return

    def _enqueue_assessment_rebuilds_for_filter(self, session, assessment_filter) -> None:
        """Invalidate assessment partitions before bulk-deleting the matching rows.

        The rows can span multiple experiments and days, so group the denormalized
        ``experiment_id``/``trace_timestamp_ms`` by experiment and enqueue one
        ASSESSMENT rebuild batch per experiment while the source rows still exist.
        """
        if not rollups_enabled():
            return

        day_bucket = func.floor(SqlAssessments.trace_timestamp_ms / MS_PER_DAY)
        cutoff_ms = get_current_time_millis() - ROLLUP_ELIGIBILITY_LAG_MS
        affected_partitions = (
            session
            .query(SqlAssessments.experiment_id, day_bucket.label("day_bucket"))
            .filter(
                assessment_filter,
                SqlAssessments.experiment_id.isnot(None),
                SqlAssessments.trace_timestamp_ms.isnot(None),
                SqlAssessments.trace_timestamp_ms <= cutoff_ms,
            )
            .distinct()
            .order_by(SqlAssessments.experiment_id, day_bucket)
        )
        enqueue_rollup_rebuild_partitions(
            session,
            (
                (RollupFamily.ASSESSMENT, int(experiment_id), [int(bucket) * MS_PER_DAY])
                for experiment_id, bucket in affected_partitions
            ),
        )

    def _trace_query(self, session, for_update_or_delete=False, workspace=None):
        query = self._tracking_store._get_query(session, SqlTraceInfo)
        if for_update_or_delete:
            return self._apply_trace_row_lock(query)
        return query

    def _apply_trace_row_lock(self, query):
        if self._tracking_store.db_type == MSSQL:
            return query.with_hint(
                SqlTraceInfo,
                "WITH (UPDLOCK, ROWLOCK)",
                dialect_name="mssql",
            )
        return query.with_for_update()

    def _trace_row_lock_query(self, session, trace_ids: list[str]):
        """
        Build a locking SELECT over the given trace_info rows, held until the transaction ends.

        log_spans() recomputes the trace-level token usage and cost from the trace's stored spans
        and overwrites the metadata, which makes it a read-modify-write. Without this lock, two
        log_spans() calls for the same trace would each miss the other's uncommitted spans, and the
        last writer would drop the other batch's usage.

        Rows are locked in request_id order so that concurrent transactions acquire them in the
        same order, and callers must lock before writing any span rows so that the trace row is
        always locked ahead of span rows, matching the order the archival paths use. Together
        these reduce the likelihood of deadlocks between overlapping transactions.

        NB: this queries SqlTraceInfo directly instead of going through `_trace_query()`, whose
        workspace-aware override joins experiments. Locking through that join would also lock the
        experiment row and serialize every trace in the experiment. Workspace scoping stays
        enforced by the `_trace_query()` fetch that resolved these trace IDs, and only rows it
        resolved are locked here.
        """
        query = (
            session
            .query(SqlTraceInfo)
            .filter(SqlTraceInfo.request_id.in_(trace_ids))
            .order_by(SqlTraceInfo.request_id)
        )
        return self._apply_trace_row_lock(query)

    def _stored_span_rows_query(self, session, trace_ids: list[str]):
        """
        Fetch the stored span rows log_spans() recomputes trace-level token usage and cost from.

        On MySQL this must be a locking read. Under InnoDB's default REPEATABLE READ, a plain
        SELECT is served from the snapshot taken at the transaction's first read, which predates
        the trace row lock, so it could miss spans committed by a log_spans() call that held the
        lock before us. A locking read always reads the latest committed rows. The other backends
        already see committed data once the lock wait ends (READ COMMITTED on PostgreSQL and SQL
        Server, a single writer on SQLite), so they keep the plain SELECT and avoid taking span
        row locks.
        """
        query = session.query(
            SqlSpan.trace_id,
            SqlSpan.span_id,
            SqlSpan.parent_span_id,
            SqlSpan.content,
        ).filter(
            SqlSpan.trace_id.in_(trace_ids),
            SqlSpan.content != "",
        )
        if self._tracking_store.db_type == MYSQL:
            query = query.with_for_update()
        return query

    def _get_trace_artifact_location_tag(self, experiment, trace_id: str) -> SqlTraceTag:
        # Validate new trace ids before embedding them in the artifact location, like
        # FileStore.start_trace does.
        _validate_new_trace_id(trace_id)
        # Trace data is stored as file artifacts regardless of the tracking backend choice.
        # We use subdirectory "/traces" under the experiment's artifact location to isolate
        # them from run artifacts.
        artifact_uri = append_to_uri_path(
            experiment.artifact_location,
            self.TRACE_FOLDER_NAME,
            trace_id,
            self.ARTIFACTS_FOLDER_NAME,
        )
        return SqlTraceTag(request_id=trace_id, key=MLFLOW_ARTIFACT_LOCATION, value=artifact_uri)

    def start_trace(self, trace_info: "TraceInfo") -> TraceInfo:
        """
        Create a trace using the V3 API format with a complete Trace object.

        Args:
            trace_info: The TraceInfo object to create in the backend.

        Returns:
            The created TraceInfo object from the backend.
        """
        # Retry on DB deadlocks so a concurrent log_spans()/start_trace() race does not
        # drop the trace. Each attempt opens a fresh managed session.
        return self._tracking_store._run_with_deadlock_retry(self._start_trace_once, trace_info)

    def _start_trace_once(self, trace_info: "TraceInfo") -> TraceInfo:
        trace_name = validate_trace_name(trace_info.tags.get(TraceTagKey.TRACE_NAME))
        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            experiment = self._tracking_store.get_experiment(trace_info.experiment_id)
            self._tracking_store._check_experiment_is_active(experiment)

            # Use the provided trace_id
            trace_id = trace_info.trace_id

            request_metadata = dict(trace_info.trace_metadata.items())
            analytics_columns = analytics_columns_from_metadata(request_metadata)

            # Create SqlTraceInfo with V3 fields directly
            sql_trace_info = SqlTraceInfo(
                request_id=trace_id,
                experiment_id=trace_info.experiment_id,
                timestamp_ms=trace_info.request_time,
                execution_time_ms=trace_info.execution_duration,
                status=trace_info.state.value,
                client_request_id=trace_info.client_request_id,
                request_preview=_replace_nul(trace_info.request_preview),
                response_preview=_replace_nul(trace_info.response_preview),
                trace_name=trace_name,
                **analytics_columns,
            )

            tags = [
                SqlTraceTag(request_id=trace_id, key=k, value=v)
                for k, v in trace_info.tags.items()
                if k != TraceTagKey.TRACE_NAME
            ] + [self._get_trace_artifact_location_tag(experiment, trace_id)]
            sql_trace_info.tags = tags

            for key in PROMOTED_TRACE_METADATA_KEYS:
                request_metadata.pop(key, None)

            # Signal that start_trace() has finalized top-level trace fields. Optional analytics
            # groups are authoritative only when supplied, so log_spans() may still fill missing
            # session, token usage, or cost values.
            request_metadata[TraceMetadataKey.TRACE_INFO_FINALIZED] = "true"

            # The caller may not always specify a trace_id on each assessment when
            # exporting traces with assessments, so backfill it on the SQL entity.
            sql_assessments = []
            for a in trace_info.assessments:
                sql_assessment = SqlAssessments.from_mlflow_entity(a)
                sql_assessment.trace_id = trace_id
                sql_assessment.experiment_id = int(trace_info.experiment_id)
                sql_assessment.trace_timestamp_ms = trace_info.request_time
                sql_assessments.append(sql_assessment)
            sql_trace_info.assessments = sql_assessments

            new_rollup_partitions = [
                (
                    family,
                    int(trace_info.experiment_id),
                    [trace_info.request_time],
                )
                for family in (RollupFamily.TRACE_METRIC, RollupFamily.ASSESSMENT)
            ]
            enqueue_rollup_rebuild_partitions(session, new_rollup_partitions)
            try:
                # Happy path: attach metadata via cascade for a single flush.
                # Emit rows in sorted key order so concurrent writers acquire the
                # trace_request_metadata PK-index locks in a consistent order across
                # transactions and cannot deadlock.
                sql_trace_info.request_metadata = [
                    SqlTraceMetadata(request_id=trace_id, key=k, value=v)
                    for k, v in sorted(request_metadata.items())
                ]
                session.add(sql_trace_info)
                session.flush()
            except IntegrityError:
                # Trace already exists (likely created by log_spans() racing with
                # start_trace()). Roll back the failed INSERT, lock and reread the
                # persistent row, then merge child rows additively so we never
                # mutate trace state from a stale snapshot or drop assessments
                # that were already attached to the trace.
                session.rollback()
                session.expunge_all()
                # Rebuild child rows after expunging the failed parent tree so later
                # per-row merges cannot drag its stale trace_info state back in.
                tags = [
                    SqlTraceTag(request_id=trace_id, key=tag.key, value=tag.value) for tag in tags
                ]
                sql_assessments = []
                for a in trace_info.assessments:
                    sql_assessment = SqlAssessments.from_mlflow_entity(a)
                    sql_assessment.trace_id = trace_id
                    sql_assessment.experiment_id = int(trace_info.experiment_id)
                    sql_assessment.trace_timestamp_ms = trace_info.request_time
                    sql_assessments.append(sql_assessment)
                trace_write_workspace = self._tracking_store._get_active_workspace()
                if rollups_enabled():
                    # Snapshot the old source partitions without taking source-row locks, then
                    # lock every old and new rebuild key before the read-modify-write lock below.
                    # A concurrent queue-first writer either commits before this lock acquisition
                    # or leaves its own marker behind, so a snapshot race cannot lose invalidation.
                    db_trace_snapshot = (
                        self
                        ._trace_query(session, workspace=trace_write_workspace)
                        .filter(SqlTraceInfo.request_id == trace_id)
                        .one_or_none()
                    )
                    if db_trace_snapshot is None:
                        raise MlflowException(
                            f"Trace with ID '{trace_id}' no longer exists.",
                            error_code=RESOURCE_DOES_NOT_EXIST,
                        )
                    previous_partition = (
                        int(db_trace_snapshot.experiment_id),
                        db_trace_snapshot.timestamp_ms,
                    )
                    previous_span_partitions = [
                        (int(experiment_id), int(start_time_unix_nano) // 1_000_000)
                        for experiment_id, start_time_unix_nano in session.query(
                            SqlSpan.experiment_id, SqlSpan.start_time_unix_nano
                        ).filter(
                            SqlSpan.trace_id == trace_id,
                            SqlSpan.experiment_id.isnot(None),
                            SqlSpan.start_time_unix_nano.isnot(None),
                        )
                    ]
                    enqueue_rollup_rebuild_partitions(
                        session,
                        [
                            *new_rollup_partitions,
                            *[
                                (family, previous_partition[0], [previous_partition[1]])
                                for family in (
                                    RollupFamily.TRACE_METRIC,
                                    RollupFamily.ASSESSMENT,
                                )
                            ],
                            *[
                                (RollupFamily.SPAN_COST, experiment_id, [timestamp_ms])
                                for experiment_id, timestamp_ms in previous_span_partitions
                            ],
                            *[
                                (
                                    RollupFamily.SPAN_COST,
                                    int(trace_info.experiment_id),
                                    [timestamp_ms],
                                )
                                for _, timestamp_ms in previous_span_partitions
                            ],
                        ],
                    )
                # Lock the reread because this is the read half of a read-modify-write
                # merge on trace_info, not a passive lookup. The db_payload_generation bump below
                # coordinates DB-backed payload generation, but this row lock also prevents
                # us from preserving top-level trace fields from a stale snapshot.
                db_sql_trace_info = (
                    self
                    ._trace_query(
                        session,
                        for_update_or_delete=True,
                        workspace=trace_write_workspace,
                    )
                    .filter(SqlTraceInfo.request_id == trace_id)
                    .one_or_none()
                )
                if db_sql_trace_info is None:
                    raise MlflowException(
                        f"Trace with ID '{trace_id}' no longer exists.",
                        error_code=RESOURCE_DOES_NOT_EXIST,
                    )
                if str(db_sql_trace_info.experiment_id) != str(trace_info.experiment_id):
                    # Only merge a racing write into the same experiment's trace; never move
                    # another experiment's trace into the caller's.
                    raise MlflowException(
                        f"Trace with ID '{trace_id}' already exists in another experiment.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                if not self._is_sql_trace_db_backed(db_sql_trace_info):
                    raise MlflowException(
                        f"Cannot update traces that are no longer DB-backed: '{trace_id}'.",
                        error_code=INVALID_STATE,
                    )
                # Assessments are merged by ID alone, so one that belongs to another trace would
                # be moved here and overwritten.
                if assessment_ids := [a.assessment_id for a in sql_assessments]:
                    if foreign := (
                        session
                        .query(SqlAssessments.assessment_id)
                        .filter(
                            SqlAssessments.assessment_id.in_(assessment_ids),
                            SqlAssessments.trace_id != trace_id,
                        )
                        .first()
                    ):
                        raise MlflowException(
                            f"Assessment with ID '{foreign.assessment_id}' belongs to another "
                            "trace.",
                            error_code=INVALID_PARAMETER_VALUE,
                        )
                # Advance the payload generation before staging ORM writes so a concurrent archival
                # finalization cannot be hidden by autoflush re-publishing TRACKING_STORE.
                self._advance_db_payload_generations_for_db_span_writes(session, [trace_id])

                db_sql_trace_info.timestamp_ms = trace_info.request_time
                db_sql_trace_info.execution_time_ms = trace_info.execution_duration
                db_sql_trace_info.status = trace_info.state.value
                db_sql_trace_info.client_request_id = trace_info.client_request_id
                if TraceTagKey.TRACE_NAME in trace_info.tags:
                    db_sql_trace_info.trace_name = trace_name
                for column, value in analytics_columns.items():
                    setattr(db_sql_trace_info, column, value)
                if trace_info.request_preview is not None:
                    db_sql_trace_info.request_preview = _replace_nul(trace_info.request_preview)
                if trace_info.response_preview is not None:
                    db_sql_trace_info.response_preview = _replace_nul(trace_info.response_preview)

                for tag in tags:
                    session.merge(tag)
                for assessment in sql_assessments:
                    session.merge(assessment)

                # Upsert metadata individually so the complete data
                # from start_trace() overwrites any partial values from log_spans().
                # Merge in sorted key order to keep PK-index lock acquisition consistent
                # across transactions and avoid deadlocks.
                for k, v in sorted(request_metadata.items()):
                    session.merge(SqlTraceMetadata(request_id=trace_id, key=k, value=v))
                session.query(SqlSpan).filter(SqlSpan.trace_id == trace_id).update(
                    {SqlSpan.experiment_id: int(trace_info.experiment_id)},
                    synchronize_session=False,
                )
                session.query(SqlAssessments).filter(SqlAssessments.trace_id == trace_id).update(
                    {
                        SqlAssessments.experiment_id: int(trace_info.experiment_id),
                        SqlAssessments.trace_timestamp_ms: trace_info.request_time,
                    },
                    synchronize_session=False,
                )
                session.flush()
                sql_trace_info = self._get_sql_trace_info(
                    session,
                    trace_id,
                    workspace=trace_write_workspace,
                )

            return sql_trace_info.to_mlflow_entity()

    def get_trace_info(self, trace_id: str) -> TraceInfo:
        """
        Fetch the trace info for the given trace id.

        Args:
            trace_id: Unique string identifier of the trace.

        Returns:
            The TraceInfo object.
        """
        with self._tracking_store.ManagedSessionMaker() as session:
            trace_row = (
                self
                ._trace_query(session)
                .with_entities(*_TRACE_INFO_COLUMNS)
                .filter(SqlTraceInfo.request_id == trace_id)
                .one_or_none()
            )
            if trace_row is None:
                raise MlflowException(
                    f"Trace with ID '{trace_id}' not found.",
                    RESOURCE_DOES_NOT_EXIST,
                )
            return _build_trace_infos_from_rows(session, [trace_row])[0]

    def _get_sql_trace_info(
        self, session, trace_id, workspace=None, *, for_update=False
    ) -> SqlTraceInfo:
        query = self._trace_query(
            session,
            for_update_or_delete=for_update,
            workspace=workspace,
        )
        # A locking reread is used after queue-key acquisition. Force SQLAlchemy to refresh an
        # object already present in the identity map so routing comparisons never use its earlier
        # optimistic snapshot.
        if for_update:
            query = query.populate_existing()
        sql_trace_info = query.filter(SqlTraceInfo.request_id == trace_id).one_or_none()
        if sql_trace_info is None:
            raise MlflowException(
                f"Trace with ID '{trace_id}' not found.",
                RESOURCE_DOES_NOT_EXIST,
            )
        return sql_trace_info

    def _apply_trace_filter_clauses(
        self,
        statement: _SqlAlchemyStatement,
        attribute_filters: list[ColumnElement],
        non_attribute_filters: list[Subquery],
        span_attribute_filters: list[ColumnElement],
        span_filters: list[Subquery],
        run_id_filter: str | None,
    ) -> _SqlAlchemyStatement:
        """
        Apply trace filter clauses to a SQLAlchemy statement.

        This helper consolidates the logic for applying trace filters that is shared
        between search_traces(), find_completed_sessions(), and correlation queries.

        Args:
            statement: SQLAlchemy statement (Select or Query) to apply filters to
            attribute_filters: List of attribute filter conditions (e.g., WHERE clauses)
            non_attribute_filters: List of subqueries for tag/metadata filters to join
            span_attribute_filters: Filters combined in a correlated SqlSpan EXISTS clause
            span_filters: List of subqueries for non-SqlSpan filters to join
            run_id_filter: Optional run_id to filter by

        Returns:
            Modified statement with all filters applied (same type as input)
        """
        # Apply non-attribute filters (tags and metadata)
        for non_attr_filter in non_attribute_filters:
            statement = statement.join(non_attr_filter)

        # Apply direct span filters in a single correlated EXISTS clause so combined
        # predicates target the same span without duplicating matching trace rows.
        if span_attribute_filters:
            statement = statement.filter(
                exists().where(
                    SqlSpan.trace_id == SqlTraceInfo.request_id,
                    *span_attribute_filters,
                )
            )

        # Apply subquery-backed span filters with explicit join condition
        for span_filter in span_filters:
            statement = statement.join(
                span_filter, SqlTraceInfo.request_id == span_filter.c.request_id
            )

        # Build the filter conditions
        filter_conditions = [*attribute_filters]

        # If run_id filter is present, we need to handle it specially to include linked traces
        if run_id_filter:
            # Create a subquery to check if a trace is linked to the run via entity associations
            linked_trace_exists = exists().where(
                (SqlEntityAssociation.source_id == SqlTraceInfo.request_id)
                & (SqlEntityAssociation.source_type == EntityAssociationType.TRACE)
                & (SqlEntityAssociation.destination_type == EntityAssociationType.RUN)
                & (SqlEntityAssociation.destination_id == run_id_filter)
            )

            # Create a subquery to check if trace has run_id in metadata
            metadata_exists = exists().where(
                (SqlTraceMetadata.request_id == SqlTraceInfo.request_id)
                & (SqlTraceMetadata.key == TraceMetadataKey.SOURCE_RUN)
                & (SqlTraceMetadata.value == run_id_filter)
            )

            # If run_id filter is present, add OR condition for linked traces
            filter_conditions.append(
                or_(
                    linked_trace_exists,  # Trace is linked via entity associations
                    metadata_exists,  # Trace has run_id in metadata
                )
            )

        # Apply all filter conditions
        if filter_conditions:
            statement = statement.filter(*filter_conditions)

        return statement

    def search_traces(
        self,
        experiment_ids: list[str] | None = None,
        filter_string: str | None = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: list[str] | None = None,
        page_token: str | None = None,
        model_id: str | None = None,
        locations: list[str] | None = None,
    ) -> tuple[list[TraceInfo], str | None]:
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
            locations: A list of locations to search over. To search over experiments, provide
                a list of experiment IDs.

        Returns:
            A tuple of a list of :py:class:`TraceInfo <mlflow.entities.TraceInfo>` objects that
            satisfy the search expressions and a pagination token for the next page of results.
        """
        locations = _resolve_experiment_ids_and_locations(experiment_ids, locations)
        self._tracking_store._validate_max_results_param(max_results)

        with self._tracking_store.ManagedSessionMaker() as session:
            if locations is not None:
                locations = self._tracking_store._filter_experiment_ids(session, locations)
                locations = [int(e) for e in locations]
                location_filter = SqlTraceInfo.experiment_id.in_(locations)
            else:
                location_filter = None

            cases_orderby, parsed_orderby, sorting_joins = _get_orderby_clauses_for_search_traces(
                order_by or [], session
            )
            stmt = self._trace_query(session).with_entities(*_TRACE_INFO_COLUMNS)
            if cases_orderby:
                stmt = stmt.add_columns(*cases_orderby)

            scoped_trace_query = self._trace_query(session)
            if locations is not None:
                scoped_trace_query = scoped_trace_query.filter(location_filter)
            (
                attribute_filters,
                non_attribute_filters,
                span_attribute_filters,
                span_filters,
                run_id_filter,
            ) = _get_filter_clauses_for_search_traces(
                filter_string, session, self._tracking_store._get_dialect(), scoped_trace_query
            )

            stmt = self._apply_trace_filter_clauses(
                stmt,
                attribute_filters,
                non_attribute_filters,
                span_attribute_filters,
                span_filters,
                run_id_filter,
            )

            # using an outer join is necessary here because we want to be able to sort
            # on a column (tag, metric or param) without removing the lines that
            # do not have a value for this column (which is what inner join would do)
            for j in sorting_joins:
                stmt = stmt.outerjoin(j)

            offset = SearchTraceUtils.parse_start_offset_from_page_token(page_token)
            if location_filter is not None:
                stmt = stmt.filter(location_filter)

            # One row past the page, so the token is issued only when something actually follows
            # it. Reading exactly `max_results` cannot tell a full page with more behind it from
            # one that ends on the boundary, and every other paginated search in this store
            # already looks ahead this way.
            stmt = stmt.order_by(*parsed_orderby).offset(offset).limit(max_results + 1)

            trace_rows = stmt.all()
            has_more = len(trace_rows) > max_results
            # Built after trimming: the look-ahead row is only ever counted, and building its
            # trace info would read its tags, assessments and request metadata for nothing.
            trace_infos = _build_trace_infos_from_rows(session, trace_rows[:max_results])

            # Compute next search token
            if has_more:
                final_offset = offset + max_results
                next_token = SearchTraceUtils.create_page_token(final_offset)
            else:
                next_token = None

            return trace_infos, next_token

    def find_completed_sessions(
        self,
        experiment_id: str,
        min_last_trace_timestamp_ms: int,
        max_last_trace_timestamp_ms: int,
        max_results: int | None = None,
        filter_string: str | None = None,
    ) -> list[CompletedSession]:
        """
        Find completed sessions based on their last trace timestamp.

        A completed session is one whose last trace timestamp falls within the specified
        time window [min_last_trace_timestamp_ms, max_last_trace_timestamp_ms] and has
        no traces after max_last_trace_timestamp_ms (i.e., the session is not ongoing).

        Sessions are ordered by (last_trace_timestamp_ms ASC, session_id ASC) to ensure
        deterministic and stable ordering, especially when timestamp ties occur. This is
        useful when repeatedly calling this method with a ``max_results`` limit.

        Args:
            experiment_id: The experiment to search.
            min_last_trace_timestamp_ms: Lower bound for session's last trace timestamp (inclusive).
                Sessions with last trace before this time are excluded.
            max_last_trace_timestamp_ms: Upper bound for session's last trace timestamp (inclusive).
                Sessions with any traces after this time are excluded.
            max_results: Maximum number of sessions to return. If None, returns all
                matching sessions.
            filter_string: Optional search filter string to apply to the first trace
                in each session. Only sessions whose first trace matches this filter
                will be returned.

        Returns:
            List of CompletedSession objects sorted by (last_trace_timestamp_ms ASC,
            session_id ASC).
        """
        with self._tracking_store.ManagedSessionMaker() as session:
            experiment_id_int = _parse_experiment_id(experiment_id)

            experiment_ids = self._tracking_store._filter_experiment_ids(
                session, [experiment_id_int]
            )
            if not experiment_ids:
                return []
            experiment_id_int = experiment_ids[0]

            # Example: Given min=200, max=400
            #   Session A (traces at [100, 200, 300]): last=300, no traces >400 → INCLUDED
            #   Session B (traces at [100, 200, 500]): has trace >400 → EXCLUDED (ongoing)
            #   Session C (traces at [50, 100]): no traces >=200 → EXCLUDED (too old)

            # Step 1: Find sessions with at least one trace >= min timestamp (optimization)
            # Example: Sessions A, B pass (have traces >=200); Session C excluded
            candidate_sessions = self._build_candidate_sessions_subquery(
                session=session,
                experiment_id=experiment_id_int,
                min_last_trace_timestamp_ms=min_last_trace_timestamp_ms,
            )

            # Step 2: Optional filter on first trace (e.g., tags.myTag = "value")
            filtered_sessions = self._build_first_trace_filter_subquery(
                session=session,
                experiment_id=experiment_id_int,
                filter_string=filter_string,
                candidate_sessions=candidate_sessions,
            )

            # Step 3: Compute first/last trace timestamps for each session
            # Example: Session A → {first: 100, last: 300}, Session B → {first: 100, last: 500}
            sessions_with_stats = self._build_session_stats_subquery(
                session=session,
                experiment_id=experiment_id_int,
                sessions=filtered_sessions,
            )

            # Step 4: Get completed sessions (last trace <= max timestamp)
            # Example: Session A (last=300 <= 400) → INCLUDED
            #          Session B (last=500 > 400) → EXCLUDED
            query = self._build_completed_sessions_query(
                session=session,
                sessions_with_stats=sessions_with_stats,
                max_last_trace_timestamp_ms=max_last_trace_timestamp_ms,
                max_results=max_results,
            )

            results = query.all()

            return [
                CompletedSession(
                    session_id=row.session_id,
                    first_trace_timestamp_ms=row.first_trace_timestamp_ms,
                    last_trace_timestamp_ms=row.last_trace_timestamp_ms,
                )
                for row in results
            ]

    def _build_candidate_sessions_subquery(
        self,
        session: Session,
        experiment_id: str,
        min_last_trace_timestamp_ms: int,
    ) -> Subquery:
        """
        Build subquery for sessions with at least one trace after the minimum timestamp.

        This optimization avoids aggregating stats for sessions with no recent traces.
        """
        return (
            session
            .query(SqlTraceInfo.session_id.label("session_id"))
            .filter(
                SqlTraceInfo.experiment_id == experiment_id,
                SqlTraceInfo.timestamp_ms >= min_last_trace_timestamp_ms,
                SqlTraceInfo.session_id.isnot(None),
            )
            .distinct()
            .subquery()
        )

    def _build_first_trace_filter_subquery(
        self,
        session: Session,
        experiment_id: str,
        filter_string: str | None,
        candidate_sessions: Subquery,
    ) -> Subquery:
        """
        Build subquery for sessions whose first trace matches the filter.

        Returns candidate_sessions unchanged if no filter_string provided.
        """
        if not filter_string:
            return candidate_sessions

        # Parse the filter string to get filter clauses
        scoped_trace_query = self._trace_query(session).filter(
            SqlTraceInfo.experiment_id == experiment_id
        )
        (
            attribute_filters,
            non_attribute_filters,
            span_attribute_filters,
            span_filters,
            run_id_filter,
        ) = _get_filter_clauses_for_search_traces(
            filter_string, session, self._tracking_store._get_dialect(), scoped_trace_query
        )

        # Subquery: first trace timestamp for each session
        first_traces = (
            self
            ._trace_query(session)
            .with_entities(
                SqlTraceInfo.session_id.label("session_id"),
                func.min(SqlTraceInfo.timestamp_ms).label("first_timestamp"),
            )
            .join(
                candidate_sessions,
                SqlTraceInfo.session_id == candidate_sessions.c.session_id,
            )
            .filter(
                SqlTraceInfo.experiment_id == experiment_id,
            )
            .group_by(SqlTraceInfo.session_id)
            .subquery()
        )

        # Subquery: filter first traces using the parsed filter
        filtered_trace_query = self._trace_query(session).with_entities(
            SqlTraceInfo.session_id.label("session_id")
        )

        filtered_trace_query = self._apply_trace_filter_clauses(
            filtered_trace_query,
            attribute_filters,
            non_attribute_filters,
            span_attribute_filters,
            span_filters,
            run_id_filter,
        )

        # Join with first_traces to match only the first trace in each session
        filtered_trace_query = filtered_trace_query.join(
            first_traces,
            (SqlTraceInfo.session_id == first_traces.c.session_id)
            & (SqlTraceInfo.timestamp_ms == first_traces.c.first_timestamp),
        )

        # Apply session-specific filters
        return (
            filtered_trace_query
            .filter(
                SqlTraceInfo.experiment_id == experiment_id,
            )
            .distinct()
            .subquery()
        )

    def _build_session_stats_subquery(
        self,
        session: Session,
        experiment_id: str,
        sessions: Subquery,
    ) -> Subquery:
        """
        Build subquery aggregating first/last trace timestamps for sessions.
        """
        stats_query = (
            self
            ._trace_query(session)
            .with_entities(
                SqlTraceInfo.session_id.label("session_id"),
                func.min(SqlTraceInfo.timestamp_ms).label("first_trace_timestamp_ms"),
                func.max(SqlTraceInfo.timestamp_ms).label("last_trace_timestamp_ms"),
            )
            .join(
                sessions,
                SqlTraceInfo.session_id == sessions.c.session_id,
            )
        )

        return (
            stats_query
            .filter(SqlTraceInfo.experiment_id == experiment_id)
            .group_by(SqlTraceInfo.session_id)
            .subquery()
        )

    def _build_completed_sessions_query(
        self,
        session: Session,
        sessions_with_stats: Subquery,
        max_last_trace_timestamp_ms: int,
        max_results: int | None,
    ) -> Query:
        """
        Build main query for completed sessions.

        Returns sessions where last trace <= max timestamp, ordered by
        (last_trace_timestamp_ms ASC, session_id ASC) for deterministic pagination.
        """
        query = (
            session
            .query(
                sessions_with_stats.c.session_id,
                sessions_with_stats.c.first_trace_timestamp_ms,
                sessions_with_stats.c.last_trace_timestamp_ms,
            )
            .filter(
                sessions_with_stats.c.last_trace_timestamp_ms <= max_last_trace_timestamp_ms,
            )
            .order_by(
                sessions_with_stats.c.last_trace_timestamp_ms.asc(),
                # Use session_id as tiebreaker for deterministic ordering when multiple
                # sessions have the same timestamp. This ensures checkpoint resume works
                # correctly when max_results is hit in the middle of a timestamp group.
                sessions_with_stats.c.session_id.asc(),
            )
        )

        if max_results is not None:
            query = query.limit(max_results)

        return query

    def query_trace_metrics(
        self,
        experiment_ids: list[str],
        view_type: MetricViewType,
        metric_name: str,
        aggregations: list[MetricAggregation],
        dimensions: list[str] | None = None,
        filters: list[str] | None = None,
        time_interval_seconds: int | None = None,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        max_results: int = MAX_RESULTS_QUERY_TRACE_METRICS,
        page_token: str | None = None,
    ) -> PagedList[list[MetricDataPoint]]:
        validate_query_trace_metrics_params(view_type, metric_name, aggregations, dimensions)

        if time_interval_seconds and (start_time_ms is None or end_time_ms is None):
            raise MlflowException.invalid_parameter_value(
                "start_time_ms and end_time_ms are required if time_interval_seconds is set"
            )

        with self._tracking_store.ManagedSessionMaker() as session:
            # Rollup routing reads coverage, rebuild state, rollup rows, and raw gaps in separate
            # statements. Establish their shared snapshot before the first database read.
            configure_rollup_read_snapshot(session, self._tracking_store.db_type)
            experiment_ids_int = self._tracking_store._filter_experiment_ids(
                session, _parse_experiment_ids(experiment_ids)
            )

            def raw_query():
                if view_type == MetricViewType.ASSESSMENTS:
                    return session.query(SqlAssessments).filter(
                        SqlAssessments.experiment_id.in_(experiment_ids_int)
                    )
                return self._trace_query(session).filter(
                    SqlTraceInfo.experiment_id.in_(experiment_ids_int)
                )

            def raw_range_points(
                ranges: list[tuple[int | None, int | None]],
                range_aggregations: list[MetricAggregation] | None = None,
            ):
                return query_metrics(
                    view_type=view_type,
                    db_type=self._tracking_store.db_type,
                    query=raw_query(),
                    metric_name=metric_name,
                    aggregations=(
                        range_aggregations if range_aggregations is not None else aggregations
                    ),
                    dimensions=dimensions,
                    filters=filters,
                    time_interval_seconds=time_interval_seconds,
                    max_results=max_results,
                    time_ranges_ms=ranges,
                    accessible_experiment_ids=experiment_ids_int,
                )

            full_raw_range = [(start_time_ms, end_time_ms)]

            # Opt-in fast path: serve whole covered UTC days from precomputed rollups and query only
            # the partial-day edges (and any not-yet-built days) raw. resolve_rollup_read returns
            # None whenever rollups are disabled or the request is not exactly rollup-servable, so
            # the raw path below stays authoritative and results match the rollups-disabled case.
            plan = resolve_rollup_read(
                view_type=view_type,
                metric_name=metric_name,
                aggregations=aggregations,
                dimensions=dimensions,
                filters=filters,
                time_interval_seconds=time_interval_seconds,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                experiment_ids=experiment_ids_int,
                db_type=self._tracking_store.db_type,
            )
            sql_grouped_merge = plan is not None and requires_sql_grouped_merge(plan)
            if sql_grouped_merge:
                served = serve_sql_grouped_span_cost_read(
                    session, plan, self._tracking_store.db_type, max_results
                )
            else:
                served = serve_rollup_read(session, plan) if plan is not None else None

            if served is not None:
                raw_points = []
                if not sql_grouped_merge:
                    range_aggregations = raw_aggregations_for_plan(plan)
                    raw_points = (
                        raw_range_points(served.raw_ranges, range_aggregations)
                        if served.raw_ranges
                        else []
                    )

                if not rollup_read_is_current(session, plan, served.served_day_starts_ms):
                    _logger.debug(
                        "SQL trace rollup routing selected the full raw path for view=%s "
                        "metric=%s: "
                        "a served day was invalidated during the read",
                        view_type.value,
                        metric_name,
                    )
                    data_points = raw_range_points(full_raw_range)
                elif sql_grouped_merge:
                    # This path already combined rollup and raw-gap contributions under the
                    # database's string collation, including global ordering and limiting.
                    data_points = served.data_points
                elif plan.bucketed:
                    # Match the single-query raw path: order by time bucket (then grouping
                    # dimensions) and apply the global max_results after merging rollup and raw
                    # contributions.
                    data_points = order_and_limit_data_points(
                        [*served.data_points, *raw_points], max_results
                    )
                else:
                    data_points = merge_unbucketed_data_points(
                        plan, served.data_points, raw_points, max_results
                    )
            else:
                data_points = raw_range_points(full_raw_range)

            # TODO: Implement pagination with page_token
            return PagedList(data_points, None)

    def set_trace_tag(self, trace_id: str, key: str, value: str):
        """
        Set a tag on the trace with the given trace_id.

        Args:
            trace_id: The ID of the trace.
            key: The string key of the tag.
            value: The string value of the tag.
        """
        key, value = _validate_trace_tag(key, value)
        if key == TraceTagKey.TRACE_NAME:
            value = validate_trace_name(value)
        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            self._validate_trace_accessible(session, trace_id)
            if key == TraceTagKey.TRACE_NAME:
                sql_trace_info = self._get_sql_trace_info(session, trace_id)
                sql_trace_info.trace_name = value
            else:
                session.merge(SqlTraceTag(request_id=trace_id, key=key, value=value))

    def delete_trace_tag(self, trace_id: str, key: str):
        """
        Delete a tag on the trace with the given trace_id.

        Args:
            trace_id: The ID of the trace.
            key: The string key of the tag.
        """
        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            self._validate_trace_accessible(session, trace_id)
            if key == TraceTagKey.TRACE_NAME:
                sql_trace_info = self._get_sql_trace_info(session, trace_id)
                if sql_trace_info.trace_name is None:
                    raise MlflowException(
                        f"No trace tag with key '{key}' for trace with ID '{trace_id}'",
                        RESOURCE_DOES_NOT_EXIST,
                    )
                sql_trace_info.trace_name = None
                return
            deleted = (
                session
                .query(SqlTraceTag)
                .filter_by(request_id=trace_id, key=key)
                .delete(synchronize_session=False)
            )
            if deleted == 0:
                raise MlflowException(
                    f"No trace tag with key '{key}' for trace with ID '{trace_id}'",
                    RESOURCE_DOES_NOT_EXIST,
                )

    def delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: int | None = None,
        max_traces: int | None = None,
        trace_ids: list[str] | None = None,
    ) -> int:
        """
        Delete traces based on the specified criteria.

        Args:
            experiment_id: ID of the associated experiment.
            max_timestamp_millis: The maximum timestamp in milliseconds since the UNIX epoch for
                deleting traces. Traces older than or equal to this timestamp will be deleted.
            max_traces: The maximum number of traces to delete.
            trace_ids: A set of request IDs to delete.

        Returns:
            The number of traces deleted.
        """
        validate_delete_traces_parameters(max_timestamp_millis, max_traces, trace_ids)
        experiment_id_int = _parse_experiment_id(experiment_id)
        deleted_db_backed_count = 0
        selected_archived_traces: list[_TraceDeleteSelection] = []
        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            filters = [SqlTraceInfo.experiment_id == experiment_id_int]
            if max_timestamp_millis is not None:
                filters.append(SqlTraceInfo.timestamp_ms <= max_timestamp_millis)
            if trace_ids:
                filters.append(SqlTraceInfo.request_id.in_(trace_ids))
            if max_traces is not None:
                limited_subquery = (
                    self
                    ._trace_query(session)
                    .with_entities(SqlTraceInfo.request_id)
                    .filter(*filters)
                    .order_by(SqlTraceInfo.timestamp_ms, SqlTraceInfo.request_id)
                    .limit(max_traces)
                    .subquery()
                )
                filters.append(SqlTraceInfo.request_id.in_(select(limited_subquery.c.request_id)))

            use_rollups = rollups_enabled()
            selected_trace_ids = self._select_trace_ids_for_delete(
                session=session,
                filters=filters,
                lock=not use_rollups,
            )
            if not selected_trace_ids:
                return 0

            if use_rollups:
                self._enqueue_rollup_rebuilds_for_trace_delete(
                    session, experiment_id_int, selected_trace_ids
                )
                selected_trace_ids = self._lock_trace_ids_for_delete(session, selected_trace_ids)
                if not selected_trace_ids:
                    return 0

            # Classify selected traces while the trace rows remain locked, then delete the
            # DB-backed subset before any slow object-store cleanup begins.
            selected_archived_traces = self._select_archived_traces_for_delete(
                session=session,
                trace_ids=selected_trace_ids,
            )
            archived_trace_ids = {
                selected_trace.trace_id for selected_trace in selected_archived_traces
            }
            db_backed_trace_ids = [
                trace_id for trace_id in selected_trace_ids if trace_id not in archived_trace_ids
            ]
            if db_backed_trace_ids:
                deleted_db_backed_count = (
                    session
                    .query(SqlTraceInfo)
                    .filter(SqlTraceInfo.request_id.in_(db_backed_trace_ids))
                    .delete(synchronize_session=False)
                )
                self._tracking_store._delete_review_queue_items_for_traces(
                    session, db_backed_trace_ids
                )
                delete_trace_associations(session, db_backed_trace_ids)

        if not selected_archived_traces:
            return deleted_db_backed_count

        # Archived traces stay for a second phase so payload cleanup can happen outside the
        # transaction. Missing archived files are still treated as successful cleanup so
        # concurrent deleters can converge safely.
        deleted_archived_trace_ids = self._delete_archived_trace_payloads(selected_archived_traces)
        if not deleted_archived_trace_ids:
            return deleted_db_backed_count

        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            self._enqueue_rollup_rebuilds_for_trace_delete(
                session, experiment_id_int, deleted_archived_trace_ids
            )
            deleted_archived_count = (
                session
                .query(SqlTraceInfo)
                .filter(SqlTraceInfo.request_id.in_(deleted_archived_trace_ids))
                .delete(synchronize_session=False)
            )
            self._tracking_store._delete_review_queue_items_for_traces(
                session, deleted_archived_trace_ids
            )
            delete_trace_associations(session, deleted_archived_trace_ids)
        return deleted_db_backed_count + deleted_archived_count

    def _enqueue_rollup_rebuilds_for_trace_delete(
        self, session: Session, experiment_id: int, trace_ids: list[str]
    ) -> None:
        """Invalidate trace, span-cost, and assessment partitions before source-row deletion."""
        if not rollups_enabled():
            return

        timestamps = [
            timestamp_ms
            for (timestamp_ms,) in session.query(SqlTraceInfo.timestamp_ms).filter(
                SqlTraceInfo.request_id.in_(trace_ids)
            )
        ]
        span_days_by_experiment: defaultdict[int, set[int]] = defaultdict(set)
        for span_experiment_id, start_time_unix_nano in session.query(
            SqlSpan.experiment_id, SqlSpan.start_time_unix_nano
        ).filter(
            SqlSpan.trace_id.in_(trace_ids),
            SqlSpan.experiment_id.isnot(None),
            SqlSpan.start_time_unix_nano.isnot(None),
        ):
            span_days_by_experiment[int(span_experiment_id)].add(
                int(start_time_unix_nano) // 1_000_000
            )
        assessment_days_by_experiment: defaultdict[int, set[int]] = defaultdict(set)
        for assessment_experiment_id, trace_timestamp_ms in session.query(
            SqlAssessments.experiment_id, SqlAssessments.trace_timestamp_ms
        ).filter(
            SqlAssessments.trace_id.in_(trace_ids),
            SqlAssessments.experiment_id.isnot(None),
            SqlAssessments.trace_timestamp_ms.isnot(None),
        ):
            assessment_days_by_experiment[int(assessment_experiment_id)].add(
                int(trace_timestamp_ms)
            )
        enqueue_rollup_rebuild_partitions(
            session,
            [
                (RollupFamily.TRACE_METRIC, experiment_id, timestamps),
                *[
                    (RollupFamily.SPAN_COST, span_experiment_id, span_timestamps)
                    for span_experiment_id, span_timestamps in span_days_by_experiment.items()
                ],
                *[
                    (RollupFamily.ASSESSMENT, assessment_experiment_id, assessment_timestamps)
                    for assessment_experiment_id, assessment_timestamps in (
                        assessment_days_by_experiment.items()
                    )
                ],
            ],
        )

    def _select_trace_ids_for_delete(
        self,
        *,
        session: Session,
        filters: list[ColumnElement[bool]],
        lock: bool = False,
    ) -> list[str]:
        rows = (
            self
            ._trace_query(session, for_update_or_delete=lock)
            .with_entities(SqlTraceInfo.request_id)
            .filter(and_(*filters))
            # Keep row locking deterministic so overlapping delete calls acquire trace_info locks
            # in the same order instead of deadlocking on row-locking backends.
            .order_by(SqlTraceInfo.timestamp_ms, SqlTraceInfo.request_id)
            .all()
        )
        return [trace_id for (trace_id,) in rows]

    def _lock_trace_ids_for_delete(self, session: Session, trace_ids: list[str]) -> list[str]:
        rows = (
            self
            ._trace_query(session, for_update_or_delete=True)
            .with_entities(SqlTraceInfo.request_id)
            .filter(SqlTraceInfo.request_id.in_(trace_ids))
            .order_by(SqlTraceInfo.timestamp_ms, SqlTraceInfo.request_id)
            .all()
        )
        return [trace_id for (trace_id,) in rows]

    def _select_archived_traces_for_delete(
        self,
        *,
        session: Session,
        trace_ids: list[str],
    ) -> list[_TraceDeleteSelection]:
        """
        Return archive-backed traces and their payload URIs for a delete pass.

        `delete_traces()` calls this while the selected `trace_info` rows are still locked so it
        can classify candidates consistently, delete DB-backed rows in the same transaction, and
        leave only archive-backed traces for the slower payload cleanup phase.
        """
        if not trace_ids:
            return []
        spans_location_tag = aliased(SqlTraceTag)
        archive_location_tag = aliased(SqlTraceTag)
        rows = (
            self
            ._trace_query(session)
            .outerjoin(
                spans_location_tag,
                and_(
                    spans_location_tag.request_id == SqlTraceInfo.request_id,
                    spans_location_tag.key == TraceTagKey.SPANS_LOCATION,
                ),
            )
            .outerjoin(
                archive_location_tag,
                and_(
                    archive_location_tag.request_id == SqlTraceInfo.request_id,
                    archive_location_tag.key == TraceTagKey.ARCHIVE_LOCATION,
                ),
            )
            .with_entities(
                SqlTraceInfo.request_id,
                archive_location_tag.value,
            )
            .filter(
                SqlTraceInfo.request_id.in_(trace_ids),
                spans_location_tag.value == SpansLocation.ARCHIVE_REPO.value,
            )
            .all()
        )
        return [
            _TraceDeleteSelection(
                trace_id=trace_id,
                archived_artifact_uri=archived_artifact_uri,
            )
            for trace_id, archived_artifact_uri in rows
        ]

    @staticmethod
    def _is_missing_archived_trace_payload_delete_error(exc: Exception) -> bool:
        if isinstance(exc, FileNotFoundError):
            return True
        if isinstance(exc, MlflowException) and exc.error_code == ErrorCode.Name(
            RESOURCE_DOES_NOT_EXIST
        ):
            return True
        return "No such file or directory" in str(exc) or "No such artifact" in str(exc)

    def _delete_archived_trace_payloads(
        self, selected_traces: list[_TraceDeleteSelection]
    ) -> list[str]:
        def delete_payload(selected_trace: _TraceDeleteSelection) -> str | None:
            if selected_trace.archived_artifact_uri is None:
                return selected_trace.trace_id
            try:
                get_artifact_repository(selected_trace.archived_artifact_uri).delete_artifacts(
                    TRACE_ARCHIVAL_FILENAME
                )
            except Exception as e:
                if not self._is_missing_archived_trace_payload_delete_error(e):
                    _logger.error(
                        "Failed to clean up archived payload for trace %s; "
                        "leaving the trace row intact.",
                        selected_trace.trace_id,
                        exc_info=True,
                    )
                    return None
            return selected_trace.trace_id

        if len(selected_traces) <= 1:
            deleted_trace_ids = [delete_payload(trace) for trace in selected_traces]
        else:
            with ThreadPoolExecutor(
                max_workers=min(MLFLOW_TRACE_DELETE_MAX_WORKERS.get(), len(selected_traces)),
                thread_name_prefix="mlflow-trace-delete",
            ) as executor:
                deleted_trace_ids = list(
                    map_with_context(executor, delete_payload, selected_traces)
                )
        return [trace_id for trace_id in deleted_trace_ids if trace_id is not None]

    def create_assessment(self, assessment: Assessment) -> Assessment:
        """
        Create a new assessment in the database.

        If the assessment has an 'overrides' field set, this will also mark the
        overridden assessment as invalid.

        Args:
            assessment: The Assessment object to create (without assessment_id).

        Returns:
            The created Assessment object with backend-generated metadata.
        """

        use_rollups = rollups_enabled()
        while True:
            with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
                self._validate_trace_accessible(session, assessment.trace_id)
                if use_rollups:
                    # Queue keys must be acquired before the trace row lock. Snapshot the routing
                    # fields optimistically, lock that assessment partition, then lock and refresh
                    # the trace. If a span writer moved the trace between those reads, retry in a
                    # fresh transaction so we never acquire another queue key while holding the
                    # source-row lock.
                    sql_trace_info = self._get_sql_trace_info(session, assessment.trace_id)
                    routing_snapshot = (
                        int(sql_trace_info.experiment_id),
                        sql_trace_info.timestamp_ms,
                    )
                    enqueue_rollup_rebuilds(
                        session,
                        RollupFamily.ASSESSMENT,
                        routing_snapshot[0],
                        [routing_snapshot[1]],
                    )
                    sql_trace_info = self._get_sql_trace_info(
                        session,
                        assessment.trace_id,
                        for_update=True,
                    )
                    current_routing = (
                        int(sql_trace_info.experiment_id),
                        sql_trace_info.timestamp_ms,
                    )
                    if current_routing != routing_snapshot:
                        session.rollback()
                        continue
                else:
                    # Preserve the pre-rollup path: one locking read protects the assessment's
                    # denormalized experiment and timestamp from concurrent span writes.
                    sql_trace_info = self._get_sql_trace_info(
                        session,
                        assessment.trace_id,
                        for_update=True,
                    )

                sql_assessment = SqlAssessments.from_mlflow_entity(assessment)
                sql_assessment.experiment_id = sql_trace_info.experiment_id
                sql_assessment.trace_timestamp_ms = sql_trace_info.timestamp_ms

                if sql_assessment.overrides:
                    update_count = (
                        session
                        .query(SqlAssessments)
                        .filter(
                            SqlAssessments.trace_id == sql_assessment.trace_id,
                            SqlAssessments.assessment_id == sql_assessment.overrides,
                        )
                        .update({"valid": False})
                    )

                    if update_count == 0:
                        raise MlflowException(
                            f"Assessment with ID '{sql_assessment.overrides}' not found "
                            f"for trace '{assessment.trace_id}'",
                            RESOURCE_DOES_NOT_EXIST,
                        )

                session.add(sql_assessment)
                # A missing trace (e.g. deleted while still attached to a review queue) trips the
                # ``trace_id`` -> ``trace_info`` foreign key on flush. Rather than leak the raw SQL
                # error, roll back and check whether the trace is actually gone: report a clean
                # "not found" if so, otherwise a generic error (the IntegrityError could also come
                # from another constraint, e.g. a caller-supplied duplicate ``assessment_id``).
                try:
                    session.flush()
                except IntegrityError as e:
                    session.rollback()
                    trace_exists = (
                        session
                        .query(SqlTraceInfo.request_id)
                        .filter(SqlTraceInfo.request_id == assessment.trace_id)
                        .first()
                    ) is not None
                    if not trace_exists:
                        raise MlflowException(
                            f"Trace with ID '{assessment.trace_id}' not found. "
                            "It may have been deleted.",
                            RESOURCE_DOES_NOT_EXIST,
                        ) from e
                    raise MlflowException(
                        f"Failed to create assessment for trace '{assessment.trace_id}' "
                        "due to a constraint violation.",
                        INTERNAL_ERROR,
                    ) from e
                return sql_assessment.to_mlflow_entity()

    def get_assessment(self, trace_id: str, assessment_id: str) -> Assessment:
        """
        Fetch the assessment for the given trace_id and assessment_id.

        Args:
            trace_id: The ID of the trace containing the assessment.
            assessment_id: The ID of the assessment to retrieve.

        Returns:
            The Assessment object.
        """
        with self._tracking_store.ManagedSessionMaker() as session:
            sql_assessment = self._get_sql_assessment(session, trace_id, assessment_id)
            return sql_assessment.to_mlflow_entity()

    def update_assessment(
        self,
        trace_id: str,
        assessment_id: str,
        name: str | None = None,
        expectation: ExpectationValue | None = None,
        feedback: FeedbackValue | None = None,
        rationale: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Assessment:
        """
        Updates an existing assessment with new values while preserving immutable fields.

        Only source and span_id are immutable.
        The last_update_time_ms will always be updated to the current timestamp.
        Metadata will be merged with the new metadata taking precedence.

        Args:
            trace_id: The unique identifier of the trace containing the assessment.
            assessment_id: The unique identifier of the assessment to update.
            name: The updated name of the assessment. If None, preserves existing name.
            expectation: Updated expectation value for expectation assessments.
            feedback: Updated feedback value for feedback assessments.
            rationale: Updated rationale text. If None, preserves existing rationale.
            metadata: Updated metadata dict. Will be merged with existing metadata.

        Returns:
            Assessment: The updated assessment object with new last_update_time_ms.

        Raises:
            MlflowException: If the assessment doesn't exist, if immutable fields have
                            changed, or if there's an error saving the assessment.
        """
        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            existing_sql = self._get_sql_assessment(session, trace_id, assessment_id)
            existing = existing_sql.to_mlflow_entity()

            enqueue_rollup_rebuilds(
                session,
                RollupFamily.ASSESSMENT,
                existing_sql.experiment_id,
                [existing_sql.trace_timestamp_ms],
            )

            if expectation is not None and feedback is not None:
                raise MlflowException.invalid_parameter_value(
                    "Cannot specify both `expectation` and `feedback` parameters."
                )

            if expectation is not None and not isinstance(existing, Expectation):
                raise MlflowException.invalid_parameter_value(
                    "Cannot update expectation value on a Feedback assessment."
                )

            if feedback is not None and not isinstance(existing, Feedback):
                raise MlflowException.invalid_parameter_value(
                    "Cannot update feedback value on an Expectation assessment."
                )

            merged_metadata = None
            if existing.metadata or metadata:
                merged_metadata = (existing.metadata or {}).copy()
                if metadata:
                    merged_metadata.update(metadata)

            updated_timestamp = get_current_time_millis()

            if isinstance(existing, Expectation):
                new_value = expectation.value if expectation is not None else existing.value

                updated_assessment = Expectation(
                    name=name if name is not None else existing.name,
                    value=new_value,
                    source=existing.source,
                    trace_id=trace_id,
                    metadata=merged_metadata,
                    span_id=existing.span_id,
                    create_time_ms=existing.create_time_ms,
                    last_update_time_ms=updated_timestamp,
                )
            else:
                if feedback is not None:
                    new_value = feedback.value
                    new_error = feedback.error
                else:
                    new_value = existing.value
                    new_error = existing.error

                updated_assessment = Feedback(
                    name=name if name is not None else existing.name,
                    value=new_value,
                    error=new_error,
                    source=existing.source,
                    trace_id=trace_id,
                    metadata=merged_metadata,
                    span_id=existing.span_id,
                    create_time_ms=existing.create_time_ms,
                    last_update_time_ms=updated_timestamp,
                    rationale=rationale if rationale is not None else existing.rationale,
                )

            updated_assessment.assessment_id = existing.assessment_id
            updated_assessment.valid = existing.valid
            updated_assessment.overrides = existing.overrides

            if hasattr(existing, "run_id"):
                updated_assessment.run_id = existing.run_id

            if updated_assessment.feedback is not None:
                value_json = json.dumps(updated_assessment.feedback.value)
                error_json = (
                    json.dumps(updated_assessment.feedback.error.to_dictionary())
                    if updated_assessment.feedback.error
                    else None
                )
            elif updated_assessment.expectation is not None:
                value_json = json.dumps(updated_assessment.expectation.value)
                error_json = None

            metadata_json = (
                json.dumps(updated_assessment.metadata) if updated_assessment.metadata else None
            )

            aggregate_value, is_numeric_value = assessment_aggregate(new_value)
            session.query(SqlAssessments).filter(
                SqlAssessments.trace_id == trace_id, SqlAssessments.assessment_id == assessment_id
            ).update({
                "name": updated_assessment.name,
                "value": value_json,
                "error": error_json,
                "last_updated_timestamp": updated_timestamp,
                "rationale": updated_assessment.rationale,
                "assessment_metadata": metadata_json,
                "aggregate_value": aggregate_value,
                "is_numeric_value": is_numeric_value,
            })

            return updated_assessment

    def delete_assessment(self, trace_id: str, assessment_id: str) -> None:
        """
        Delete an assessment from a trace.

        If the deleted assessment was overriding another assessment, the overridden
        assessment will be restored to valid=True.

        Args:
            trace_id: The ID of the trace containing the assessment.
            assessment_id: The ID of the assessment to delete.
        """
        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            self._validate_trace_accessible(session, trace_id)

            assessment_to_delete = (
                session
                .query(SqlAssessments)
                .filter_by(trace_id=trace_id, assessment_id=assessment_id)
                .first()
            )

            if assessment_to_delete is None:
                # Assessment doesn't exist - this is idempotent, so just return
                return

            enqueue_rollup_rebuilds(
                session,
                RollupFamily.ASSESSMENT,
                assessment_to_delete.experiment_id,
                [assessment_to_delete.trace_timestamp_ms],
            )

            # If this assessment was overriding another assessment, restore the original
            if assessment_to_delete.overrides:
                session.query(SqlAssessments).filter_by(
                    assessment_id=assessment_to_delete.overrides
                ).update({"valid": True})

            session.delete(assessment_to_delete)
            session.commit()

    def _get_sql_assessment(self, session, trace_id: str, assessment_id: str) -> SqlAssessments:
        """Helper method to get SqlAssessments object."""
        sql_assessment = (
            session
            .query(SqlAssessments)
            .filter(
                SqlAssessments.trace_id == trace_id, SqlAssessments.assessment_id == assessment_id
            )
            .one_or_none()
        )
        if sql_assessment is None:
            trace_exists = (
                session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).first()
                is not None
            )
            if not trace_exists:
                raise MlflowException(
                    f"Trace with request_id '{trace_id}' not found",
                    RESOURCE_DOES_NOT_EXIST,
                )
            else:
                raise MlflowException(
                    f"Assessment with ID '{assessment_id}' not found for trace '{trace_id}'",
                    RESOURCE_DOES_NOT_EXIST,
                )
        return sql_assessment

    def calculate_trace_filter_correlation(
        self,
        experiment_ids: list[str],
        filter_string1: str,
        filter_string2: str,
        base_filter: str | None = None,
    ) -> TraceFilterCorrelationResult:
        """
        Calculate correlation between two trace filter conditions using NPMI.

        Args:
            experiment_ids: List of experiment_ids to search over
            filter_string1: First filter condition in search_traces filter syntax
            filter_string2: Second filter condition in search_traces filter syntax
            base_filter: Optional base filter that both filter1 and filter2 are tested on top of
                        (e.g. 'request_time > ... and request_time < ...' for time windows)

        Returns:
            TraceFilterCorrelationResult which containst the NPMI analytics data.
        """

        with self._tracking_store.ManagedSessionMaker() as session:
            experiment_ids = self._tracking_store._filter_experiment_ids(
                session, _parse_experiment_ids(experiment_ids)
            )
            # Keep IDs as integers when filtering trace_info.experiment_id. psycopg v3
            # rejects VARCHAR parameters compared with this INTEGER column.

            filter1_combined = (
                f"{base_filter} and {filter_string1}" if base_filter else filter_string1
            )
            filter2_combined = (
                f"{base_filter} and {filter_string2}" if base_filter else filter_string2
            )

            filter1_subquery = self._build_trace_filter_subquery(
                session, experiment_ids, filter1_combined
            )
            filter2_subquery = self._build_trace_filter_subquery(
                session, experiment_ids, filter2_combined
            )

            counts = self._get_trace_correlation_counts(
                session, experiment_ids, filter1_subquery, filter2_subquery, base_filter
            )

            npmi_result = trace_correlation.calculate_npmi_from_counts(
                counts.joint_count,
                counts.filter1_count,
                counts.filter2_count,
                counts.total_count,
            )

            return TraceFilterCorrelationResult(
                npmi=npmi_result.npmi,
                npmi_smoothed=npmi_result.npmi_smoothed,
                filter1_count=counts.filter1_count,
                filter2_count=counts.filter2_count,
                joint_count=counts.joint_count,
                total_count=counts.total_count,
            )

    def _build_trace_filter_subquery(self, session, experiment_ids: list[int], filter_string: str):
        """Build a subquery for traces that match a given filter in the specified experiments."""
        stmt = select(SqlTraceInfo.request_id).where(SqlTraceInfo.experiment_id.in_(experiment_ids))

        if filter_string:
            scoped_trace_query = self._trace_query(session).filter(
                SqlTraceInfo.experiment_id.in_(experiment_ids)
            )
            (
                attribute_filters,
                non_attribute_filters,
                span_attribute_filters,
                span_filters,
                run_id_filter,
            ) = _get_filter_clauses_for_search_traces(
                filter_string, session, self._tracking_store._get_dialect(), scoped_trace_query
            )

            stmt = self._apply_trace_filter_clauses(
                stmt,
                attribute_filters,
                non_attribute_filters,
                span_attribute_filters,
                span_filters,
                run_id_filter,
            )

        return stmt

    def _get_trace_correlation_counts(
        self,
        session,
        experiment_ids: list[int],
        filter1_subquery,
        filter2_subquery,
        base_filter: str | None = None,
    ) -> trace_correlation.TraceCorrelationCounts:
        """
        Get trace counts for correlation analysis using a single SQL query.

        This method efficiently calculates all necessary counts for NPMI calculation
        in a single database round-trip using LEFT JOINs instead of EXISTS subqueries
        for MSSQL compatibility.

        When base_filter is provided, the total count refers to traces matching the base filter.
        """
        f1_subq = filter1_subquery.subquery()
        f2_subq = filter2_subquery.subquery()

        filter1_alias = aliased(f1_subq)
        filter2_alias = aliased(f2_subq)

        # If base_filter is provided, use traces matching the base filter as the universe
        # Otherwise, use all traces in the experiments
        if base_filter:
            base_subquery = self._build_trace_filter_subquery(session, experiment_ids, base_filter)
            base_subq = base_subquery.subquery()
            base_table = aliased(base_subq)
            base_request_id = base_table.c.request_id
        else:
            base_table = SqlTraceInfo
            base_request_id = SqlTraceInfo.request_id

        # NB: MSSQL does not support exists queries within subjoins so a slightly
        # less efficient subquery LEFT JOIN is used to support all backends.
        query = (
            session
            .query(
                func.count(base_request_id).label("total"),
                func.count(filter1_alias.c.request_id).label("filter1"),
                func.count(filter2_alias.c.request_id).label("filter2"),
                func.count(
                    case(
                        (
                            (filter1_alias.c.request_id.isnot(None))
                            & (filter2_alias.c.request_id.isnot(None)),
                            base_request_id,
                        ),
                        else_=None,
                    )
                ).label("joint"),
            )
            .select_from(base_table)
            .outerjoin(filter1_alias, base_request_id == filter1_alias.c.request_id)
            .outerjoin(filter2_alias, base_request_id == filter2_alias.c.request_id)
        )

        # Only add experiment filter if we're using SqlTraceInfo directly (no base_filter)
        if not base_filter:
            query = query.filter(SqlTraceInfo.experiment_id.in_(experiment_ids))

        result = query.one()

        # Cast to int (some databases return Decimal)
        # Handle None values from empty result sets
        total_count = int(result.total or 0)
        filter1_count = int(result.filter1 or 0)
        filter2_count = int(result.filter2 or 0)
        joint_count = int(result.joint or 0)

        return trace_correlation.TraceCorrelationCounts(
            total_count=total_count,
            filter1_count=filter1_count,
            filter2_count=filter2_count,
            joint_count=joint_count,
        )

    def log_spans(self, location: str, spans: list[Span], tracking_uri=None) -> list[Span]:
        """
        Log multiple span entities to the tracking store.

        Spans may belong to different traces; they are grouped by trace_id internally
        and processed in a single DB session to minimize round-trips.

        Args:
            location: Experiment ID of an MLflow experiment.
            spans: List of Span entities to log.
            tracking_uri: The tracking URI to use. Default to None.

        Returns:
            List of logged Span entities.
        """
        # Retry on DB deadlocks so a concurrent start_trace()/log_spans() race does not
        # drop metadata. Each attempt opens a fresh managed session.
        return self._tracking_store._run_with_deadlock_retry(
            self._log_spans_once, location, spans, tracking_uri=tracking_uri
        )

    def _log_spans_once(self, location: str, spans: list[Span], tracking_uri=None) -> list[Span]:
        if not spans:
            return []

        # Group spans by trace_id to handle multi-trace batches in a single session
        spans_by_trace: dict[str, list[Span]] = defaultdict(list)
        for span in spans:
            spans_by_trace[span.trace_id].append(span)

        all_trace_ids = list(spans_by_trace.keys())

        # Pre-compute per-trace aggregates outside the DB session (pure Python, no I/O)
        trace_aggregates: dict[str, _TraceAggregate] = {}
        all_span_rows = []
        for trace_id, trace_spans in spans_by_trace.items():
            min_start_ms = min(s.start_time_ns for s in trace_spans) // 1_000_000
            end_times = [s.end_time_ns for s in trace_spans if s.end_time_ns is not None]
            max_end_ms = (max(end_times) // 1_000_000) if end_times else None
            root_span_status = self._get_trace_status_from_root_span(trace_spans)

            usage_nodes: list[SpanAggregationNode] = []
            cost_nodes: list[SpanAggregationNode] = []
            session_id = None
            user_id = None
            root_span_dict = None
            trace_tags_from_root_attr: dict[str, str] = {}
            for span in trace_spans:
                span_dict = translate_span_when_storing(span)
                span_cost = None
                span_attributes = span_dict.get("attributes", {})
                # Collect a node for every span, including spans without usage or cost, so that
                # the tree-aware aggregation can traverse parent chains through them.
                usage_nodes.append(
                    _span_aggregation_node(
                        span.span_id, span.parent_id, span_attributes, SpanAttributeKey.CHAT_USAGE
                    )
                )
                cost_nodes.append(
                    _span_aggregation_node(
                        span.span_id, span.parent_id, span_attributes, SpanAttributeKey.LLM_COST
                    )
                )
                if span_attributes:
                    # Session ID from OTel semantic conventions:
                    # https://opentelemetry.io/docs/specs/semconv/registry/attributes/session/#session-id
                    if session_id is None and (
                        span_session_id := (
                            span_attributes.get(SpanAttributeKey.SESSION_ID)
                            or span_attributes.get(GenAiSemconvKey.CONVERSATION_ID)
                        )
                    ):
                        session_id = validate_session_id(_try_parse_json_string(span_session_id))
                    # user id used by OTel semantic conventions: https://opentelemetry.io/docs/specs/semconv/registry/attributes/user/#user-id
                    if user_id is None and (span_user_id := span_attributes.get("user.id")):
                        user_id = _try_parse_json_string(span_user_id)
                    # Get cost for span metrics
                    span_cost = span_attributes.get(SpanAttributeKey.LLM_COST)

                # Keep non-ASCII text unescaped so `trace.text` / `span.content` LIKE filters
                # can match it: stored as a \uXXXX escape, it never matches the text users type.
                # MSSQL keeps the escaped form: its `spans.content` column is a non-Unicode
                # VARCHAR, which would replace characters outside its code page with "?".
                content_json = json.dumps(
                    span_dict,
                    cls=TraceJSONEncoder,
                    ensure_ascii=self._tracking_store.db_type == MSSQL,
                )

                model_name = bounded_model_dimension(
                    _try_parse_json_string(span_attributes.get(SpanAttributeKey.MODEL))
                )
                model_provider = bounded_model_dimension(
                    _try_parse_json_string(span_attributes.get(SpanAttributeKey.MODEL_PROVIDER))
                )
                span_cost_values = {}
                if span_cost:
                    try:
                        span_cost_values = json.loads(span_cost)
                    except (TypeError, ValueError):
                        _logger.debug("Skipping malformed cost for span %s", span.span_id)

                # experiment_id filled in after we resolve trace infos
                all_span_rows.append({
                    "trace_id": span.trace_id,
                    "experiment_id": None,
                    "span_id": span.span_id,
                    "parent_span_id": span.parent_id,
                    "name": span.name,
                    "type": span.span_type,
                    "status": span.status.status_code,
                    "start_time_unix_nano": span.start_time_ns,
                    "end_time_unix_nano": span.end_time_ns,
                    "content": content_json,
                    "input_cost": finite_float_or_none(span_cost_values.get("input_cost")),
                    "output_cost": finite_float_or_none(span_cost_values.get("output_cost")),
                    "total_cost": finite_float_or_none(span_cost_values.get("total_cost")),
                    "model_name": model_name,
                    "model_provider": model_provider,
                })

                if span.parent_id is None:
                    root_span_dict = span_dict
                    # Extract user-defined tags emitted by OtelSpanProcessor as
                    # individual "mlflow.traceTag.<key>" attributes on the root span (OTLP path).
                    # from_otel_proto JSON-encodes every attribute value once, so one
                    # _try_parse_json_string call is sufficient to unwrap the plain string value.
                    prefix = SpanAttributeKey.TRACE_TAG_PREFIX
                    for attr_key, attr_value in span_attributes.items():
                        if attr_key.startswith(prefix):
                            tag_key = attr_key[len(prefix) :]
                            tag_value = str(_try_parse_json_string(attr_value))
                            try:
                                tag_key, tag_value = _validate_trace_tag(tag_key, tag_value)
                            except Exception:
                                _logger.debug(
                                    "Skipping invalid trace tag from OTLP attribute %r", attr_key
                                )
                                continue
                            if tag_key == TraceTagKey.TRACE_NAME:
                                tag_value = validate_trace_name(tag_value)
                            trace_tags_from_root_attr[tag_key] = tag_value

            # Tree-aware aggregation matching the client-side aggregator
            # (aggregate_usage_from_spans): rollup autologgers (e.g. pydantic_ai, agno,
            # dspy) set cumulative usage on parent spans, so a flat sum over all spans
            # would double-count parent and children.
            trace_aggregates[trace_id] = _TraceAggregate(
                min_start_ms=min_start_ms,
                max_end_ms=max_end_ms,
                root_span_status=root_span_status,
                trace_status=root_span_status or TraceState.IN_PROGRESS.value,
                aggregated_token_usage=aggregate_usage_from_span_nodes(usage_nodes) or {},
                aggregated_cost=aggregate_cost_from_span_nodes(cost_nodes) or {},
                session_id=session_id,
                user_id=user_id,
                root_span_dict=root_span_dict,
                trace_tags=trace_tags_from_root_attr,
                usage_nodes=usage_nodes,
                cost_nodes=cost_nodes,
            )

        batch_span_keys = {(row["trace_id"], row["span_id"]) for row in all_span_rows}

        def enqueue_span_write_rollup_partitions(session: Session) -> None:
            if not rollups_enabled():
                return
            trace_snapshots = {
                request_id: (int(experiment_id), timestamp_ms)
                for request_id, experiment_id, timestamp_ms in self
                ._trace_query(session)
                .with_entities(
                    SqlTraceInfo.request_id,
                    SqlTraceInfo.experiment_id,
                    SqlTraceInfo.timestamp_ms,
                )
                .filter(SqlTraceInfo.request_id.in_(all_trace_ids))
            }
            finalized_trace_snapshots = {
                request_id
                for (request_id,) in session.query(SqlTraceMetadata.request_id).filter(
                    SqlTraceMetadata.request_id.in_(all_trace_ids),
                    SqlTraceMetadata.key == TraceMetadataKey.TRACE_INFO_FINALIZED,
                )
            }
            partitions = []
            for trace_id in all_trace_ids:
                agg = trace_aggregates[trace_id]
                snapshot = trace_snapshots.get(trace_id)
                experiment_id, old_timestamp_ms = snapshot or (int(location), agg.min_start_ms)
                timestamps = [old_timestamp_ms]
                if snapshot is None or trace_id not in finalized_trace_snapshots:
                    timestamps.append(agg.min_start_ms)
                partitions.extend(
                    (family, experiment_id, timestamps)
                    for family in (RollupFamily.TRACE_METRIC, RollupFamily.ASSESSMENT)
                )

            if batch_span_keys:
                batch_span_ids = {span_id for _, span_id in batch_span_keys}
                previous_span_rows = session.query(
                    SqlSpan.trace_id,
                    SqlSpan.span_id,
                    SqlSpan.experiment_id,
                    SqlSpan.start_time_unix_nano,
                ).filter(
                    SqlSpan.trace_id.in_(all_trace_ids),
                    SqlSpan.span_id.in_(batch_span_ids),
                )
                partitions.extend(
                    (
                        RollupFamily.SPAN_COST,
                        int(row.experiment_id),
                        [int(row.start_time_unix_nano) // 1_000_000],
                    )
                    for row in previous_span_rows
                    if (
                        (row.trace_id, row.span_id) in batch_span_keys
                        and row.experiment_id is not None
                        and row.start_time_unix_nano is not None
                    )
                )
            for row in all_span_rows:
                experiment_id = trace_snapshots.get(row["trace_id"], (int(location), None))[0]
                partitions.append((
                    RollupFamily.SPAN_COST,
                    experiment_id,
                    [int(row["start_time_unix_nano"]) // 1_000_000],
                ))
            enqueue_rollup_rebuild_partitions(session, partitions)

        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            # The rebuild queue is the first lock domain for both writers and publishers.
            enqueue_span_write_rollup_partitions(session)
            # --- Phase 1: Lock and batch-fetch all existing trace infos (1 query) ---
            existing_traces = {
                t.request_id: t
                for t in self
                ._trace_query(session, for_update_or_delete=True)
                .filter(SqlTraceInfo.request_id.in_(all_trace_ids))
                .order_by(SqlTraceInfo.request_id)
                .all()
            }

            # --- Phase 2: Create missing traces ---
            # On IntegrityError (concurrent start_trace race), roll back and retry so that
            # previously flushed trace_infos (which session.rollback() would undo) are
            # re-created in the next attempt. Loop until all traces are in existing_traces.
            # Traces created by this call cannot have spans stored by earlier calls, so
            # their batch-local aggregates are authoritative (no recompute needed later).
            created_trace_ids: set[str] = set()
            if any(tid not in existing_traces for tid in all_trace_ids):
                experiment = self._tracking_store.get_experiment(location)
                for _attempt in range(_LOG_SPANS_MAX_TRACE_CREATE_RETRIES):
                    pending = [tid for tid in all_trace_ids if tid not in existing_traces]
                    if not pending:
                        break
                    conflict = None
                    for trace_id in pending:
                        agg = trace_aggregates[trace_id]
                        sql_trace_info = SqlTraceInfo(
                            request_id=trace_id,
                            experiment_id=location,
                            timestamp_ms=agg.min_start_ms,
                            execution_time_ms=(
                                (agg.max_end_ms - agg.min_start_ms) if agg.max_end_ms else None
                            ),
                            status=agg.trace_status,
                            client_request_id=None,
                        )
                        sql_trace_info.tags = [
                            self._get_trace_artifact_location_tag(experiment, trace_id)
                        ]
                        session.add(sql_trace_info)
                        try:
                            session.flush()
                        except IntegrityError:
                            # A concurrent start_trace() created this trace. Roll back
                            # the entire transaction (undoing any trace_infos we just
                            # flushed) and re-fetch to rebuild existing_traces.
                            session.rollback()
                            enqueue_span_write_rollup_partitions(session)
                            conflict = trace_id
                            break
                        existing_traces[trace_id] = sql_trace_info
                        created_trace_ids.add(trace_id)
                    if conflict is not None:
                        # The rollback undid every trace this attempt created. Traces found by
                        # the re-fetch below were created by another process.
                        created_trace_ids.clear()
                        # Re-fetch whatever now exists in DB (created by start_trace or us)
                        existing_traces = {
                            t.request_id: t
                            for t in self
                            ._trace_query(session, for_update_or_delete=True)
                            .filter(SqlTraceInfo.request_id.in_(all_trace_ids))
                            .order_by(SqlTraceInfo.request_id)
                            .all()
                        }

            # Log a warning for any traces we still couldn't create after retries
            # (e.g., concurrent start_trace hasn't committed yet). Skip their spans
            # rather than crashing with KeyError.
            if missing_trace_ids := {tid for tid in all_trace_ids if tid not in existing_traces}:
                _logger.warning(
                    "Could not create trace_info for %d trace(s) after %d retries; "
                    "spans for these traces will be dropped: %s",
                    len(missing_trace_ids),
                    _LOG_SPANS_MAX_TRACE_CREATE_RETRIES,
                    missing_trace_ids,
                )
                all_span_rows = [r for r in all_span_rows if r["trace_id"] not in missing_trace_ids]

            # Keep downstream per-trace updates aligned with the surviving span rows.
            all_trace_ids = [trace_id for trace_id in all_trace_ids if trace_id in existing_traces]

            # Only add spans to traces in the experiment being logged to; never write into a
            # trace stored in another experiment.
            if foreign_trace_ids := sorted(
                trace_id
                for trace_id in all_trace_ids
                if str(existing_traces[trace_id].experiment_id) != str(location)
            ):
                raise MlflowException(
                    f"Trace with ID '{foreign_trace_ids[0]}' already exists in another experiment.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            # Serialize concurrent log_spans() calls for the same trace, so that Phase 5's
            # recompute of the trace-level token usage/cost sees every span committed for the
            # trace (see _trace_row_lock_query). Traces created by this call are not visible to
            # other transactions yet and cannot have spans stored by an earlier call, so only
            # pre-existing rows need the lock. Taken before any span rows are written to keep the
            # trace-row -> span-row lock order consistent with the archival paths.
            if preexisting_trace_ids := [
                trace_id for trace_id in all_trace_ids if trace_id not in created_trace_ids
            ]:
                self._trace_row_lock_query(session, preexisting_trace_ids).all()

            # Fill in experiment_id on span rows now that we have trace infos
            for row in all_span_rows:
                row["experiment_id"] = existing_traces[row["trace_id"]].experiment_id

            # --- Phase 3: Bulk upsert all spans ---
            _bulk_upsert(session, SqlSpan, all_span_rows)

            # --- Phase 4: Fetch trace finalization markers ---
            trace_ids_with_token_usage = {
                tid for tid in all_trace_ids if trace_aggregates[tid].aggregated_token_usage
            }
            trace_ids_with_cost = {
                tid for tid in all_trace_ids if trace_aggregates[tid].aggregated_cost
            }
            trace_ids_with_session = {
                tid for tid in all_trace_ids if trace_aggregates[tid].session_id
            }

            # Traces where start_trace() has already written the authoritative values.
            # log_spans() must not accumulate on top of those to avoid double-counting.
            finalized_trace_ids: set[str] = set()
            if trace_ids_with_token_usage or trace_ids_with_cost or trace_ids_with_session:
                all_finalized_ids = sorted(
                    trace_ids_with_token_usage | trace_ids_with_cost | trace_ids_with_session
                )
                rows = (
                    session
                    .query(SqlTraceMetadata)
                    .filter(
                        SqlTraceMetadata.request_id.in_(all_finalized_ids),
                        SqlTraceMetadata.key == TraceMetadataKey.TRACE_INFO_FINALIZED,
                    )
                    .all()
                )
                finalized_trace_ids.update(row.request_id for row in rows)

            authoritative_token_usage_ids = {
                trace_id
                for trace_id in trace_ids_with_token_usage & finalized_trace_ids
                if any(
                    getattr(existing_traces[trace_id], column) is not None
                    for column in TOKEN_COLUMN_BY_KEY.values()
                )
            }
            authoritative_cost_ids = {
                trace_id
                for trace_id in trace_ids_with_cost & finalized_trace_ids
                if any(
                    getattr(existing_traces[trace_id], column) is not None
                    for column in COST_COLUMN_BY_KEY.values()
                )
            }

            # For traces that existed before this call, the batch-local aggregate is not
            # authoritative: a rollup parent and its children may arrive in separate
            # batches (e.g. BatchSpanProcessor exports of a distributed trace), and
            # accumulating per-batch sums onto the stored value would double-count.
            # Fetch the trace's already-stored spans so Phase 5 can recompute the
            # tree-aware aggregate over the full span tree (stored + batch).
            #
            # Derived from preexisting_trace_ids so the recompute only reads spans for traces
            # locked above. That keeps the read from observing another log_spans() call's
            # uncommitted spans, and keeps recompute_trace_ids a subset of the locked rows by
            # construction rather than by coincidence.
            preexisting_trace_id_set = set(preexisting_trace_ids)
            recompute_token_usage_ids = (
                trace_ids_with_token_usage & preexisting_trace_id_set
            ) - authoritative_token_usage_ids
            recompute_cost_ids = (
                trace_ids_with_cost & preexisting_trace_id_set
            ) - authoritative_cost_ids
            recompute_trace_ids = recompute_token_usage_ids | recompute_cost_ids
            stored_usage_nodes: defaultdict[str, list[SpanAggregationNode]] = defaultdict(list)
            stored_cost_nodes: defaultdict[str, list[SpanAggregationNode]] = defaultdict(list)
            if recompute_trace_ids:
                stored_span_rows = self._stored_span_rows_query(
                    session, sorted(recompute_trace_ids)
                ).all()
                for row_trace_id, row_span_id, row_parent_span_id, row_content in stored_span_rows:
                    # Phase 3 already upserted this batch's spans, so a span re-sent in this batch
                    # is read back here as well. Skip it, otherwise it would be counted twice: once
                    # from its stored row and once from the batch node in trace_aggregates.
                    if (row_trace_id, row_span_id) in batch_span_keys:
                        continue
                    try:
                        row_attributes = json.loads(row_content).get("attributes") or {}
                    except (json.JSONDecodeError, AttributeError):
                        _logger.debug("Skipping malformed span content for span %s", row_span_id)
                        continue
                    if row_trace_id in recompute_token_usage_ids:
                        stored_usage_nodes[row_trace_id].append(
                            _span_aggregation_node(
                                row_span_id,
                                row_parent_span_id,
                                row_attributes,
                                SpanAttributeKey.CHAT_USAGE,
                            )
                        )
                    if row_trace_id in recompute_cost_ids:
                        stored_cost_nodes[row_trace_id].append(
                            _span_aggregation_node(
                                row_span_id,
                                row_parent_span_id,
                                row_attributes,
                                SpanAttributeKey.LLM_COST,
                            )
                        )

            # --- Phase 5: Per-trace updates (UPDATE + merges) ---
            # Iterate trace_ids in sorted order, and within each trace emit
            # trace_request_metadata merges in sorted key order, so concurrent
            # start_trace()/log_spans() transactions acquire PK-index locks consistently and
            # cannot deadlock.
            for trace_id in sorted(all_trace_ids):
                agg = trace_aggregates[trace_id]
                sql_trace_info = existing_traces[trace_id]
                min_start_ms = agg.min_start_ms
                max_end_ms = agg.max_end_ms
                root_span_status = agg.root_span_status
                metadata_writes: dict[str, str] = {}

                # Atomic update of trace time range using SQLAlchemy's case expressions.
                # This is necessary to handle concurrent span additions from multiple
                # processes/threads without race conditions. The database performs the
                # min/max comparisons atomically, ensuring the trace always reflects the
                # earliest start and latest end times across all spans.
                # Skip if start_trace() has already written the authoritative timestamp
                # and duration (indicated by TRACE_INFO_FINALIZED flag).
                update_dict = {}
                trace_timestamp_may_change = False
                if trace_id not in finalized_trace_ids:
                    timestamp_update_expr = case(
                        (SqlTraceInfo.timestamp_ms > min_start_ms, min_start_ms),
                        else_=SqlTraceInfo.timestamp_ms,
                    )
                    update_dict[SqlTraceInfo.timestamp_ms] = timestamp_update_expr
                    trace_timestamp_may_change = sql_trace_info.timestamp_ms > min_start_ms
                if max_end_ms is not None and trace_id not in finalized_trace_ids:
                    update_dict[SqlTraceInfo.execution_time_ms] = (
                        case(
                            (
                                (SqlTraceInfo.timestamp_ms + SqlTraceInfo.execution_time_ms)
                                > max_end_ms,
                                SqlTraceInfo.timestamp_ms + SqlTraceInfo.execution_time_ms,
                            ),
                            else_=max_end_ms,
                        )
                        - timestamp_update_expr
                    )

                # If trace status is IN_PROGRESS or unspecified, check for root span to update it
                if sql_trace_info.status in (
                    TraceState.IN_PROGRESS.value,
                    TraceState.STATE_UNSPECIFIED.value,
                ):
                    if root_span_status:
                        update_dict[SqlTraceInfo.status] = root_span_status

                if root_span_dict := agg.root_span_dict:
                    update_dict.update(
                        self._update_trace_info_attributes(sql_trace_info, root_span_dict)
                    )
                if TraceTagKey.TRACE_NAME in agg.trace_tags:
                    update_dict[SqlTraceInfo.trace_name] = agg.trace_tags[TraceTagKey.TRACE_NAME]

                # start_trace() values take precedence over values inferred from spans.
                if aggregated_token_usage := agg.aggregated_token_usage:
                    if trace_id not in authoritative_token_usage_ids:
                        if trace_id in created_trace_ids:
                            trace_token_usage = aggregated_token_usage
                        else:
                            # Recompute over the full span tree (stored + batch) instead
                            # of accumulating onto the stored value, so rollup parents
                            # arriving in a different batch than their children are
                            # de-duplicated. Spans logged by earlier calls are included
                            # via stored_usage_nodes, so overwriting is lossless.
                            trace_token_usage = aggregate_usage_from_span_nodes(
                                agg.usage_nodes + stored_usage_nodes[trace_id]
                            )
                        if trace_token_usage:
                            for key, column in TOKEN_COLUMN_BY_KEY.items():
                                value = token_count_or_none(trace_token_usage.get(key))
                                if value is not None:
                                    update_dict[getattr(SqlTraceInfo, column)] = value

                if aggregated_cost := agg.aggregated_cost:
                    if trace_id not in authoritative_cost_ids:
                        if trace_id in created_trace_ids:
                            recorded_cost = aggregated_cost
                        else:
                            recorded_cost = aggregate_cost_from_span_nodes(
                                agg.cost_nodes + stored_cost_nodes[trace_id]
                            )
                        if recorded_cost:
                            for key, column in COST_COLUMN_BY_KEY.items():
                                value = finite_float_or_none(recorded_cost.get(key))
                                if value is not None:
                                    update_dict[getattr(SqlTraceInfo, column)] = value

                if agg.session_id and sql_trace_info.session_id is None:
                    update_dict[SqlTraceInfo.session_id] = agg.session_id

                # User ID metadata
                if agg.user_id:
                    existing_user_id = (
                        session
                        .query(SqlTraceMetadata)
                        .filter(
                            SqlTraceMetadata.request_id == trace_id,
                            SqlTraceMetadata.key == TraceMetadataKey.TRACE_USER,
                        )
                        .one_or_none()
                    )
                    if not existing_user_id:
                        metadata_writes[TraceMetadataKey.TRACE_USER] = agg.user_id

                # Emit custom metadata merges in sorted key order so PK-index lock
                # acquisition is deterministic across transactions (#24332).
                for key, value in sorted(metadata_writes.items()):
                    session.merge(SqlTraceMetadata(request_id=trace_id, key=key, value=value))

                if update_dict:
                    # `trace_id` was selected through workspace-scoped reads earlier in this
                    # call, so we can update by PK here without paying for another workspace
                    # filter on the hot path.
                    (
                        session
                        .query(SqlTraceInfo)
                        .filter(SqlTraceInfo.request_id == trace_id)
                        .update(
                            update_dict,
                            # Skip session synchronization for performance — we don't
                            # use the ORM object afterward.
                            synchronize_session=False,
                        )
                    )
                if trace_timestamp_may_change:
                    (
                        session
                        .query(SqlAssessments)
                        .filter(SqlAssessments.trace_id == trace_id)
                        .update(
                            {SqlAssessments.trace_timestamp_ms: min_start_ms},
                            synchronize_session=False,
                        )
                    )
            # Keep the authoritative archived/non-DB-backed check after the writes so the
            # generation bump closes the TOCTOU window; if it fails, the surrounding transaction
            # rolls back the earlier span and trace updates.
            self._advance_db_payload_generations_for_db_span_writes(session, all_trace_ids)

            # Re-publish TRACKING_STORE only after the conditional db_payload_generation bump
            # succeeds so
            # span writes, including span-only changes that did not update trace_info, commit
            # atomically with the new DB-backed payload generation.
            for trace_id in all_trace_ids:
                agg = trace_aggregates[trace_id]
                session.merge(
                    SqlTraceTag(
                        request_id=trace_id,
                        key=TraceTagKey.SPANS_LOCATION,
                        value=SpansLocation.TRACKING_STORE.value,
                    )
                )

                # Persist OTel resource attributes (e.g., service.name) as trace tags so
                # they are visible in the UI and available for filtering. Resource is attached
                # to every span produced from the same OTLP ResourceSpans block; use any span
                # in this trace so multi-trace batches are handled correctly.
                # These are written first so that user-defined trace tags (below) take
                # precedence over resource attributes on key collision.
                resource = next(
                    (
                        r
                        for span in spans_by_trace[trace_id]
                        if (r := getattr(span._span, "resource", None)) is not None and r.attributes
                    ),
                    None,
                )
                if resource is not None:
                    for key, value in resource.attributes.items():
                        # Skip OTel SDK internal metadata and the reserved mlflow.*
                        # namespace so a client cannot clobber bookkeeping tags
                        # (e.g. SPANS_LOCATION) via resource attributes.
                        if key.startswith(("telemetry.sdk.", "mlflow.")):
                            continue
                        str_value = value if isinstance(value, str) else json.dumps(value)
                        try:
                            key, str_value = _validate_trace_tag(key, str_value)
                        except Exception:
                            _logger.debug("Skipping invalid resource attribute %r", key)
                            continue
                        session.merge(
                            SqlTraceTag(
                                request_id=trace_id,
                                key=key,
                                value=str_value,
                            )
                        )

                # Restore user-defined tags carried via mlflow.traceTag.* attributes on the root
                # span (set by OtelSpanProcessor when the trace was exported over OTLP).
                # Written after resource attributes so user tags take precedence on collision.
                for tag_key, tag_value in agg.trace_tags.items():
                    if tag_key != TraceTagKey.TRACE_NAME:
                        session.merge(
                            SqlTraceTag(request_id=trace_id, key=tag_key, value=tag_value)
                        )

        return spans

    def _update_trace_info_attributes(
        self, sql_trace_info: SqlTraceInfo, span_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Update trace info attributes based on span dictionary.

        Args:
            sql_trace_info: SqlTraceInfo object
            span_dict: Dictionary of span

        Returns:
            Dictionary of update attributes
        """
        update_dict = {}
        try:
            if sql_trace_info.request_preview is None and (
                trace_inputs := span_dict.get("attributes", {}).get(SpanAttributeKey.INPUTS)
            ):
                update_dict[SqlTraceInfo.request_preview] = _get_truncated_preview(
                    trace_inputs,
                    role="user",
                )

            if sql_trace_info.response_preview is None and (
                trace_outputs := span_dict.get("attributes", {}).get(SpanAttributeKey.OUTPUTS)
            ):
                update_dict[SqlTraceInfo.response_preview] = _get_truncated_preview(
                    trace_outputs,
                    role="assistant",
                )
        except Exception:
            _logger.debug(f"Failed to update trace info attributes: {span_dict}", exc_info=True)

        return update_dict

    def _get_trace_status_from_root_span(self, spans: list[Span]) -> str | None:
        """
        Infer trace status from root span if present.

        Returns the mapped trace status string or None if no root span found.
        """
        for span in spans:
            if span.parent_id is None:  # Found root span (no parent)
                # Map span status to trace status
                span_status = span.status.status_code
                if span_status == SpanStatusCode.ERROR:
                    return TraceState.ERROR.value
                else:
                    # Beyond ERROR, the only other valid span statuses are OK and UNSET.
                    # For both OK and UNSET span statuses, return OK trace status.
                    # UNSET is unexpected in production but we handle it gracefully.
                    return TraceState.OK.value
        return None

    def get_trace(self, trace_id: str, *, allow_partial: bool = False) -> Trace:
        if not allow_partial:
            for retry_count in range(3):
                # only retry if the spans are not fully exported
                if trace := self._get_trace(trace_id, allow_partial):
                    return trace
                elif retry_count < 2:
                    time.sleep(2**retry_count)
            raise MlflowException(
                message=f"Trace with ID {trace_id} is not fully exported yet, "
                "please try again later.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return self._get_trace(trace_id, allow_partial)

    def _get_trace(self, trace_id: str, allow_partial: bool) -> Trace | None:
        """
        Get the trace with spans for given trace id. This function should
        only return None when the spans are not fully exported. If the trace
        info doesn't exist, it should raise an exception.
        """
        with self._tracking_store.ManagedSessionMaker() as session:
            sql_trace_info = (
                self
                ._trace_query(session)
                .options(
                    selectinload(SqlTraceInfo.tags),
                    selectinload(SqlTraceInfo.request_metadata),
                    selectinload(SqlTraceInfo.assessments),
                )
                .filter(SqlTraceInfo.request_id == trace_id)
                .one_or_none()
            )

            if sql_trace_info is None:
                raise MlflowException(
                    message=f"Trace with ID {trace_id} is not found.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

            trace_info = sql_trace_info.to_mlflow_entity()
            span_snapshots = []
            traces_with_cleared_payloads: set[str] = set()
            if (
                trace_info.tags.get(TraceTagKey.SPANS_LOCATION)
                == SpansLocation.TRACKING_STORE.value
            ):
                tracking_store_span_snapshots, traces_with_cleared_payloads = (
                    self._load_tracking_store_span_snapshots(session, [trace_id])
                )
                span_snapshots = tracking_store_span_snapshots.get(trace_id, [])
            trace_snapshot = _TraceReadSnapshot(
                trace_info=trace_info,
                spans=span_snapshots,
            )

        # If archival finalized after we read trace_info but before we loaded spans, reread
        # metadata so archived traces dispatch to archive storage instead of empty DB payloads.
        trace_snapshot = self._refresh_transitioning_trace_snapshot(
            trace_snapshot,
            traces_with_cleared_payloads=traces_with_cleared_payloads,
        )

        spans = self._get_spans_with_trace_info(
            trace_snapshot.trace_info,
            trace_snapshot.spans,
            allow_partial=allow_partial,
        )
        is_archive_backed = (
            trace_snapshot.trace_info.tags.get(TraceTagKey.SPANS_LOCATION)
            == SpansLocation.ARCHIVE_REPO.value
        )
        # Archive-backed reads may legitimately resolve to [] when the archived payload is gone;
        # keep reserving None for "not ready yet" DB-backed traces that should be retried/skipped.
        if allow_partial or spans or (spans is not None and is_archive_backed):
            return Trace(info=trace_snapshot.trace_info, data=TraceData(spans=spans))
        return None

    def _query_trace_infos_in_batches(
        self,
        session,
        trace_ids: list[str],
        experiment_ids: list[str] | None,
        query_options=(),
    ) -> list[SqlTraceInfo]:
        """
        Run the batch trace-info query, chunking both ``trace_ids`` and the
        optional ``experiment_ids`` scope to stay under SQLite's bound-parameter cap.
        Every trace-ID chunk is queried against every experiment-ID chunk,
        then the combined results are restored to the requested trace-ID order.
        """
        trace_id_order = {trace_id: idx for idx, trace_id in enumerate(trace_ids)}
        unique_trace_ids = list(trace_id_order)
        experiment_id_chunks: list[list[int] | None] = [None]
        if experiment_ids is not None:
            parsed_ids = list(dict.fromkeys(_parse_experiment_ids(experiment_ids)))
            experiment_id_chunks = [
                parsed_ids[i : i + self._TRACE_BATCH_QUERY_ID_CHUNK_SIZE]
                for i in range(0, len(parsed_ids), self._TRACE_BATCH_QUERY_ID_CHUNK_SIZE)
            ]

        sql_trace_infos: list[SqlTraceInfo] = []
        for trace_id_start in range(
            0, len(unique_trace_ids), self._TRACE_BATCH_QUERY_ID_CHUNK_SIZE
        ):
            trace_id_chunk = unique_trace_ids[
                trace_id_start : trace_id_start + self._TRACE_BATCH_QUERY_ID_CHUNK_SIZE
            ]
            for experiment_id_chunk in experiment_id_chunks:
                filters = [SqlTraceInfo.request_id.in_(trace_id_chunk)]
                if experiment_id_chunk is not None:
                    filters.append(SqlTraceInfo.experiment_id.in_(experiment_id_chunk))
                query = self._trace_query(session).options(*query_options).filter(*filters)
                sql_trace_infos.extend(query.all())

        sql_trace_infos.sort(key=lambda sti: trace_id_order[sti.request_id])
        return sql_trace_infos

    def batch_get_traces(
        self,
        trace_ids: list[str],
        location: str | None = None,
        experiment_ids: list[str] | None = None,
    ) -> list[Trace]:
        """
        Get complete traces with spans for given trace ids.

        Args:
            trace_ids: The trace IDs to get.
            location: Location of the trace. Should be None for SQLAlchemy backend.
            experiment_ids: Optional list of experiment IDs to scope the query.

        Returns:
            List of Trace objects for the given trace IDs.
        """
        if not trace_ids:
            return []

        with self._tracking_store.ManagedSessionMaker() as session:
            # Load trace metadata first; DB-backed span rows are fetched separately only for traces
            # that still read from the tracking store.
            sql_trace_infos = self._query_trace_infos_in_batches(
                session,
                trace_ids,
                experiment_ids,
                query_options=[
                    selectinload(SqlTraceInfo.tags),
                    selectinload(SqlTraceInfo.request_metadata),
                    selectinload(SqlTraceInfo.assessments),
                ],
            )
            trace_infos = [sql_trace_info.to_mlflow_entity() for sql_trace_info in sql_trace_infos]
            tracking_store_trace_ids = [
                trace_info.trace_id
                for trace_info in trace_infos
                if trace_info.tags.get(TraceTagKey.SPANS_LOCATION)
                == SpansLocation.TRACKING_STORE.value
            ]
            tracking_store_span_snapshots, traces_with_cleared_payloads = (
                self._load_tracking_store_span_snapshots(session, tracking_store_trace_ids)
            )
            trace_snapshots = [
                _TraceReadSnapshot(
                    trace_info=trace_info,
                    spans=tracking_store_span_snapshots.get(trace_info.trace_id, []),
                )
                for trace_info in trace_infos
            ]

        # If archival finalized after we read trace_info but before we loaded spans, reread
        # metadata so archived traces dispatch to archive storage instead of empty DB payloads.
        trace_snapshots = [
            self._refresh_transitioning_trace_snapshot(
                trace_snapshot,
                traces_with_cleared_payloads=traces_with_cleared_payloads,
            )
            for trace_snapshot in trace_snapshots
        ]

        traces = []
        for trace_snapshot in trace_snapshots:
            trace_info = trace_snapshot.trace_info
            # batch_get_traces is depended by search_traces, so we need to return
            # complete traces only.
            spans = self._get_spans_with_trace_info(
                trace_info, trace_snapshot.spans, allow_partial=False
            )
            is_archive_backed = (
                trace_info.tags.get(TraceTagKey.SPANS_LOCATION) == SpansLocation.ARCHIVE_REPO.value
            )
            # Archive-backed traces stay visible even when the payload read resolves to [] so
            # batch reads only skip DB-backed traces that are still incomplete (`spans is None`).
            if spans or (spans is not None and is_archive_backed):
                traces.append(Trace(info=trace_info, data=TraceData(spans=spans)))

        return traces

    def batch_get_trace_infos(
        self,
        trace_ids: list[str],
        location: str | None = None,
        experiment_ids: list[str] | None = None,
    ) -> list[TraceInfo]:
        """
        Get trace metadata (TraceInfo) for given trace IDs without loading spans.

        Args:
            trace_ids: The trace IDs to get.
            location: Location of the trace. Should be None for SQLAlchemy backend.
            experiment_ids: Optional list of experiment IDs to scope the query.

        Returns:
            List of TraceInfo objects for the given trace IDs.
        """
        if not trace_ids:
            return []

        with self._tracking_store.ManagedSessionMaker() as session:
            sql_trace_infos = self._query_trace_infos_in_batches(session, trace_ids, experiment_ids)

            return [sql_trace_info.to_mlflow_entity() for sql_trace_info in sql_trace_infos]

    def _get_trace_ids_outside_tracking_store(
        self, session: Session, trace_ids: Iterable[str]
    ) -> dict[str, str]:
        trace_ids = list(dict.fromkeys(trace_ids))
        if not trace_ids:
            return {}

        blocked_rows = (
            session
            .query(SqlTraceTag.request_id, SqlTraceTag.value)
            .filter(
                SqlTraceTag.request_id.in_(trace_ids),
                SqlTraceTag.key == TraceTagKey.SPANS_LOCATION,
                SqlTraceTag.value != SpansLocation.TRACKING_STORE.value,
            )
            .all()
        )
        return dict(blocked_rows)

    def _get_existing_trace_ids(self, session: Session, trace_ids: Iterable[str]) -> set[str]:
        trace_ids = list(dict.fromkeys(trace_ids))
        if not trace_ids:
            return set()
        return {
            request_id
            for (request_id,) in session
            .query(SqlTraceInfo.request_id)
            .filter(SqlTraceInfo.request_id.in_(trace_ids))
            .all()
        }

    def _raise_log_spans_trace_write_conflict_error(
        self,
        *,
        blocked_trace_ids: dict[str, str],
        missing_trace_ids: list[str],
    ) -> None:
        archived_trace_ids = sorted(
            trace_id
            for trace_id, spans_location in blocked_trace_ids.items()
            if spans_location == SpansLocation.ARCHIVE_REPO.value
        )
        other_non_db_backed_trace_ids = sorted(
            trace_id
            for trace_id, spans_location in blocked_trace_ids.items()
            if spans_location != SpansLocation.ARCHIVE_REPO.value
        )

        if archived_trace_ids and not other_non_db_backed_trace_ids and not missing_trace_ids:
            archived_trace_list = ", ".join(f"'{trace_id}'" for trace_id in archived_trace_ids)
            raise MlflowException(
                f"Cannot log spans to archived traces: {archived_trace_list}.",
                error_code=INVALID_STATE,
            )

        if missing_trace_ids and not archived_trace_ids and not other_non_db_backed_trace_ids:
            missing_trace_list = ", ".join(f"'{trace_id}'" for trace_id in missing_trace_ids)
            raise MlflowException(
                f"Cannot log spans to traces that no longer exist: {missing_trace_list}.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        detail_groups = []
        if archived_trace_ids:
            detail_groups.append(
                "archived=" + ", ".join(f"'{trace_id}'" for trace_id in archived_trace_ids)
            )
        if other_non_db_backed_trace_ids:
            detail_groups.append(
                "other=" + ", ".join(f"'{trace_id}'" for trace_id in other_non_db_backed_trace_ids)
            )
        if missing_trace_ids:
            detail_groups.append(
                "missing=" + ", ".join(f"'{trace_id}'" for trace_id in missing_trace_ids)
            )

        detail_suffix = f": {'; '.join(detail_groups)}" if detail_groups else "."
        raise MlflowException(
            "Cannot log spans because one or more traces are unavailable for DB-backed span "
            f"writes{detail_suffix}",
            error_code=INVALID_STATE,
        )

    def _advance_db_payload_generations_for_db_span_writes(
        self, session: Session, trace_ids: Iterable[str]
    ) -> None:
        """
        Atomically advance the DB-backed payload generation for each touched trace.

        The generation bump is the writer-side guard against archival finalization: if a trace is
        no longer DB-backed at commit time, the conditional UPDATE affects fewer rows than
        expected and the whole write transaction must roll back.
        """
        trace_ids = list(dict.fromkeys(trace_ids))
        if not trace_ids:
            return

        spans_outside_tracking_store = exists().where(
            and_(
                SqlTraceTag.request_id == SqlTraceInfo.request_id,
                SqlTraceTag.key == TraceTagKey.SPANS_LOCATION,
                SqlTraceTag.value != SpansLocation.TRACKING_STORE.value,
            )
        )
        updated_rows = (
            session
            .query(SqlTraceInfo)
            .filter(
                SqlTraceInfo.request_id.in_(trace_ids),
                ~spans_outside_tracking_store,
            )
            .update(
                {SqlTraceInfo.db_payload_generation: SqlTraceInfo.db_payload_generation + 1},
                synchronize_session=False,
            )
        )
        if updated_rows == len(trace_ids):
            return

        # We intentionally pay for this reread only on the failure path because
        # archival races are rare, and we still need this post-UPDATE check to
        # distinguish archived traces from other non-DB-backed states.
        blocked_trace_ids = self._get_trace_ids_outside_tracking_store(session, trace_ids)
        existing_trace_ids = self._get_existing_trace_ids(session, trace_ids)
        missing_trace_ids = [
            trace_id for trace_id in trace_ids if trace_id not in existing_trace_ids
        ]
        self._raise_log_spans_trace_write_conflict_error(
            blocked_trace_ids=blocked_trace_ids,
            missing_trace_ids=missing_trace_ids,
        )

    def archive_traces(
        self,
        *,
        resolved_trace_archival_location: str,
        broader_retention: str,
        long_retention_allowlist: set[str] | list[str] | None = None,
        max_traces_per_pass: int | None = None,
    ) -> int:
        if max_traces_per_pass is not None and max_traces_per_pass <= 0:
            raise MlflowException.invalid_parameter_value(
                f"`max_traces_per_pass` must be a positive integer, received {max_traces_per_pass}."
            )
        if not resolved_trace_archival_location:
            raise MlflowException.invalid_parameter_value(
                "`resolved_trace_archival_location` must be provided."
            )
        resolved_trace_archival_location = _validate_trace_archival_location(
            resolved_trace_archival_location,
            parameter_name="resolved_trace_archival_location",
        )
        if not broader_retention:
            raise MlflowException.invalid_parameter_value("`broader_retention` must be provided.")
        broader_retention = _validate_trace_archival_retention_string(
            broader_retention, parameter_name="broader_retention"
        )
        now_millis = self._get_archive_traces_now_millis()
        long_retention_allowlist = {
            str(experiment_id) for experiment_id in long_retention_allowlist or []
        }
        broader_retention_millis = _parse_trace_archival_duration_millis(broader_retention)
        if broader_retention_millis is None:
            raise MlflowException(
                "Trace archival config resolution returned no archival retention.",
                error_code=INTERNAL_ERROR,
            )
        with self._tracking_store.ManagedSessionMaker() as session:
            archive_now_requests, candidates = self._plan_trace_archival(
                session=session,
                now_millis=now_millis,
                broader_retention_millis=broader_retention_millis,
                long_retention_allowlist=long_retention_allowlist,
                max_traces_per_pass=max_traces_per_pass,
            )
        return self._execute_trace_archival_plan(
            archive_now_requests=archive_now_requests,
            candidates=candidates,
            resolved_trace_archival_location=resolved_trace_archival_location,
            max_traces_per_pass=max_traces_per_pass,
            now_millis=now_millis,
        )

    def _plan_trace_archival(
        self,
        *,
        session: Session,
        now_millis: int,
        broader_retention_millis: int,
        long_retention_allowlist: set[str],
        max_traces_per_pass: int | None,
    ) -> tuple[list[_ArchiveNowCleanupRequest], list[_TraceArchiveCandidate]]:
        archive_now_requests: list[_ArchiveNowCleanupRequest] = []
        # Group archive-now experiments by cutoff so experiments requesting the same urgency
        # can share a candidate query instead of scanning one experiment at a time.
        archive_now_cutoff_groups: dict[int | None, list[str]] = defaultdict(list)
        regular_cutoff_groups: dict[int, list[str]] = defaultdict(list)
        for experiment_id, experiment_tags in self._get_active_experiment_trace_archival_tags(
            session
        ):
            archive_now_raw = experiment_tags.get(TraceExperimentTagKey.ARCHIVE_NOW)
            archive_now = _ArchiveNowRequest.from_tag_value(archive_now_raw)
            if archive_now and archive_now_raw is not None:
                archive_now_requests.append(
                    _ArchiveNowCleanupRequest(
                        experiment_id=experiment_id,
                        raw_value=archive_now_raw,
                        parsed_request=archive_now,
                    )
                )
                archive_now_cutoff_groups[
                    now_millis - archive_now.older_than_millis
                    if archive_now.older_than_millis is not None
                    else None
                ].append(experiment_id)

            retention_millis = self._resolve_effective_trace_archival_retention_millis(
                experiment_id=experiment_id,
                experiment_tags=experiment_tags,
                broader_retention_millis=broader_retention_millis,
                long_retention_allowlist=long_retention_allowlist,
            )
            archive_now_covers_retention = archive_now is not None and (
                archive_now.older_than_millis is None
                or archive_now.older_than_millis <= retention_millis
            )
            if retention_millis and not archive_now_covers_retention:
                regular_cutoff_groups[now_millis - retention_millis].append(experiment_id)

        archive_now_candidates = self._collect_grouped_trace_archive_candidates(
            session=session,
            cutoff_groups=archive_now_cutoff_groups,
            max_traces_per_pass=max_traces_per_pass,
        )

        regular_candidates: list[_TraceArchiveCandidate] = []
        if max_traces_per_pass is None or len(archive_now_candidates) < max_traces_per_pass:
            regular_candidates = self._collect_grouped_trace_archive_candidates(
                session=session,
                cutoff_groups=regular_cutoff_groups,
                max_traces_per_pass=max_traces_per_pass,
            )

        return (
            archive_now_requests,
            self._dedupe_trace_archive_candidates(archive_now_candidates, regular_candidates),
        )

    def _collect_grouped_trace_archive_candidates(
        self,
        *,
        session: Session,
        cutoff_groups: dict[int | None, list[str]],
        max_traces_per_pass: int | None,
    ) -> list[_TraceArchiveCandidate]:
        candidates: list[_TraceArchiveCandidate] = []
        for cutoff, experiment_ids in cutoff_groups.items():
            candidates = self._merge_limited_trace_archive_candidates(
                candidates,
                self._find_archivable_trace_candidates_for_experiments(
                    session=session,
                    experiment_ids=experiment_ids,
                    max_timestamp_millis=cutoff,
                    limit=max_traces_per_pass,
                ),
                limit=max_traces_per_pass,
            )
        return candidates

    def _execute_trace_archival_plan(
        self,
        *,
        archive_now_requests: list[_ArchiveNowCleanupRequest],
        candidates: list[_TraceArchiveCandidate],
        resolved_trace_archival_location: str,
        max_traces_per_pass: int | None,
        now_millis: int,
    ) -> int:
        archive_now_experiment_ids = {request.experiment_id for request in archive_now_requests}
        retryable_failure_experiment_ids: set[str] = set()
        archived_count = 0
        candidates_to_archive = (
            candidates if max_traces_per_pass is None else candidates[:max_traces_per_pass]
        )

        try:
            for candidate in candidates_to_archive:
                try:
                    if self._archive_trace_candidate(
                        trace_id=candidate.trace_id,
                        resolved_trace_archival_location=resolved_trace_archival_location,
                    ):
                        archived_count += 1
                except (MlflowException, SQLAlchemyError):
                    _logger.warning(
                        "Failed to archive trace %s; leaving it eligible for retry.",
                        candidate.trace_id,
                        exc_info=True,
                    )
                    if candidate.experiment_id in archive_now_experiment_ids:
                        retryable_failure_experiment_ids.add(candidate.experiment_id)
        finally:
            self._clear_completed_archive_now_requests(
                archive_now_requests=archive_now_requests,
                now_millis=now_millis,
                retryable_failure_experiment_ids=retryable_failure_experiment_ids,
            )

        return archived_count

    def _get_archive_traces_now_millis(self) -> int:
        """Return the wall-clock cutoff time for this archival pass.

        This small wrapper exists so tests can freeze the scheduler clock without
        patching the shared time utility.
        """
        return get_current_time_millis()

    def _resolve_effective_trace_archival_retention_millis(
        self,
        *,
        experiment_id: str,
        experiment_tags: dict[str, str],
        broader_retention_millis: int,
        long_retention_allowlist: set[str],
    ) -> int:
        resolved_retention = _resolve_effective_trace_archival_retention(
            experiment_id=experiment_id,
            experiment_tags=experiment_tags,
            broader_retention=_validate_trace_archival_retention_string(
                _format_trace_archival_duration_millis(broader_retention_millis)
            ),
            long_retention_allowlist=long_retention_allowlist,
        )
        resolved_retention_millis = _parse_trace_archival_duration_millis(resolved_retention)
        if resolved_retention_millis is None:
            return broader_retention_millis
        return resolved_retention_millis

    def _get_active_experiment_trace_archival_tags(
        self, session: Session
    ) -> list[tuple[str, dict[str, str]]]:
        # Fetch active experiments with only the archival-related tags consulted by the scheduler.
        # The outer join keeps experiments that inherit defaults, and the single query avoids both
        # joined-loading unrelated tags and building a large Python-side IN list.
        rows = (
            self._tracking_store
            ._get_query(session, SqlExperiment)
            .outerjoin(
                SqlExperimentTag,
                and_(
                    SqlExperimentTag.experiment_id == SqlExperiment.experiment_id,
                    SqlExperimentTag.key.in_([
                        TraceExperimentTagKey.ARCHIVE_NOW,
                        TraceExperimentTagKey.ARCHIVAL_RETENTION,
                    ]),
                ),
            )
            .with_entities(
                SqlExperiment.experiment_id, SqlExperimentTag.key, SqlExperimentTag.value
            )
            .filter(SqlExperiment.lifecycle_stage == LifecycleStage.ACTIVE)
            .order_by(SqlExperiment.experiment_id.asc(), SqlExperimentTag.key.asc())
            .all()
        )

        experiments: list[tuple[str, dict[str, str]]] = []
        current_experiment_id: str | None = None
        current_tags: dict[str, str] | None = None
        for experiment_id, key, value in rows:
            experiment_id = str(experiment_id)
            if experiment_id != current_experiment_id:
                current_experiment_id = experiment_id
                current_tags = {}
                experiments.append((experiment_id, current_tags))
            if key is not None:
                current_tags[key] = value

        return experiments

    @staticmethod
    def _merge_limited_trace_archive_candidates(
        existing_candidates: list[_TraceArchiveCandidate],
        new_candidates: list[_TraceArchiveCandidate],
        *,
        limit: int | None,
    ) -> list[_TraceArchiveCandidate]:
        if limit is not None and limit <= 0:
            return []

        if not existing_candidates:
            candidates = new_candidates
        elif not new_candidates:
            candidates = existing_candidates
        else:
            candidates = [*existing_candidates, *new_candidates]

        sorted_candidates = sorted(
            candidates, key=lambda candidate: (candidate.timestamp_ms, candidate.trace_id)
        )
        return sorted_candidates if limit is None else sorted_candidates[:limit]

    @staticmethod
    def _dedupe_trace_archive_candidates(
        archive_now_candidates: list[_TraceArchiveCandidate],
        regular_candidates: list[_TraceArchiveCandidate],
    ) -> list[_TraceArchiveCandidate]:
        deduped: list[_TraceArchiveCandidate] = []
        seen_trace_ids: set[str] = set()
        for candidates in (
            sorted(archive_now_candidates, key=lambda c: (c.timestamp_ms, c.trace_id)),
            sorted(regular_candidates, key=lambda c: (c.timestamp_ms, c.trace_id)),
        ):
            for candidate in candidates:
                if candidate.trace_id in seen_trace_ids:
                    continue
                deduped.append(candidate)
                seen_trace_ids.add(candidate.trace_id)
        return deduped

    def _find_archivable_trace_candidates_for_experiments(
        self,
        *,
        session: Session,
        experiment_ids: list[str],
        max_timestamp_millis: int | None,
        limit: int | None = None,
    ) -> list[_TraceArchiveCandidate]:
        if not experiment_ids:
            return []

        spans_outside_tracking_store = exists().where(
            and_(
                SqlTraceTag.request_id == SqlTraceInfo.request_id,
                SqlTraceTag.key == TraceTagKey.SPANS_LOCATION,
                SqlTraceTag.value != SpansLocation.TRACKING_STORE.value,
            )
        )
        archival_failure_exists = exists().where(
            and_(
                SqlTraceTag.request_id == SqlTraceInfo.request_id,
                SqlTraceTag.key == TraceTagKey.ARCHIVAL_FAILURE,
            )
        )
        non_empty_span_exists = exists().where(
            and_(
                SqlSpan.trace_id == SqlTraceInfo.request_id,
                SqlSpan.content != "",
            )
        )

        candidates: list[_TraceArchiveCandidate] = []
        for chunk_start in range(0, len(experiment_ids), _TRACE_ARCHIVAL_EXPERIMENT_ID_CHUNK_SIZE):
            # Query experiment ids in chunks to keep the SQL IN list bounded.
            experiment_id_chunk = [
                int(experiment_id)
                for experiment_id in experiment_ids[
                    chunk_start : chunk_start + _TRACE_ARCHIVAL_EXPERIMENT_ID_CHUNK_SIZE
                ]
            ]
            statement = (
                self
                ._trace_query(session)
                .with_entities(
                    SqlTraceInfo.request_id, SqlTraceInfo.experiment_id, SqlTraceInfo.timestamp_ms
                )
                .filter(
                    SqlTraceInfo.experiment_id.in_(experiment_id_chunk),
                    SqlTraceInfo.status != TraceState.IN_PROGRESS.value,
                    ~spans_outside_tracking_store,
                    ~archival_failure_exists,
                    non_empty_span_exists,
                )
                .order_by(SqlTraceInfo.timestamp_ms.asc(), SqlTraceInfo.request_id.asc())
            )
            if max_timestamp_millis is not None:
                statement = statement.filter(SqlTraceInfo.timestamp_ms <= max_timestamp_millis)
            if limit is not None:
                statement = statement.limit(limit)

            chunk_candidates = [
                _TraceArchiveCandidate(
                    trace_id=trace_id,
                    experiment_id=str(experiment_id),
                    timestamp_ms=timestamp_ms,
                )
                for trace_id, experiment_id, timestamp_ms in statement.all()
            ]
            if limit is None:
                candidates.extend(chunk_candidates)
            else:
                candidates = self._merge_limited_trace_archive_candidates(
                    candidates, chunk_candidates, limit=limit
                )

        if limit is None:
            return sorted(
                candidates, key=lambda candidate: (candidate.timestamp_ms, candidate.trace_id)
            )
        return candidates

    def _archive_trace_candidate(
        self,
        *,
        trace_id: str,
        resolved_trace_archival_location: str,
    ) -> bool:
        archival_data = self._load_trace_archival_data(trace_id)
        if archival_data is None:
            return False
        trace_info, db_payload_generation, span_rows = archival_data

        try:
            archived_pb = self._serialize_trace_archival_span_rows_to_pb(span_rows)
        except MlflowTraceArchivalMalformedTrace:
            _logger.warning("Marking trace %s as MALFORMED_TRACE during archival.", trace_id)
            self._mark_trace_archival_failure(
                trace_id=trace_id,
                failure_reason=TraceArchivalFailureReason.MALFORMED_TRACE.value,
                db_payload_generation=db_payload_generation,
            )
            return False

        artifact_uri = append_to_uri_path(
            resolved_trace_archival_location,
            str(trace_info.experiment_id),
            self.TRACE_FOLDER_NAME,
            trace_info.trace_id,
            self.ARTIFACTS_FOLDER_NAME,
        )
        artifact_repo = get_artifact_repository(artifact_uri)

        try:
            artifact_repo.upload_archived_trace_data_bytes(archived_pb)
        except Exception as e:
            self._delete_unreferenced_archived_trace_payload(
                trace_id=trace_id,
                artifact_uri=artifact_uri,
                artifact_repo=artifact_repo,
            )
            # Normalize backend-specific upload errors so the outer archival loop
            # can treat them as retryable without depending on repository internals.
            raise MlflowException("Trace archival upload failed.") from e

        try:
            finalized = self._finalize_archived_trace(
                trace_id=trace_id,
                artifact_uri=artifact_uri,
                db_payload_generation=db_payload_generation,
            )
        except Exception as e:
            self._delete_unreferenced_archived_trace_payload(
                trace_id=trace_id,
                artifact_uri=artifact_uri,
                artifact_repo=artifact_repo,
            )
            if isinstance(e, MlflowException):
                raise
            # Normalize unexpected finalize-time errors after cleanup so the
            # outer archival loop can retry them consistently.
            raise MlflowException("Trace archival finalization failed.") from e

        if not finalized:
            self._delete_unreferenced_archived_trace_payload(
                trace_id=trace_id,
                artifact_uri=artifact_uri,
                artifact_repo=artifact_repo,
            )

        return finalized

    def _load_trace_archival_data(
        self, trace_id: str
    ) -> tuple[TraceInfo, int, list[tuple[str, str]]] | None:
        with self._tracking_store.ManagedSessionMaker() as session:
            sql_trace_info = (
                self
                ._trace_query(session)
                .options(selectinload(SqlTraceInfo.tags))
                .filter(SqlTraceInfo.request_id == trace_id)
                .one_or_none()
            )
            if sql_trace_info is None:
                return None

            span_rows = [
                (span_id, content)
                for span_id, content in (
                    session
                    .query(SqlSpan.span_id, SqlSpan.content)
                    .filter(SqlSpan.trace_id == trace_id)
                    .order_by(SqlSpan.span_id)
                    .all()
                )
            ]

            trace_info = sql_trace_info.to_mlflow_entity()
            if not self._is_trace_actionable_for_archival(trace_info, span_rows):
                return None

            return trace_info, sql_trace_info.db_payload_generation, span_rows

    def _serialize_trace_archival_span_rows_to_pb(self, span_rows: list[tuple[str, str]]) -> bytes:
        try:
            spans = [
                Span.from_dict(translate_loaded_span(json.loads(content)))
                for _, content in span_rows
            ]
            return spans_to_traces_data_pb(spans)
        except (MlflowException, TypeError, ValueError, AttributeError) as e:
            raise MlflowTraceArchivalMalformedTrace(str(e)) from e

    def _is_trace_actionable_for_archival(
        self, trace_info: TraceInfo, span_rows: list[tuple[str, str]]
    ) -> bool:
        spans_location = trace_info.tags.get(TraceTagKey.SPANS_LOCATION)
        if not self._is_db_backed_spans_location(spans_location):
            return False
        if TraceTagKey.ARCHIVAL_FAILURE in trace_info.tags:
            return False
        if trace_info.state == TraceState.IN_PROGRESS:
            return False
        return any(content != "" for _, content in span_rows)

    @staticmethod
    def _is_db_backed_spans_location(spans_location: str | None) -> bool:
        return spans_location in (None, SpansLocation.TRACKING_STORE.value)

    def _is_sql_trace_db_backed(self, sql_trace_info: SqlTraceInfo) -> bool:
        spans_location = next(
            (tag.value for tag in sql_trace_info.tags if tag.key == TraceTagKey.SPANS_LOCATION),
            None,
        )
        return self._is_db_backed_spans_location(spans_location)

    def _is_trace_metadata_actionable_for_archival(self, sql_trace_info: SqlTraceInfo) -> bool:
        if not self._is_sql_trace_db_backed(sql_trace_info):
            return False
        if any(tag.key == TraceTagKey.ARCHIVAL_FAILURE for tag in sql_trace_info.tags):
            return False
        return sql_trace_info.status in {
            TraceState.OK.value,
            TraceState.ERROR.value,
            TraceState.STATE_UNSPECIFIED.value,
        }

    def _trace_has_non_empty_span_content(self, session: Session, trace_id: str) -> bool:
        # Use .first() instead of a top-level EXISTS query for MSSQL compatibility.
        return (
            session
            .query(SqlSpan.span_id)
            .filter(
                SqlSpan.trace_id == trace_id,
                SqlSpan.content != "",
            )
            .first()
            is not None
        )

    def _delete_unreferenced_archived_trace_payload(
        self,
        *,
        trace_id: str,
        artifact_uri: str,
        artifact_repo,
    ) -> None:
        with self._tracking_store.ManagedSessionMaker() as session:
            sql_trace_info = (
                self
                ._trace_query(session)
                .options(joinedload(SqlTraceInfo.tags))
                .filter(SqlTraceInfo.request_id == trace_id)
                .one_or_none()
            )
            if sql_trace_info is not None:
                trace_info = sql_trace_info.to_mlflow_entity()
                if (
                    trace_info.tags.get(TraceTagKey.SPANS_LOCATION)
                    == SpansLocation.ARCHIVE_REPO.value
                    and trace_info.tags.get(TraceTagKey.ARCHIVE_LOCATION) == artifact_uri
                ):
                    return

        try:
            artifact_repo.delete_artifacts(TRACE_ARCHIVAL_FILENAME)
        except Exception:
            _logger.warning(
                "Failed to delete unreferenced archived payload for trace %s.",
                trace_id,
                exc_info=True,
            )

    def _finalize_archived_trace(
        self,
        *,
        trace_id: str,
        artifact_uri: str,
        db_payload_generation: int,
    ) -> bool:
        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            sql_trace_info = (
                self
                ._trace_query(session, for_update_or_delete=True)
                .options(selectinload(SqlTraceInfo.tags))
                .filter(SqlTraceInfo.request_id == trace_id)
                .one_or_none()
            )
            if sql_trace_info is None:
                return False

            if sql_trace_info.db_payload_generation != db_payload_generation:
                return False
            if not self._is_trace_metadata_actionable_for_archival(sql_trace_info):
                return False
            # The db_payload_generation check is the primary race guard; this cheap EXISTS probe is
            # defense in depth that confirms DB-backed span content still exists without
            # triggering a second full load of potentially large span payloads.
            if not self._trace_has_non_empty_span_content(session, trace_id):
                return False

            (
                session
                .query(SqlSpan)
                .filter(SqlSpan.trace_id == trace_id)
                .update({SqlSpan.content: ""}, synchronize_session=False)
            )
            session.merge(
                SqlTraceTag(
                    request_id=trace_id,
                    key=TraceTagKey.SPANS_LOCATION,
                    value=SpansLocation.ARCHIVE_REPO.value,
                )
            )
            session.merge(
                SqlTraceTag(
                    request_id=trace_id,
                    key=TraceTagKey.ARCHIVE_LOCATION,
                    value=artifact_uri,
                )
            )
            (
                session
                .query(SqlTraceTag)
                .filter(
                    SqlTraceTag.request_id == trace_id,
                    SqlTraceTag.key == TraceTagKey.ARCHIVAL_FAILURE,
                )
                .delete(synchronize_session=False)
            )
            return True

    def _mark_trace_archival_failure(
        self, *, trace_id: str, failure_reason: str, db_payload_generation: int | None = None
    ) -> None:
        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            sql_trace_info = (
                self
                ._trace_query(session, for_update_or_delete=True)
                # Keep tags out of the locking SELECT: PostgreSQL rejects FOR UPDATE on
                # LEFT OUTER JOIN-loaded collections from the nullable side of the join.
                .options(selectinload(SqlTraceInfo.tags))
                .filter(SqlTraceInfo.request_id == trace_id)
                .one_or_none()
            )
            if sql_trace_info is None:
                return

            if not self._is_sql_trace_db_backed(sql_trace_info):
                return
            if (
                db_payload_generation is not None
                and sql_trace_info.db_payload_generation != db_payload_generation
            ):
                return

            session.merge(
                SqlTraceTag(
                    request_id=trace_id,
                    key=TraceTagKey.ARCHIVAL_FAILURE,
                    value=failure_reason,
                )
            )

    def _clear_completed_archive_now_requests(
        self,
        *,
        archive_now_requests: list[_ArchiveNowCleanupRequest],
        now_millis: int,
        retryable_failure_experiment_ids: set[str] | None = None,
    ) -> None:
        if not archive_now_requests:
            return

        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            for request in archive_now_requests:
                if (
                    retryable_failure_experiment_ids
                    and request.experiment_id in retryable_failure_experiment_ids
                ):
                    continue

                older_than_cutoff = (
                    now_millis - request.parsed_request.older_than_millis
                    if request.parsed_request.older_than_millis is not None
                    else None
                )
                remaining_state = self._get_archive_now_remaining_state(
                    session=session,
                    experiment_id=request.experiment_id,
                    max_timestamp_millis=older_than_cutoff,
                )
                if remaining_state in (
                    _ArchiveNowRemainingState.ARCHIVABLE,
                    _ArchiveNowRemainingState.TRANSIENT,
                ):
                    continue
                if remaining_state == _ArchiveNowRemainingState.BLOCKED_UNMARKED:
                    _logger.warning(
                        "Clearing archive-now request %r on experiment %s. Some matching traces "
                        "still remain in the tracking store but are not currently archivable "
                        "and are not marked with an archival failure.",
                        request.raw_value,
                        request.experiment_id,
                    )

                (
                    session
                    .query(SqlExperimentTag)
                    .filter(
                        SqlExperimentTag.experiment_id == int(request.experiment_id),
                        SqlExperimentTag.key == TraceExperimentTagKey.ARCHIVE_NOW,
                        # Only clear the request we started with; a newer archive-now value may
                        # have been written while this archival pass was still running.
                        SqlExperimentTag.value == request.raw_value,
                    )
                    .delete(synchronize_session=False)
                )

    def _get_archive_now_remaining_state(
        self,
        *,
        session: Session,
        experiment_id: str,
        max_timestamp_millis: int | None,
    ) -> _ArchiveNowRemainingState:
        spans_outside_tracking_store = exists().where(
            and_(
                SqlTraceTag.request_id == SqlTraceInfo.request_id,
                SqlTraceTag.key == TraceTagKey.SPANS_LOCATION,
                SqlTraceTag.value != SpansLocation.TRACKING_STORE.value,
            )
        )
        archival_failure_exists = exists().where(
            and_(
                SqlTraceTag.request_id == SqlTraceInfo.request_id,
                SqlTraceTag.key == TraceTagKey.ARCHIVAL_FAILURE,
            )
        )
        non_empty_span_exists = exists().where(
            and_(
                SqlSpan.trace_id == SqlTraceInfo.request_id,
                SqlSpan.content != "",
            )
        )

        remaining_statement = (
            self
            ._trace_query(session)
            .with_entities(SqlTraceInfo.request_id)
            .filter(
                SqlTraceInfo.experiment_id == int(experiment_id),
                ~spans_outside_tracking_store,
            )
        )
        if max_timestamp_millis is not None:
            remaining_statement = remaining_statement.filter(
                SqlTraceInfo.timestamp_ms <= max_timestamp_millis
            )

        if remaining_statement.first() is None:
            return _ArchiveNowRemainingState.DONE

        if (
            remaining_statement.filter(
                SqlTraceInfo.status != TraceState.IN_PROGRESS.value,
                ~archival_failure_exists,
                non_empty_span_exists,
            ).first()
            is not None
        ):
            return _ArchiveNowRemainingState.ARCHIVABLE

        if (
            remaining_statement.filter(
                SqlTraceInfo.status == TraceState.IN_PROGRESS.value,
                ~archival_failure_exists,
            ).first()
            is not None
        ):
            return _ArchiveNowRemainingState.TRANSIENT

        # Only clear archive-now once the remaining traces are either fully processed
        # or explicitly marked as terminal archival failures.
        if remaining_statement.filter(~archival_failure_exists).first() is not None:
            return _ArchiveNowRemainingState.BLOCKED_UNMARKED

        return _ArchiveNowRemainingState.TERMINAL_FAILURES_ONLY

    def _get_spans_with_trace_info(
        self, trace_info: TraceInfo, spans: list[_TraceSpanSnapshot], allow_partial: bool = True
    ) -> list[Span] | None:
        spans_location = trace_info.tags.get(TraceTagKey.SPANS_LOCATION)
        if spans_location == SpansLocation.ARCHIVE_REPO.value:
            artifact_repo = get_artifact_repository(get_archive_uri_for_trace(trace_info))
            return artifact_repo.download_archived_trace_data().spans

        # ARTIFACT_REPO traces still rely on the existing handler/client fallback path because the
        # trace artifact URI may use a proxy-only scheme such as mlflow-artifacts://.
        if spans_location != SpansLocation.TRACKING_STORE.value:
            raise MlflowTracingException("Trace data not stored in tracking store")

        span_snapshots = sorted(
            spans,
            key=lambda span: (
                # Root spans come first, then sort by start time
                0 if span.parent_span_id is None else 1,
                span.start_time_unix_nano,
            ),
        )
        # check whether all spans are logged before returning if not allow partial
        if not allow_partial and (
            trace_stats := trace_info.trace_metadata.get(TraceMetadataKey.SIZE_STATS)
        ):
            trace_stats = json.loads(trace_stats)
            num_spans = trace_stats.get(TraceSizeStatsKey.NUM_SPANS, 0)
            if len(span_snapshots) < num_spans:
                _logger.debug(
                    f"Trace {trace_info.trace_id} is not fully exported yet, "
                    f"expecting {num_spans} spans but got {len(span_snapshots)}"
                )
                return

        # Defer OTel Span reconstruction until a caller needs properties or
        # to_otel_proto(). Callers that only need dicts (e.g. TraceData.to_dict /
        # get-trace-artifact) skip Span.from_dict entirely. Unmodified stored
        # JSON is retained for artifact passthrough via TraceData.to_json_bytes().
        return [LazySpan.from_stored_content(span.content) for span in span_snapshots]

    def _load_tracking_store_span_snapshots(
        self, session: Session, trace_ids: Iterable[str]
    ) -> tuple[dict[str, list[_TraceSpanSnapshot]], set[str]]:
        trace_ids = list(dict.fromkeys(trace_ids))
        if not trace_ids:
            return {}, set()

        span_snapshots_by_trace_id: dict[str, list[_TraceSpanSnapshot]] = defaultdict(list)
        traces_with_cleared_payloads: set[str] = set()
        rows = (
            session
            .query(
                SqlSpan.trace_id,
                SqlSpan.content,
                SqlSpan.parent_span_id,
                SqlSpan.start_time_unix_nano,
            )
            .filter(SqlSpan.trace_id.in_(trace_ids))
            .all()
        )
        for trace_id, content, parent_span_id, start_time_unix_nano in rows:
            if content == "":
                traces_with_cleared_payloads.add(trace_id)
                continue
            span_snapshots_by_trace_id[trace_id].append(
                _TraceSpanSnapshot(
                    content=content,
                    parent_span_id=parent_span_id,
                    start_time_unix_nano=start_time_unix_nano,
                )
            )
        return span_snapshots_by_trace_id, traces_with_cleared_payloads

    def _refresh_transitioning_trace_snapshot(
        self,
        trace_snapshot: _TraceReadSnapshot,
        *,
        traces_with_cleared_payloads: set[str],
    ) -> _TraceReadSnapshot:
        """
        Refresh a DB-backed read snapshot when archival may have finalized mid-read.

        A trace can be observed as TRACKING_STORE in the initial metadata read, then have its
        span payloads cleared and its tags flipped to ARCHIVE_REPO before the span rows are
        loaded. When that happens, reread the trace metadata and return an archive-backed
        snapshot so the caller can fetch spans from the archive repository instead of treating
        the cleared DB rows as missing data.
        """
        trace_info = trace_snapshot.trace_info
        if (
            trace_info.tags.get(TraceTagKey.SPANS_LOCATION) != SpansLocation.TRACKING_STORE.value
            or trace_info.trace_id not in traces_with_cleared_payloads
        ):
            return trace_snapshot

        refreshed_trace_info = self.get_trace_info(trace_info.trace_id)
        if (
            refreshed_trace_info.tags.get(TraceTagKey.SPANS_LOCATION)
            == SpansLocation.TRACKING_STORE.value
        ):
            return trace_snapshot

        return _TraceReadSnapshot(
            trace_info=refreshed_trace_info,
            spans=[],
        )

    def deprecated_start_trace_v2(
        self,
        experiment_id: str,
        timestamp_ms: int,
        request_metadata: dict[str, str],
        tags: dict[str, str],
    ) -> TraceInfoV2:
        """
        DEPRECATED. DO NOT USE.

        Create an initial TraceInfo object in the database.

        Args:
            experiment_id: String id of the experiment for this run.
            timestamp_ms: Start time of the trace, in milliseconds since the UNIX epoch.
            request_metadata: Metadata of the trace.
            tags: Tags of the trace.

        Returns:
            The created TraceInfo object.
        """
        trace_name = validate_trace_name(tags.get(TraceTagKey.TRACE_NAME))
        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            experiment = self._tracking_store.get_experiment(experiment_id)
            self._tracking_store._check_experiment_is_active(experiment)

            request_id = generate_request_id_v2()
            request_metadata = dict(request_metadata)
            trace_info = SqlTraceInfo(
                request_id=request_id,
                experiment_id=experiment_id,
                timestamp_ms=timestamp_ms,
                execution_time_ms=None,
                status=TraceStatus.IN_PROGRESS,
                trace_name=trace_name,
                **analytics_columns_from_metadata(request_metadata),
            )

            trace_info.tags = [
                SqlTraceTag(key=k, value=v) for k, v in tags.items() if k != TraceTagKey.TRACE_NAME
            ]
            trace_info.tags.append(self._get_trace_artifact_location_tag(experiment, request_id))

            for key in PROMOTED_TRACE_METADATA_KEYS:
                request_metadata.pop(key, None)

            # Emit metadata rows in sorted key order to keep PK-index lock acquisition
            # consistent with the other trace-metadata writers and avoid deadlocks (#24332).
            trace_info.request_metadata = [
                SqlTraceMetadata(key=k, value=v) for k, v in sorted(request_metadata.items())
            ]
            enqueue_rollup_rebuilds(
                session,
                RollupFamily.TRACE_METRIC,
                int(experiment_id),
                [timestamp_ms],
            )
            session.add(trace_info)

            return TraceInfoV2.from_v3(trace_info.to_mlflow_entity())

    def deprecated_end_trace_v2(
        self,
        request_id: str,
        timestamp_ms: int,
        status: TraceStatus,
        request_metadata: dict[str, str],
        tags: dict[str, str],
    ) -> TraceInfoV2:
        """
        DEPRECATED. DO NOT USE.

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
        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            sql_trace_info = self._get_sql_trace_info(session, request_id)
            trace_start_time_ms = sql_trace_info.timestamp_ms
            enqueue_rollup_rebuilds(
                session,
                RollupFamily.TRACE_METRIC,
                int(sql_trace_info.experiment_id),
                [trace_start_time_ms],
            )
            execution_time_ms = timestamp_ms - trace_start_time_ms
            sql_trace_info.execution_time_ms = execution_time_ms
            sql_trace_info.status = status
            request_metadata = dict(request_metadata)
            analytics_columns = analytics_columns_from_metadata(request_metadata)
            for column, value in analytics_columns.items():
                setattr(sql_trace_info, column, value)
            for key in PROMOTED_TRACE_METADATA_KEYS:
                request_metadata.pop(key, None)

            tags = dict(tags)
            if TraceTagKey.TRACE_NAME in tags:
                sql_trace_info.trace_name = validate_trace_name(tags.pop(TraceTagKey.TRACE_NAME))
            session.merge(sql_trace_info)
            # Merge metadata in sorted key order so concurrent writers acquire the
            # trace_request_metadata PK-index locks in a consistent order and cannot
            # deadlock.
            for k, v in sorted(request_metadata.items()):
                session.merge(SqlTraceMetadata(request_id=request_id, key=k, value=v))
            for k, v in tags.items():
                session.merge(SqlTraceTag(request_id=request_id, key=k, value=v))
            return TraceInfoV2.from_v3(sql_trace_info.to_mlflow_entity())

    def create_issue(
        self,
        experiment_id: str,
        name: str,
        description: str,
        status: IssueStatus = IssueStatus.PENDING,
        severity: IssueSeverity | None = None,
        root_causes: list[str] | None = None,
        source_run_id: str | None = None,
        categories: list[str] | None = None,
        created_by: str | None = None,
        *,
        issue_id: str | None = None,
        last_seen_timestamp: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Issue:
        """
        Create a new issue in the database.

        Args:
            experiment_id: The experiment ID.
            name: Short descriptive name for the issue.
            description: Detailed description of the issue.
            status: Issue status. Defaults to IssueStatus.PENDING.
            severity: Optional severity level indicator.
            root_causes: Optional list of root cause analyses.
            source_run_id: Optional run ID that discovered this issue.
            categories: Optional list of categories for the issue.
            created_by: Optional identifier for who created this issue.
            issue_id: Optional iss-* ID (at most 36 characters); reuse returns a conflict.
            last_seen_timestamp: Latest observation in milliseconds; defaults to creation time.
            metadata: String metadata map (at most 100 entries).

        Returns:
            The created Issue entity.
        """
        validate_issue_fields(
            name=name,
            status=status,
            severity=severity,
            categories=categories,
            metadata=metadata,
            last_seen_timestamp=last_seen_timestamp,
            issue_id=issue_id,
            created_by=created_by,
        )
        status = IssueStatus(status)
        severity = IssueSeverity(severity) if severity is not None else None
        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            # Verify experiment exists
            self._tracking_store._get_experiment(session, experiment_id, ViewType.ACTIVE_ONLY)

            # Generate issue ID
            issue_id = issue_id or f"iss-{uuid.uuid4().hex}"
            if (
                self._tracking_store
                ._get_query(session, SqlIssue)
                .filter(SqlIssue.issue_id == issue_id)
                .first()
            ):
                raise MlflowException("Issue ID already exists", RESOURCE_ALREADY_EXISTS)
            if source_run_id is not None:
                source_run = self._tracking_store._get_run(session, source_run_id)
                if str(source_run.experiment_id) != str(experiment_id):
                    raise MlflowException.invalid_parameter_value(
                        "Source run must belong to the Issue experiment"
                    )

            # Get current timestamp
            current_time = get_current_time_millis()

            # Serialize root_causes and categories to JSON
            root_causes_json = json.dumps(root_causes) if root_causes else None
            categories_json = json.dumps(categories) if categories else None

            # Create SqlIssue record
            sql_issue = SqlIssue(
                issue_id=issue_id,
                experiment_id=_parse_experiment_id(experiment_id),
                name=name,
                description=description,
                status=status.value,
                severity=severity.value if severity else None,
                root_causes=root_causes_json,
                source_run_id=source_run_id,
                categories=categories_json,
                created_timestamp=current_time,
                last_updated_timestamp=current_time,
                created_by=created_by,
                last_seen_timestamp=last_seen_timestamp
                if last_seen_timestamp is not None
                else current_time,
                last_updated_by=created_by,
                issue_metadata=json.dumps(metadata) if metadata else None,
                issue_categories=[
                    SqlIssueCategory(category=category, category_hash=category_hash)
                    for category_hash, category in {
                        issue_category_hash(category): category for category in categories or []
                    }.items()
                ],
            )

            session.add(sql_issue)
            try:
                session.flush()
            except IntegrityError as e:
                if any(
                    term in str(e.orig).lower() for term in ("unique", "duplicate", "primary key")
                ):
                    raise MlflowException("Issue ID already exists", RESOURCE_ALREADY_EXISTS) from e
                raise

            # Return Issue entity
            return sql_issue.to_mlflow_entity()

    def get_issue(self, issue_id: str) -> Issue:
        """
        Get an issue by ID.

        Args:
            issue_id: The issue ID to fetch.

        Returns:
            The Issue entity.
        """
        with self._tracking_store.ManagedSessionMaker() as session:
            sql_issue = (
                self._tracking_store
                ._get_query(session, SqlIssue)
                .filter(SqlIssue.issue_id == issue_id)
                .first()
            )
            if not sql_issue:
                raise MlflowException(
                    f"Issue with ID '{issue_id}' not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

            return sql_issue.to_mlflow_entity()

    @record_usage_event(UpdateIssueEvent)
    def update_issue(
        self,
        issue_id: str,
        status: IssueStatus | None = None,
        name: str | None = None,
        description: str | None = None,
        severity: IssueSeverity | None = None,
        *,
        categories: list[str] | None = None,
        last_seen_timestamp: int | None = None,
        last_updated_by: str | None = None,
        metadata: dict[str, str] | None = None,
        delete_metadata_keys: list[str] | None = None,
    ) -> Issue:
        """
        Update an existing issue.

        Args:
            issue_id: The ID of the issue to update.
            status: Optional new status.
            name: Optional new name for the issue.
            description: Optional new description.
            severity: Optional new severity level.
            categories: Replacement categories; [] clears them and None leaves them unchanged.
            last_seen_timestamp: Latest observation in milliseconds; never moves backwards.
            last_updated_by: Update actor, overridden by the authenticated HTTP user when available.
            metadata: Entries to merge with existing metadata.
            delete_metadata_keys: Metadata keys to remove; cannot overlap metadata updates.

        Returns:
            The updated Issue entity.
        """
        validate_issue_fields(
            name=name,
            status=status,
            severity=severity,
            categories=categories,
            metadata=metadata,
            last_seen_timestamp=last_seen_timestamp,
            delete_metadata_keys=delete_metadata_keys,
            last_updated_by=last_updated_by,
        )
        with self._tracking_store.ManagedSessionMaker(read_only=False) as session:
            if self._tracking_store.db_type == SQLITE:
                # SQLite ignores FOR UPDATE. Start a write transaction before reading maps so
                # concurrent per-key patches cannot overwrite each other's unrelated entries.
                # If the workspace-scoped query misses, the same scoped read below raises before
                # applying any changes.
                scoped_ids = (
                    self._tracking_store
                    ._get_query(session, SqlIssue)
                    .filter(SqlIssue.issue_id == issue_id)
                    .with_entities(SqlIssue.issue_id)
                )
                session.execute(
                    sqlalchemy
                    .update(SqlIssue)
                    .where(SqlIssue.issue_id.in_(scoped_ids))
                    .values(last_updated_timestamp=SqlIssue.last_updated_timestamp)
                    .execution_options(synchronize_session=False)
                )
            # Fetch the existing issue
            sql_issue = (
                self._tracking_store
                ._get_query(session, SqlIssue)
                .filter(SqlIssue.issue_id == issue_id)
                .with_hint(SqlIssue, "WITH (UPDLOCK, ROWLOCK)", dialect_name="mssql")
                .with_for_update(of=SqlIssue)
                .first()
            )
            if not sql_issue:
                raise MlflowException(
                    f"Issue with ID '{issue_id}' not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

            # Update fields if provided
            if status is not None:
                sql_issue.status = IssueStatus(status).value
            if name is not None:
                sql_issue.name = name
            if description is not None:
                sql_issue.description = description
            if severity is not None:
                sql_issue.severity = IssueSeverity(severity).value

            if categories is not None:
                sql_issue.categories = (
                    json.dumps(list(dict.fromkeys(categories))) if categories else None
                )
                # Keep searchable categories and the legacy JSON representation in one transaction.
                existing = {entry.category_hash: entry for entry in sql_issue.issue_categories}
                indexed_categories = []
                for category_hash, category in {
                    issue_category_hash(category): category for category in categories
                }.items():
                    entry = existing.get(category_hash) or SqlIssueCategory(
                        category_hash=category_hash
                    )
                    entry.category = category
                    indexed_categories.append(entry)
                sql_issue.issue_categories = indexed_categories
            if last_seen_timestamp is not None:
                sql_issue.last_seen_timestamp = max(
                    sql_issue.last_seen_timestamp
                    if sql_issue.last_seen_timestamp is not None
                    else sql_issue.created_timestamp,
                    last_seen_timestamp,
                )
            if last_updated_by is not None:
                sql_issue.last_updated_by = last_updated_by
            if metadata is not None or delete_metadata_keys:
                merged = json.loads(sql_issue.issue_metadata) if sql_issue.issue_metadata else {}
                merged.update(metadata or {})
                for key in delete_metadata_keys or []:
                    merged.pop(key, None)
                validate_issue_fields(metadata=merged)
                sql_issue.issue_metadata = json.dumps(merged)
            # Update last_updated_timestamp
            sql_issue.last_updated_timestamp = get_current_time_millis()

            session.flush()

            return sql_issue.to_mlflow_entity()

    def search_issues(
        self,
        experiment_id: str | None = None,
        filter_string: str | None = None,
        max_results: int | None = None,
        page_token: str | None = None,
        include_trace_count: bool = False,
        *,
        name: str | None = None,
        statuses: list[str] | None = None,
        severities: list[str] | None = None,
        categories: list[str] | None = None,
        order_by: list[str] | None = None,
        include_total_count: bool = False,
        view: Literal["basic", "full"] | None = None,
    ) -> PagedList[Issue]:
        """
        Search for issues matching the given filters.

        Args:
            experiment_id: Optional experiment ID to filter by.
            filter_string: Optional filter string for advanced filtering.
                Supported filters: status, source_run_id
                Supported comparators: =, !=
                Examples:
                    - "status = 'resolved'"
                    - "source_run_id = 'run123'"
                    - "status = 'pending' AND source_run_id != 'run456'"
            max_results: Maximum number of results to return.
            page_token: Token for pagination.
            include_trace_count: Whether to include the count of traces impacted by each issue.
            name: Case-insensitive literal substring of the name (not a wildcard pattern).
            statuses: Accepted status values, ORed within the list.
            severities: Accepted severity values, ORed within the list.
            categories: Match any category, ignoring case and surrounding whitespace.
                Different filter fields are ANDed.
            order_by: Field and optional ASC/DESC; supports name, status, severity and timestamps.
                Issue ID breaks ties. Explicit ordering uses a cursor over the last row's sort
                values, so inserts or removals before it do not shift subsequent pages.
                This is a live list, not a snapshot: edits to sort/filter fields can move rows
                across the cursor. Omit to retain the legacy severity/count/creation ordering.
            include_total_count: Return the matching count before pagination in total_count.
            view: "basic" returns an empty description and omits root_causes and metadata.
                The default is "full".

        Returns:
            A PagedList of Issue entities.
        """
        self._tracking_store._validate_max_results_param(max_results, allow_null=True)
        if max_results is None:
            max_results = SEARCH_ISSUES_DEFAULT_MAX_RESULTS

        validate_issue_search(name, statuses, severities, order_by)
        validate_issue_fields(categories=categories)
        for value in statuses or []:
            validate_issue_fields(status=value)
        for value in severities or []:
            validate_issue_fields(severity=value)
        if view not in (None, "basic", "full"):
            raise MlflowException.invalid_parameter_value("view must be basic or full")
        enhanced = (
            any(
                value is not None
                for value in (name, statuses, severities, categories, order_by, view)
            )
            or include_total_count
        )
        fingerprint = issue_query_fingerprint(
            workspace=get_request_workspace(),
            experiment_id=experiment_id,
            filter_string=filter_string,
            name=name,
            statuses=sorted(set(statuses or [])),
            severities=sorted(set(severities or [])),
            categories=sorted({issue_category_hash(category) for category in categories or []}),
            order_by=order_by or [],
            include_trace_count=include_trace_count,
            include_total_count=include_total_count,
            view=view,
        )
        with self._tracking_store.ManagedSessionMaker() as session:
            position = issue_page_position(page_token, fingerprint, allow_legacy=not enhanced)

            query = self._tracking_store._get_query(session, SqlIssue)

            if include_trace_count:
                trace_count_label = func.count(func.distinct(SqlAssessments.trace_id)).label(
                    "trace_count"
                )
                query = (
                    query
                    .add_columns(trace_count_label)
                    .outerjoin(
                        SqlAssessments,
                        and_(
                            SqlAssessments.name == SqlIssue.issue_id,
                            SqlAssessments.assessment_type == "issue",
                        ),
                    )
                    .group_by(SqlIssue.issue_id)
                )

            if experiment_id is not None:
                query = query.filter(SqlIssue.experiment_id == _parse_experiment_id(experiment_id))

            if filter_string:
                parsed_filters = SearchIssuesUtils.parse_search_filter(filter_string)
                filter_clauses = _get_search_issues_filter_clauses(
                    parsed_filters, self._tracking_store._get_dialect()
                )
                query = query.filter(*filter_clauses)

            if name is not None:
                escaped = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                if self._tracking_store.db_type == MSSQL:
                    escaped = escaped.replace("[", "\\[")
                query = query.filter(SqlIssue.name.ilike(f"%{escaped}%", escape="\\"))
            if statuses:
                query = query.filter(SqlIssue.status.in_(statuses))
            if severities:
                query = query.filter(SqlIssue.severity.in_(severities))
            if categories:
                query = query.filter(
                    exists().where(
                        SqlIssueCategory.issue_id == SqlIssue.issue_id,
                        SqlIssueCategory.category_hash.in_([
                            issue_category_hash(category) for category in categories
                        ]),
                    )
                )
            total_count = query.count() if include_total_count else None

            # IssueSeverity enum is ordered from lowest to highest severity
            severity_priorities = {severity.value: severity._rank for severity in IssueSeverity}
            severity_order = case(
                severity_priorities,
                value=SqlIssue.severity,
                else_=-1,
            )

            order_clauses = [severity_order.desc()]
            if include_trace_count:
                order_clauses.append(trace_count_label.desc())
            order_clauses.extend([
                SqlIssue.created_timestamp.desc(),
                SqlIssue.issue_id.desc(),
            ])
            sort_columns = []
            if order_by:
                sortable = {
                    "name": func.lower(SqlIssue.name),
                    "status": case(
                        {"pending": 0, "rejected": 1, "resolved": 2}, value=SqlIssue.status, else_=3
                    ),
                    "severity": severity_order,
                    "created_timestamp": SqlIssue.created_timestamp,
                    "last_updated_timestamp": SqlIssue.last_updated_timestamp,
                    "last_seen_timestamp": func.coalesce(
                        SqlIssue.last_seen_timestamp, SqlIssue.created_timestamp
                    ),
                }
                order_clauses = []
                for order in order_by:
                    parts = order.split()
                    if (
                        not 1 <= len(parts) <= 2
                        or parts[0] not in sortable
                        or (len(parts) == 2 and parts[1].upper() not in ("ASC", "DESC"))
                    ):
                        raise MlflowException.invalid_parameter_value(
                            f"Invalid issue order_by: {order!r}"
                        )
                    column = sortable[parts[0]]
                    descending = len(parts) == 2 and parts[1].upper() == "DESC"
                    sort_columns.append((column, descending))
                sort_columns.append((SqlIssue.issue_id, True))
                order_clauses = [
                    column.desc() if descending else column.asc()
                    for column, descending in sort_columns
                ]
            if position.values is not None:
                value_types = [
                    str if order.split()[0] == "name" else int for order in order_by or []
                ] + [str]
                if (
                    not sort_columns
                    or len(position.values) != len(value_types)
                    or any(
                        type(value) is not expected
                        for value, expected in zip(position.values, value_types)
                    )
                ):
                    raise MlflowException.invalid_parameter_value("Invalid issue page token")
                equal_prefix = []
                after_cursor = []
                for (column, descending), value in zip(sort_columns, position.values):
                    after_cursor.append(
                        and_(*equal_prefix, column < value if descending else column > value)
                    )
                    equal_prefix.append(column == value)
                query = query.filter(or_(*after_cursor))
            if sort_columns:
                # Read the effective keys from SQL (including LOWER/COALESCE/CASE), not from
                # the entity or the anchor row on a later request; the anchor may be deleted.
                query = query.add_columns(*(column for column, _ in sort_columns))
            query = query.order_by(*order_clauses)
            if view == "basic":
                query = query.options(
                    defer(SqlIssue.description),
                    defer(SqlIssue.root_causes),
                    defer(SqlIssue.issue_metadata),
                )

            query = query.offset(position.offset).limit(max_results + 1)

            results = query.all()
            has_next_page = len(results) > max_results
            next_token = None
            if has_next_page:
                if sort_columns:
                    next_token = issue_keyset_token(
                        list(results[max_results - 1][-len(sort_columns) :]), fingerprint
                    )
                elif enhanced:
                    next_token = issue_page_token(position.offset + max_results, fingerprint)
                else:
                    next_token = SearchTraceUtils.create_page_token(position.offset + max_results)
            if sort_columns:
                # Sort keys follow the optional trace count: (SqlIssue, trace_count, *sort_keys)
                # when requested, otherwise (SqlIssue, *sort_keys).
                results = [(row[0], row[1]) if include_trace_count else row[0] for row in results]

            if include_trace_count:
                issues = [
                    sql_issue.to_mlflow_entity(trace_count=trace_count, basic=view == "basic")
                    for sql_issue, trace_count in results[:max_results]
                ]
            else:
                issues = [
                    sql_issue.to_mlflow_entity(basic=view == "basic")
                    for sql_issue in results[:max_results]
                ]

            return PagedList(issues, token=next_token, total_count=total_count)

    def sum_gateway_trace_cost(
        self,
        start_time_ms: int,
        end_time_ms: int,
        workspace: str | None = None,
        endpoint_id: str | None = None,
        username: str | None = None,
    ) -> float:
        with self._tracking_store.ManagedSessionMaker() as session:
            query = (
                session
                .query(func.coalesce(func.sum(SqlTraceInfo.total_cost), 0.0))
                .join(
                    SqlTraceMetadata,
                    SqlTraceMetadata.request_id == SqlTraceInfo.request_id,
                )
                .filter(
                    SqlTraceMetadata.key == TraceMetadataKey.GATEWAY_ENDPOINT_ID,
                    SqlTraceInfo.timestamp_ms >= start_time_ms,
                    SqlTraceInfo.timestamp_ms < end_time_ms,
                )
            )

            if endpoint_id is not None:
                query = query.filter(SqlTraceMetadata.value == endpoint_id)

            if workspace is not None:
                query = query.join(
                    SqlExperiment,
                    SqlExperiment.experiment_id == SqlTraceInfo.experiment_id,
                ).filter(SqlExperiment.workspace == workspace)

            if username is not None:
                # Filter to traces whose recorded auth username matches the username.
                # Uses a separate aliased join because SqlTraceMetadata is already
                # joined above for the gateway-endpoint marker.
                user_metadata = aliased(SqlTraceMetadata)
                query = query.join(
                    user_metadata,
                    user_metadata.request_id == SqlTraceInfo.request_id,
                ).filter(
                    user_metadata.key == TraceMetadataKey.AUTH_USERNAME,
                    user_metadata.value == username,
                )

            return float(query.scalar())


_TRACE_INFO_COLUMNS = tuple(SqlTraceInfo.__table__.columns)


_TRACE_CHILD_QUERY_BATCH_SIZE = 500


def _build_trace_infos_from_rows(session, trace_rows):
    if not trace_rows:
        return []

    trace_ids = [row.request_id for row in trace_rows]
    tags_by_trace = defaultdict(dict)
    metadata_by_trace = defaultdict(dict)
    assessments_by_trace = defaultdict(list)
    for trace_id_batch in chunk_list(trace_ids, _TRACE_CHILD_QUERY_BATCH_SIZE):
        for trace_id, key, value in session.execute(
            select(SqlTraceTag.request_id, SqlTraceTag.key, SqlTraceTag.value).where(
                SqlTraceTag.request_id.in_(trace_id_batch)
            )
        ):
            if key != TraceTagKey.TRACE_NAME:
                tags_by_trace[trace_id][key] = value

        for trace_id, key, value in session.execute(
            select(SqlTraceMetadata.request_id, SqlTraceMetadata.key, SqlTraceMetadata.value).where(
                SqlTraceMetadata.request_id.in_(trace_id_batch)
            )
        ):
            if key not in PROMOTED_TRACE_METADATA_KEYS:
                metadata_by_trace[trace_id][key] = value

        assessments = (
            session.query(SqlAssessments).filter(SqlAssessments.trace_id.in_(trace_id_batch)).all()
        )
        for assessment in assessments:
            assessments_by_trace[assessment.trace_id].append(assessment.to_mlflow_entity())

    trace_infos = []
    for row in trace_rows:
        tags = tags_by_trace[row.request_id]
        if row.trace_name is not None:
            tags[TraceTagKey.TRACE_NAME] = row.trace_name
        trace_metadata = metadata_by_trace[row.request_id]
        trace_metadata.update(compatibility_metadata_from_columns(row))
        trace_infos.append(
            TraceInfo(
                trace_id=row.request_id,
                trace_location=TraceLocation.from_experiment_id(str(row.experiment_id)),
                request_time=row.timestamp_ms,
                execution_duration=row.execution_time_ms,
                state=TraceState(row.status),
                tags=tags,
                trace_metadata=trace_metadata,
                client_request_id=row.client_request_id,
                request_preview=row.request_preview,
                response_preview=row.response_preview,
                assessments=assessments_by_trace[row.request_id],
            )
        )
    return trace_infos


@lru_cache(maxsize=32)
def _parse_trace_order_by(order_by_clause):
    return SearchTraceUtils.parse_order_by_for_search_traces(order_by_clause)


@dataclass(frozen=True, slots=True)
class _ParsedTraceFilter:
    key_type: str
    key: str
    value: Any
    comparator: str


@lru_cache(maxsize=64)
def _parse_trace_filter(filter_string):
    return tuple(
        _ParsedTraceFilter(
            key_type=clause.get("type"),
            key=clause.get("key"),
            value=clause.get("value"),
            comparator=clause.get("comparator"),
        )
        for clause in SearchTraceUtils.parse_search_filter_for_search_traces(filter_string)
    )


def _get_orderby_clauses_for_search_traces(order_by_list: list[str], session):
    """Sorts a set of traces based on their natural ordering and an overriding set of order_bys.
    Traces are ordered first by timestamp_ms descending, then by trace_id for tie-breaking.
    """
    clauses = []
    ordering_joins = []
    observed_order_by_clauses = set()
    select_clauses = []

    for clause_id, order_by_clause in enumerate(order_by_list):
        key_type, key, ascending = _parse_trace_order_by(order_by_clause)

        if (
            SearchTraceUtils.is_request_metadata(key_type, "=")
            and key in TRACE_ANALYTICS_COLUMNS_BY_METADATA_KEY
        ):
            raise MlflowException.invalid_parameter_value(
                f"Ordering by reserved metadata '{key}' is not supported because it is "
                "represented by multiple columns."
            )
        needs_null_ordering = True
        if SearchTraceUtils.is_attribute(key_type, key, "=") and key == "end_time_ms":
            order_value = SqlTraceInfo.timestamp_ms + func.coalesce(
                SqlTraceInfo.execution_time_ms, 0
            )
            needs_null_ordering = False
        elif SearchTraceUtils.is_attribute(key_type, key, "="):
            order_value = getattr(SqlTraceInfo, key)
            needs_null_ordering = order_value.nullable
        elif SearchTraceUtils.is_tag(key_type, "=") and key == TraceTagKey.TRACE_NAME:
            order_value = SqlTraceInfo.trace_name
        elif (
            SearchTraceUtils.is_request_metadata(key_type, "=")
            and key == TraceMetadataKey.TRACE_SESSION
        ):
            order_value = SqlTraceInfo.session_id
        else:
            if SearchTraceUtils.is_tag(key_type, "="):
                entity = SqlTraceTag
            elif SearchTraceUtils.is_request_metadata(key_type, "="):
                entity = SqlTraceMetadata
            else:
                raise MlflowException(
                    f"Invalid identifier type '{key_type}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            # Tags and request metadata requires a join to the main table (trace_info)
            subquery = session.query(entity).filter(entity.key == key).subquery()
            ordering_joins.append(subquery)
            order_value = subquery.c.value

        if needs_null_ordering:
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


def _get_session_scoped_trace_ids(scoped_trace_query: Query, assessment_filters):
    """Find all trace IDs covered by session-scoped assessments matching the given filters.

    The supplied trace query must be limited to the active workspace and requested experiments.

    Two-step approach:
    1. Find session IDs that have a matching session-scoped assessment.
    2. Find all trace IDs belonging to those sessions.
    """
    session_ids = (
        scoped_trace_query
        .with_entities(SqlTraceInfo.session_id)
        .join(SqlAssessments, SqlAssessments.trace_id == SqlTraceInfo.request_id)
        .filter(
            SqlTraceInfo.session_id.isnot(None),
            *assessment_filters,
            SqlAssessments.assessment_metadata.isnot(None),
            SqlAssessments.assessment_metadata.contains(f'"{TraceMetadataKey.TRACE_SESSION}":'),
        )
    )
    return scoped_trace_query.with_entities(SqlTraceInfo.request_id).filter(
        SqlTraceInfo.session_id.in_(session_ids),
    )


def _get_filter_clauses_for_search_traces(filter_string, session, dialect, scoped_trace_query):
    """
    Creates trace attribute filters and subqueries that will be inner-joined
    to SqlTraceInfo to act as multi-clause filters and return them as a tuple.
    Also extracts run_id filter if present for special handling.

    Returns:
        attribute_filters: Direct filters on SqlTraceInfo attributes
        non_attribute_filters: Subqueries for tags and metadata
        span_attribute_filters: Direct filters on SqlSpan
        span_filters: Subqueries for non-SqlSpan filters
        run_id_filter: Special run_id value for linked trace handling
    """
    attribute_filters = []
    non_attribute_filters = []
    span_attribute_filters = []
    span_filters = []
    run_id_filter = None
    span_start_time_lower_bound_ms = None

    for clause in _parse_trace_filter(filter_string):
        key_type = clause.key_type
        key_name = clause.key
        value = clause.value
        comparator = clause.comparator.upper()

        # Check if this is an issue filter (stored in assessments table)
        # Note: Issue filters use the format 'issue.id = "issue-123"', which differs
        # from assessment filters that use 'feedback/expectation.<key_name> <operator> <value>'.
        # Issue filters match on issue ID, which is the assessment name instead of value.
        if SearchTraceUtils.is_issue(key_type, key_name, comparator):
            # Query assessments table for issue references
            # IssueReference assessments have assessment_type='issue' and name=issue_id
            issue_subquery = (
                session
                .query(SqlAssessments.trace_id.label("request_id"))
                .filter(
                    SqlAssessments.assessment_type == "issue",
                    SqlAssessments.name == value,
                )
                .distinct()
                .subquery()
            )
            span_filters.append(issue_subquery)
            continue

        if SearchTraceUtils.is_attribute(key_type, key_name, comparator):
            if key_name == "timestamp_ms" and comparator in (">", ">="):
                span_start_time_lower_bound_ms = (
                    value
                    if span_start_time_lower_bound_ms is None
                    else max(span_start_time_lower_bound_ms, value)
                )
            if key_name in ("end_time_ms", "end_time"):
                # end_time = timestamp_ms + execution_time_ms
                attribute = SqlTraceInfo.timestamp_ms + func.coalesce(
                    SqlTraceInfo.execution_time_ms, 0
                )
            else:
                attribute = getattr(SqlTraceInfo, key_name)
            attr_filter = SearchTraceUtils.get_sql_comparison_func(comparator, dialect)(
                attribute, value
            )
            attribute_filters.append(attr_filter)
        else:
            if SearchTraceUtils.is_tag(key_type, comparator) and key_name == TraceTagKey.TRACE_NAME:
                if comparator == "IS NULL":
                    attribute_filters.append(SqlTraceInfo.trace_name.is_(None))
                elif comparator == "IS NOT NULL":
                    attribute_filters.append(SqlTraceInfo.trace_name.isnot(None))
                else:
                    attribute_filters.append(
                        SearchTraceUtils.get_sql_comparison_func(comparator, dialect)(
                            SqlTraceInfo.trace_name, value
                        )
                    )
                continue
            if (
                SearchTraceUtils.is_request_metadata(key_type, comparator)
                and key_name == TraceMetadataKey.TRACE_SESSION
            ):
                if comparator == "IS NULL":
                    attribute_filters.append(SqlTraceInfo.session_id.is_(None))
                elif comparator == "IS NOT NULL":
                    attribute_filters.append(SqlTraceInfo.session_id.isnot(None))
                else:
                    attribute_filters.append(
                        SearchTraceUtils.get_sql_comparison_func(comparator, dialect)(
                            SqlTraceInfo.session_id, value
                        )
                    )
                continue
            if (
                SearchTraceUtils.is_request_metadata(key_type, comparator)
                and key_name in TRACE_ANALYTICS_COLUMNS_BY_METADATA_KEY
            ):
                attribute_filters.append(
                    get_trace_analytics_metadata_filter(key_name, comparator, value, SqlTraceInfo)
                )
                continue
            # Check if this is a run_id filter (stored as SOURCE_RUN in metadata)
            if (
                SearchTraceUtils.is_request_metadata(key_type, comparator)
                and key_name == TraceMetadataKey.SOURCE_RUN
                and comparator == "="
            ):
                run_id_filter = value
                # Don't add run_id filter to non_attribute_filters since we handle it specially
                continue

            if SearchTraceUtils.is_tag(key_type, comparator):
                # Special handling for prompts filter: only support exact match with name/version
                # Uses EntityAssociation table for linkage
                if key_name == TraceTagKey.LINKED_PROMPTS:
                    # Only support = comparator for prompts filter
                    if comparator != "=":
                        raise MlflowException(
                            f"Invalid comparator '{comparator}' for prompts filter. "
                            "Only '=' is supported with format: prompt = \"name/version\"",
                            error_code=INVALID_PARAMETER_VALUE,
                        )
                    # Parse the filter value to extract name/version
                    # Expected format: "name/version"
                    if value.count("/") != 1:
                        raise MlflowException(
                            f"Invalid prompts filter value '{value}'. "
                            'Expected format: prompt = "name/version"',
                            error_code=INVALID_PARAMETER_VALUE,
                        )
                    # Query EntityAssociation table for trace<>prompt linkage
                    # The prompt version ID is stored as "name/version"
                    prompt_id = value
                    association_subquery = (
                        session
                        .query(SqlEntityAssociation.source_id.label("request_id"))
                        .filter(
                            SqlEntityAssociation.source_type == EntityAssociationType.TRACE,
                            SqlEntityAssociation.destination_type
                            == EntityAssociationType.PROMPT_VERSION,
                            SqlEntityAssociation.destination_id == prompt_id,
                        )
                        .distinct()
                        .subquery()
                    )
                    span_filters.append(association_subquery)
                    continue
                if comparator in ("IS NULL", "IS NOT NULL"):
                    tag_exists_subquery = session.query(SqlTraceTag.request_id).filter(
                        SqlTraceTag.request_id == SqlTraceInfo.request_id,
                        SqlTraceTag.key == key_name,
                    )
                    if comparator == "IS NULL":
                        attribute_filters.append(~tag_exists_subquery.exists())
                    else:
                        attribute_filters.append(tag_exists_subquery.exists())
                    continue
                entity = SqlTraceTag
            elif SearchTraceUtils.is_request_metadata(key_type, comparator):
                # Handle IS NULL / IS NOT NULL for metadata
                if comparator in ("IS NULL", "IS NOT NULL"):
                    metadata_exists_subquery = session.query(SqlTraceMetadata.request_id).filter(
                        SqlTraceMetadata.request_id == SqlTraceInfo.request_id,
                        SqlTraceMetadata.key == key_name,
                    )
                    if comparator == "IS NULL":
                        attribute_filters.append(~metadata_exists_subquery.exists())
                    else:
                        attribute_filters.append(metadata_exists_subquery.exists())
                    continue
                entity = SqlTraceMetadata
            elif SearchTraceUtils.is_span(key_type, key_name, comparator):
                # Spans have specialized columns (name, type, status) unlike tags/metadata
                # which have key-value structure, so we need specialized handling

                # Handle span.attributes.<attribute> format
                if key_name.startswith("attributes."):
                    attr_name = SearchTraceUtils._trim_backticks(key_name[len("attributes.") :])
                    # Search within the content JSON for the specific attribute
                    # TODO: we should improve this by saving only the attributes into the table.
                    if comparator == "RLIKE":
                        # For RLIKE, transform the user pattern to match within JSON structure
                        # The JSON structure is: "<attr>": "\"<value>\""
                        # Values are JSON-encoded strings with escaped quotes
                        transformed_value = value
                        if value.startswith("^"):
                            transformed_value = transformed_value[1:]
                            search_prefix = '"\\\\"'
                        else:
                            search_prefix = '"\\\\".*'
                        if value.endswith("$"):
                            transformed_value = transformed_value[:-1]
                            search_suffix = '\\\\"'
                        else:
                            search_suffix = ""
                        search_pattern = (
                            f'"{attr_name}": {search_prefix}{transformed_value}{search_suffix}'
                        )
                        val_filter = SearchTraceUtils.get_sql_comparison_func(comparator, dialect)(
                            SqlSpan.content, search_pattern
                        )
                    else:
                        # For LIKE/ILIKE, use wildcards for broad matching
                        val_filter = SearchTraceUtils.get_sql_comparison_func(comparator, dialect)(
                            SqlSpan.content, f'%"{attr_name}"{value}%'
                        )
                else:
                    span_column = getattr(SqlSpan, key_name)
                    val_filter = SearchTraceUtils.get_sql_comparison_func(comparator, dialect)(
                        span_column, value
                    )
                span_attribute_filters.append(val_filter)
                continue
            elif SearchTraceUtils.is_assessment(key_type, key_name, comparator):
                # Create subquery to find traces with matching assessments
                # Filter by assessment name and check the value
                assessment_filters = [
                    SqlAssessments.assessment_type == key_type,
                    SqlAssessments.name == key_name,
                    SqlAssessments.valid == sqlalchemy.true(),
                ]
                if comparator in ("IS NULL", "IS NOT NULL"):
                    assessment_exists_subquery = session.query(SqlAssessments.trace_id).filter(
                        SqlAssessments.trace_id == SqlTraceInfo.request_id,
                        *assessment_filters,
                    )
                    session_covered = _get_session_scoped_trace_ids(
                        scoped_trace_query, assessment_filters
                    )

                    if comparator == "IS NULL":
                        exists_clause = assessment_exists_subquery.exists()
                        attribute_filters.append(
                            sqlalchemy.and_(
                                ~exists_clause,
                                SqlTraceInfo.request_id.notin_(session_covered),
                            )
                        )
                    else:
                        direct_matches = (
                            session
                            .query(SqlAssessments.trace_id.label("request_id"))
                            .filter(*assessment_filters)
                            .distinct()
                        )
                        combined = direct_matches.union(session_covered).subquery()
                        span_filters.append(combined)
                    continue

                # Other comparators: filter by value
                if comparator in ("<", "<=", ">", ">="):
                    value_filter = sqlalchemy.and_(
                        SqlAssessments.is_numeric_value == sqlalchemy.true(),
                        SearchTraceUtils.get_sql_comparison_func(comparator, dialect)(
                            SqlAssessments.aggregate_value, value
                        ),
                    )
                else:
                    value_filter = SearchTraceUtils._get_sql_json_comparison_func(
                        comparator, dialect
                    )(SqlAssessments.value, value)
                assessment_filters_with_value = [*assessment_filters, value_filter]
                direct_matches = (
                    session
                    .query(SqlAssessments.trace_id.label("request_id"))
                    .filter(*assessment_filters_with_value)
                    .distinct()
                )
                session_siblings = _get_session_scoped_trace_ids(
                    scoped_trace_query, assessment_filters_with_value
                )
                feedback_subquery = direct_matches.union(session_siblings).subquery()
                span_filters.append(feedback_subquery)
                continue
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

    if span_attribute_filters:
        span_scope_filters = [SqlSpan.experiment_id == SqlTraceInfo.experiment_id]
        if span_start_time_lower_bound_ms is not None:
            # Allow minor clock skew between trace and span timestamps while still pruning spans
            # that are well outside the trace search range.
            span_start_time_lower_bound_ns = (
                span_start_time_lower_bound_ms - _SPAN_SEARCH_TIMESTAMP_TOLERANCE_MS
            ) * 1_000_000
            span_scope_filters.append(
                SqlSpan.start_time_unix_nano >= span_start_time_lower_bound_ns
            )
        span_attribute_filters[:] = [*span_scope_filters, *span_attribute_filters]

    return (
        attribute_filters,
        non_attribute_filters,
        span_attribute_filters,
        span_filters,
        run_id_filter,
    )


def _get_search_issues_filter_clauses(parsed_filters, dialect):
    """
    Creates filter clauses for searching issues.

    Args:
        parsed_filters: List of parsed filter dictionaries from SearchIssuesUtils
        dialect: Database dialect for SQL comparison functions

    Returns:
        List of SQLAlchemy filter clauses
    """
    filter_clauses = []

    for f in parsed_filters:
        key = f["key"]
        comparator = f["comparator"]
        value = f["value"]

        # Get the appropriate SqlIssue attribute
        attr = getattr(SqlIssue, key)

        # Use SearchUtils to get the comparison function and apply it
        filter_clause = SearchUtils.get_sql_comparison_func(comparator, dialect)(attr, value)
        filter_clauses.append(filter_clause)

    return filter_clauses


def _try_parse_json_string(value: str) -> str:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
    return parsed if isinstance(parsed, str) else value


def _span_aggregation_node(
    span_id: str, parent_span_id: str | None, attributes: dict[str, Any], attribute_key: str
) -> SpanAggregationNode:
    """
    Build the aggregation node for a span, whether it comes from the incoming batch or a
    stored span row. Span attributes may hold the usage/cost payload either as a dict or as
    a JSON string, depending on how the span reached the store.
    """
    value = attributes.get(attribute_key)
    if isinstance(value, str):
        value = try_json_loads(value)
    return SpanAggregationNode(
        span_id=span_id,
        parent_id=parent_span_id,
        data=value if isinstance(value, dict) else None,
    )


@dataclass
class _TraceAggregate:
    """Pre-computed per-trace aggregates used by log_spans() to minimize DB round-trips."""

    min_start_ms: int
    max_end_ms: int | None
    root_span_status: str | None
    trace_status: str
    aggregated_token_usage: dict[str, Any] = field(default_factory=dict)
    aggregated_cost: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    user_id: str | None = None
    root_span_dict: dict[str, Any] | None = None
    # User-defined tags from mlflow.update_current_trace(tags=...) carried via OTLP export
    trace_tags: dict[str, str] = field(default_factory=dict)
    # Per-span usage/cost nodes of this batch, kept so pre-existing traces can recompute
    # the tree-aware aggregate across batches (batch nodes + spans already in the DB).
    usage_nodes: list[SpanAggregationNode] = field(default_factory=list)
    cost_nodes: list[SpanAggregationNode] = field(default_factory=list)


_LOG_SPANS_MAX_TRACE_CREATE_RETRIES = 10
