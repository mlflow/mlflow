import asyncio
import logging
from abc import ABC
from typing import TYPE_CHECKING, Literal

from mlflow.entities import Assessment, Issue, IssueSeverity, IssueStatus
from mlflow.entities.trace import Span, Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_metrics import MetricAggregation, MetricDataPoint, MetricViewType
from mlflow.entities.workspace import WorkspaceDeletionMode
from mlflow.exceptions import MlflowException, MlflowNotImplementedException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import MAX_RESULTS_QUERY_TRACE_METRICS, SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.tracing.analysis import TraceFilterCorrelationResult

if TYPE_CHECKING:
    from mlflow.genai.scorers.online.entities import CompletedSession
    from mlflow.store.tracking.abstract_store import AbstractStore

_logger = logging.getLogger(__name__)


class AbstractTraceBackend(ABC):
    """Trace persistence, retrieval, and analytics independently of the tracking metadata store.

    Backends own traces, assessments, issues, and their search and analytics. The hosting
    tracking store owns other metadata, associations, and cross-store lifecycle coordination.
    """

    def set_tracking_store(self, tracking_store: "AbstractStore") -> None:
        """Bind optional host context after the tracking store finishes initialization.

        Overrides should call ``super().set_tracking_store(tracking_store)``. Hybrid backends can
        use ``with tracking_store.disable_trace_backend():`` for native host trace operations.
        A backend instance must not be shared between tracking store instances.
        """
        if hasattr(self, "_tracking_store") and self._tracking_store is not tracking_store:
            raise MlflowException.invalid_parameter_value(
                "A trace backend instance cannot be shared between tracking stores."
            )
        self._tracking_store = tracking_store

    def on_traces_deleted(self, trace_ids: list[str]) -> None:
        """Best-effort cleanup of host references after successful backend trace deletion.

        Call this with actual deleted IDs, not request IDs that may not have existed. Overrides
        can call ``super().on_traces_deleted(trace_ids)`` for host association/queue cleanup.
        The backend must remove its own spans, assessments, indexes, and aggregates.
        """
        if not trace_ids:
            return
        try:
            self._tracking_store.on_traces_deleted(trace_ids)
        except Exception:
            _logger.exception("Failed to clean up tracking-store references for deleted traces.")

    def on_run_deleted(self, run_id: str) -> None:
        """Remove assessments produced by a successfully soft-deleted run; retain traces."""
        raise MlflowNotImplementedException()

    def on_run_permanently_deleted(self, run_id: str) -> None:
        """Clean assessments and detach issue source-run references after permanent deletion.

        Preserve issues whose source run was deleted, matching SQL's ON DELETE SET NULL.
        The run no longer exists, so cleanup must not require it.
        """
        raise MlflowNotImplementedException()

    def on_experiment_deleted(self, experiment_id: str) -> None:
        """Remove assessments produced by experiment runs, preserving restorable trace data."""
        raise MlflowNotImplementedException()

    def on_experiment_permanently_deleted(self, experiment_id: str) -> None:
        """Remove all backend data for an experiment, including issue-only data and aggregates."""
        raise MlflowNotImplementedException()

    def before_workspace_deleted(self, workspace_name: str, mode: WorkspaceDeletionMode) -> None:
        """For RESTRICT reject any backend data in the named workspace before mutation."""
        raise MlflowNotImplementedException()

    def on_workspace_deleted(self, workspace_name: str) -> None:
        """Remove all backend data after cascading workspace deletion; parent rows are gone."""
        raise MlflowNotImplementedException()

    @property
    def supports_trace_archival(self) -> bool:
        """Return whether server-owned trace archival is supported by this tracking store."""
        return False

    def sum_gateway_trace_cost(
        self,
        start_time_ms: int,
        end_time_ms: int,
        workspace: str | None = None,
        endpoint_id: str | None = None,
        username: str | None = None,
    ) -> float:
        """Sum gateway trace costs for budget reconstruction over ``[start_time_ms, end_time_ms)``.

        Include all backend-owned trace storage tiers. Optional workspace, endpoint, and
        authenticated username filters are combined with AND; ``None`` leaves that dimension
        unfiltered. A workspace of ``None`` includes all workspaces for global policies.
        Implement this operation to support historical gateway budget reconstruction.
        """
        raise MlflowNotImplementedException()

    def start_trace(self, trace_info: TraceInfo) -> TraceInfo:
        """
        Create a trace using the V3 API format with a complete Trace object.

        Args:
            trace_info: The TraceInfo object to create in the backend.

        Returns:
            The returned TraceInfo object from the backend.
        """
        raise MlflowNotImplementedException()

    def delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: int | None = None,
        max_traces: int | None = None,
        trace_ids: list[str] | None = None,
    ) -> int:
        """
        Delete traces based on the specified criteria.

        - Either `max_timestamp_millis` or `trace_ids` must be specified, but not both.
        - `max_traces` can't be specified if `trace_ids` is specified.

        Implementations should call
        :func:`mlflow.utils.validation.validate_delete_traces_parameters` before deleting traces
        to validate these constraints.
        After deleting related backend-owned data, call ``on_traces_deleted`` with the actual
        deleted IDs to clean host references. The public return value remains a count.

        Args:
            experiment_id: ID of the associated experiment.
            max_timestamp_millis: The maximum timestamp in milliseconds since the UNIX epoch for
                deleting traces. Traces older than or equal to this timestamp will be deleted.
            max_traces: The maximum number of traces to delete. If max_traces is specified, and
                it is less than the number of traces that would be deleted based on the
                max_timestamp_millis, the oldest traces will be deleted first.
            trace_ids: A set of trace IDs to delete.

        Returns:
            The number of traces deleted.
        """
        raise MlflowNotImplementedException()

    def get_trace_info(self, trace_id: str) -> TraceInfo:
        """
        Get the trace matching the `trace_id`.

        Args:
            trace_id: String id of the trace to fetch.

        Returns:
            The fetched Trace object, of type ``mlflow.entities.TraceInfo``.
        """
        raise MlflowNotImplementedException()

    def get_trace(self, trace_id: str, *, allow_partial: bool = False) -> Trace:
        """
        Get a trace with spans for given trace id.

        Args:
            trace_id: String id of the trace to fetch.
            allow_partial: Whether to allow partial traces. If True, the trace will be returned
                even if it is not fully exported yet. If False, MLflow retries and returns
                the trace until all spans are exported or retries are exhausted. Default
                to False.

        Returns:
            The fetched Trace object, of type ``mlflow.entities.Trace``.
        """
        raise MlflowNotImplementedException()

    def batch_get_traces(
        self,
        trace_ids: list[str],
        location: str | None = None,
        experiment_ids: list[str] | None = None,
    ) -> list[Trace]:
        """
        Get a batch of complete traces with spans for given trace ids.

        Args:
            trace_ids: List of trace IDs to fetch.
            location: Location of the trace. For example, "catalog.schema" for UC schema.
            experiment_ids: Optional list of experiment IDs to scope the query. When
                provided, only traces belonging to these experiments are returned.
                ``SqlAlchemyStore`` enforces this directly against the database.
                ``RestStore`` forwards it to the remote backend when explicitly set,
                so it takes effect there if the remote server enforces it. Not
                supported against a Databricks-hosted backend, since that API has no
                corresponding field.

        Returns:
            List of Trace objects.
        """
        raise MlflowNotImplementedException()

    def batch_get_trace_infos(
        self,
        trace_ids: list[str],
        location: str | None = None,
        experiment_ids: list[str] | None = None,
    ) -> list[TraceInfo]:
        """
        Get trace metadata (TraceInfo) for given trace IDs without loading spans.

        This is more efficient than batch_get_traces when only metadata is needed,
        as it avoids loading potentially large span data.
        The default implementation calls ``get_trace_info`` for each unique ID and omits
        missing traces. Backends can override it with an optimized batch query or location support.

        Args:
            trace_ids: List of trace IDs to fetch.
            location: Location of the trace. For example, "catalog.schema" for UC schema.
            experiment_ids: Optional list of experiment IDs to scope the query. When
                provided, only traces belonging to these experiments are returned.
                ``SqlAlchemyStore`` enforces this directly against the database.
                ``RestStore`` forwards it to the remote backend when explicitly set,
                so it takes effect there if the remote server enforces it. Not
                supported against a Databricks-hosted backend, since that API has no
                corresponding field.

        Returns:
            List of TraceInfo objects containing only metadata (no spans).
        """
        if location is not None:
            raise MlflowNotImplementedException(
                f"{self.__class__.__name__}'s default batch_get_trace_infos does not support "
                "location-scoped retrieval. Override the method to support locations."
            )
        allowed_experiments = set(experiment_ids) if experiment_ids is not None else None
        if allowed_experiments is not None and not allowed_experiments:
            return []
        infos = []
        for trace_id in dict.fromkeys(trace_ids):
            try:
                info = self.get_trace_info(trace_id)
            except MlflowException as e:
                if e.error_code != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                    raise
            else:
                if allowed_experiments is None or info.experiment_id in allowed_experiments:
                    infos.append(info)
        return infos

    def archive_traces(
        self,
        *,
        resolved_trace_archival_location: str,
        broader_retention: str,
        long_retention_allowlist: set[str] | list[str] | None = None,
        max_traces_per_pass: int | None = None,
    ) -> int:
        """
        Archive eligible DB-backed trace payloads into the archival repository.

        Concurrent executions of this method against the same backing store / trace population
        are not supported.

        Args:
            resolved_trace_archival_location: Final archival repository root for this archival
                pass after broader-scope server and workspace resolution has already been applied.
            broader_retention: Resolved broader-scope retention in the form ``<int><unit>`` where
                unit is one of ``m``, ``h``, or ``d``. Experiment-level retention overrides are
                resolved against this value during the pass.
            long_retention_allowlist: Experiment IDs allowed to exceed the broader-scope
                retention with a longer experiment-level retention override.
            max_traces_per_pass: Maximum number of traces to archive in this invocation. When the
                scheduler applies a pass-level budget across multiple scopes, it passes the
                remaining budget for the current scope here. If unset, archive all eligible
                traces.

        Returns:
            The number of traces archived during the pass.
        """
        raise MlflowNotImplementedException()

    def get_online_trace_details(
        self,
        trace_id: str,
        source_inference_table: str,
        source_databricks_request_id: str,
    ) -> str:
        raise MlflowNotImplementedException(
            f"{self.__class__.__name__} does not support `get_online_trace_details`."
        )

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
            model_id: If specified, return traces associated with the model ID.
            locations: A list of locations to search over.

        Returns:
            A tuple of a list of :py:class:`TraceInfo <mlflow.entities.TraceInfo>` objects that
            satisfy the search expressions and a pagination token for the next page of results.
            If the underlying tracking store supports pagination, the token for the
            next page may be obtained via the ``token`` attribute of the returned object; however,
            some store implementations may not support pagination and thus the returned token would
            not be meaningful in such cases.
        """
        raise MlflowNotImplementedException()

    def find_completed_sessions(
        self,
        experiment_id: str,
        min_last_trace_timestamp_ms: int,
        max_last_trace_timestamp_ms: int,
        max_results: int | None = None,
        filter_string: str | None = None,
    ) -> list["CompletedSession"]:
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
            filter_string: Optional filter string to apply to the first trace in each session.
                Uses the same syntax as search_traces. If provided, only sessions whose
                first trace matches this filter will be included.

        Returns:
            List of CompletedSession objects sorted by (last_trace_timestamp_ms ASC,
            session_id ASC).
        """
        raise MlflowNotImplementedException(self.__class__.__name__)

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
        """
        Query trace metrics for the given experiment ids.

        Args:
            experiment_ids: List of experiment ids to query metrics for.
            view_type: The view type to query metrics for.
            metric_name: The metric name to query metrics for.
            aggregations: The aggregations to apply to the metrics.
            dimensions: The dimensions to group metrics by.
            filters: The filters to apply to the traces.
            time_interval_seconds: The time interval in seconds to group traces metrics by.
            start_time_ms: The start time to query traces metrics for.
            end_time_ms: The end time to query traces metrics for.
            max_results: The maximum number of traces metrics to return. Default is 1000.
            page_token: The page token to use for pagination.
        """
        raise MlflowNotImplementedException()

    def set_trace_tag(self, trace_id: str, key: str, value: str):
        """
        Set a tag on the trace with the given trace_id.

        Args:
            trace_id: The ID of the trace.
            key: The string key of the tag.
            value: The string value of the tag.
        """
        raise MlflowNotImplementedException()

    def delete_trace_tag(self, trace_id: str, key: str):
        """
        Delete a tag on the trace with the given trace_id.

        Args:
            trace_id: The ID of the trace.
            key: The string key of the tag.
        """
        raise MlflowNotImplementedException()

    def get_assessment(self, trace_id: str, assessment_id: str) -> Assessment:
        """
        Retrieve an assessment from a given trace.

        Args:
            trace_id: The ID of the trace.
            assessment_id: The assessment identifier that denotes a unique assessment entry
                for a given trace.

        Returns:
            The Assessment object for the given trace and assessment ids.
        """
        raise MlflowNotImplementedException()

    def create_assessment(self, assessment: Assessment) -> Assessment:
        """
        Logs an Assessment for a given trace or a span within a trace.

        Args:
            assessment: An :py:class:`Assessment <mlflow.entities.Assessment>` object that
                contains the key value mappings of assessment criteria comprised of either
                expectations or user/system/scorer-provided feedback (label data) on the quality
                of the trace response or for a span within a trace.

        Returns:
            The Assessment object for the logging operation.
        """
        raise MlflowNotImplementedException()

    def update_assessment(
        self,
        trace_id: str,
        assessment_id: str,
        name: str | None = None,
        expectation: str | None = None,
        feedback: str | None = None,
        rationale: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Assessment:
        """
        Updates the given Assessment's mutable values to overwrite updated values
        for the given trace and Assessment data.

        Args:
            trace_id: The ID of the trace.
            assessment_id: The ID of the assessment upon which overrides will be applied to
                mutable attributes.
            name: An Optional override to the name of the assessment.
            expectation: An Optional override of the expectation for the assessment.
            feedback: An Optional override to the feedback for a given assessment.
            rationale: An Optional string defining the reasoning behind the override of
                the assessment.
            metadata: An Optional mapping of additional customizable metadata for the assessment.

        Returns:
            The Assessment object representing the updated state of an assessment for a given trace.
        """
        raise MlflowNotImplementedException()

    def delete_assessment(self, trace_id: str, assessment_id: str) -> None:
        """
        Delete an assessment for a given trace.

        Args:
            trace_id: The ID of the trace.
            assessment_id: The ID of the assessment to be deleted.
        """
        raise MlflowNotImplementedException()

    def log_spans(self, location: str, spans: list[Span], tracking_uri=None) -> list[Span]:
        """
        Log multiple span entities to the tracking store.

        Args:
            location: The location to log spans to. Can be either experiment ID or the
                full UC table name.
            spans: List of Span entities to log. Spans may belong to different traces;
                the store will group them by trace_id internally.
            tracking_uri: The tracking URI to use. Default to None.

        Returns:
            List of logged Span entities.
        """
        raise MlflowNotImplementedException()

    async def log_spans_async(self, location: str, spans: list[Span]) -> list[Span]:
        """
        Asynchronously log multiple span entities to the tracking store.

        The default implementation offloads ``log_spans()`` to a worker thread so
        async callers do not stall the event loop. Stores that implement
        ``log_spans()`` inherit this behavior.

        Args:
            location: The location to log spans to.
            spans: List of Span entities to log. Spans may belong to different traces.

        Returns:
            List of logged Span entities.
        """
        return await asyncio.to_thread(self.log_spans, location, spans)

    def calculate_trace_filter_correlation(
        self,
        experiment_ids: list[str],
        filter_string1: str,
        filter_string2: str,
        base_filter: str | None = None,
    ) -> TraceFilterCorrelationResult:
        """
        Calculate correlation between two trace filter conditions using NPMI.

        This method analyzes the correlation between traces matching two different
        filter conditions using Normalized Pointwise Mutual Information (NPMI).

        Args:
            experiment_ids: List of experiment IDs to analyze traces from.
            filter_string1: First filter condition (MLflow search filter syntax).
            filter_string2: Second filter condition (MLflow search filter syntax).
            base_filter: Optional base filter that both filter1 and filter2 are tested on top of
                        (e.g. 'request_time > ... and request_time < ...' for time windows).

        Returns:
            TraceFilterCorrelationResult containing:
            - npmi: Correlation score from -1 (never co-occur) to 1 (always co-occur),
                   or NaN if undefined (when a filter has zero matches)
            - filter1_count: Number of traces matching filter1
            - filter2_count: Number of traces matching filter2
            - joint_count: Number of traces matching both filters
            - total_count: Total number of traces in the experiments

        Raises:
            MlflowException: If filters are invalid or experiments don't exist.
        """
        raise MlflowNotImplementedException(
            f"The Correlations API is not implemented for {self.__class__.__name__}. "
            "A SQL backend is required to use this feature."
        )

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
        Create a new issue.

        Issue metadata and assessments are backend-owned so counted search can filter, order,
        and paginate within the backend. Use tracking-store context to validate the experiment
        and optional source run as needed.

        Args:
            experiment_id: The experiment ID.
            name: Short descriptive name for the issue.
            description: Detailed description of the issue.
            status: Issue status. Defaults to IssueStatus.PENDING.
            severity: Optional severity level indicator.
            root_causes: Optional list of root cause analyses.
            source_run_id: Optional MLflow run ID that discovered this issue.
            categories: Optional list of categories for the issue.
            created_by: Optional identifier for who created this issue.
            issue_id: Optional iss-* ID (at most 36 characters); reuse returns a conflict.
            last_seen_timestamp: Latest observation in milliseconds; defaults to creation time.
            metadata: String metadata map (at most 100 entries).

        Returns:
            The created Issue entity.
        """
        raise MlflowNotImplementedException()

    def get_issue(self, issue_id: str) -> Issue:
        """
        Get an issue by ID.

        Args:
            issue_id: The ID of the issue to retrieve.

        Returns:
            The Issue entity.
        """
        raise MlflowNotImplementedException()

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
        raise MlflowNotImplementedException()

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
            max_results: Maximum number of results to return.
            page_token: Token for pagination.
            include_trace_count: Whether to include the count of traces impacted by each issue.
            name: Case-insensitive literal substring of the name (not a wildcard pattern).
            statuses: Accepted status values, ORed within the list.
            severities: Accepted severity values, ORed within the list.
            categories: Match any category, ignoring case and surrounding whitespace.
                Different filter fields are ANDed.
            order_by: Field and optional ASC/DESC; supports name, status, severity and timestamps.
                Issue ID breaks ties. Omit to retain the legacy severity/count/creation ordering.
            include_total_count: Return the matching count before pagination in total_count.
            view: "basic" returns an empty description and omits root_causes and metadata.
                The default is "full".

        Returns:
            A PagedList of Issue entities.
        """
        raise MlflowNotImplementedException()
