import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import NamedTuple, Sequence

import mlflow
from mlflow.entities.assessment import Assessment
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.span import NO_OP_SPAN_TRACE_ID, Span
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import UCSchemaLocation
from mlflow.environment_variables import (
    _MLFLOW_SEARCH_TRACES_MAX_BATCH_SIZE,
    MLFLOW_SEARCH_TRACES_MAX_THREADS,
    MLFLOW_TRACING_SQL_WAREHOUSE_ID,
)
from mlflow.exceptions import (
    MlflowException,
    MlflowTraceDataCorrupted,
    MlflowTraceDataException,
    MlflowTraceDataNotFound,
)
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INVALID_PARAMETER_VALUE,
    NOT_FOUND,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.telemetry.events import LogAssessmentEvent, StartTraceEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.tracing.constant import GET_TRACE_V4_RETRY_TIMEOUT_SECONDS, TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import TraceJSONEncoder, exclude_immutable_tags, parse_trace_id_v4
from mlflow.tracing.utils.artifact_utils import get_artifact_uri_for_trace
from mlflow.tracking._tracking_service.utils import _get_store, _resolve_tracking_uri
from mlflow.utils import is_uuid
from mlflow.utils.mlflow_tags import IMMUTABLE_TAGS
from mlflow.utils.uri import add_databricks_profile_info_to_artifact_uri, is_databricks_uri

_logger = logging.getLogger(__name__)


class TraceInfoGroups(NamedTuple):
    tracking_store_trace_infos: list[TraceInfo]
    artifact_repo_trace_infos: list[TraceInfo]


class TracingClient:
    """
    Client of an MLflow Tracking Server that creates and manages experiments and runs.
    """

    def __init__(self, tracking_uri: str | None = None):
        """
        Args:
            tracking_uri: Address of local or remote tracking server.
        """
        self.tracking_uri = _resolve_tracking_uri(tracking_uri)
        # NB: Fetch the tracking store (`self.store`) upon client initialization to ensure that
        # the tracking URI is valid and the store can be properly resolved. We define `store` as a
        # property method to ensure that the client is serializable, even if the store is not
        # self.store
        self.store

    @property
    def store(self):
        return _get_store(self.tracking_uri)

    @record_usage_event(StartTraceEvent)
    def start_trace(self, trace_info: TraceInfo) -> TraceInfo:
        """
        Create a new trace in the backend.

        Args:
            trace_info: The TraceInfo object to record in the backend.

        Returns:
            The returned TraceInfoV3 object from the backend.
        """
        return self.store.start_trace(trace_info=trace_info)

    def log_spans(self, location: str, spans: list[Span]) -> list[Span]:
        """
        Log spans to the backend.

        Args:
            location: The location to log spans to. It should either be an experiment ID or a
                Unity Catalog table name.
            spans: List of Span objects to log.

        Returns:
            List of logged Span objects from the backend.
        """
        return self.store.log_spans(
            location=location,
            spans=spans,
            tracking_uri=self.tracking_uri if is_databricks_uri(self.tracking_uri) else None,
        )

    def delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: int | None = None,
        max_traces: int | None = None,
        trace_ids: list[str] | None = None,
    ) -> int:
        return self.store.delete_traces(
            experiment_id=experiment_id,
            max_timestamp_millis=max_timestamp_millis,
            max_traces=max_traces,
            trace_ids=trace_ids,
        )

    def get_trace_info(self, trace_id: str) -> TraceInfo:
        """
        Get the trace info matching the ``trace_id``.

        Args:
            trace_id: String id of the trace to fetch.

        Returns:
            TraceInfo object, of type ``mlflow.entities.trace_info.TraceInfo``.
        """
        with InMemoryTraceManager.get_instance().get_trace(trace_id) as trace:
            if trace is not None:
                return trace.info

        return self.store.get_trace_info(trace_id)

    def get_trace(self, trace_id: str) -> Trace:
        """
        Get the trace matching the ``trace_id``.

        Args:
            trace_id: String id of the trace to fetch.

        Returns:
            The fetched Trace object, of type ``mlflow.entities.Trace``.
        """
        location, _ = parse_trace_id_v4(trace_id)
        if location is not None:
            start_time = time.time()
            attempt = 0
            while time.time() - start_time < GET_TRACE_V4_RETRY_TIMEOUT_SECONDS:
                # For a V4 trace, load spans from the v4 BatchGetTraces endpoint.
                # BatchGetTraces returns an empty list if the trace is not found, which will be
                # retried up to GET_TRACE_V4_RETRY_TIMEOUT_SECONDS seconds.
                if traces := self.store.batch_get_traces([trace_id], location):
                    return traces[0]

                attempt += 1
                interval = 2**attempt
                _logger.debug(
                    f"Trace not found, retrying in {interval} seconds (attempt {attempt})"
                )
                time.sleep(interval)

            raise MlflowException(
                message=f"Trace with ID {trace_id} is not found.",
                error_code=NOT_FOUND,
            )
        else:
            # V3 trace, load spans from artifact repository.
            try:
                trace_info = self.get_trace_info(trace_id)
                trace_data = self._download_trace_data(trace_info)
            except MlflowTraceDataNotFound:
                raise MlflowException(
                    message=(
                        f"Trace with ID {trace_id} cannot be loaded because it is missing span "
                        "data. Please try creating or loading another trace."
                    ),
                    error_code=BAD_REQUEST,
                ) from None  # Ensure the original spammy exception is not included in the traceback
            except MlflowTraceDataCorrupted:
                raise MlflowException(
                    message=(
                        f"Trace with ID {trace_id} cannot be loaded because its span data"
                        " is corrupted. Please try creating or loading another trace."
                    ),
                    error_code=BAD_REQUEST,
                ) from None  # Ensure the original spammy exception is not included in the traceback
            return Trace(trace_info, trace_data)

    def get_online_trace_details(
        self,
        trace_id: str,
        source_inference_table: str,
        source_databricks_request_id: str,
    ) -> str:
        return self.store.get_online_trace_details(
            trace_id=trace_id,
            source_inference_table=source_inference_table,
            source_databricks_request_id=source_databricks_request_id,
        )

    def _search_traces(
        self,
        experiment_ids: list[str] | None = None,
        filter_string: str | None = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: list[str] | None = None,
        page_token: str | None = None,
        model_id: str | None = None,
        locations: list[str] | None = None,
    ):
        return self.store.search_traces(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
            model_id=model_id,
            locations=locations,
        )

    def search_traces(
        self,
        experiment_ids: list[str] | None = None,
        filter_string: str | None = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: list[str] | None = None,
        page_token: str | None = None,
        run_id: str | None = None,
        include_spans: bool = True,
        model_id: str | None = None,
        locations: list[str] | None = None,
    ) -> PagedList[Trace]:
        """
        Return traces that match the given list of search expressions within the experiments.

        Args:
            experiment_ids: List of experiment ids to scope the search. Deprecated,
                use `locations` instead.
            filter_string: A search filter string.
            max_results: Maximum number of traces desired.
            order_by: List of order_by clauses.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_traces`` call.
            run_id: A run id to scope the search. When a trace is created under an active run,
                it will be associated with the run and you can filter on the run id to retrieve
                the trace.
            include_spans: If ``True``, include spans in the returned traces. Otherwise, only
                the trace metadata is returned, e.g., trace ID, start time, end time, etc,
                without any spans.
            model_id: If specified, return traces associated with the model ID.
            locations: A list of locations to search over. To search over experiments, provide
                a list of experiment IDs. To search over UC tables on databricks, provide
                a list of locations in the format `<catalog_name>.<schema_name>`.

        Returns:
            A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
            :py:class:`Trace <mlflow.entities.Trace>` objects that satisfy the search
            expressions. If the underlying tracking store supports pagination, the token for the
            next page may be obtained via the ``token`` attribute of the returned object; however,
            some store implementations may not support pagination and thus the returned token would
            not be meaningful in such cases.
        """
        if model_id is not None:
            if filter_string:
                raise MlflowException(
                    message=(
                        "Cannot specify both `model_id` or `filter_string` in the search_traces "
                        "call."
                    ),
                    error_code=INVALID_PARAMETER_VALUE,
                )

            filter_string = (
                f"request_metadata.`mlflow.modelId` = '{model_id}'"
                if MLFLOW_TRACING_SQL_WAREHOUSE_ID.get() is None
                else None
            )

        if run_id:
            run = self.store.get_run(run_id)
            if run.info.experiment_id not in locations:
                raise MlflowException(
                    f"Run {run_id} belongs to experiment {run.info.experiment_id}, which is not "
                    f"in the list of locations provided: {locations}. Please include "
                    f"experiment {run.info.experiment_id} in the `locations` parameter to "
                    "search for traces from this run.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            additional_filter = f"attribute.run_id = '{run_id}'"
            if filter_string:
                if TraceMetadataKey.SOURCE_RUN in filter_string:
                    raise MlflowException(
                        "You cannot filter by run_id when it is already part of the filter string."
                        f"Please remove the {TraceMetadataKey.SOURCE_RUN} filter from the filter "
                        "string and try again.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                filter_string += f" AND {additional_filter}"
            else:
                filter_string = additional_filter

        traces = []
        next_max_results = max_results
        next_token = page_token

        max_workers = MLFLOW_SEARCH_TRACES_MAX_THREADS.get()
        executor = (
            ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="MlflowTracingSearch")
            if include_spans
            else nullcontext()
        )
        with executor:
            while len(traces) < max_results:
                trace_infos, next_token = self._search_traces(
                    experiment_ids=experiment_ids,
                    filter_string=filter_string,
                    max_results=next_max_results,
                    order_by=order_by,
                    page_token=next_token,
                    model_id=model_id,
                    locations=locations,
                )

                if include_spans:
                    location_to_trace_infos = self._group_trace_infos_by_location(trace_infos)
                    for location, location_trace_infos in location_to_trace_infos.items():
                        if "." in location:
                            # UC schema location. Get full traces from v4 BatchGetTraces.
                            # All traces in a single call must be located in the same table.
                            trace_ids = [t.trace_id for t in location_trace_infos]
                            traces.extend(
                                self._download_spans_from_batch_get_traces(
                                    trace_ids, location, executor
                                )
                            )
                        else:
                            # MLflow experiment location. Load spans from artifact repository.
                            traces.extend(
                                trace
                                for trace in executor.map(
                                    self._download_spans_from_artifact_repo,
                                    location_trace_infos,
                                )
                                if trace
                            )
                else:
                    traces.extend(Trace(t, TraceData(spans=[])) for t in trace_infos)

                if not next_token:
                    break

                next_max_results = max_results - len(traces)

        return PagedList(traces, next_token)

    def _download_spans_from_batch_get_traces(
        self, trace_ids: list[str], location: str, executor: ThreadPoolExecutor
    ) -> list[Trace]:
        """
        Fetch full traces including spans from the BatchGetTrace v4 endpoint.
        BatchGetTrace endpoint only support up to 10 traces in a single call.
        """
        traces = []

        def _fetch_minibatch(ids: list[str]) -> list[Trace]:
            return self.store.batch_get_traces(ids, location) or []

        batch_size = _MLFLOW_SEARCH_TRACES_MAX_BATCH_SIZE.get()
        batches = [trace_ids[i : i + batch_size] for i in range(0, len(trace_ids), batch_size)]
        for minibatch_traces in executor.map(_fetch_minibatch, batches):
            traces.extend(minibatch_traces)
        return traces

    def _download_spans_from_artifact_repo(self, trace_info: TraceInfo) -> Trace | None:
        """
        Download trace data for the given trace_info and returns a Trace object.
        If the download fails (e.g., the trace data is missing or corrupted), returns None.

        This is used for traces logged via v3 endpoint, where spans are stored in artifact store.
        """
        is_online_trace = is_uuid(trace_info.trace_id)
        is_databricks = is_databricks_uri(self.tracking_uri)

        # For online traces in Databricks, we need to get trace data from a different endpoint
        try:
            if is_databricks and is_online_trace:
                # For online traces, get data from the online API
                trace_data = self.get_online_trace_details(
                    trace_id=trace_info.trace_id,
                    source_inference_table=trace_info.request_metadata.get("mlflow.sourceTable"),
                    source_databricks_request_id=trace_info.request_metadata.get(
                        "mlflow.databricksRequestId"
                    ),
                )
                trace_data = TraceData.from_dict(json.loads(trace_data))
            else:
                # For offline traces, download data from artifact storage
                trace_data = self._download_trace_data(trace_info)
        except MlflowTraceDataException as e:
            _logger.warning(
                (
                    f"Failed to download trace data for trace {trace_info.trace_id!r} "
                    f"with {e.ctx}. For full traceback, set logging level to DEBUG."
                ),
                exc_info=_logger.isEnabledFor(logging.DEBUG),
            )
            return None
        else:
            return Trace(trace_info, trace_data)

    def _group_trace_infos_by_location(
        self, trace_infos: list[TraceInfo]
    ) -> dict[str, list[TraceInfo]]:
        """
        Group the trace infos based on where the trace data is stored.

        Returns:
            A dictionary mapping location to a list of trace infos.
        """
        location_to_trace_infos = defaultdict(list)
        for trace_info in trace_infos:
            if uc_schema := trace_info.trace_location.uc_schema:
                location = f"{uc_schema.catalog_name}.{uc_schema.schema_name}"
                location_to_trace_infos[location].append(trace_info)
            elif mlflow_experiment := trace_info.trace_location.mlflow_experiment:
                location_to_trace_infos[mlflow_experiment.experiment_id].append(trace_info)
            else:
                _logger.warning(f"Unsupported location: {trace_info.trace_location}. Skipping.")
        return location_to_trace_infos

    def calculate_trace_filter_correlation(
        self,
        experiment_ids: list[str],
        filter_string1: str,
        filter_string2: str,
        base_filter: str | None = None,
    ):
        """
        Calculate the correlation (NPMI) between two trace filter conditions.

        This method computes the Normalized Pointwise Mutual Information (NPMI)
        between traces matching two different filter conditions, which measures
        how much more (or less) likely traces are to satisfy both conditions
        compared to if the conditions were independent.

        Args:
            experiment_ids: List of experiment IDs to search within.
            filter_string1: First filter condition (e.g., "span.type = 'LLM'").
            filter_string2: Second filter condition (e.g., "feedback.quality > 0.8").
            base_filter: Optional base filter that both filter1 and filter2 are tested on top of
                        (e.g., 'request_time > ... and request_time < ...' for time windows).

        Returns:
            TraceFilterCorrelationResult containing:
                - npmi: NPMI score from -1 (never co-occur) to 1 (always co-occur)
                - npmi_smoothed: Smoothed NPMI value with Jeffreys prior for robustness
                - filter1_count: Number of traces matching filter_string1
                - filter2_count: Number of traces matching filter_string2
                - joint_count: Number of traces matching both filters
                - total_count: Total number of traces in the experiments

        .. code-block:: python

            from mlflow.tracing.client import TracingClient

            client = TracingClient()
            result = client.calculate_trace_filter_correlation(
                experiment_ids=["123"],
                filter_string1="span.type = 'LLM'",
                filter_string2="feedback.quality > 0.8",
            )
            print(f"NPMI: {result.npmi:.3f}")
            # Output: NPMI: 0.456
        """
        return self.store.calculate_trace_filter_correlation(
            experiment_ids=experiment_ids,
            filter_string1=filter_string1,
            filter_string2=filter_string2,
            base_filter=base_filter,
        )

    def set_trace_tags(self, trace_id: str, tags: dict[str, str]):
        """
        Set tags on the trace with the given trace_id.

        Args:
            trace_id: The ID of the trace.
            tags: A dictionary of key-value pairs.
        """
        tags = exclude_immutable_tags(tags)
        for k, v in tags.items():
            self.set_trace_tag(trace_id, k, v)

    def set_trace_tag(self, trace_id: str, key: str, value: str):
        """
        Set a tag on the trace with the given trace ID.

        Args:
            trace_id: The ID of the trace to set the tag on.
            key: The string key of the tag. Must be at most 250 characters long, otherwise
                it will be truncated when stored.
            value: The string value of the tag. Must be at most 250 characters long, otherwise
                it will be truncated when stored.
        """
        if not isinstance(value, str):
            _logger.warning(
                "Received non-string value for trace tag. Please note that non-string tag values"
                "will automatically be stringified when the trace is logged."
            )

        # Trying to set the tag on the active trace first
        with InMemoryTraceManager.get_instance().get_trace(trace_id) as trace:
            if trace:
                trace.info.tags[key] = str(value)
                return

        if key in IMMUTABLE_TAGS:
            _logger.warning(f"Tag '{key}' is immutable and cannot be set on a trace.")
        else:
            self.store.set_trace_tag(trace_id, key, str(value))

    def delete_trace_tag(self, trace_id: str, key: str):
        """
        Delete a tag on the trace with the given trace ID.

        Args:
            trace_id: The ID of the trace to delete the tag from.
            key: The string key of the tag. Must be at most 250 characters long, otherwise
                it will be truncated when stored.
        """
        # Trying to delete the tag on the active trace first
        with InMemoryTraceManager.get_instance().get_trace(trace_id) as trace:
            if trace:
                if key in trace.info.tags:
                    trace.info.tags.pop(key)
                    return
                else:
                    raise MlflowException(
                        f"Tag with key {key} not found in trace with ID {trace_id}.",
                        error_code=RESOURCE_DOES_NOT_EXIST,
                    )

        if key in IMMUTABLE_TAGS:
            _logger.warning(f"Tag '{key}' is immutable and cannot be deleted on a trace.")
        else:
            self.store.delete_trace_tag(trace_id, key)

    def get_assessment(self, trace_id: str, assessment_id: str) -> Assessment:
        """
        Get an assessment entity from the backend store.

        Args:
            trace_id: The ID of the trace.
            assessment_id: The ID of the assessment to get.

        Returns:
            The Assessment object.
        """

        return self.store.get_assessment(trace_id, assessment_id)

    @record_usage_event(LogAssessmentEvent)
    def log_assessment(self, trace_id: str, assessment: Assessment) -> Assessment:
        """
        Log an assessment to a trace.

        Args:
            trace_id: The ID of the trace.
            assessment: The assessment object to log.

        Returns:
            The logged Assessment object.
        """
        assessment.trace_id = trace_id

        if trace_id is None or trace_id == NO_OP_SPAN_TRACE_ID:
            _logger.debug(
                "Skipping assessment logging for NO_OP_SPAN_TRACE_ID. This is expected when "
                "tracing is disabled."
            )
            return assessment

        # If the trace is the active trace, add the assessment to it in-memory
        if trace_id == mlflow.get_active_trace_id():
            with InMemoryTraceManager.get_instance().get_trace(trace_id) as trace:
                if trace is None:
                    _logger.debug(
                        f"Trace {trace_id} is active but not found in the in-memory buffer. "
                        "Something is wrong with trace handling. Skipping assessment logging."
                    )
                trace.info.assessments.append(assessment)
            return assessment
        return self.store.create_assessment(assessment)

    def update_assessment(
        self,
        trace_id: str,
        assessment_id: str,
        assessment: Assessment,
    ):
        """
        Update an existing assessment entity in the backend store.

        Args:
            trace_id: The ID of the trace.
            assessment_id: The ID of the feedback assessment to update.
            assessment: The updated assessment.
        """

        return self.store.update_assessment(
            trace_id=trace_id,
            assessment_id=assessment_id,
            name=assessment.name,
            expectation=assessment.expectation,
            feedback=assessment.feedback,
            rationale=assessment.rationale,
            metadata=assessment.metadata,
        )

    def delete_assessment(self, trace_id: str, assessment_id: str):
        """
        Delete an assessment associated with a trace.

        Args:
            trace_id: The ID of the trace.
            assessment_id: The ID of the assessment to delete.
        """

        self.store.delete_assessment(trace_id=trace_id, assessment_id=assessment_id)

    def _get_artifact_repo_for_trace(self, trace_info: TraceInfo):
        artifact_uri = get_artifact_uri_for_trace(trace_info)
        artifact_uri = add_databricks_profile_info_to_artifact_uri(artifact_uri, self.tracking_uri)
        return get_artifact_repository(artifact_uri)

    def _download_trace_data(self, trace_info: TraceInfo) -> TraceData:
        """
        Download trace data from artifact repository.

        Args:
            trace_info: Either a TraceInfo or TraceInfoV3 object containing trace metadata.

        Returns:
            TraceData object representing the downloaded trace data.
        """
        artifact_repo = self._get_artifact_repo_for_trace(trace_info)
        return TraceData.from_dict(artifact_repo.download_trace_data())

    def _upload_trace_data(self, trace_info: TraceInfo, trace_data: TraceData) -> None:
        artifact_repo = self._get_artifact_repo_for_trace(trace_info)
        trace_data_json = json.dumps(trace_data.to_dict(), cls=TraceJSONEncoder, ensure_ascii=False)
        return artifact_repo.upload_trace_data(trace_data_json)

    # TODO: Migrate this to the new association table
    def link_prompt_versions_to_trace(
        self, trace_id: str, prompts: Sequence[PromptVersion]
    ) -> None:
        """
        Link multiple prompt versions to a trace.

        Args:
            trace_id: The ID of the trace to link prompts to.
            prompts: List of PromptVersion objects to link to the trace.
        """
        from mlflow.tracking._model_registry.utils import _get_store as _get_model_registry_store

        registry_store = _get_model_registry_store()
        registry_store.link_prompts_to_trace(prompt_versions=prompts, trace_id=trace_id)

    def _set_experiment_trace_location(
        self,
        location: UCSchemaLocation,
        experiment_id: str,
        sql_warehouse_id: str | None = None,
    ) -> UCSchemaLocation:
        if is_databricks_uri(self.tracking_uri):
            return self.store.set_experiment_trace_location(
                experiment_id=str(experiment_id),
                location=location,
                sql_warehouse_id=sql_warehouse_id,
            )
        raise MlflowException(
            "Setting storage location is not supported on non-Databricks backends."
        )

    def _unset_experiment_trace_location(
        self, experiment_id: str, location: UCSchemaLocation
    ) -> None:
        if is_databricks_uri(self.tracking_uri):
            self.store.unset_experiment_trace_location(str(experiment_id), location)
        else:
            raise MlflowException(
                "Clearing storage location is not supported on non-Databricks backends."
            )
