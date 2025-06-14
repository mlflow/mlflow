import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Optional, Sequence, Union

import mlflow
from mlflow.entities.assessment import (
    Assessment,
)
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.span import NO_OP_SPAN_TRACE_ID
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_info_v2 import TraceInfoV2
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_SEARCH_TRACES_MAX_THREADS
from mlflow.exceptions import (
    MlflowException,
    MlflowTraceDataCorrupted,
    MlflowTraceDataException,
    MlflowTraceDataNotFound,
)
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import TraceJSONEncoder, exclude_immutable_tags
from mlflow.tracing.utils.artifact_utils import get_artifact_uri_for_trace
from mlflow.tracking._tracking_service.utils import _get_store, _resolve_tracking_uri
from mlflow.utils import is_uuid
from mlflow.utils.mlflow_tags import IMMUTABLE_TAGS
from mlflow.utils.uri import add_databricks_profile_info_to_artifact_uri, is_databricks_uri

_logger = logging.getLogger(__name__)


class TracingClient:
    """
    Client of an MLflow Tracking Server that creates and manages experiments and runs.
    """

    def __init__(self, tracking_uri: Optional[str] = None):
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

    def start_trace(
        self,
        experiment_id: str,
        timestamp_ms: int,
        request_metadata: dict[str, str],
        tags: dict[str, str],
    ):
        """
        Start an initial TraceInfo object in the backend store.

        Args:
            experiment_id: String id of the experiment for this run.
            timestamp_ms: Start time of the trace, in milliseconds since the UNIX epoch.
            request_metadata: Metadata of the trace.
            tags: Tags of the trace.

        Returns:
            The created TraceInfo object.
        """
        tags = exclude_immutable_tags(tags or {})
        return self.store.start_trace(
            experiment_id=experiment_id,
            timestamp_ms=timestamp_ms,
            request_metadata=request_metadata,
            tags=tags,
        )

    def start_trace_v3(self, trace: Trace) -> TraceInfo:
        """
        Start a trace using the V3 API format.
        NB: This method is named "Start" for internal reason in the backend, but actually
        should be called at the end of the trace. We will migrate this to "CreateTrace"
        API in the future to avoid confusion.

        Args:
            trace: The Trace object to create.

        Returns:
            The returned TraceInfoV3 object from the backend.
        """
        return self.store.start_trace_v3(trace=trace)

    def end_trace(
        self,
        request_id: str,
        timestamp_ms: int,
        status: TraceStatus,
        request_metadata: dict[str, str],
        tags: dict[str, str],
    ) -> TraceInfoV2:
        """
        Update the TraceInfo object in the backend store with the completed trace info.

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
        tags = exclude_immutable_tags(tags or {})
        return self.store.end_trace(
            request_id=request_id,
            timestamp_ms=timestamp_ms,
            status=status,
            request_metadata=request_metadata,
            tags=tags,
        )

    def delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: Optional[int] = None,
        max_traces: Optional[int] = None,
        request_ids: Optional[list[str]] = None,
    ) -> int:
        return self.store.delete_traces(
            experiment_id=experiment_id,
            max_timestamp_millis=max_timestamp_millis,
            max_traces=max_traces,
            request_ids=request_ids,
        )

    def get_trace_info(self, request_id, should_query_v3: bool = False) -> TraceInfoV2:
        """
        Get the trace info matching the ``request_id``.

        Args:
            request_id: String id of the trace to fetch.
            should_query_v3: If True, the backend store will query the V3 API for the trace info.
                TODO: Remove this flag once the V3 API is the default in OSS.

        Returns:
            TraceInfo object, of type ``mlflow.entities.trace_info.TraceInfo``.
        """
        return self.store.get_trace_info(request_id, should_query_v3=should_query_v3)

    def get_trace(self, request_id) -> Trace:
        """
        Get the trace matching the ``request_id``.

        Args:
            request_id: String id of the trace to fetch.

        Returns:
            The fetched Trace object, of type ``mlflow.entities.Trace``.
        """
        trace_info = self.get_trace_info(
            request_id=request_id, should_query_v3=is_databricks_uri(self.tracking_uri)
        )
        try:
            trace_data = self._download_trace_data(trace_info)
        except MlflowTraceDataNotFound:
            raise MlflowException(
                message=(
                    f"Trace with ID {request_id} cannot be loaded because it is missing span data."
                    " Please try creating or loading another trace."
                ),
                error_code=BAD_REQUEST,
            ) from None  # Ensure the original spammy exception is not included in the traceback
        except MlflowTraceDataCorrupted:
            raise MlflowException(
                message=(
                    f"Trace with ID {request_id} cannot be loaded because its span data"
                    " is corrupted. Please try creating or loading another trace."
                ),
                error_code=BAD_REQUEST,
            ) from None  # Ensure the original spammy exception is not included in the traceback
        return Trace(trace_info, trace_data)

    def get_online_trace_details(
        self,
        trace_id: str,
        sql_warehouse_id: str,
        source_inference_table: str,
        source_databricks_request_id: str,
    ) -> str:
        return self.store.get_online_trace_details(
            trace_id=trace_id,
            sql_warehouse_id=sql_warehouse_id,
            source_inference_table=source_inference_table,
            source_databricks_request_id=source_databricks_request_id,
        )

    def _search_traces(
        self,
        experiment_ids: list[str],
        filter_string: Optional[str] = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
        model_id: Optional[str] = None,
        sql_warehouse_id: Optional[str] = None,
    ):
        return self.store.search_traces(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
            model_id=model_id,
            sql_warehouse_id=sql_warehouse_id,
        )

    def search_traces(
        self,
        experiment_ids: list[str],
        filter_string: Optional[str] = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
        run_id: Optional[str] = None,
        include_spans: bool = True,
        model_id: Optional[str] = None,
        sql_warehouse_id: Optional[str] = None,
    ) -> PagedList[Trace]:
        """
        Return traces that match the given list of search expressions within the experiments.

        Args:
            experiment_ids: List of experiment ids to scope the search.
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
            sql_warehouse_id: Only used in Databricks. The ID of the SQL warehouse to use for
                searching traces in inference tables.


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
                if sql_warehouse_id is None
                else None
            )

        if run_id:
            run = self.store.get_run(run_id)
            if run.info.experiment_id not in experiment_ids:
                raise MlflowException(
                    f"Run {run_id} belongs to experiment {run.info.experiment_id}, which is not "
                    f"in the list of experiment IDs provided: {experiment_ids}. Please include "
                    f"experiment {run.info.experiment_id} in the `experiment_ids` parameter to "
                    "search for traces from this run.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            additional_filter = f"metadata.{TraceMetadataKey.SOURCE_RUN} = '{run_id}'"
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

        is_databricks = is_databricks_uri(self.tracking_uri)

        def download_trace_extra_fields(
            trace_info: Union[TraceInfoV2, TraceInfo],
        ) -> Optional[Trace]:
            """
            Download trace data and assessments for the given trace_info and returns a Trace object.
            If the download fails (e.g., the trace data is missing or corrupted), returns None.

            The trace_info parameter can be either TraceInfo or TraceInfoV3 object.
            """
            from mlflow.entities.trace_info import TraceInfo

            # Determine if this is TraceInfo or TraceInfoV3
            # Helps while transitioning to V3 traces for offline & online
            is_v3 = isinstance(trace_info, TraceInfo)
            trace_id = trace_info.trace_id if is_v3 else trace_info.request_id
            is_online_trace = is_uuid(trace_id)

            # For online traces in Databricks, we need to get trace data from a different endpoint
            try:
                if is_databricks and is_online_trace:
                    # For online traces, get data from the online API
                    trace_data = self.get_online_trace_details(
                        trace_id=trace_id,
                        sql_warehouse_id=sql_warehouse_id,
                        source_inference_table=trace_info.request_metadata.get(
                            "mlflow.sourceTable"
                        ),
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
                        f"Failed to download trace data for trace {trace_id!r} "
                        f"with {e.ctx}. For full traceback, set logging level to DEBUG."
                    ),
                    exc_info=_logger.isEnabledFor(logging.DEBUG),
                )
                return None
            else:
                return Trace(trace_info, trace_data)

        traces = []
        next_max_results = max_results
        next_token = page_token

        max_workers = MLFLOW_SEARCH_TRACES_MAX_THREADS.get()
        executor = ThreadPoolExecutor(max_workers=max_workers) if include_spans else nullcontext()
        with executor:
            while len(traces) < max_results:
                trace_infos, next_token = self._search_traces(
                    experiment_ids=experiment_ids,
                    filter_string=filter_string,
                    max_results=next_max_results,
                    order_by=order_by,
                    page_token=next_token,
                    model_id=model_id,
                    sql_warehouse_id=sql_warehouse_id,
                )

                if include_spans:
                    traces.extend(
                        t for t in executor.map(download_trace_extra_fields, trace_infos) if t
                    )
                else:
                    traces.extend(
                        Trace(trace_info, TraceData(spans=[])) for trace_info in trace_infos
                    )

                if not next_token:
                    break

                next_max_results = max_results - len(traces)

        return PagedList(traces, next_token)

    def set_trace_tags(self, request_id, tags):
        """
        Set tags on the trace with the given request_id.

        Args:
            request_id: The ID of the trace.
            tags: A dictionary of key-value pairs.
        """
        tags = exclude_immutable_tags(tags)
        for k, v in tags.items():
            self.set_trace_tag(request_id, k, v)

    def set_trace_tag(self, request_id, key, value):
        """
        Set a tag on the trace with the given trace ID.

        Args:
            request_id: The ID of the trace to set the tag on.
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
        with InMemoryTraceManager.get_instance().get_trace(request_id) as trace:
            if trace:
                trace.info.tags[key] = str(value)
                return

        if key in IMMUTABLE_TAGS:
            _logger.warning(f"Tag '{key}' is immutable and cannot be set on a trace.")
        else:
            self.store.set_trace_tag(request_id, key, str(value))

    def delete_trace_tag(self, request_id, key):
        """
        Delete a tag on the trace with the given trace ID.

        Args:
            request_id: The ID of the trace to delete the tag from.
            key: The string key of the tag. Must be at most 250 characters long, otherwise
                it will be truncated when stored.
        """
        # Trying to delete the tag on the active trace first
        with InMemoryTraceManager.get_instance().get_trace(request_id) as trace:
            if trace:
                if key in trace.info.tags:
                    trace.info.tags.pop(key)
                    return
                else:
                    raise MlflowException(
                        f"Tag with key {key} not found in trace with ID {request_id}.",
                        error_code=RESOURCE_DOES_NOT_EXIST,
                    )

        if key in IMMUTABLE_TAGS:
            _logger.warning(f"Tag '{key}' is immutable and cannot be deleted on a trace.")
        else:
            self.store.delete_trace_tag(request_id, key)

    def get_assessment(self, trace_id: str, assessment_id: str) -> Assessment:
        """
        Get an assessment entity from the backend store.

        Args:
            trace_id: The ID of the trace.
            assessment_id: The ID of the assessment to get.

        Returns:
            The Assessment object.
        """
        if not is_databricks_uri(self.tracking_uri):
            raise MlflowException(
                "This API is currently only available for Databricks Managed MLflow. This "
                "will be available in the open-source version of MLflow in a future release."
            )

        return self.store.get_assessment(trace_id, assessment_id)

    def log_assessment(self, trace_id: str, assessment: Assessment) -> Assessment:
        if not is_databricks_uri(self.tracking_uri):
            raise MlflowException(
                "This API is currently only available for Databricks Managed MLflow. This "
                "will be available in the open-source version of MLflow in a future release."
            )

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
        if not is_databricks_uri(self.tracking_uri):
            raise MlflowException(
                "This API is currently only available for Databricks Managed MLflow. This "
                "will be available in the open-source version of MLflow in a future release."
            )

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
        if not is_databricks_uri(self.tracking_uri):
            raise MlflowException(
                "This API is currently only available for Databricks Managed MLflow. This "
                "will be available in the open-source version of MLflow in a future release."
            )

        self.store.delete_assessment(trace_id=trace_id, assessment_id=assessment_id)

    def _get_artifact_repo_for_trace(self, trace_info: TraceInfoV2):
        artifact_uri = get_artifact_uri_for_trace(trace_info)
        artifact_uri = add_databricks_profile_info_to_artifact_uri(artifact_uri, self.tracking_uri)
        return get_artifact_repository(artifact_uri)

    def _download_trace_data(self, trace_info: Union[TraceInfoV2, TraceInfo]) -> TraceData:
        """
        Download trace data from artifact repository.

        Args:
            trace_info: Either a TraceInfo or TraceInfoV3 object containing trace metadata.

        Returns:
            TraceData object representing the downloaded trace data.
        """
        artifact_repo = self._get_artifact_repo_for_trace(trace_info)
        return TraceData.from_dict(artifact_repo.download_trace_data())

    def _upload_trace_data(self, trace_info: TraceInfoV2, trace_data: TraceData) -> None:
        artifact_repo = self._get_artifact_repo_for_trace(trace_info)
        trace_data_json = json.dumps(trace_data.to_dict(), cls=TraceJSONEncoder, ensure_ascii=False)
        return artifact_repo.upload_trace_data(trace_data_json)

    def _upload_ended_trace_info(
        self,
        trace_info: TraceInfoV2,
    ) -> TraceInfoV2:
        """
        Update the TraceInfo object in the backend store with the completed trace info.

        Args:
            trace_info: Updated TraceInfo object to be stored in the backend store.

        Returns:
            The updated TraceInfo object.
        """
        return self.end_trace(
            request_id=trace_info.request_id,
            timestamp_ms=trace_info.timestamp_ms + trace_info.execution_time_ms,
            status=trace_info.status,
            request_metadata=trace_info.request_metadata,
            tags=trace_info.tags or {},
        )

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
