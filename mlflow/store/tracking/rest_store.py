import json
import logging
from typing import Any, Optional

from mlflow.entities import (
    DatasetInput,
    Experiment,
    LoggedModel,
    LoggedModelInput,
    LoggedModelOutput,
    LoggedModelParameter,
    LoggedModelStatus,
    LoggedModelTag,
    Metric,
    Run,
    RunInfo,
    TraceInfoV2,
    ViewType,
)
from mlflow.entities.assessment import Assessment, Expectation, Feedback
from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_info_v2 import TraceInfoV2
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import (
    _MLFLOW_CREATE_LOGGED_MODEL_PARAMS_BATCH_SIZE,
    _MLFLOW_LOG_LOGGED_MODEL_PARAMS_BATCH_SIZE,
    MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT,
)
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.protos.service_pb2 import (
    CreateAssessment,
    CreateExperiment,
    CreateLoggedModel,
    CreateRun,
    DeleteAssessment,
    DeleteExperiment,
    DeleteLoggedModel,
    DeleteLoggedModelTag,
    DeleteRun,
    DeleteTag,
    DeleteTraces,
    DeleteTraceTag,
    EndTrace,
    FinalizeLoggedModel,
    GetAssessmentRequest,
    GetExperiment,
    GetExperimentByName,
    GetLoggedModel,
    GetMetricHistory,
    GetOnlineTraceDetails,
    GetRun,
    GetTraceInfo,
    GetTraceInfoV3,
    LogBatch,
    LogInputs,
    LogLoggedModelParamsRequest,
    LogMetric,
    LogModel,
    LogOutputs,
    LogParam,
    MlflowService,
    RestoreExperiment,
    RestoreRun,
    SearchExperiments,
    SearchLoggedModels,
    SearchRuns,
    SearchTraces,
    SearchTracesV3Request,
    SearchUnifiedTraces,
    SetExperimentTag,
    SetLoggedModelTags,
    SetTag,
    SetTraceTag,
    StartTrace,
    StartTraceV3,
    TraceRequestMetadata,
    TraceTag,
    UpdateAssessment,
    UpdateExperiment,
    UpdateRun,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    call_endpoint,
    extract_api_info_for_service,
    get_create_assessment_endpoint,
    get_logged_model_endpoint,
    get_search_traces_v3_endpoint,
    get_set_trace_tag_endpoint,
    get_single_assessment_endpoint,
    get_single_trace_endpoint,
    get_trace_assessment_endpoint,
    get_trace_info_endpoint,
)

_METHOD_TO_INFO = extract_api_info_for_service(MlflowService, _REST_API_PATH_PREFIX)
_logger = logging.getLogger(__name__)


class RestStore(AbstractStore):
    """
    Client for a remote tracking server accessed via REST API calls

    Args
        get_host_creds: Method to be invoked prior to every REST request to get the
            :py:class:`mlflow.rest_utils.MlflowHostCreds` for the request. Note that this
            is a function so that we can obtain fresh credentials in the case of expiry.
    """

    def __init__(self, get_host_creds):
        super().__init__()
        self.get_host_creds = get_host_creds

    def _call_endpoint(
        self,
        api,
        json_body=None,
        endpoint=None,
        retry_timeout_seconds=None,
    ):
        if endpoint:
            # Allow customizing the endpoint for compatibility with dynamic endpoints, such as
            # /mlflow/traces/{request_id}/info.
            _, method = _METHOD_TO_INFO[api]
        else:
            endpoint, method = _METHOD_TO_INFO[api]
        response_proto = api.Response()
        return call_endpoint(
            self.get_host_creds(),
            endpoint,
            method,
            json_body,
            response_proto,
            retry_timeout_seconds=retry_timeout_seconds,
        )

    def search_experiments(
        self,
        view_type=ViewType.ACTIVE_ONLY,
        max_results=None,
        filter_string=None,
        order_by=None,
        page_token=None,
    ):
        req_body = message_to_json(
            SearchExperiments(
                view_type=view_type,
                max_results=max_results,
                page_token=page_token,
                order_by=order_by,
                filter=filter_string,
            )
        )
        response_proto = self._call_endpoint(SearchExperiments, req_body)
        experiments = [Experiment.from_proto(x) for x in response_proto.experiments]
        token = (
            response_proto.next_page_token if response_proto.HasField("next_page_token") else None
        )
        return PagedList(experiments, token)

    def create_experiment(self, name, artifact_location=None, tags=None):
        """
        Create a new experiment.
        If an experiment with the given name already exists, throws exception.

        Args:
            name: Desired name for an experiment.
            artifact_location: Location to store run artifacts.
            tags: A list of :py:class:`mlflow.entities.ExperimentTag` instances to set for the
                experiment.

        Returns:
            experiment_id for the newly created experiment if successful, else None
        """
        tag_protos = [tag.to_proto() for tag in tags] if tags else []
        req_body = message_to_json(
            CreateExperiment(name=name, artifact_location=artifact_location, tags=tag_protos)
        )
        response_proto = self._call_endpoint(CreateExperiment, req_body)
        return response_proto.experiment_id

    def get_experiment(self, experiment_id):
        """
        Fetch the experiment from the backend store.

        Args:
            experiment_id: String id for the experiment

        Returns:
            A single :py:class:`mlflow.entities.Experiment` object if it exists,
            otherwise raises an Exception.
        """
        req_body = message_to_json(GetExperiment(experiment_id=str(experiment_id)))
        response_proto = self._call_endpoint(GetExperiment, req_body)
        return Experiment.from_proto(response_proto.experiment)

    def delete_experiment(self, experiment_id):
        req_body = message_to_json(DeleteExperiment(experiment_id=str(experiment_id)))
        self._call_endpoint(DeleteExperiment, req_body)

    def restore_experiment(self, experiment_id):
        req_body = message_to_json(RestoreExperiment(experiment_id=str(experiment_id)))
        self._call_endpoint(RestoreExperiment, req_body)

    def rename_experiment(self, experiment_id, new_name):
        req_body = message_to_json(
            UpdateExperiment(experiment_id=str(experiment_id), new_name=new_name)
        )
        self._call_endpoint(UpdateExperiment, req_body)

    def get_run(self, run_id):
        """
        Fetch the run from backend store

        Args:
            run_id: Unique identifier for the run

        Returns:
            A single Run object if it exists, otherwise raises an Exception
        """
        req_body = message_to_json(GetRun(run_uuid=run_id, run_id=run_id))
        response_proto = self._call_endpoint(GetRun, req_body)
        return Run.from_proto(response_proto.run)

    def update_run_info(self, run_id, run_status, end_time, run_name):
        """Updates the metadata of the specified run."""
        req_body = message_to_json(
            UpdateRun(
                run_uuid=run_id,
                run_id=run_id,
                status=run_status,
                end_time=end_time,
                run_name=run_name,
            )
        )
        response_proto = self._call_endpoint(UpdateRun, req_body)
        return RunInfo.from_proto(response_proto.run_info)

    def create_run(self, experiment_id, user_id, start_time, tags, run_name):
        """
        Create a run under the specified experiment ID, setting the run's status to "RUNNING"
        and the start time to the current time.

        Args:
            experiment_id: ID of the experiment for this run.
            user_id: ID of the user launching this run.
            start_time: timestamp of the initialization of the run.
            tags: tags to apply to this run at initialization.
            run_name: Name of this run.

        Returns:
            The created Run object.
        """

        tag_protos = [tag.to_proto() for tag in tags]
        req_body = message_to_json(
            CreateRun(
                experiment_id=str(experiment_id),
                user_id=user_id,
                start_time=start_time,
                tags=tag_protos,
                run_name=run_name,
            )
        )
        response_proto = self._call_endpoint(CreateRun, req_body)
        return Run.from_proto(response_proto.run)

    def start_trace(
        self,
        experiment_id: str,
        timestamp_ms: int,
        request_metadata: dict[str, str],
        tags: dict[str, str],
    ) -> TraceInfoV2:
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
        request_metadata_proto = []
        for key, value in request_metadata.items():
            attr = TraceRequestMetadata()
            attr.key = key
            attr.value = str(value)
            request_metadata_proto.append(attr)

        tags_proto = []
        for key, value in tags.items():
            tag = TraceTag()
            tag.key = key
            tag.value = str(value)
            tags_proto.append(tag)

        req_body = message_to_json(
            StartTrace(
                experiment_id=str(experiment_id),
                timestamp_ms=timestamp_ms,
                request_metadata=request_metadata_proto,
                tags=tags_proto,
            )
        )
        response_proto = self._call_endpoint(StartTrace, req_body)
        return TraceInfoV2.from_proto(response_proto.trace_info)

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
        req_body = message_to_json(StartTraceV3(trace=trace.to_proto()))
        response_proto = self._call_endpoint(
            # NB: _call_endpoint doesn't handle versioning between v2 and v3 endpoint
            # yet, so manually passing the v3 endpoint here.
            StartTraceV3,
            req_body,
            endpoint="/api/3.0/mlflow/traces",
            retry_timeout_seconds=MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT.get(),
        )
        return TraceInfo.from_proto(response_proto.trace.trace_info)

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
        request_metadata_proto = []
        for key, value in request_metadata.items():
            attr = TraceRequestMetadata()
            attr.key = key
            attr.value = str(value)
            request_metadata_proto.append(attr)

        tags_proto = []
        for key, value in tags.items():
            tag = TraceTag()
            tag.key = key
            tag.value = str(value)
            tags_proto.append(tag)

        req_body = message_to_json(
            EndTrace(
                request_id=request_id,
                timestamp_ms=timestamp_ms,
                status=status.to_proto(),
                request_metadata=request_metadata_proto,
                tags=tags_proto,
            )
        )
        # EndTrace endpoint is a dynamic path built with the request_id
        # Always use v2 endpoint (not v3) for this endpoint to maintain compatibility
        endpoint = get_single_trace_endpoint(request_id, use_v3=False)
        response_proto = self._call_endpoint(EndTrace, req_body, endpoint=endpoint)
        return TraceInfoV2.from_proto(response_proto.trace_info)

    def _delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: Optional[int] = None,
        max_traces: Optional[int] = None,
        request_ids: Optional[list[str]] = None,
    ) -> int:
        req_body = message_to_json(
            DeleteTraces(
                experiment_id=experiment_id,
                max_timestamp_millis=max_timestamp_millis,
                max_traces=max_traces,
                request_ids=request_ids,
            )
        )
        res = self._call_endpoint(DeleteTraces, req_body)
        return res.traces_deleted

    def get_trace_info(self, request_id, should_query_v3: bool = False):
        """
        Get the trace matching the `request_id`.

        Args:
            request_id: String id of the trace to fetch.
            should_query_v3: If True, the backend store will query the V3 API for the trace info.
                TODO: Remove this flag once the V3 API is the default in OSS.

        Returns:
            The fetched Trace object, of type ``mlflow.entities.TraceInfo``.
        """
        # If explicitly set, use the provided value, otherwise detect based on the tracking URI

        if should_query_v3:
            trace_v3_req_body = message_to_json(GetTraceInfoV3(trace_id=request_id))
            trace_v3_endpoint = get_trace_assessment_endpoint(
                request_id, is_databricks=should_query_v3
            )
            try:
                trace_v3_response_proto = self._call_endpoint(
                    GetTraceInfoV3, trace_v3_req_body, endpoint=trace_v3_endpoint
                )
                return TraceInfo.from_proto(trace_v3_response_proto.trace.trace_info)
            except Exception:
                # TraceV3 endpoint is not globally enabled yet; graceful fallback path.
                _logger.debug(
                    f"Failed to fetch trace info from V3 API for request ID {request_id!r}.",
                    exc_info=True,
                )
        req_body = message_to_json(GetTraceInfo(request_id=request_id))
        endpoint = get_trace_info_endpoint(request_id)
        response_proto = self._call_endpoint(GetTraceInfo, req_body, endpoint=endpoint)
        return TraceInfoV2.from_proto(response_proto.trace_info)

    def _is_databricks_tracking_uri(self):
        """
        Check if the tracking URI associated with this store is a Databricks URI.
        """
        from mlflow.tracking._tracking_service.utils import get_tracking_uri
        from mlflow.utils.uri import is_databricks_uri

        try:
            tracking_uri = get_tracking_uri()
            return is_databricks_uri(tracking_uri)
        except Exception:
            return False

    def get_online_trace_details(
        self,
        trace_id: str,
        sql_warehouse_id: str,
        source_inference_table: str,
        source_databricks_request_id: str,
    ):
        req = GetOnlineTraceDetails(
            trace_id=trace_id,
            sql_warehouse_id=sql_warehouse_id,
            source_inference_table=source_inference_table,
            source_databricks_request_id=source_databricks_request_id,
        )
        req_body = message_to_json(req)
        response_proto = self._call_endpoint(GetOnlineTraceDetails, req_body)
        return response_proto.trace_data

    def search_traces(
        self,
        experiment_ids: list[str],
        filter_string: Optional[str] = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
        model_id: Optional[str] = None,
        sql_warehouse_id: Optional[str] = None,
    ):
        if sql_warehouse_id is None:
            is_databricks = self._is_databricks_tracking_uri()

            if is_databricks:
                # Create trace_locations from experiment_ids for the V3 API
                trace_locations = []
                for exp_id in experiment_ids:
                    try:
                        location = TraceLocation.from_experiment_id(exp_id)
                        proto_location = location.to_proto()
                        trace_locations.append(proto_location)
                    except Exception as e:
                        raise MlflowException(
                            f"Invalid experiment ID format: {exp_id}. Error: {e!s}"
                        ) from e

                # Create V3 request message using protobuf
                request = SearchTracesV3Request(
                    locations=trace_locations,
                    filter=filter_string,
                    max_results=max_results,
                    order_by=order_by,
                    page_token=page_token,
                )

                req_body = message_to_json(request)
                endpoint = get_search_traces_v3_endpoint(is_databricks=is_databricks)

                try:
                    response_proto = self._call_endpoint(
                        SearchTracesV3Request, req_body, endpoint=endpoint
                    )
                    trace_infos = [TraceInfo.from_proto(t) for t in response_proto.traces]
                except Exception as e:
                    _logger.error(f"Error searching traces: {e!s}")
                    raise
            else:
                # Non-Databricks environment - use V2 request format
                request = SearchTraces(
                    experiment_ids=experiment_ids,
                    filter=filter_string,
                    max_results=max_results,
                    order_by=order_by,
                    page_token=page_token,
                )

                req_body = message_to_json(request)

                try:
                    response_proto = self._call_endpoint(SearchTraces, req_body)
                    trace_infos = [TraceInfoV2.from_proto(t).to_v3() for t in response_proto.traces]
                except Exception as e:
                    _logger.error(f"Error searching traces: {e!s}")
                    raise
        else:
            response_proto = self._search_unified_traces(
                model_id=model_id,
                sql_warehouse_id=sql_warehouse_id,
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
                page_token=page_token,
            )
            # Convert TraceInfo (v2) objects to TraceInfoV3 objects for consistency
            trace_infos = [TraceInfoV2.from_proto(t).to_v3() for t in response_proto.traces]
        return trace_infos, response_proto.next_page_token or None

    def _search_unified_traces(
        self,
        model_id: str,
        sql_warehouse_id: str,
        experiment_ids: list[str],
        filter_string: Optional[str] = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
    ):
        request = SearchUnifiedTraces(
            model_id=model_id,
            sql_warehouse_id=sql_warehouse_id,
            experiment_ids=experiment_ids,
            filter=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )
        req_body = message_to_json(request)
        return self._call_endpoint(SearchUnifiedTraces, req_body)

    def set_trace_tag(self, request_id: str, key: str, value: str):
        """
        Set a tag on the trace with the given request_id.

        Args:
            request_id: The ID of the trace.
            key: The string key of the tag.
            value: The string value of the tag.
        """
        # Always use v2 endpoint
        req_body = message_to_json(SetTraceTag(key=key, value=value))
        self._call_endpoint(
            SetTraceTag,
            req_body,
            endpoint=get_set_trace_tag_endpoint(request_id, is_databricks=False),
        )

    def delete_trace_tag(self, request_id: str, key: str):
        """
        Delete a tag on the trace with the given request_id.

        Args:
            request_id: The ID of the trace.
            key: The string key of the tag.
        """
        # Always use v2 endpoint
        req_body = message_to_json(DeleteTraceTag(key=key))
        self._call_endpoint(
            DeleteTraceTag,
            req_body,
            endpoint=get_set_trace_tag_endpoint(request_id, is_databricks=False),
        )

    def get_assessment(self, trace_id: str, assessment_id: str) -> Assessment:
        """
        Get an assessment entity from the backend store.
        """
        is_databricks = self._is_databricks_tracking_uri()
        req_body = message_to_json(
            GetAssessmentRequest(trace_id=trace_id, assessment_id=assessment_id)
        )
        response_proto = self._call_endpoint(
            GetAssessmentRequest,
            req_body,
            endpoint=get_single_assessment_endpoint(
                trace_id, assessment_id, is_databricks=is_databricks
            ),
        )
        return Assessment.from_proto(response_proto.assessment)

    def create_assessment(self, assessment: Assessment) -> Assessment:
        """
        Create an assessment entity in the backend store.

        Args:
            assessment: The assessment to log (without an assessment_id).

        Returns:
            The created Assessment object.
        """
        is_databricks = self._is_databricks_tracking_uri()
        req_body = message_to_json(CreateAssessment(assessment=assessment.to_proto()))
        response_proto = self._call_endpoint(
            CreateAssessment,
            req_body,
            endpoint=get_create_assessment_endpoint(
                assessment.trace_id, is_databricks=is_databricks
            ),
        )
        return Assessment.from_proto(response_proto.assessment)

    def update_assessment(
        self,
        trace_id: str,
        assessment_id: str,
        name: Optional[str] = None,
        expectation: Optional[Expectation] = None,
        feedback: Optional[Feedback] = None,
        rationale: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Assessment:
        """
        Update an existing assessment entity in the backend store.

        Args:
            trace_id: The ID of the trace.
            assessment_id: The ID of the assessment to update.
            name: The updated name of the assessment.
            expectation: The updated expectation value of the assessment.
            feedback: The updated feedback value of the assessment.
            rationale: The updated rationale of the feedback. Not applicable for expectations.
            metadata: Additional metadata for the assessment.
        """
        if expectation is not None and feedback is not None:
            raise MlflowException.invalid_parameter_value(
                "Exactly one of `expectation` or `feedback` should be specified."
            )

        is_databricks = self._is_databricks_tracking_uri()
        update = UpdateAssessment()

        # The assessment object to be sent to the backend (only contains fields to update and IDs)
        assessment = update.assessment
        # Field mask specifies which fields to update.
        mask = update.update_mask

        assessment.assessment_id = assessment_id
        assessment.trace_id = trace_id

        if name is not None:
            assessment.assessment_name = name
            mask.paths.append("assessment_name")
        if expectation is not None:
            assessment.expectation.CopyFrom(expectation.to_proto())
            mask.paths.append("expectation")
        if feedback is not None:
            assessment.feedback.CopyFrom(feedback.to_proto())
            mask.paths.append("feedback")
        if rationale is not None:
            assessment.rationale = rationale
            mask.paths.append("rationale")
        if metadata is not None:
            assessment.metadata.update(metadata)
            mask.paths.append("metadata")

        req_body = message_to_json(update)
        response_proto = self._call_endpoint(
            UpdateAssessment,
            req_body,
            endpoint=get_single_assessment_endpoint(
                trace_id, assessment_id, is_databricks=is_databricks
            ),
        )
        return Assessment.from_proto(response_proto.assessment)

    def delete_assessment(self, trace_id: str, assessment_id: str):
        """
        Delete an assessment associated with a trace.

        Args:
            trace_id: String ID of the trace.
            assessment_id: String ID of the assessment to delete.
        """
        is_databricks = self._is_databricks_tracking_uri()
        req_body = message_to_json(DeleteAssessment(trace_id=trace_id, assessment_id=assessment_id))
        self._call_endpoint(
            DeleteAssessment,
            req_body,
            endpoint=get_single_assessment_endpoint(
                trace_id, assessment_id, is_databricks=is_databricks
            ),
        )

    def log_metric(self, run_id: str, metric: Metric):
        """
        Log a metric for the specified run

        Args:
            run_id: String id for the run
            metric: Metric instance to log
        """
        req_body = message_to_json(
            LogMetric(
                run_uuid=run_id,
                run_id=run_id,
                key=metric.key,
                value=metric.value,
                timestamp=metric.timestamp,
                step=metric.step,
                model_id=metric.model_id,
                dataset_name=metric.dataset_name,
                dataset_digest=metric.dataset_digest,
            )
        )
        self._call_endpoint(LogMetric, req_body)

    def log_param(self, run_id, param):
        """
        Log a param for the specified run

        Args:
            run_id: String id for the run
            param: Param instance to log
        """
        req_body = message_to_json(
            LogParam(run_uuid=run_id, run_id=run_id, key=param.key, value=param.value)
        )
        self._call_endpoint(LogParam, req_body)

    def set_experiment_tag(self, experiment_id, tag):
        """
        Set a tag for the specified experiment

        Args:
            experiment_id: String ID of the experiment
            tag: ExperimentRunTag instance to log
        """
        req_body = message_to_json(
            SetExperimentTag(experiment_id=experiment_id, key=tag.key, value=tag.value)
        )
        self._call_endpoint(SetExperimentTag, req_body)

    def set_tag(self, run_id, tag):
        """
        Set a tag for the specified run

        Args:
            run_id: String ID of the run
            tag: RunTag instance to log
        """
        req_body = message_to_json(
            SetTag(run_uuid=run_id, run_id=run_id, key=tag.key, value=tag.value)
        )
        self._call_endpoint(SetTag, req_body)

    def delete_tag(self, run_id, key):
        """
        Delete a tag from a run. This is irreversible.

        Args:
            run_id: String ID of the run.
            key: Name of the tag.
        """
        req_body = message_to_json(DeleteTag(run_id=run_id, key=key))
        self._call_endpoint(DeleteTag, req_body)

    def get_metric_history(self, run_id, metric_key, max_results=None, page_token=None):
        """
        Return all logged values for a given metric.

        Args:
            run_id: Unique identifier for run.
            metric_key: Metric name within the run.
            max_results: Maximum number of metric history events (steps) to return per paged
                query. Only supported in 'databricks' backend.
            page_token: A Token specifying the next paginated set of results of metric history.

        Returns:
            A PagedList of :py:class:`mlflow.entities.Metric` entities if a paginated request
            is made by setting ``max_results`` to a value other than ``None``, a List of
            :py:class:`mlflow.entities.Metric` entities if ``max_results`` is None, else, if no
            metrics of the ``metric_key`` have been logged to the ``run_id``, an empty list.
        """
        req_body = message_to_json(
            GetMetricHistory(
                run_uuid=run_id,
                run_id=run_id,
                metric_key=metric_key,
                max_results=max_results,
                page_token=page_token,
            )
        )
        response_proto = self._call_endpoint(GetMetricHistory, req_body)

        metric_history = [Metric.from_proto(metric) for metric in response_proto.metrics]
        return PagedList(metric_history, response_proto.next_page_token or None)

    def _search_runs(
        self, experiment_ids, filter_string, run_view_type, max_results, order_by, page_token
    ):
        experiment_ids = [str(experiment_id) for experiment_id in experiment_ids]
        sr = SearchRuns(
            experiment_ids=experiment_ids,
            filter=filter_string,
            run_view_type=ViewType.to_proto(run_view_type),
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )
        req_body = message_to_json(sr)
        response_proto = self._call_endpoint(SearchRuns, req_body)
        runs = [Run.from_proto(proto_run) for proto_run in response_proto.runs]
        # If next_page_token is not set, we will see it as "". We need to convert this to None.
        next_page_token = None
        if response_proto.next_page_token:
            next_page_token = response_proto.next_page_token
        return runs, next_page_token

    def delete_run(self, run_id):
        req_body = message_to_json(DeleteRun(run_id=run_id))
        self._call_endpoint(DeleteRun, req_body)

    def restore_run(self, run_id):
        req_body = message_to_json(RestoreRun(run_id=run_id))
        self._call_endpoint(RestoreRun, req_body)

    def get_experiment_by_name(self, experiment_name):
        try:
            req_body = message_to_json(GetExperimentByName(experiment_name=experiment_name))
            response_proto = self._call_endpoint(GetExperimentByName, req_body)
            return Experiment.from_proto(response_proto.experiment)
        except MlflowException as e:
            if e.error_code == databricks_pb2.ErrorCode.Name(
                databricks_pb2.RESOURCE_DOES_NOT_EXIST
            ):
                return None
            else:
                raise

    def log_batch(self, run_id, metrics, params, tags):
        metric_protos = [metric.to_proto() for metric in metrics]
        param_protos = [param.to_proto() for param in params]
        tag_protos = [tag.to_proto() for tag in tags]
        req_body = message_to_json(
            LogBatch(metrics=metric_protos, params=param_protos, tags=tag_protos, run_id=run_id)
        )
        self._call_endpoint(LogBatch, req_body)

    def record_logged_model(self, run_id, mlflow_model):
        req_body = message_to_json(
            LogModel(run_id=run_id, model_json=json.dumps(mlflow_model.get_tags_dict()))
        )
        self._call_endpoint(LogModel, req_body)

    def create_logged_model(
        self,
        experiment_id: str,
        name: Optional[str] = None,
        source_run_id: Optional[str] = None,
        tags: Optional[list[LoggedModelTag]] = None,
        params: Optional[list[LoggedModelParameter]] = None,
        model_type: Optional[str] = None,
    ) -> LoggedModel:
        """
        Create a new logged model.

        Args:
            experiment_id: ID of the experiment to which the model belongs.
            name: Name of the model. If not specified, a random name will be generated.
            source_run_id: ID of the run that produced the model.
            tags: Tags to set on the model.
            params: Parameters to set on the model.
            model_type: Type of the model.

        Returns:
            The created model.
        """
        # Include the first 100 params in the initial request
        initial_params = []
        remaining_params = []
        if params:
            initial_batch_size = _MLFLOW_CREATE_LOGGED_MODEL_PARAMS_BATCH_SIZE.get()
            initial_params = params[:initial_batch_size]
            remaining_params = params[initial_batch_size:]

        req_body = message_to_json(
            CreateLoggedModel(
                experiment_id=experiment_id,
                name=name,
                model_type=model_type,
                source_run_id=source_run_id,
                params=[p.to_proto() for p in initial_params],
                tags=[t.to_proto() for t in tags or []],
            )
        )

        response_proto = self._call_endpoint(CreateLoggedModel, req_body)
        model = LoggedModel.from_proto(response_proto.model)

        # Log remaining params if there are any
        if remaining_params:
            self.log_logged_model_params(model_id=model.model_id, params=remaining_params)
            model = self.get_logged_model(model_id=model.model_id)

        return model

    def log_logged_model_params(self, model_id: str, params: list[LoggedModelParameter]) -> None:
        """
        Log parameters for a logged model in batches of 100.

        Args:
            model_id: ID of the model to log parameters for.
            params: List of parameters to log.

        Returns:
            None
        """
        # Process params in batches to avoid exceeding per-request backend limits
        batch_size = _MLFLOW_LOG_LOGGED_MODEL_PARAMS_BATCH_SIZE.get()
        endpoint = get_logged_model_endpoint(model_id)
        for i in range(0, len(params), batch_size):
            batch = params[i : i + batch_size]
            req_body = message_to_json(
                LogLoggedModelParamsRequest(
                    model_id=model_id,
                    params=[p.to_proto() for p in batch],
                )
            )
            self._call_endpoint(
                LogLoggedModelParamsRequest, json_body=req_body, endpoint=f"{endpoint}/params"
            )

    def get_logged_model(self, model_id: str) -> LoggedModel:
        """
        Fetch the logged model with the specified ID.

        Args:
            model_id: ID of the model to fetch.

        Returns:
            The fetched model.
        """
        endpoint = get_logged_model_endpoint(model_id)
        response_proto = self._call_endpoint(GetLoggedModel, endpoint=endpoint)
        return LoggedModel.from_proto(response_proto.model)

    def delete_logged_model(self, model_id) -> None:
        request = DeleteLoggedModel(model_id=model_id)
        endpoint = get_logged_model_endpoint(model_id)
        self._call_endpoint(
            DeleteLoggedModel, endpoint=endpoint, json_body=message_to_json(request)
        )

    def search_logged_models(
        self,
        experiment_ids: list[str],
        filter_string: Optional[str] = None,
        datasets: Optional[list[dict[str, Any]]] = None,
        max_results: Optional[int] = None,
        order_by: Optional[list[dict[str, Any]]] = None,
        page_token: Optional[str] = None,
    ) -> PagedList[LoggedModel]:
        """
        Search for logged models that match the specified search criteria.

        Args:
            experiment_ids: List of experiment ids to scope the search.
            filter_string: A search filter string.
            datasets: List of dictionaries to specify datasets on which to apply metrics filters.
                The following fields are supported:

                name (str): Required. Name of the dataset.
                digest (str): Optional. Digest of the dataset.
            max_results: Maximum number of logged models desired.
            order_by: List of dictionaries to specify the ordering of the search results.
                The following fields are supported:

                field_name (str): Required. Name of the field to order by, e.g. "metrics.accuracy".
                ascending: (bool): Optional. Whether the order is ascending or not.
                dataset_name: (str): Optional. If ``field_name`` refers to a metric, this field
                    specifies the name of the dataset associated with the metric. Only metrics
                    associated with the specified dataset name will be considered for ordering.
                    This field may only be set if ``field_name`` refers to a metric.
                dataset_digest (str): Optional. If ``field_name`` refers to a metric, this field
                    specifies the digest of the dataset associated with the metric. Only metrics
                    associated with the specified dataset name and digest will be considered for
                    ordering. This field may only be set if ``dataset_name`` is also set.
            page_token: Token specifying the next page of results.

        Returns:
            A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
            :py:class:`LoggedModel <mlflow.entities.LoggedModel>` objects.
        """
        req_body = message_to_json(
            SearchLoggedModels(
                experiment_ids=experiment_ids,
                filter=filter_string,
                datasets=[
                    SearchLoggedModels.Dataset(
                        dataset_name=d["dataset_name"],
                        dataset_digest=d.get("dataset_digest"),
                    )
                    for d in datasets or []
                ],
                max_results=max_results,
                order_by=[
                    SearchLoggedModels.OrderBy(
                        field_name=d["field_name"],
                        ascending=d.get("ascending", True),
                        dataset_name=d.get("dataset_name"),
                        dataset_digest=d.get("dataset_digest"),
                    )
                    for d in order_by or []
                ],
                page_token=page_token,
            )
        )
        response_proto = self._call_endpoint(SearchLoggedModels, req_body)
        models = [LoggedModel.from_proto(x) for x in response_proto.models]
        return PagedList(models, response_proto.next_page_token or None)

    def finalize_logged_model(self, model_id: str, status: LoggedModelStatus) -> LoggedModel:
        """
        Finalize a model by updating its status.

        Args:
            model_id: ID of the model to finalize.
            status: Final status to set on the model.

        Returns:
            The updated model.
        """
        endpoint = get_logged_model_endpoint(model_id)
        json_body = message_to_json(
            FinalizeLoggedModel(model_id=model_id, status=status.to_proto())
        )
        response_proto = self._call_endpoint(
            FinalizeLoggedModel, json_body=json_body, endpoint=endpoint
        )
        return LoggedModel.from_proto(response_proto.model)

    def set_logged_model_tags(self, model_id: str, tags: list[LoggedModelTag]) -> None:
        """
        Set tags on the specified logged model.

        Args:
            model_id: ID of the model.
            tags: Tags to set on the model.

        Returns:
            None
        """
        endpoint = get_logged_model_endpoint(model_id)
        json_body = message_to_json(SetLoggedModelTags(tags=[tag.to_proto() for tag in tags]))
        self._call_endpoint(SetLoggedModelTags, json_body=json_body, endpoint=f"{endpoint}/tags")

    def delete_logged_model_tag(self, model_id: str, key: str) -> None:
        """
        Delete a tag from the specified logged model.

        Args:
            model_id: ID of the model.
            key: Key of the tag to delete.

        Returns:
            The model with the specified tag removed.
        """
        endpoint = get_logged_model_endpoint(model_id)
        self._call_endpoint(DeleteLoggedModelTag, endpoint=f"{endpoint}/tags/{key}")

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
            models: List of :py:class:`mlflow.entities.LoggedModelInput` instances to log.

        Returns:
            None.
        """
        datasets_protos = [dataset.to_proto() for dataset in datasets or []]
        models_protos = [model.to_proto() for model in models or []]
        req_body = message_to_json(
            LogInputs(
                run_id=run_id,
                datasets=datasets_protos,
                models=models_protos,
            )
        )
        self._call_endpoint(LogInputs, req_body)

    def log_outputs(self, run_id: str, models: list[LoggedModelOutput]):
        """
        Log outputs, such as models, to the specified run.

        Args:
            run_id: String id for the run
            models: List of :py:class:`mlflow.entities.LoggedModelOutput` instances to log
                as outputs of the run.

        Returns:
            None.
        """
        req_body = message_to_json(LogOutputs(run_id=run_id, models=[m.to_proto() for m in models]))
        self._call_endpoint(LogOutputs, req_body)
