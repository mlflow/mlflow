import functools
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.entities import DatasetRecord, EvaluationDataset

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from packaging.version import Version

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
    ScorerVersion,
    ViewType,
)

# Constants for Databricks API disabled decorator
_DATABRICKS_DATASET_API_NAME = "Evaluation dataset APIs"
_DATABRICKS_DATASET_ALTERNATIVE = "Use the databricks-agents library for dataset operations."
from mlflow.entities.assessment import Assessment, Expectation, Feedback
from mlflow.entities.span import Span
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
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
    AddDatasetToExperiments,
    CalculateTraceFilterCorrelation,
    CreateAssessment,
    CreateDataset,
    CreateExperiment,
    CreateLoggedModel,
    CreateRun,
    DeleteAssessment,
    DeleteDataset,
    DeleteDatasetTag,
    DeleteExperiment,
    DeleteExperimentTag,
    DeleteLoggedModel,
    DeleteLoggedModelTag,
    DeleteRun,
    DeleteScorer,
    DeleteTag,
    DeleteTraces,
    DeleteTraceTag,
    EndTrace,
    FinalizeLoggedModel,
    GetAssessmentRequest,
    GetDataset,
    GetDatasetExperimentIds,
    GetDatasetRecords,
    GetExperiment,
    GetExperimentByName,
    GetLoggedModel,
    GetMetricHistory,
    GetRun,
    GetScorer,
    GetTraceInfo,
    GetTraceInfoV3,
    LinkTracesToRun,
    ListScorers,
    ListScorerVersions,
    LogBatch,
    LogInputs,
    LogLoggedModelParamsRequest,
    LogMetric,
    LogModel,
    LogOutputs,
    LogParam,
    MlflowService,
    RegisterScorer,
    RemoveDatasetFromExperiments,
    RestoreExperiment,
    RestoreRun,
    SearchEvaluationDatasets,
    SearchExperiments,
    SearchLoggedModels,
    SearchRuns,
    SearchTraces,
    SearchTracesV3,
    SetDatasetTags,
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
    UpsertDatasetRecords,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.analysis import TraceFilterCorrelationResult
from mlflow.tracing.utils.otlp import MLFLOW_EXPERIMENT_ID_HEADER, OTLP_TRACES_PATH
from mlflow.utils.databricks_utils import databricks_api_disabled
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    _V3_TRACE_REST_API_PATH_PREFIX,
    MlflowHostCreds,
    call_endpoint,
    extract_api_info_for_service,
    get_logged_model_endpoint,
    get_single_assessment_endpoint,
    get_single_trace_endpoint,
    get_trace_tag_endpoint,
    http_request,
    verify_rest_response,
)
from mlflow.utils.validation import _resolve_experiment_ids_and_locations

_logger = logging.getLogger(__name__)


class RestStore(AbstractStore):
    """
    Client for a remote tracking server accessed via REST API calls

    Args
        get_host_creds: Method to be invoked prior to every REST request to get the
            :py:class:`mlflow.rest_utils.MlflowHostCreds` for the request. Note that this
            is a function so that we can obtain fresh credentials in the case of expiry.
    """

    _METHOD_TO_INFO = extract_api_info_for_service(MlflowService, _REST_API_PATH_PREFIX)

    def __init__(self, get_host_creds):
        super().__init__()
        self.get_host_creds = get_host_creds

    @staticmethod
    @functools.lru_cache
    def _get_server_version(host_creds: MlflowHostCreds) -> Version | None:
        """
        Get the MLflow server version with caching.

        Args:
            host_creds: MlflowHostCreds object

        Returns:
            Version object if successful, None if failed to retrieve version.
        """
        try:
            response = http_request(
                host_creds=host_creds,
                endpoint="/version",
                method="GET",
                timeout=3,  # Short timeout to fail fast if server version API isn't available
                # Disable non-DB SDK retries; default retry policy takes minutes, which is too long
                max_retries=0,
                # Approximately disable DB SDK retries (0 is interpreted as 'unspecified', so use 1)
                retry_timeout_seconds=1,
                raise_on_status=True,
            )
            return Version(response.text)
        except Exception as e:
            _logger.debug(f"Failed to retrieve server version: {e}")

        return None

    def _call_endpoint(
        self,
        api,
        json_body=None,
        endpoint=None,
        retry_timeout_seconds=None,
        response_proto=None,
    ):
        if endpoint:
            # Allow customizing the endpoint for compatibility with dynamic endpoints, such as
            # /mlflow/traces/{trace_id}/info.
            _, method = self._METHOD_TO_INFO[api]
        else:
            endpoint, method = self._METHOD_TO_INFO[api]
        response_proto = response_proto or api.Response()
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

    def start_trace(self, trace_info: TraceInfo) -> TraceInfo:
        """
        Create a new trace using the V3 API format.

        NB: The backend API is named "StartTraceV3" for some internal reason, but actually
        it is supposed to be called at the end of the trace.

        Args:
            trace_info: The TraceInfo object to create in the backend.

        Returns:
            The returned TraceInfo object from the backend.
        """
        # NB: The Databricks backend expects a Trace object, not a TraceInfo object, although
        # it doesn't use the data field at all. Trace data increases the payload size significantly,
        # so we create a Trace object with an empty data field here.
        trace = Trace(info=trace_info, data=TraceData(spans=[]))
        req_body = message_to_json(StartTraceV3(trace=trace.to_proto()))

        try:
            response_proto = self._call_endpoint(
                # NB: _call_endpoint doesn't handle versioning between v2 and v3 endpoint
                # yet, so manually passing the v3 endpoint here.
                StartTraceV3,
                req_body,
                endpoint=_V3_TRACE_REST_API_PATH_PREFIX,
                retry_timeout_seconds=MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT.get(),
            )
            return TraceInfo.from_proto(response_proto.trace.trace_info)
        except MlflowException as e:
            if e.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.ENDPOINT_NOT_FOUND):
                _logger.debug(
                    "Server does not support StartTraceV3 API yet. Falling back to V2 API."
                )
                return self._create_trace_v2_fallback(trace_info)
            raise

    def _create_trace_v2_fallback(self, trace_info: TraceInfo) -> TraceInfo:
        """
        Create a new trace using the V2 API format. This is a fallback for the case where the
        client is v3 but the tracking server does not support v3 yet(<= 3.2.0).
        """
        trace_info_v2 = self.deprecated_start_trace_v2(
            experiment_id=trace_info.experiment_id,
            timestamp_ms=trace_info.request_time,
            request_metadata=trace_info.trace_metadata,
            tags=trace_info.tags,
        )
        self.deprecated_end_trace_v2(
            request_id=trace_info_v2.request_id,
            timestamp_ms=trace_info.request_time + trace_info.execution_duration,
            status=trace_info.status,
            request_metadata=trace_info.trace_metadata,
            tags=trace_info.tags,
        )
        return trace_info_v2.to_v3()

    def _delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: int | None = None,
        max_traces: int | None = None,
        trace_ids: list[str] | None = None,
    ) -> int:
        req_body = message_to_json(
            DeleteTraces(
                experiment_id=experiment_id,
                max_timestamp_millis=max_timestamp_millis,
                max_traces=max_traces,
                request_ids=trace_ids,
            )
        )
        res = self._call_endpoint(DeleteTraces, req_body)
        return res.traces_deleted

    def get_trace_info(self, trace_id: str) -> TraceInfo:
        """
        Get the trace matching the `trace_id`.

        Args:
            trace_id: String id of the trace to fetch.

        Returns:
            The fetched Trace object, of type ``mlflow.entities.TraceInfo``.
        """
        trace_v3_req_body = message_to_json(GetTraceInfoV3(trace_id=trace_id))
        trace_v3_endpoint = get_single_trace_endpoint(trace_id)
        try:
            trace_v3_response_proto = self._call_endpoint(
                GetTraceInfoV3, trace_v3_req_body, endpoint=trace_v3_endpoint
            )
            return TraceInfo.from_proto(trace_v3_response_proto.trace.trace_info)
        except MlflowException as e:
            # If the tracking server does not support V3 trace API yet, fallback to V2 API.
            if e.error_code != databricks_pb2.ErrorCode.Name(databricks_pb2.ENDPOINT_NOT_FOUND):
                raise
            _logger.debug("Server does not support GetTraceInfoV3 API yet. Falling back to V2 API.")

        req_body = message_to_json(GetTraceInfo(request_id=trace_id))
        endpoint = get_single_trace_endpoint(trace_id, use_v3=False)
        response_proto = self._call_endpoint(GetTraceInfo, req_body, endpoint=endpoint)
        return TraceInfoV2.from_proto(response_proto.trace_info).to_v3()

    def search_traces(
        self,
        experiment_ids: list[str] | None = None,
        filter_string: str | None = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: list[str] | None = None,
        page_token: str | None = None,
        model_id: str | None = None,
        locations: list[str] | None = None,
    ):
        locations = _resolve_experiment_ids_and_locations(experiment_ids, locations)

        if model_id is not None:
            raise MlflowException.invalid_parameter_value(
                "Searching traces by model_id is not supported on the current tracking server.",
            )

        return self._search_traces(
            locations=locations,
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )

    def _search_traces(
        self,
        locations: list[str],
        filter_string: str | None = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> tuple[list[TraceInfo], str | None]:
        # Create trace_locations from experiment_ids for the V3 API
        trace_locations = []
        for exp_id in locations:
            try:
                location = TraceLocation.from_experiment_id(exp_id)
                proto_location = location.to_proto()
                trace_locations.append(proto_location)
            except Exception as e:
                raise MlflowException(
                    f"Invalid experiment ID format: {exp_id}. Error: {e!s}"
                ) from e

        # Create V3 request message using protobuf
        request = SearchTracesV3(
            locations=trace_locations,
            filter=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )

        req_body = message_to_json(request)
        v3_endpoint = f"{_V3_TRACE_REST_API_PATH_PREFIX}/search"

        try:
            response_proto = self._call_endpoint(SearchTracesV3, req_body, v3_endpoint)
        except MlflowException as e:
            if e.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.ENDPOINT_NOT_FOUND):
                _logger.debug(
                    "Server does not support SearchTracesV3 API yet. Falling back to V2 API."
                )
                response_proto = self._call_endpoint(SearchTraces, req_body)
            else:
                raise

        trace_infos = [TraceInfo.from_proto(t) for t in response_proto.traces]
        return trace_infos, response_proto.next_page_token or None

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

        Returns:
            TraceFilterCorrelationResult containing NPMI analytics data.
        """
        request = CalculateTraceFilterCorrelation(
            experiment_ids=experiment_ids,
            filter_string1=filter_string1,
            filter_string2=filter_string2,
        )
        if base_filter is not None:
            request.base_filter = base_filter

        req_body = message_to_json(request)
        v3_endpoint = f"{_V3_TRACE_REST_API_PATH_PREFIX}/calculate-filter-correlation"
        response_proto = self._call_endpoint(CalculateTraceFilterCorrelation, req_body, v3_endpoint)
        return TraceFilterCorrelationResult.from_proto(response_proto)

    def set_trace_tag(self, trace_id: str, key: str, value: str):
        """
        Set a tag on the trace with the given trace_id.

        Args:
            trace_id: The ID of the trace.
            key: The string key of the tag.
            value: The string value of the tag.
        """
        # Always use v2 endpoint
        req_body = message_to_json(SetTraceTag(key=key, value=value))
        self._call_endpoint(SetTraceTag, req_body, endpoint=get_trace_tag_endpoint(trace_id))

    def delete_trace_tag(self, trace_id: str, key: str):
        """
        Delete a tag on the trace with the given trace_id.

        Args:
            trace_id: The ID of the trace.
            key: The string key of the tag.
        """
        # Always use v2 endpoint
        req_body = message_to_json(DeleteTraceTag(key=key))
        self._call_endpoint(DeleteTraceTag, req_body, endpoint=get_trace_tag_endpoint(trace_id))

    def get_assessment(self, trace_id: str, assessment_id: str) -> Assessment:
        """
        Get an assessment entity from the backend store.
        """
        req_body = message_to_json(
            GetAssessmentRequest(trace_id=trace_id, assessment_id=assessment_id)
        )
        response_proto = self._call_endpoint(
            GetAssessmentRequest,
            req_body,
            endpoint=get_single_assessment_endpoint(trace_id, assessment_id),
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
        req_body = message_to_json(CreateAssessment(assessment=assessment.to_proto()))
        response_proto = self._call_endpoint(
            CreateAssessment,
            req_body,
            endpoint=f"{_V3_TRACE_REST_API_PATH_PREFIX}/{assessment.trace_id}/assessments",
        )
        return Assessment.from_proto(response_proto.assessment)

    def update_assessment(
        self,
        trace_id: str,
        assessment_id: str,
        name: str | None = None,
        expectation: Expectation | None = None,
        feedback: Feedback | None = None,
        rationale: str | None = None,
        metadata: dict[str, str] | None = None,
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
            endpoint=get_single_assessment_endpoint(trace_id, assessment_id),
        )
        return Assessment.from_proto(response_proto.assessment)

    def delete_assessment(self, trace_id: str, assessment_id: str):
        """
        Delete an assessment associated with a trace.

        Args:
            trace_id: String ID of the trace.
            assessment_id: String ID of the assessment to delete.
        """
        req_body = message_to_json(DeleteAssessment(trace_id=trace_id, assessment_id=assessment_id))
        self._call_endpoint(
            DeleteAssessment,
            req_body,
            endpoint=get_single_assessment_endpoint(trace_id, assessment_id),
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

    def delete_experiment_tag(self, experiment_id, key):
        """
        Delete a tag from the specified experiment

        Args:
            experiment_id: String ID of the experiment
            key: String name of the tag to be deleted
        """
        req_body = message_to_json(DeleteExperimentTag(experiment_id=experiment_id, key=key))
        self._call_endpoint(DeleteExperimentTag, req_body)

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
        name: str | None = None,
        source_run_id: str | None = None,
        tags: list[LoggedModelTag] | None = None,
        params: list[LoggedModelParameter] | None = None,
        model_type: str | None = None,
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
        filter_string: str | None = None,
        datasets: list[dict[str, Any]] | None = None,
        max_results: int | None = None,
        order_by: list[dict[str, Any]] | None = None,
        page_token: str | None = None,
    ) -> PagedList[LoggedModel]:
        """
        Search for logged models that match the specified search criteria.

        Args:
            experiment_ids: List of experiment ids to scope the search.
            filter_string: A search filter string.
            datasets: List of dictionaries to specify datasets on which to apply metrics filters.
                The following fields are supported:

                dataset_name (str): Required. Name of the dataset.
                dataset_digest (str): Optional. Digest of the dataset.
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
        datasets: list[DatasetInput] | None = None,
        models: list[LoggedModelInput] | None = None,
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

    ############################################################################################
    # Scorer Management APIs
    ############################################################################################

    def register_scorer(self, experiment_id: str, name: str, serialized_scorer: str) -> int:
        """
        Register a scorer for an experiment.

        Args:
            experiment_id: String ID of the experiment.
            name: String name of the scorer.
            serialized_scorer: String containing the serialized scorer data.

        Returns:
            The new version number for the scorer.
        """
        req_body = message_to_json(
            RegisterScorer(
                experiment_id=experiment_id,
                name=name,
                serialized_scorer=serialized_scorer,
            )
        )
        # Scorer APIs are v3.0 endpoints
        response_proto = self._call_endpoint(
            RegisterScorer,
            req_body,
            endpoint="/api/3.0/mlflow/scorers/register",
        )
        return response_proto.version

    def list_scorers(self, experiment_id: str) -> list[ScorerVersion]:
        """
        List all scorers for an experiment (latest version for each scorer name).

        Args:
            experiment_id: String ID of the experiment.

        Returns:
            List of Scorer entities.
        """
        req_body = message_to_json(ListScorers(experiment_id=experiment_id))
        # Scorer APIs are v3.0 endpoints
        response_proto = self._call_endpoint(
            ListScorers,
            req_body,
            endpoint="/api/3.0/mlflow/scorers/list",
        )
        return [ScorerVersion.from_proto(scorer) for scorer in response_proto.scorers]

    def list_scorer_versions(self, experiment_id: str, name: str) -> list[ScorerVersion]:
        """
        List all versions of a specific scorer for an experiment.

        Args:
            experiment_id: String ID of the experiment.
            name: String name of the scorer.

        Returns:
            List of Scorer entities for all versions.
        """
        req_body = message_to_json(ListScorerVersions(experiment_id=experiment_id, name=name))
        # Scorer APIs are v3.0 endpoints
        response_proto = self._call_endpoint(
            ListScorerVersions,
            req_body,
            endpoint="/api/3.0/mlflow/scorers/versions",
        )
        return [ScorerVersion.from_proto(scorer) for scorer in response_proto.scorers]

    def get_scorer(
        self, experiment_id: str, name: str, version: int | None = None
    ) -> ScorerVersion:
        """
        Get a specific scorer for an experiment.

        Args:
            experiment_id: String ID of the experiment.
            name: String name of the scorer.
            version: Integer version of the scorer. If None, returns the scorer
               with maximum version.

        Returns:
            A ScorerVersion entity object.
        """
        req_body = message_to_json(
            GetScorer(experiment_id=experiment_id, name=name, version=version)
        )
        # Scorer APIs are v3.0 endpoints
        response_proto = self._call_endpoint(
            GetScorer,
            req_body,
            endpoint="/api/3.0/mlflow/scorers/get",
        )
        return ScorerVersion.from_proto(response_proto.scorer)

    def delete_scorer(self, experiment_id: str, name: str, version: int | None = None) -> None:
        """
        Delete a scorer for an experiment.

        Args:
            experiment_id: String ID of the experiment.
            name: String name of the scorer.
            version: Integer version of the scorer to delete. If None, deletes all versions.

        Returns:
            None.
        """
        req_body = message_to_json(
            DeleteScorer(experiment_id=experiment_id, name=name, version=version)
        )
        # Scorer APIs are v3.0 endpoints
        self._call_endpoint(
            DeleteScorer,
            req_body,
            endpoint="/api/3.0/mlflow/scorers/delete",
        )

    ############################################################################################
    # Deprecated MLflow Tracing APIs. Kept for backward compatibility but do not use.
    ############################################################################################
    def deprecated_start_trace_v2(
        self,
        experiment_id: str,
        timestamp_ms: int,
        request_metadata: dict[str, str],
        tags: dict[str, str],
    ) -> TraceInfoV2:
        """
        DEPRECATED. DO NOT USE.

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
        endpoint = f"{_REST_API_PATH_PREFIX}/mlflow/traces/{request_id}"
        response_proto = self._call_endpoint(EndTrace, req_body, endpoint=endpoint)
        return TraceInfoV2.from_proto(response_proto.trace_info)

    @databricks_api_disabled(_DATABRICKS_DATASET_API_NAME, _DATABRICKS_DATASET_ALTERNATIVE)
    def create_dataset(
        self,
        name: str,
        tags: dict[str, str] | None = None,
        experiment_ids: list[str] | None = None,
    ) -> "EvaluationDataset":
        """
        Create an evaluation dataset.

        Args:
            name: The name of the evaluation dataset.
            tags: Optional tags to associate with the dataset.
            experiment_ids: List of experiment IDs to associate with the dataset.

        Returns:
            The created EvaluationDataset.
        """
        from mlflow.entities import EvaluationDataset

        req = CreateDataset(
            name=name,
            experiment_ids=experiment_ids or [],
        )

        if tags:
            req.tags = json.dumps(tags)

        req_body = message_to_json(req)
        # Dataset APIs are v3.0 endpoints
        response_proto = self._call_endpoint(
            CreateDataset, req_body, endpoint="/api/3.0/mlflow/datasets/create"
        )
        return EvaluationDataset.from_proto(response_proto.dataset)

    @databricks_api_disabled(_DATABRICKS_DATASET_API_NAME, _DATABRICKS_DATASET_ALTERNATIVE)
    def get_dataset(self, dataset_id: str) -> "EvaluationDataset":
        """
        Get an evaluation dataset by ID.

        Args:
            dataset_id: The ID of the dataset to retrieve.

        Returns:
            The EvaluationDataset object.
        """
        from mlflow.entities import EvaluationDataset

        # GetDataset uses path parameter, not request body
        # Dataset APIs are v3.0 endpoints
        response_proto = self._call_endpoint(
            GetDataset, None, endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}"
        )
        return EvaluationDataset.from_proto(response_proto.dataset)

    @databricks_api_disabled(_DATABRICKS_DATASET_API_NAME, _DATABRICKS_DATASET_ALTERNATIVE)
    def delete_dataset(self, dataset_id: str) -> None:
        """
        Delete an evaluation dataset.

        Args:
            dataset_id: The ID of the dataset to delete.
        """
        # DeleteDataset uses path parameter, not request body
        # Dataset APIs are v3.0 endpoints
        self._call_endpoint(DeleteDataset, None, endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}")

    @databricks_api_disabled(_DATABRICKS_DATASET_API_NAME, _DATABRICKS_DATASET_ALTERNATIVE)
    def search_datasets(
        self,
        experiment_ids: list[str] | None = None,
        filter_string: str | None = None,
        max_results: int = 1000,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList["EvaluationDataset"]:
        """
        Search for evaluation datasets.

        Args:
            experiment_ids: List of experiment IDs to filter by.
            filter_string: Filter string for dataset names.
            max_results: Maximum number of results to return.
            order_by: Ordering criteria.
            page_token: Token for retrieving the next page of results.

        Returns:
            A PagedList of evaluation datasets.
        """
        from mlflow.entities import EvaluationDataset

        req = SearchEvaluationDatasets(
            experiment_ids=experiment_ids or [],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by or [],
            page_token=page_token,
        )
        req_body = message_to_json(req)
        # Dataset APIs are v3.0 endpoints
        response_proto = self._call_endpoint(
            SearchEvaluationDatasets, req_body, endpoint="/api/3.0/mlflow/datasets/search"
        )
        datasets = [EvaluationDataset.from_proto(ds) for ds in response_proto.datasets]
        return PagedList(datasets, response_proto.next_page_token)

    @databricks_api_disabled(_DATABRICKS_DATASET_API_NAME, _DATABRICKS_DATASET_ALTERNATIVE)
    def upsert_dataset_records(
        self, dataset_id: str, records: list[dict[str, Any]]
    ) -> dict[str, int]:
        """
        Upsert evaluation dataset records.

        Args:
            dataset_id: The ID of the dataset.
            records: List of record dictionaries to upsert.

        Returns:
            Dictionary with 'inserted' and 'updated' counts.
        """
        req = UpsertDatasetRecords(
            records=json.dumps(records),
        )
        req_body = message_to_json(req)
        # Dataset APIs are v3.0 endpoints - dataset_id is in path
        response_proto = self._call_endpoint(
            UpsertDatasetRecords,
            req_body,
            endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}/records",
        )
        return {
            "inserted": response_proto.inserted_count,
            "updated": response_proto.updated_count,
        }

    @databricks_api_disabled(_DATABRICKS_DATASET_API_NAME, _DATABRICKS_DATASET_ALTERNATIVE)
    def set_dataset_tags(self, dataset_id: str, tags: dict[str, Any]) -> None:
        """
        Set tags for an evaluation dataset.

        This implements an upsert operation - existing tags are merged with new tags.

        Args:
            dataset_id: The ID of the dataset to update.
            tags: Dictionary of tags to update.
        """
        req = SetDatasetTags(
            tags=json.dumps(tags),
        )
        req_body = message_to_json(req)
        # Dataset APIs are v3.0 endpoints - dataset_id is in path
        self._call_endpoint(
            SetDatasetTags, req_body, endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}/tags"
        )

    @databricks_api_disabled(_DATABRICKS_DATASET_API_NAME, _DATABRICKS_DATASET_ALTERNATIVE)
    def delete_dataset_tag(self, dataset_id: str, key: str) -> None:
        """
        Delete a tag from an evaluation dataset.

        Args:
            dataset_id: The ID of the dataset.
            key: The tag key to delete.
        """
        # DeleteDatasetTag uses path parameters for both dataset_id and key
        # Dataset APIs are v3.0 endpoints
        self._call_endpoint(
            DeleteDatasetTag, None, endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}/tags/{key}"
        )

    @databricks_api_disabled(_DATABRICKS_DATASET_API_NAME, _DATABRICKS_DATASET_ALTERNATIVE)
    def get_dataset_experiment_ids(self, dataset_id: str) -> list[str]:
        """
        Get experiment IDs associated with an evaluation dataset.

        Args:
            dataset_id: The ID of the dataset.

        Returns:
            List of experiment IDs associated with the dataset.
        """
        # GetDatasetExperimentIds uses path parameter
        # Dataset APIs are v3.0 endpoints
        response_proto = self._call_endpoint(
            GetDatasetExperimentIds,
            None,
            endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}/experiment-ids",
        )
        return list(response_proto.experiment_ids)

    def _load_dataset_records(
        self, dataset_id: str, max_results: int | None = None, page_token: str | None = None
    ) -> tuple["list[DatasetRecord]", str | None]:
        """
        Load dataset records with pagination support.

        Args:
            dataset_id: The ID of the dataset.
            max_results: Maximum number of records to return. If None, returns all records.
            page_token: Token for pagination. If None, starts from the beginning.

        Returns:
            Tuple of (list of DatasetRecord objects, next_page_token).
            next_page_token is None if there are no more records.
        """
        from mlflow.entities.dataset_record import DatasetRecord

        if max_results is None:
            # No pagination requested - fetch all records
            all_records = []
            current_page_token = page_token

            while True:
                req = GetDatasetRecords(max_results=1000)
                if current_page_token:
                    req.page_token = current_page_token

                req_body = message_to_json(req)
                response_proto = self._call_endpoint(
                    GetDatasetRecords,
                    req_body,
                    endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}/records",
                )

                if response_proto.records:
                    records_dicts = json.loads(response_proto.records)
                    for record_dict in records_dicts:
                        all_records.append(DatasetRecord.from_dict(record_dict))

                if response_proto.next_page_token:
                    current_page_token = response_proto.next_page_token
                else:
                    break

            return all_records, None
        else:
            # Paginated request - fetch only requested page
            req = GetDatasetRecords(max_results=max_results)
            if page_token:
                req.page_token = page_token

            req_body = message_to_json(req)
            response_proto = self._call_endpoint(
                GetDatasetRecords,
                req_body,
                endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}/records",
            )

            records = []
            if response_proto.records:
                records_dicts = json.loads(response_proto.records)
                for record_dict in records_dicts:
                    records.append(DatasetRecord.from_dict(record_dict))

            next_page_token = (
                response_proto.next_page_token if response_proto.next_page_token else None
            )
            return records, next_page_token

    def link_traces_to_run(self, trace_ids: list[str], run_id: str) -> None:
        """
        Link multiple traces to a run by creating entity associations.

        Args:
            trace_ids: List of trace IDs to link to the run. Maximum 100 traces allowed.
            run_id: ID of the run to link traces to.

        Raises:
            MlflowException: If more than 100 traces are provided.
        """
        req_body = message_to_json(
            LinkTracesToRun(
                trace_ids=trace_ids,
                run_id=run_id,
            )
        )
        self._call_endpoint(LinkTracesToRun, req_body)

    def add_dataset_to_experiments(
        self, dataset_id: str, experiment_ids: list[str]
    ) -> "EvaluationDataset":
        """
        Add a dataset to additional experiments via REST API.
        """
        req_body = message_to_json(
            AddDatasetToExperiments(
                dataset_id=dataset_id,
                experiment_ids=experiment_ids,
            )
        )
        response = self._call_endpoint(AddDatasetToExperiments, req_body)
        return EvaluationDataset.from_proto(response.dataset)

    def remove_dataset_from_experiments(
        self, dataset_id: str, experiment_ids: list[str]
    ) -> "EvaluationDataset":
        """
        Remove a dataset from experiments via REST API.
        """
        req_body = message_to_json(
            RemoveDatasetFromExperiments(
                dataset_id=dataset_id,
                experiment_ids=experiment_ids,
            )
        )
        response = self._call_endpoint(RemoveDatasetFromExperiments, req_body)
        return EvaluationDataset.from_proto(response.dataset)

    def log_spans(self, location: str, spans: list[Span], tracking_uri=None) -> list[Span]:
        """
        Log multiple span entities to the tracking store via the OTel API.

        Args:
            location: The location to log spans to. It should be experiment ID of an MLflow
                experiment.
            spans: List of Span entities to log. All spans must belong to the same trace.
            tracking_uri: The tracking URI to use. Default to None.

        Returns:
            List of logged Span entities.

        Raises:
            MlflowException: If spans belong to different traces or the OTel API call fails.
        """
        if not spans:
            return []

        server_version = self._get_server_version(self.get_host_creds())
        if server_version is None:
            raise NotImplementedError(
                "log_spans is not supported: could not identify MLflow server version"
            )
        elif server_version < Version("3.4"):
            raise NotImplementedError(
                f"log_spans is not supported: MLflow server version {server_version} is"
                f" less than 3.4"
            )

        trace_ids = {span.trace_id for span in spans}
        if len(trace_ids) > 1:
            raise MlflowException(
                f"All spans must belong to the same trace. Found trace IDs: {trace_ids}",
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )

        request = ExportTraceServiceRequest()
        resource_spans = request.resource_spans.add()
        scope_spans = resource_spans.scope_spans.add()
        scope_spans.spans.extend(span.to_otel_proto() for span in spans)

        response = http_request(
            host_creds=self.get_host_creds(),
            endpoint=OTLP_TRACES_PATH,
            method="POST",
            data=request.SerializeToString(),
            extra_headers={
                "Content-Type": "application/x-protobuf",
                MLFLOW_EXPERIMENT_ID_HEADER: location,
            },
        )

        verify_rest_response(response, OTLP_TRACES_PATH)
        return spans

    async def log_spans_async(self, location: str, spans: list[Span]) -> list[Span]:
        """
        Async wrapper for log_spans method.

        Args:
            location: The location to log spans to. It should be experiment ID of an MLflow
                experiment.
            spans: List of Span entities to log. All spans must belong to the same trace.

        Returns:
            List of logged Span entities.

        Raises:
            MlflowException: If spans belong to different traces or the OTel API call fails.
        """
        return self.log_spans(location, spans)
