import logging
from collections import defaultdict
from urllib.parse import quote

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from mlflow.entities import Assessment, Span, Trace, TraceInfo, TraceLocation
from mlflow.entities.assessment import ExpectationValue, FeedbackValue
from mlflow.entities.trace_location import UCSchemaLocation as UCSchemaLocationEntity
from mlflow.environment_variables import (
    MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT,
    MLFLOW_TRACING_SQL_WAREHOUSE_ID,
)
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.databricks_pb2 import (
    ALREADY_EXISTS,
    BAD_REQUEST,
    ENDPOINT_NOT_FOUND,
    INVALID_PARAMETER_VALUE,
    ErrorCode,
)
from mlflow.protos.databricks_tracing_pb2 import Assessment as ProtoAssessment
from mlflow.protos.databricks_tracing_pb2 import (
    BatchGetTraces,
    BatchLinkTraceToRun,
    BatchUnlinkTraceFromRun,
    CreateAssessment,
    CreateTraceInfo,
    CreateTraceUCStorageLocation,
    DatabricksTrackingService,
    DeleteAssessment,
    DeleteTraceTag,
    GetAssessment,
    GetTraceInfo,
    LinkExperimentToUCTraceLocation,
    SearchTraces,
    SetTraceTag,
    UnLinkExperimentToUCTraceLocation,
    UpdateAssessment,
)
from mlflow.protos.databricks_tracing_pb2 import TraceInfo as ProtoTraceInfo
from mlflow.protos.service_pb2 import GetOnlineTraceDetails, MlflowService, SearchUnifiedTraces
from mlflow.store.tracking import SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracing.utils import parse_trace_id_v4
from mlflow.tracing.utils.otlp import OTLP_TRACES_PATH
from mlflow.utils.databricks_tracing_utils import (
    assessment_to_proto,
    trace_from_proto,
    trace_location_to_proto,
    uc_schema_location_from_proto,
    uc_schema_location_to_proto,
)
from mlflow.utils.databricks_utils import get_databricks_workspace_client_config
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    _V4_REST_API_PATH_PREFIX,
    _V4_TRACE_REST_API_PATH_PREFIX,
    extract_api_info_for_service,
    get_single_assessment_endpoint_v4,
    get_single_trace_endpoint_v4,
    http_request,
    verify_rest_response,
)

DATABRICKS_UC_TABLE_HEADER = "X-Databricks-UC-Table-Name"

_logger = logging.getLogger(__name__)


class DatabricksTracingRestStore(RestStore):
    """
    Client for a databricks tracking server accessed via REST API calls.
    This is only used for Databricks-specific tracing APIs, all other APIs including
    runs, experiments, models etc. should be implemented in the RestStore.

    Args
        get_host_creds: Method to be invoked prior to every REST request to get the
            :py:class:`mlflow.rest_utils.MlflowHostCreds` for the request. Note that this
            is a function so that we can obtain fresh credentials in the case of expiry.
    """

    _METHOD_TO_INFO = extract_api_info_for_service(
        MlflowService, _REST_API_PATH_PREFIX
    ) | extract_api_info_for_service(DatabricksTrackingService, _V4_REST_API_PATH_PREFIX)

    def __init__(self, get_host_creds):
        super().__init__(get_host_creds)

    def _call_endpoint(
        self,
        api,
        json_body=None,
        endpoint=None,
        retry_timeout_seconds=None,
        response_proto=None,
    ):
        try:
            return super()._call_endpoint(
                api,
                json_body=json_body,
                endpoint=endpoint,
                retry_timeout_seconds=retry_timeout_seconds,
                response_proto=response_proto,
            )
        except RestException as e:
            if (
                e.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
                and "Could not resolve a SQL warehouse ID" in e.message
            ):
                raise MlflowException(
                    message=(
                        "SQL warehouse ID is required for accessing traces in UC tables.\n"
                        f"Please set the {MLFLOW_TRACING_SQL_WAREHOUSE_ID.name} environment "
                        "variable to your SQL warehouse ID.\n"
                        "```\nexport MLFLOW_TRACING_SQL_WAREHOUSE_ID=<your_sql_warehouse_id>\n```\n"
                        "See https://docs.databricks.com/compute/sql-warehouse for how to "
                        "set up a SQL warehouse and get its ID."
                    ),
                    error_code=BAD_REQUEST,
                ) from e
            raise

    def start_trace(self, trace_info: TraceInfo) -> TraceInfo:
        """
        Create a new trace using the V4 API format.

        Args:
            trace_info: The TraceInfo object to create in the backend. Currently, this
                only supports trace_location with uc_schema, or mlflow_experiment that's
                linked to a UC table.

        Returns:
            The returned TraceInfo object from the backend.
        """
        try:
            if trace_info._is_v4():
                return self._start_trace_v4(trace_info)

        # Temporarily we capture all exceptions and fallback to v3 if the trace location is not uc
        # TODO: remove this once the endpoint is fully rolled out
        except Exception as e:
            if trace_info.trace_location.mlflow_experiment is None:
                _logger.debug("MLflow experiment is not set for trace, cannot fallback to V3 API.")
                raise
            _logger.debug(f"Falling back to V3 API due to {e!s}")
        return super().start_trace(trace_info)

    def _start_trace_v4(self, trace_info: TraceInfo) -> TraceInfo:
        location, otel_trace_id = parse_trace_id_v4(trace_info.trace_id)
        if location is None:
            raise MlflowException("Invalid trace ID format for v4 API.")

        req_body = message_to_json(trace_info.to_proto())
        response_proto = self._call_endpoint(
            CreateTraceInfo,
            req_body,
            endpoint=f"{_V4_REST_API_PATH_PREFIX}/mlflow/traces/{location}/{otel_trace_id}/info",
            retry_timeout_seconds=MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT.get(),
            response_proto=ProtoTraceInfo(),
        )
        return TraceInfo.from_proto(response_proto)

    def batch_get_traces(self, trace_ids: list[str], location: str) -> list[Trace]:
        """
        Get a batch of complete traces with spans for given trace ids.

        Args:
            trace_ids: List of trace IDs to fetch.
            location: Location of the trace. For example, "catalog.schema" for UC schema.

        Returns:
            List of Trace objects.
        """
        trace_ids = [parse_trace_id_v4(trace_id)[1] for trace_id in trace_ids]
        req_body = message_to_json(
            BatchGetTraces(
                location_id=location,
                trace_ids=trace_ids,
                sql_warehouse_id=MLFLOW_TRACING_SQL_WAREHOUSE_ID.get(),
            )
        )
        response_proto = self._call_endpoint(
            BatchGetTraces,
            req_body,
            endpoint=f"{_V4_TRACE_REST_API_PATH_PREFIX}/{location}/batchGet",
        )
        return [trace_from_proto(proto, location) for proto in response_proto.traces]

    def get_trace_info(self, trace_id: str) -> TraceInfo:
        """
        Get the trace info matching the `trace_id`.

        Args:
            trace_id: String id of the trace to fetch.

        Returns:
            The fetched ``mlflow.entities.TraceInfo`` object.
        """
        location, trace_id = parse_trace_id_v4(trace_id)
        if location is not None:
            sql_warehouse_id = MLFLOW_TRACING_SQL_WAREHOUSE_ID.get()
            trace_v4_req_body = message_to_json(
                GetTraceInfo(
                    trace_id=trace_id, location=location, sql_warehouse_id=sql_warehouse_id
                )
            )
            endpoint = f"{get_single_trace_endpoint_v4(location, trace_id)}/info"
            response_proto = self._call_endpoint(GetTraceInfo, trace_v4_req_body, endpoint=endpoint)
            return TraceInfo.from_proto(response_proto.trace.trace_info)

        return super().get_trace_info(trace_id)

    def set_trace_tag(self, trace_id: str, key: str, value: str):
        """
        Set a tag on the trace with the given trace_id.

        Args:
            trace_id: The ID of the trace.
            key: The string key of the tag.
            value: The string value of the tag.
        """
        location, trace_id = parse_trace_id_v4(trace_id)
        if location is not None:
            endpoint = f"{get_single_trace_endpoint_v4(location, trace_id)}/tags"
            req_body = message_to_json(
                SetTraceTag(
                    key=key,
                    value=value,
                )
            )
            self._call_endpoint(SetTraceTag, req_body, endpoint=endpoint)
            return
        return super().set_trace_tag(trace_id, key, value)

    def delete_trace_tag(self, trace_id: str, key: str):
        """
        Delete a tag on the trace with the given trace_id.

        Args:
            trace_id: The ID of the trace.
            key: The string key of the tag.
        """
        location, trace_id = parse_trace_id_v4(trace_id)
        if location is not None:
            sql_warehouse_id = MLFLOW_TRACING_SQL_WAREHOUSE_ID.get()
            encoded_key = quote(key, safe="")
            endpoint = f"{get_single_trace_endpoint_v4(location, trace_id)}/tags/{encoded_key}"
            req_body = message_to_json(DeleteTraceTag(sql_warehouse_id=sql_warehouse_id))
            self._call_endpoint(DeleteTraceTag, req_body, endpoint=endpoint)
            return
        return super().delete_trace_tag(trace_id, key)

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
        # This API is not client-facing, so we should always use `locations`.
        if experiment_ids is not None:
            raise MlflowException("`experiment_ids` is deprecated, use `locations` instead.")
        if not locations:
            raise MlflowException.invalid_parameter_value(
                "At least one location must be specified for searching traces."
            )

        # model_id is only supported by V3 API
        if model_id is not None:
            return self._search_unified_traces(
                model_id=model_id,
                locations=locations,
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
                page_token=page_token,
            )

        contain_uc_schemas = False
        trace_locations = []
        for location in locations:
            match location.split("."):
                case [experiment_id]:
                    trace_locations.append(
                        trace_location_to_proto(TraceLocation.from_experiment_id(experiment_id))
                    )
                case [catalog, schema]:
                    trace_locations.append(
                        trace_location_to_proto(
                            TraceLocation.from_databricks_uc_schema(catalog, schema)
                        )
                    )
                    contain_uc_schemas = True
                case _:
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid location type: {location}. Expected type: "
                        "`<catalog_name>.<schema_name>` or `<experiment_id>`."
                    )

        request = SearchTraces(
            locations=trace_locations,
            filter=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
            sql_warehouse_id=MLFLOW_TRACING_SQL_WAREHOUSE_ID.get(),
        )
        req_body = message_to_json(request)
        try:
            response_proto = self._call_endpoint(
                SearchTraces,
                req_body,
                endpoint=f"{_V4_TRACE_REST_API_PATH_PREFIX}/search",
            )
        except MlflowException as e:
            # There are 2 expected failure cases:
            # 1. Server does not support SearchTracesV4 API yet.
            # 2. Server supports V4 API but the experiment location is not supported yet.
            # For these known cases, MLflow fallback to V3 API.
            if e.error_code == ErrorCode.Name(ENDPOINT_NOT_FOUND):
                if contain_uc_schemas:
                    raise MlflowException.invalid_parameter_value(
                        "Searching traces in UC tables is not supported yet. Only experiment IDs "
                        "are supported for searching traces."
                    )
                _logger.debug("SearchTracesV4 API is not available yet. Falling back to V3 API.")
            elif (
                e.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
                and "locations not yet supported" in e.message
            ):
                if contain_uc_schemas:
                    raise MlflowException.invalid_parameter_value(
                        "The `locations` parameter cannot contain both MLflow experiment and UC "
                        "schema in the same request. Please specify only one type of location "
                        "at a time."
                    )
                _logger.debug("Experiment locations are not supported yet. Falling back to V3 API.")
            else:
                raise

            return self._search_traces(
                locations=locations,
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
                page_token=page_token,
            )

        trace_infos = [TraceInfo.from_proto(t) for t in response_proto.trace_infos]
        return trace_infos, response_proto.next_page_token or None

    def _search_unified_traces(
        self,
        model_id: str,
        locations: list[str],
        filter_string: str | None = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> tuple[list[TraceInfo], str | None]:
        request = SearchUnifiedTraces(
            model_id=model_id,
            sql_warehouse_id=MLFLOW_TRACING_SQL_WAREHOUSE_ID.get(),
            experiment_ids=locations,
            filter=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )
        req_body = message_to_json(request)
        response_proto = self._call_endpoint(SearchUnifiedTraces, req_body)
        # Convert TraceInfo (v2) objects to TraceInfoV3 objects for consistency
        trace_infos = [TraceInfo.from_proto(t) for t in response_proto.traces]
        return trace_infos, response_proto.next_page_token or None

    def get_online_trace_details(
        self,
        trace_id: str,
        source_inference_table: str,
        source_databricks_request_id: str,
    ):
        req = GetOnlineTraceDetails(
            trace_id=trace_id,
            sql_warehouse_id=MLFLOW_TRACING_SQL_WAREHOUSE_ID.get(),
            source_inference_table=source_inference_table,
            source_databricks_request_id=source_databricks_request_id,
        )
        req_body = message_to_json(req)
        response_proto = self._call_endpoint(GetOnlineTraceDetails, req_body)
        return response_proto.trace_data

    def set_experiment_trace_location(
        self,
        location: UCSchemaLocationEntity,
        experiment_id: str,
        sql_warehouse_id: str | None = None,
    ) -> UCSchemaLocationEntity:
        req_body = message_to_json(
            CreateTraceUCStorageLocation(
                uc_schema=uc_schema_location_to_proto(location),
                sql_warehouse_id=sql_warehouse_id or MLFLOW_TRACING_SQL_WAREHOUSE_ID.get(),
            )
        )
        try:
            response = self._call_endpoint(
                CreateTraceUCStorageLocation,
                req_body,
                endpoint=f"{_V4_TRACE_REST_API_PATH_PREFIX}/location",
            )
            location = uc_schema_location_from_proto(response.uc_schema)
        except MlflowException as e:
            if e.error_code == ErrorCode.Name(ALREADY_EXISTS):
                _logger.debug(f"Trace UC storage location already exists: {location}")
            else:
                raise
        _logger.debug(f"Created trace UC storage location: {location}")

        # link experiment to uc trace location
        req_body = message_to_json(
            LinkExperimentToUCTraceLocation(
                experiment_id=experiment_id,
                uc_schema=uc_schema_location_to_proto(location),
            )
        )

        self._call_endpoint(
            LinkExperimentToUCTraceLocation,
            req_body,
            endpoint=f"{_V4_TRACE_REST_API_PATH_PREFIX}/{experiment_id}/link-location",
        )
        _logger.debug(f"Linked experiment {experiment_id} to UC trace location: {location}")
        return location

    def unset_experiment_trace_location(
        self, experiment_id: str, location: UCSchemaLocationEntity
    ) -> None:
        request = UnLinkExperimentToUCTraceLocation(
            experiment_id=experiment_id,
            uc_schema=uc_schema_location_to_proto(location),
        )
        endpoint = f"{_V4_TRACE_REST_API_PATH_PREFIX}/{experiment_id}/unlink-location"
        req_body = message_to_json(request)
        self._call_endpoint(
            UnLinkExperimentToUCTraceLocation,
            req_body,
            endpoint=endpoint,
        )
        _logger.debug(f"Unlinked experiment {experiment_id} from trace location: {location}")

    def log_spans(self, location: str, spans: list[Span], tracking_uri=None) -> list[Span]:
        if not spans:
            return []

        if tracking_uri is None:
            raise MlflowException(
                "`tracking_uri` must be provided to log spans to with Databricks tracking server."
            )

        endpoint = f"/api/2.0/otel{OTLP_TRACES_PATH}"
        try:
            config = get_databricks_workspace_client_config(tracking_uri)
        except Exception as e:
            raise MlflowException(
                "Failed to log spans to UC table: could not identify Databricks workspace "
                "configuration"
            ) from e

        request = ExportTraceServiceRequest()
        resource_spans = request.resource_spans.add()
        scope_spans = resource_spans.scope_spans.add()
        scope_spans.spans.extend(span.to_otel_proto() for span in spans)

        response = http_request(
            host_creds=self.get_host_creds(),
            endpoint=endpoint,
            method="POST",
            data=request.SerializeToString(),
            extra_headers={
                "Content-Type": "application/x-protobuf",
                DATABRICKS_UC_TABLE_HEADER: location,
                **config.authenticate(),
            },
        )
        verify_rest_response(response, endpoint)
        return spans

    def create_assessment(self, assessment: Assessment) -> Assessment:
        """
        Create an assessment entity in the backend store.

        Args:
            assessment: The assessment to log (without an assessment_id).

        Returns:
            The created Assessment object.
        """
        location, trace_id = parse_trace_id_v4(assessment.trace_id)
        if location is not None:
            req_body = message_to_json(assessment_to_proto(assessment))
            endpoint = self._append_sql_warehouse_id_param(
                f"{get_single_trace_endpoint_v4(location, trace_id)}/assessments",
            )
            response_proto = self._call_endpoint(
                CreateAssessment,
                req_body,
                endpoint=endpoint,
                response_proto=ProtoAssessment(),
            )
            return Assessment.from_proto(response_proto)

        return super().create_assessment(assessment)

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

        location, parsed_trace_id = parse_trace_id_v4(trace_id)
        if location is not None:
            assessment = UpdateAssessment().assessment
            assessment.assessment_id = assessment_id
            catalog, schema = location.split(".")
            assessment.trace_location.CopyFrom(
                trace_location_to_proto(TraceLocation.from_databricks_uc_schema(catalog, schema)),
            )
            assessment.trace_id = parsed_trace_id
            # Field mask specifies which fields to update.
            mask = UpdateAssessment().update_mask
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

            endpoint = get_single_assessment_endpoint_v4(location, parsed_trace_id, assessment_id)
            endpoint = self._append_sql_warehouse_id_param(endpoint)

            if mask.paths:
                mask_param = ",".join(mask.paths)
                endpoint = f"{endpoint}&update_mask={mask_param}"

            req_body = message_to_json(assessment)
            response_proto = self._call_endpoint(
                UpdateAssessment,
                req_body,
                endpoint=endpoint,
                response_proto=ProtoAssessment(),
            )
            return Assessment.from_proto(response_proto)
        else:
            return super().update_assessment(
                trace_id, assessment_id, name, expectation, feedback, rationale, metadata
            )

    def get_assessment(self, trace_id: str, assessment_id: str) -> Assessment:
        """
        Get an assessment entity from the backend store.
        """

        location, trace_id = parse_trace_id_v4(trace_id)
        if location is not None:
            endpoint = self._append_sql_warehouse_id_param(
                get_single_assessment_endpoint_v4(location, trace_id, assessment_id)
            )
            response_proto = self._call_endpoint(
                GetAssessment, endpoint=endpoint, response_proto=ProtoAssessment()
            )
            return Assessment.from_proto(response_proto)

        return super().get_assessment(trace_id, assessment_id)

    def delete_assessment(self, trace_id: str, assessment_id: str):
        """
        Delete an assessment associated with a trace.

        Args:
            trace_id: String ID of the trace.
            assessment_id: String ID of the assessment to delete.
        """
        location, trace_id = parse_trace_id_v4(trace_id)
        if location is not None:
            endpoint = self._append_sql_warehouse_id_param(
                get_single_assessment_endpoint_v4(location, trace_id, assessment_id)
            )
            self._call_endpoint(DeleteAssessment, endpoint=endpoint)
        else:
            return super().delete_assessment(trace_id, assessment_id)

    def _group_traces_by_location(self, trace_ids: list[str]) -> dict[str | None, list[str]]:
        """
        Group trace IDs by location to separate V3 and V4 traces.

        Args:
            trace_ids: List of trace IDs (can be V3 or V4 format).

        Returns:
            Dict mapping location to list of trace IDs where:
            - None key: List of V3 trace IDs (without location prefix)
            - str keys: Location IDs (e.g., "catalog.schema") mapping to OTEL trace IDs
        """
        traces_by_location: dict[str | None, list[str]] = defaultdict(list)

        for trace_id in trace_ids:
            location_id, trace_id = parse_trace_id_v4(trace_id)
            traces_by_location[location_id].append(trace_id)

        return traces_by_location

    def _batch_link_traces_to_run(
        self, location_id: str, otel_trace_ids: list[str], run_id: str
    ) -> None:
        """
        Link multiple traces to a run by creating internal trace-to-run relationships.

        Args:
            location_id: The location ID (e.g., "catalog.schema") for the traces.
            otel_trace_ids: List of OTEL trace IDs to link to the run.
            run_id: ID of the run to link traces to.
        """
        if not otel_trace_ids:
            return

        req_body = message_to_json(
            BatchLinkTraceToRun(
                location_id=location_id,
                trace_ids=otel_trace_ids,
                run_id=run_id,
            )
        )
        endpoint = f"{_V4_TRACE_REST_API_PATH_PREFIX}/{location_id}/link-to-run/batchCreate"
        self._call_endpoint(BatchLinkTraceToRun, req_body, endpoint=endpoint)

    def _batch_unlink_traces_from_run(
        self, location_id: str, otel_trace_ids: list[str], run_id: str
    ) -> None:
        """
        Unlink multiple traces from a run by removing the internal trace-to-run relationships.

        Args:
            location_id: The location ID (e.g., "catalog.schema") for the traces.
            otel_trace_ids: List of OTEL trace IDs to unlink from the run.
            run_id: ID of the run to unlink traces from.
        """
        if not otel_trace_ids:
            return

        req_body = message_to_json(
            BatchUnlinkTraceFromRun(
                location_id=location_id,
                trace_ids=otel_trace_ids,
                run_id=run_id,
            )
        )
        endpoint = f"{_V4_TRACE_REST_API_PATH_PREFIX}/{location_id}/unlink-from-run/batchDelete"
        self._call_endpoint(BatchUnlinkTraceFromRun, req_body, endpoint=endpoint)

    def link_traces_to_run(self, trace_ids: list[str], run_id: str) -> None:
        """
        Link multiple traces to a run by creating trace-to-run relationships.

        Args:
            trace_ids: List of trace IDs to link to the run.
            run_id: ID of the run to link traces to.
        """
        if not trace_ids:
            return

        traces_by_location = self._group_traces_by_location(trace_ids)

        for location_id, batch_trace_ids in traces_by_location.items():
            if location_id is None:
                super().link_traces_to_run(batch_trace_ids, run_id)
            else:
                self._batch_link_traces_to_run(location_id, batch_trace_ids, run_id)

    def unlink_traces_from_run(self, trace_ids: list[str], run_id: str) -> None:
        """
        Unlink multiple traces from a run by removing trace-to-run relationships.

        Args:
            trace_ids: List of trace IDs to unlink from the run.
            run_id: ID of the run to unlink traces from.
        """
        if not trace_ids:
            return

        traces_by_location = self._group_traces_by_location(trace_ids)

        if v3_trace_ids := traces_by_location.pop(None, []):
            raise MlflowException(
                "Unlinking traces from runs is only supported for traces with UC schema "
                f"locations. Unsupported trace IDs: {v3_trace_ids}"
            )

        for location_id, batch_trace_ids in traces_by_location.items():
            self._batch_unlink_traces_from_run(location_id, batch_trace_ids, run_id)

    def _append_sql_warehouse_id_param(self, endpoint: str) -> str:
        if sql_warehouse_id := MLFLOW_TRACING_SQL_WAREHOUSE_ID.get():
            return f"{endpoint}?sql_warehouse_id={sql_warehouse_id}"
        return endpoint
