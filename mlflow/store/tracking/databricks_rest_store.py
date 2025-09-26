import logging

from mlflow.entities import Trace, TraceInfo, TraceLocation
from mlflow.environment_variables import (
    MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT,
    MLFLOW_TRACING_SQL_WAREHOUSE_ID,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND, ErrorCode
from mlflow.protos.databricks_tracing_pb2 import (
    CreateTrace,
    DatabricksTrackingService,
    GetTraces,
    SearchTraces,
    TraceIdentifier,
    UCSchemaLocation,
)
from mlflow.protos.service_pb2 import MlflowService, SearchUnifiedTraces
from mlflow.store.tracking import SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracing.constant import TRACE_ID_V4_PREFIX
from mlflow.tracing.utils import parse_trace_id_v4
from mlflow.utils.databricks_tracing_utils import (
    trace_from_proto,
    trace_info_to_proto,
    trace_location_from_databricks_uc_schema,
    trace_location_to_proto,
)
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    _V4_REST_API_PATH_PREFIX,
    _V4_TRACE_REST_API_PATH_PREFIX,
    extract_api_info_for_service,
)

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
            sql_warehouse_id = MLFLOW_TRACING_SQL_WAREHOUSE_ID.get()
            req_body = message_to_json(
                CreateTrace(
                    trace_info=trace_info_to_proto(trace_info), sql_warehouse_id=sql_warehouse_id
                )
            )
            if uc_schema := trace_info.trace_location.uc_schema:
                location = f"{uc_schema.catalog_name}.{uc_schema.schema_name}"
            # TODO: we should check if the experiment has a span location tag
            elif mlflow_experiment := trace_info.trace_location.mlflow_experiment:
                location = mlflow_experiment.experiment_id
            else:
                raise MlflowException("Invalid trace location")
            response_proto = self._call_endpoint(
                CreateTrace,
                req_body,
                endpoint=f"{_V4_REST_API_PATH_PREFIX}/mlflow/traces/{location}",
                retry_timeout_seconds=MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT.get(),
            )
            return TraceInfo.from_proto(response_proto.trace_info)
        # Temporarily we capture all exceptions and fallback to v3 if the trace location is not uc
        # TODO: remove this once the endpoint is fully rolled out
        except Exception as e:
            if isinstance(e, MlflowException) and e.error_code == ErrorCode.Name(
                ENDPOINT_NOT_FOUND
            ):
                _logger.debug("Server does not support CreateTrace API yet.")
            if trace_info.trace_location.mlflow_experiment is not None:
                _logger.debug("Falling back to V3 API.")
                return super().start_trace(trace_info)
            else:
                raise

    def get_traces(self, trace_ids: list[str]) -> list[Trace]:
        """
        Get complete traces with spans for given trace ids.

        Args:
            trace_ids: List of trace IDs to fetch.

        Returns:
            List of Trace objects.
        """
        sql_warehouse_id = MLFLOW_TRACING_SQL_WAREHOUSE_ID.get()
        trace_identifiers = [self._construct_trace_identifier(trace_id) for trace_id in trace_ids]
        req_body = message_to_json(
            GetTraces(trace_ids=trace_identifiers, sql_warehouse_id=sql_warehouse_id)
        )
        response_proto = self._call_endpoint(
            GetTraces, req_body, endpoint=f"{_V4_TRACE_REST_API_PATH_PREFIX}/batch"
        )
        return [trace_from_proto(proto) for proto in response_proto.traces]

    def _construct_trace_identifier(self, trace_identifier: str) -> TraceIdentifier:
        location, trace_id = parse_trace_id_v4(trace_identifier)
        # location is only None when trace_id does not starts with 'trace:/'
        if location is None:
            return TraceIdentifier(trace_id=trace_id)
        match location.split("."):
            case [catalog, schema]:
                return TraceIdentifier(
                    uc_schema=UCSchemaLocation(catalog_name=catalog, schema_name=schema),
                    trace_id=trace_id,
                )
            case _:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid trace_id format: {trace_identifier}, should be in the format of "
                    f"{TRACE_ID_V4_PREFIX}<catalog.schema>/<trace_id>"
                )

    def search_traces(
        self,
        experiment_ids: list[str] | None = None,
        filter_string: str | None = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: list[str] | None = None,
        page_token: str | None = None,
        model_id: str | None = None,
        sql_warehouse_id: str | None = None,
        locations: list[str] | None = None,
    ) -> tuple[list[TraceInfo], str | None]:
        # This API is not client-facing, so we should always use `locations`.
        if experiment_ids is not None:
            raise MlflowException.invalid_parameter_value(
                "`experiment_ids` is deprecated, use `locations` instead."
            )
        if not locations:
            raise MlflowException.invalid_parameter_value(
                "`locations` must be specified for searching traces in Databricks."
            )

        contain_uc_schemas = False
        trace_locations = []
        # model_id is only supported by V3 API
        if model_id is None:
            for location in locations:
                if "." not in location:
                    trace_locations.append(
                        trace_location_to_proto(TraceLocation.from_experiment_id(location))
                    )
                else:
                    match location.split("."):
                        case [catalog, schema]:
                            trace_locations.append(
                                trace_location_to_proto(
                                    trace_location_from_databricks_uc_schema(catalog, schema)
                                )
                            )
                            contain_uc_schemas = True
                        case _:
                            raise MlflowException.invalid_parameter_value(
                                f"Invalid location format: {location}. Expected format: "
                                "`<catalog_name>.<schema_name>` or `<experiment_id>`."
                            )

            request = SearchTraces(
                locations=trace_locations,
                filter=filter_string,
                max_results=max_results,
                order_by=order_by,
                page_token=page_token,
                sql_warehouse_id=sql_warehouse_id or MLFLOW_TRACING_SQL_WAREHOUSE_ID.get(),
            )
            req_body = message_to_json(request)
            try:
                response_proto = self._call_endpoint(
                    SearchTraces,
                    req_body,
                    endpoint=f"{_V4_TRACE_REST_API_PATH_PREFIX}/search",
                )
            except MlflowException as e:
                if e.error_code == ErrorCode.Name(ENDPOINT_NOT_FOUND):
                    _logger.debug(
                        "Server does not support SearchTracesV4 API yet. Falling back to V3 API."
                    )
                    if contain_uc_schemas:
                        raise MlflowException(
                            "Searching traces by locations including UC schemas is not supported "
                            "on the current tracking server. Only locations with experiment IDs "
                            "are supported."
                        )
                    # fallback to v3 API
                    return self._search_traces(
                        locations=locations,
                        filter_string=filter_string,
                        max_results=max_results,
                        order_by=order_by,
                        page_token=page_token,
                    )
                else:
                    raise
            trace_infos = [
                TraceInfo.from_proto(trace.trace_info) for trace in response_proto.traces
            ]
            return trace_infos, response_proto.next_page_token or None
        else:
            return self._search_unified_traces(
                model_id=model_id,
                locations=locations,
                sql_warehouse_id=sql_warehouse_id or MLFLOW_TRACING_SQL_WAREHOUSE_ID.get(),
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
                page_token=page_token,
            )

    def _search_unified_traces(
        self,
        model_id: str,
        locations: list[str],
        sql_warehouse_id: str | None = None,
        filter_string: str | None = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> tuple[list[TraceInfo], str | None]:
        request = SearchUnifiedTraces(
            model_id=model_id,
            sql_warehouse_id=sql_warehouse_id,
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
