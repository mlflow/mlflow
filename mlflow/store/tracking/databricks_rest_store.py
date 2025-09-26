import logging

from mlflow.entities import Trace, TraceInfo
from mlflow.environment_variables import (
    MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT,
    MLFLOW_TRACING_SQL_WAREHOUSE_ID,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND, ErrorCode
from mlflow.protos.databricks_tracing_pb2 import (
    CreateTrace,
    DatabricksTrackingService,
    DeleteTraceTag,
    GetTraceInfo,
    GetTraces,
    SetTraceTag,
    TraceIdentifier,
)
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
    _V4_REST_API_PATH_PREFIX,
    _V4_TRACE_REST_API_PATH_PREFIX,
    extract_api_info_for_service,
    get_single_trace_endpoint_v4,
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
        DatabricksTrackingService, _V4_REST_API_PATH_PREFIX
    )

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
                    trace_location=trace_location_to_proto(
                        trace_location_from_databricks_uc_schema(catalog, schema)
                    ),
                    trace_id=trace_id,
                )
            case _:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid trace_id format: {trace_identifier}, should be in the format of "
                    f"{TRACE_ID_V4_PREFIX}<catalog.schema>/<trace_id>"
                )

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
            endpoint = f"{get_single_trace_endpoint_v4(location, trace_id)}/tags/{key}"
            req_body = message_to_json(DeleteTraceTag(sql_warehouse_id=sql_warehouse_id))
            self._call_endpoint(DeleteTraceTag, req_body, endpoint=endpoint)
            return
        return super().delete_trace_tag(trace_id, key)
