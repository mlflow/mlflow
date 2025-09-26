import logging

from mlflow.entities import Assessment, Trace, TraceInfo
from mlflow.entities.assessment import ExpectationValue, FeedbackValue
from mlflow.environment_variables import (
    MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT,
    MLFLOW_TRACING_SQL_WAREHOUSE_ID,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND, ErrorCode
from mlflow.protos.databricks_tracing_pb2 import (
    CreateAssessment,
    CreateTrace,
    DatabricksTrackingService,
    DeleteAssessment,
    GetAssessment,
    GetTraces,
    TraceIdentifier,
    UpdateAssessment,
)
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracing.constant import TRACE_ID_V4_PREFIX
from mlflow.tracing.utils import parse_trace_id_v4
from mlflow.utils.databricks_tracing_utils import (
    assessment_to_proto,
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
    get_single_assessment_endpoint_v4,
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
            req_body = message_to_json(
                CreateAssessment(
                    assessment=assessment_to_proto(assessment),
                    sql_warehouse_id=MLFLOW_TRACING_SQL_WAREHOUSE_ID.get(),
                )
            )
            endpoint = f"{get_single_trace_endpoint_v4(location, trace_id)}/assessment"
            response_proto = self._call_endpoint(
                CreateAssessment,
                req_body,
                endpoint=endpoint,
            )
            return Assessment.from_proto(response_proto.assessment)

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
            update = UpdateAssessment(
                sql_warehouse_id=MLFLOW_TRACING_SQL_WAREHOUSE_ID.get(),
            )
            endpoint = get_single_assessment_endpoint_v4(location, parsed_trace_id, assessment_id)
            assessment = update.assessment
            assessment.assessment_id = assessment_id
            catalog, schema = location.split(".")
            assessment.trace_identifier.CopyFrom(
                TraceIdentifier(
                    trace_location=trace_location_to_proto(
                        trace_location_from_databricks_uc_schema(catalog, schema)
                    ),
                    trace_id=parsed_trace_id,
                )
            )
            # Field mask specifies which fields to update.
            mask = update.update_mask

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
                endpoint=endpoint,
            )
            return Assessment.from_proto(response_proto.assessment)
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
            req_body = message_to_json(
                GetAssessment(
                    sql_warehouse_id=MLFLOW_TRACING_SQL_WAREHOUSE_ID.get(),
                )
            )
            endpoint = get_single_assessment_endpoint_v4(location, trace_id, assessment_id)
            response_proto = self._call_endpoint(
                GetAssessment,
                req_body,
                endpoint=endpoint,
            )
            return Assessment.from_proto(response_proto.assessment)

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
            req_body = message_to_json(
                DeleteAssessment(
                    sql_warehouse_id=MLFLOW_TRACING_SQL_WAREHOUSE_ID.get(),
                )
            )
            endpoint = get_single_assessment_endpoint_v4(location, trace_id, assessment_id)
            self._call_endpoint(
                DeleteAssessment,
                req_body,
                endpoint=endpoint,
            )
        else:
            return super().delete_assessment(trace_id, assessment_id)
