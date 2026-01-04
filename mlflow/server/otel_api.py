"""
OpenTelemetry REST API endpoints for MLflow FastAPI server.

This module implements the OpenTelemetry Protocol (OTLP) REST API for ingesting spans
according to the OTel specification:
https://opentelemetry.io/docs/specs/otlp/#otlphttp

Note: This is a minimal implementation that serves as a placeholder for the OTel endpoint.
The actual span ingestion logic would need to properly convert incoming OTel format spans
to MLflow spans, which requires more complex conversion logic.
"""

from collections import defaultdict

from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

from mlflow.entities.span import Span
from mlflow.exceptions import MlflowException
from mlflow.server.handlers import _get_tracking_store
from mlflow.telemetry.events import TraceSource, TracesReceivedByServerEvent
from mlflow.telemetry.track import _record_event
from mlflow.tracing.utils.otlp import (
    MLFLOW_EXPERIMENT_ID_HEADER,
    OTLP_TRACES_PATH,
    decompress_otlp_body,
)
from mlflow.tracking.request_header.default_request_header_provider import (
    _MLFLOW_PYTHON_CLIENT_USER_AGENT_PREFIX,
    _USER_AGENT,
)

# Create FastAPI router for OTel endpoints
otel_router = APIRouter(prefix=OTLP_TRACES_PATH, tags=["OpenTelemetry"])

def _get_experiment_id_from_service_name(
    store,
    parsed_request: ExportTraceServiceRequest,
) -> str:
    """Derive the MLflow experiment ID from the OTLP request's `service.name`.

    Args:
        store: Tracking store used to look up and create experiments.
        parsed_request: The parsed OTLP trace export request.

    Returns:
        The experiment ID corresponding to the single `service.name` value found in the request.

    Raises:
        fastapi.HTTPException: If no `service.name` resource attribute is found in the request, or
            if multiple distinct `service.name` values are present.
    """
    attributes = []
    for resource_span in parsed_request.resource_spans:
        attributes.extend(resource_span.resource.attributes)

    service_names = {
        attribute.value.string_value
        for attribute in attributes
        if attribute.key == "service.name"
        and attribute.value.WhichOneof("value") == "string_value"
        and attribute.value.string_value
    }

    if not service_names:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No `service.name` resource attribute was found in the OTLP request. "
                f"Set `service.name` (e.g. via `OTEL_SERVICE_NAME`) or provide the "
                f"'{MLFLOW_EXPERIMENT_ID_HEADER}' header."
            ),
        )

    if len(service_names) > 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Multiple 'service.name' resource attributes were found in the OTLP request "
                f"({sorted(service_names)}). Set the '{MLFLOW_EXPERIMENT_ID_HEADER}' header to "
                "disambiguate which MLflow experiment to log to."
            ),
        )

    experiment_name = next(iter(service_names))
    return _get_or_create_experiment_id(store, experiment_name)


def _get_or_create_experiment_id(store, experiment_name: str) -> str:
    """Return the experiment ID, creating the experiment if needed.

    If the experiment does not exist, this function attempts to create it. If creation fails (e.g.
    due to concurrent creation), it retries a lookup by name.

    Args:
        store: Tracking store used to look up and create experiments.
        experiment_name: Name of the experiment.

    Returns:
        The experiment ID.

    Raises:
        MlflowException: If experiment creation fails and the experiment still cannot be found.
    """
    experiment = store.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id

    try:
        return store.create_experiment(experiment_name)
    except MlflowException:
        # Experiment may have been created concurrently.
        experiment = store.get_experiment_by_name(experiment_name)
        if experiment is not None:
            return experiment.experiment_id
        raise


@otel_router.post("", status_code=200)
async def export_traces(
    request: Request,
    x_mlflow_experiment_id: str | None = Header(default=None, alias=MLFLOW_EXPERIMENT_ID_HEADER),
    content_type: str | None = Header(default=None),
    content_encoding: str | None = Header(default=None),
    user_agent: str | None = Header(None, alias=_USER_AGENT),
) -> Response:
    """
    Export trace spans to MLflow via the OpenTelemetry protocol.

    This endpoint accepts OTLP/HTTP protobuf trace export requests.
    Protobuf format reference: https://opentelemetry.io/docs/specs/otlp/#binary-protobuf-encoding

    Args:
        request: OTel ExportTraceServiceRequest in protobuf format
        x_mlflow_experiment_id: Optional header containing the MLflow experiment ID. If omitted,
            MLflow attempts to infer the experiment name from the OpenTelemetry resource attribute
            `service.name` (typically set via `OTEL_SERVICE_NAME`) and uses or creates the
            corresponding experiment. Note: inferring from `service.name` requires scanning the
            OTLP payload for resource attributes, so setting the experiment ID explicitly (e.g. via
            `OTEL_EXPORTER_OTLP_HEADERS=x-mlflow-experiment-id=<id>`) avoids that extra work.
        content_type: Content-Type header from the request
        content_encoding: Content-Encoding header from the request
        user_agent: User-Agent header (used to identify MLflow Python client)

    Returns:
        FastAPI Response with ExportTraceServiceResponse in protobuf format

    Raises:
        HTTPException: If the request is invalid or span logging fails
    """
    # Validate Content-Type header
    if content_type != "application/x-protobuf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid Content-Type: {content_type}. Expected: application/x-protobuf",
        )

    # Read & decompress request body
    body = await request.body()
    if content_encoding:
        body = decompress_otlp_body(body, content_encoding.lower())

    # Parse protobuf payload
    parsed_request = ExportTraceServiceRequest()

    try:
        # In Python protobuf library 5.x, ParseFromString may not raise DecodeError on invalid data
        parsed_request.ParseFromString(body)

        # Check if we actually parsed any data
        # If no resource_spans were parsed, the data was likely invalid
        if not parsed_request.resource_spans:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid OpenTelemetry protobuf format - no spans found",
            )

    except DecodeError:
        # This will catch errors in Python protobuf library 3.x
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OpenTelemetry protobuf format",
        )

    # Group spans by trace_id to support BatchSpanProcessor
    # log_spans requires all spans in a batch to have the same trace_id
    spans_by_trace_id = defaultdict(list)
    for resource_span in parsed_request.resource_spans:
        for scope_span in resource_span.scope_spans:
            for otel_proto_span in scope_span.spans:
                try:
                    mlflow_span = Span.from_otel_proto(otel_proto_span)
                    spans_by_trace_id[mlflow_span.trace_id].append(mlflow_span)
                except Exception:
                    raise HTTPException(
                        status_code=422,
                        detail="Cannot convert OpenTelemetry span to MLflow span",
                    )

    if spans_by_trace_id:
        store = _get_tracking_store()
        experiment_id = (
            x_mlflow_experiment_id
            if x_mlflow_experiment_id is not None
            else _get_experiment_id_from_service_name(store, parsed_request)
        )

        # Note: Benchmarking shows that ThreadPoolExecutor does not improve performance
        # for SQLite backends and can actually degrade performance due to write contention.
        # Sequential logging is simpler and faster for typical use cases.
        errors = {}
        completed_trace_ids = set()
        for trace_id, trace_spans in spans_by_trace_id.items():
            try:
                store.log_spans(experiment_id, trace_spans)
                for span in trace_spans:
                    if span.parent_id is None:
                        # Only count traces with a root span as completed
                        # (logging of the root span indicates a completed trace)
                        completed_trace_ids.add(trace_id)
                        break
            except NotImplementedError:
                store_name = store.__class__.__name__
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    # NB: this error message must be the same as the one used in span exporter
                    # to avoid emitting warnings for unsupported stores
                    detail=f"REST OTLP span logging is not supported by {store_name}",
                )
            except Exception as e:
                errors[trace_id] = e

        if errors:
            error_msg = "\n".join(
                [f"Trace {trace_id}: {error}" for trace_id, error in errors.items()]
            )
            raise HTTPException(
                status_code=422,
                detail=f"Failed to log OpenTelemetry spans: {error_msg}",
            )

        if completed_trace_ids:
            trace_source = (
                TraceSource.MLFLOW_PYTHON_CLIENT
                if user_agent and user_agent.startswith(_MLFLOW_PYTHON_CLIENT_USER_AGENT_PREFIX)
                else TraceSource.UNKNOWN
            )

            _record_event(
                TracesReceivedByServerEvent,
                {
                    "source": trace_source,
                    "count": len(completed_trace_ids),
                },
            )

    # Return protobuf response as per OTLP specification
    response_message = ExportTraceServiceResponse()
    response_bytes = response_message.SerializeToString()
    return Response(
        content=response_bytes,
        media_type="application/x-protobuf",
        status_code=200,
    )
