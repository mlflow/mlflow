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
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from pydantic import BaseModel, Field

from mlflow.entities.span import Span
from mlflow.server.handlers import _get_tracking_store
from mlflow.tracing.utils.otlp import MLFLOW_EXPERIMENT_ID_HEADER, OTLP_TRACES_PATH

# Create FastAPI router for OTel endpoints
otel_router = APIRouter(prefix=OTLP_TRACES_PATH, tags=["OpenTelemetry"])


class OTelExportTraceServiceResponse(BaseModel):
    """
    Pydantic model for the OTLP/HTTP ExportTraceServiceResponse.

    This matches the OpenTelemetry protocol specification for trace export responses.
    Reference: https://opentelemetry.io/docs/specs/otlp/
    """

    partialSuccess: dict[str, Any] | None = Field(
        None, description="Details about partial success of the export operation"
    )


@otel_router.post("", response_model=OTelExportTraceServiceResponse, status_code=200)
async def export_traces(
    request: Request,
    response: Response,
    x_mlflow_experiment_id: str = Header(..., alias=MLFLOW_EXPERIMENT_ID_HEADER),
    content_type: str = Header(None),
) -> OTelExportTraceServiceResponse:
    """
    Export trace spans to MLflow via the OpenTelemetry protocol.

    This endpoint accepts OTLP/HTTP protobuf trace export requests.
    Protobuf format reference: https://opentelemetry.io/docs/specs/otlp/#binary-protobuf-encoding

    Args:
        request: OTel ExportTraceServiceRequest in protobuf format
        response: FastAPI Response object for setting headers
        x_mlflow_experiment_id: Required header containing the experiment ID
        content_type: Content-Type header from the request

    Returns:
        OTel ExportTraceServiceResponse indicating success

    Raises:
        HTTPException: If the request is invalid or span logging fails
    """
    # Validate Content-Type header
    if content_type != "application/x-protobuf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid Content-Type: {content_type}. Expected: application/x-protobuf",
        )

    # Set response Content-Type header
    response.headers["Content-Type"] = "application/x-protobuf"

    body = await request.body()
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

        # Note: Benchmarking shows that ThreadPoolExecutor does not improve performance
        # for SQLite backends and can actually degrade performance due to write contention.
        # Sequential logging is simpler and faster for typical use cases.
        errors = {}
        for trace_id, trace_spans in spans_by_trace_id.items():
            try:
                store.log_spans(x_mlflow_experiment_id, trace_spans)
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

    return OTelExportTraceServiceResponse()
