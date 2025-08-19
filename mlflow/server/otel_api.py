"""
OpenTelemetry REST API endpoints for MLflow FastAPI server.

This module implements the OpenTelemetry Protocol (OTLP) REST API for ingesting spans
according to the OTel specification:
https://opentelemetry.io/docs/specs/otlp/#otlphttp

Note: This is a minimal implementation that serves as a placeholder for the OTel endpoint.
The actual span ingestion logic would need to properly convert incoming OTel format spans
to MLflow spans, which requires more complex conversion logic.
"""

from typing import Any

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from mlflow.tracing.utils.otlp import MLFLOW_EXPERIMENT_ID_HEADER

# Create FastAPI router for OTel endpoints
otel_router = APIRouter(prefix="/v1/traces", tags=["OpenTelemetry"])


class OTelExportTraceServiceRequest(BaseModel):
    """
    Pydantic model for the OTLP/HTTP ExportTraceServiceRequest.

    This matches the OpenTelemetry protocol specification for trace export requests.
    """

    resourceSpans: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Collection of resource spans from instrumented applications",
        alias="resource_spans",
    )

    class Config:
        populate_by_name = True  # Allow both field name and alias


class OTelExportTraceServiceResponse(BaseModel):
    """
    Pydantic model for the OTLP/HTTP ExportTraceServiceResponse.

    This matches the OpenTelemetry protocol specification for trace export responses.
    """

    partialSuccess: dict[str, Any] | None = Field(
        None, description="Details about partial success of the export operation"
    )


@otel_router.post("", response_model=OTelExportTraceServiceResponse, status_code=200)
def export_traces(
    request: OTelExportTraceServiceRequest,
    x_mlflow_experiment_id: str = Header(..., alias=MLFLOW_EXPERIMENT_ID_HEADER),
) -> OTelExportTraceServiceResponse:
    """
    Export trace spans to MLflow via the OpenTelemetry protocol.

    This endpoint accepts OTLP/HTTP trace export requests.

    Note: This is a minimal placeholder implementation. A full implementation would need to:
    1. Convert incoming OTel JSON format spans to MLflow Span entities
    2. Call store.log_spans() to persist them

    Args:
        request: OTel ExportTraceServiceRequest in JSON format
        x_mlflow_experiment_id: Optional header containing the experiment ID

    Returns:
        OTel ExportTraceServiceResponse indicating success or partial success

    Raises:
        HTTPException: If the request is invalid or span logging fails
    """
    mlflow_spans = []
    for resource_span in request.resourceSpans:
        for scope_span in resource_span.get("scope_spans", resource_span.get("scopeSpans", [])):
            for span_dict in scope_span.get("spans", []):
                from google.protobuf.json_format import ParseDict
                from opentelemetry.proto.trace.v1.trace_pb2 import Span as OTelProtoSpan

                try:
                    otel_proto_span = ParseDict(span_dict, OTelProtoSpan())
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid OpenTelemetry span format",
                    )

                try:
                    from mlflow.entities.span import Span

                    mlflow_span = Span._from_otel_proto(otel_proto_span)
                    mlflow_spans.append(mlflow_span)
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="Cannot convert OpenTelemetry span to MLflow span",
                    )

    if mlflow_spans:
        from mlflow.server.handlers import _get_tracking_store

        store = _get_tracking_store()

        try:
            store.log_spans(x_mlflow_experiment_id, mlflow_spans)
        except NotImplementedError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="REST OTLP span logging is not supported by the current tracing store",
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Cannot store OpenTelemetry spans",
            )

    return OTelExportTraceServiceResponse()
