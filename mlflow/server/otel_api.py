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

from fastapi import APIRouter, Header, HTTPException, Request, status
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from pydantic import BaseModel, Field

from mlflow.entities.span import Span
from mlflow.server.handlers import _get_tracking_store
from mlflow.tracing.utils.otlp import MLFLOW_EXPERIMENT_ID_HEADER

# Create FastAPI router for OTel endpoints
otel_router = APIRouter(prefix="/v1/traces", tags=["OpenTelemetry"])


class OTelExportTraceServiceResponse(BaseModel):
    """
    Pydantic model for the OTLP/HTTP ExportTraceServiceResponse.

    This matches the OpenTelemetry protocol specification for trace export responses.
    """

    partialSuccess: dict[str, Any] | None = Field(
        None, description="Details about partial success of the export operation"
    )


@otel_router.post("", response_model=OTelExportTraceServiceResponse, status_code=200)
async def export_traces(
    request: Request,
    x_mlflow_experiment_id: str = Header(..., alias=MLFLOW_EXPERIMENT_ID_HEADER),
) -> OTelExportTraceServiceResponse:
    """
    Export trace spans to MLflow via the OpenTelemetry protocol.

    This endpoint accepts OTLP/HTTP protobuf trace export requests.

    Args:
        request: OTel ExportTraceServiceRequest in protobuf format
        x_mlflow_experiment_id: Required header containing the experiment ID

    Returns:
        OTel ExportTraceServiceResponse indicating success

    Raises:
        HTTPException: If the request is invalid or span logging fails
    """
    body = await request.body()
    parsed_request = ExportTraceServiceRequest()

    try:
        parsed_request.ParseFromString(body)
    except DecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OpenTelemetry protobuf format",
        )

    mlflow_spans = []
    for resource_span in parsed_request.resource_spans:
        for scope_span in resource_span.scope_spans:
            for otel_proto_span in scope_span.spans:
                try:
                    mlflow_span = Span._from_otel_proto(otel_proto_span)
                    mlflow_spans.append(mlflow_span)
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="Cannot convert OpenTelemetry span to MLflow span",
                    )

    if mlflow_spans:
        store = _get_tracking_store()

        try:
            store.log_spans(x_mlflow_experiment_id, mlflow_spans)
        except NotImplementedError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="REST OTLP span logging is not supported by the current tracing store",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Cannot store OpenTelemetry spans: {e}",
            )

    return OTelExportTraceServiceResponse()
