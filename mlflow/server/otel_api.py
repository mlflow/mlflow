"""
OpenTelemetry REST API endpoints for MLflow FastAPI server.

This module implements the OpenTelemetry Protocol (OTLP) REST API for ingesting spans
according to the OTel specification:
https://opentelemetry.io/docs/specs/otlp/#otlphttp

Note: This is a minimal implementation that serves as a placeholder for the OTel endpoint.
The actual span ingestion logic would need to properly convert incoming OTel format spans
to MLflow spans, which requires more complex conversion logic.
"""

import logging
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
        ..., description="Collection of resource spans from instrumented applications"
    )


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
    request: OTelExportTraceServiceRequest,
    x_mlflow_experiment_id: str | None = Header(None, alias=MLFLOW_EXPERIMENT_ID_HEADER),
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
    try:
        # TODO: Implement conversion from OTel JSON format to MLflow spans
        # This would require:
        # 1. Getting the tracking store via _get_tracking_store()
        # 2. Parsing the incoming OTel format and creating MLflow Span objects
        # 3. Calling store.log_spans(experiment_id, spans) to persist them
        #    - The experiment_id is now available from the x_mlflow_experiment_id header

        # Log the experiment ID if provided (for debugging)
        if x_mlflow_experiment_id:
            logger = logging.getLogger(__name__)
            logger.debug(f"Received spans for experiment ID: {x_mlflow_experiment_id}")

        # For now, just return success
        return OTelExportTraceServiceResponse()

    except Exception as e:
        # Log the error and return an HTTP error response
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to export traces: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export traces: {e!s}",
        )
