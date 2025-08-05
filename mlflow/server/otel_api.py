"""
OpenTelemetry REST API endpoints for MLflow FastAPI server.

This module implements the OpenTelemetry Protocol (OTLP) REST API for ingesting spans
according to the OTel specification:
https://opentelemetry.io/docs/specs/otlp/#otlphttp
"""

import base64
import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from mlflow.entities.span import Span, create_mlflow_span
from mlflow.server.handlers import _get_tracking_store
from mlflow.tracing.utils import encode_trace_id

# Import OpenTelemetry protobuf definitions
try:
    pass  # We don't actually use the protobuf imports directly in this minimal version
except ImportError as e:
    raise ImportError(
        "OpenTelemetry protobuf definitions are required for OTel API support. "
        "Please install them with: pip install opentelemetry-proto"
    ) from e

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

    partialSuccess: Optional[dict[str, Any]] = Field(
        None, description="Details about partial success of the export operation"
    )


def _convert_otel_span_to_mlflow_span(
    otel_span: dict[str, Any], trace_id: str, resource_attrs: Optional[list[dict[str, Any]]] = None
) -> Span:
    """
    Convert an OpenTelemetry span from the OTLP format to an MLflow Span entity.

    Args:
        otel_span: OpenTelemetry span in dictionary format
        trace_id: MLflow trace ID
        resource_attrs: Optional resource attributes

    Returns:
        MLflow Span entity
    """
    # OpenTelemetry uses base64-encoded trace and span IDs in JSON format
    # We need to decode these and convert to MLflow's hex format

    # Decode trace and span IDs from base64
    otel_trace_id_bytes = base64.b64decode(otel_span.get("traceId", ""))
    otel_span_id_bytes = base64.b64decode(otel_span.get("spanId", ""))

    # Convert to integers for OpenTelemetry span context
    otel_trace_id = int.from_bytes(otel_trace_id_bytes, byteorder="big")
    otel_span_id = int.from_bytes(otel_span_id_bytes, byteorder="big")

    # Get parent span ID if present
    parent_span_id = None
    if parent_id_b64 := otel_span.get("parentSpanId"):
        parent_span_id_bytes = base64.b64decode(parent_id_b64)
        parent_span_id = int.from_bytes(parent_span_id_bytes, byteorder="big")

    # Extract trace state
    trace_state = otel_span.get("traceState", "")

    # Convert timestamps from string to int (nanoseconds)
    start_time = int(otel_span.get("startTimeUnixNano", 0))
    end_time = int(otel_span.get("endTimeUnixNano", 0)) if "endTimeUnixNano" in otel_span else None

    # Create OpenTelemetry span context
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.trace.span import TraceState

    # Parse trace state if present
    otel_trace_state = None
    if trace_state:
        # Parse trace state string (format: "key1=value1,key2=value2")
        trace_state_items = []
        for item in trace_state.split(","):
            if "=" in item:
                key, value = item.split("=", 1)
                trace_state_items.append((key.strip(), value.strip()))
        if trace_state_items:
            otel_trace_state = TraceState(trace_state_items)

    # Create span context
    span_context = trace_api.SpanContext(
        trace_id=otel_trace_id,
        span_id=otel_span_id,
        is_remote=True,
        trace_flags=trace_api.TraceFlags(int(otel_span.get("flags", 0))),
        trace_state=otel_trace_state,
    )

    # Create parent context if parent span ID exists
    parent_context = None
    if parent_span_id is not None:
        parent_context = trace_api.SpanContext(
            trace_id=otel_trace_id,
            span_id=parent_span_id,
            is_remote=True,
            trace_flags=trace_api.TraceFlags(0),
        )

    # Extract attributes
    attributes = {}
    span_type = None  # Initialize span_type

    if "attributes" in otel_span:
        for attr in otel_span["attributes"]:
            key = attr.get("key", "")
            value = attr.get("value", {})
            # Handle different value types
            if "stringValue" in value:
                attributes[key] = value["stringValue"]

                # Extract span type if this is the mlflow.spanType attribute
                if key == "mlflow.spanType":
                    span_type = value["stringValue"]

            elif "intValue" in value:
                attributes[key] = str(value["intValue"])
            elif "doubleValue" in value:
                attributes[key] = str(value["doubleValue"])
            elif "boolValue" in value:
                attributes[key] = str(value["boolValue"])
            elif "arrayValue" in value:
                # Convert array values to JSON string
                attributes[key] = json.dumps(value["arrayValue"])
            elif "kvlistValue" in value:
                # Convert kvlist values to JSON string
                attributes[key] = json.dumps(value["kvlistValue"])

    # Ensure trace request ID is set
    attributes["mlflow.traceRequestId"] = json.dumps(trace_id)

    # Add experiment ID from resource attributes if provided
    if resource_attrs:
        for attr in resource_attrs:
            if attr.get("key") == "mlflow.experimentId":
                attributes["mlflow.experimentId"] = json.dumps(
                    attr.get("value", {}).get("stringValue", "0")
                )
                break

    # Extract status
    status_code = trace_api.StatusCode.UNSET
    status_message = None
    if "status" in otel_span:
        status_dict = otel_span["status"]
        code = status_dict.get("code", "STATUS_CODE_UNSET")
        # Map OTel status codes to OpenTelemetry StatusCode enum
        if code == "STATUS_CODE_OK" or code == 1:
            status_code = trace_api.StatusCode.OK
        elif code == "STATUS_CODE_ERROR" or code == 2:
            status_code = trace_api.StatusCode.ERROR
        status_message = status_dict.get("message")

    # Create a ReadableSpan
    readable_span = ReadableSpan(
        name=otel_span.get("name", ""),
        context=span_context,
        parent=parent_context,
        attributes=attributes,
        start_time=start_time,
        end_time=end_time,
        status=trace_api.Status(status_code, status_message),
    )

    # Convert to MLflow span with span_type
    return create_mlflow_span(readable_span, trace_id, span_type)


@otel_router.post("", response_model=OTelExportTraceServiceResponse, status_code=200)
async def export_traces(request: OTelExportTraceServiceRequest) -> OTelExportTraceServiceResponse:
    """
    Export trace spans to MLflow via the OpenTelemetry protocol.

    This endpoint accepts OTLP/HTTP trace export requests and logs the spans
    to the MLflow tracking store.

    Args:
        request: OTel ExportTraceServiceRequest in JSON format

    Returns:
        OTel ExportTraceServiceResponse indicating success or partial success

    Raises:
        HTTPException: If the request is invalid or span logging fails
    """
    try:
        # Get the tracking store
        store = _get_tracking_store()

        # Process each resource span batch
        for resource_span in request.resourceSpans:
            # Extract resource attributes if present
            resource_attrs = None
            if "resource" in resource_span and "attributes" in resource_span["resource"]:
                resource_attrs = resource_span["resource"]["attributes"]

            # Process each scope span
            for scope_span in resource_span.get("scopeSpans", []):
                # Convert OTel spans to MLflow spans
                mlflow_spans = []
                trace_id = None

                for otel_span_dict in scope_span.get("spans", []):
                    # Extract trace ID from first span to use for all spans
                    if trace_id is None and "traceId" in otel_span_dict:
                        trace_id_bytes = base64.b64decode(otel_span_dict["traceId"])
                        trace_id_int = int.from_bytes(trace_id_bytes, byteorder="big")
                        # Convert to MLflow trace ID format (hex string)
                        trace_id = encode_trace_id(trace_id_int)

                    # Convert to MLflow span
                    mlflow_span = _convert_otel_span_to_mlflow_span(
                        otel_span_dict, trace_id, resource_attrs
                    )
                    mlflow_spans.append(mlflow_span)

                # Log spans to the tracking store
                if mlflow_spans:
                    await store.log_spans(mlflow_spans)

        # Return success response
        return OTelExportTraceServiceResponse()

    except Exception as e:
        # Log the error and return an HTTP error response
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to export traces: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export traces: {e!s}",
        )
