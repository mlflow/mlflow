"""
OpenTelemetry REST API endpoints for MLflow FastAPI server.

This module implements the OpenTelemetry Protocol (OTLP) REST API for ingesting spans
according to the OTel specification:
https://opentelemetry.io/docs/specs/otlp/#otlphttp
"""

import asyncio
import base64
import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from mlflow.entities.span import Span, create_mlflow_span
from mlflow.server.handlers import _get_tracking_store
from mlflow.tracing.utils import encode_trace_id

# Import OpenTelemetry protobuf definitions
try:
    from google.protobuf.json_format import MessageToDict, ParseDict
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
        ExportTraceServiceResponse,
    )
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


def get_tracking_store():
    """FastAPI dependency to get the tracking store instance."""
    return _get_tracking_store()


@otel_router.post("", response_model=OTelExportTraceServiceResponse)
async def export_traces(
    request: OTelExportTraceServiceRequest,
    tracking_store=Depends(get_tracking_store),
) -> OTelExportTraceServiceResponse:
    """
    Export trace data in OpenTelemetry format.

    This endpoint implements the OTLP/HTTP protocol for receiving trace data.
    It converts OpenTelemetry spans to MLflow spans and stores them using
    the tracking store's log_span method.

    Args:
        request: OTLP ExportTraceServiceRequest containing spans to export
        tracking_store: MLflow tracking store instance

    Returns:
        OTLP ExportTraceServiceResponse indicating success or partial success
    """
    failed_count = 0
    error_messages = []

    try:
        # Process each resource span batch
        for resource_span in request.resourceSpans:
            # Extract resource attributes if needed
            resource_attrs = resource_span.get("resource", {}).get("attributes", [])

            # Process scope spans
            for scope_span in resource_span.get("scopeSpans", []):
                # Process individual spans
                # Group spans by trace ID for efficient batch processing
                spans_by_trace = {}
                for otel_span in scope_span.get("spans", []):
                    try:
                        # Extract trace ID from the span and convert to MLflow format
                        trace_id_b64 = otel_span.get("traceId", "")
                        if not trace_id_b64:
                            failed_count += 1
                            error_messages.append("Span missing trace ID")
                            continue

                        # Decode and convert to MLflow trace ID format
                        trace_id_bytes = base64.b64decode(trace_id_b64)
                        trace_id_int = int.from_bytes(trace_id_bytes, byteorder="big")
                        trace_id = f"tr-{encode_trace_id(trace_id_int)}"

                        # Convert OTel span to MLflow span
                        mlflow_span = _convert_otel_span_to_mlflow_span(
                            otel_span, trace_id, resource_attrs
                        )

                        # Group by trace ID
                        if trace_id not in spans_by_trace:
                            spans_by_trace[trace_id] = []
                        spans_by_trace[trace_id].append(mlflow_span)

                    except Exception as e:
                        failed_count += 1
                        error_messages.append(f"Failed to process span: {e!s}")

                # Log spans for each trace
                tasks = []
                for trace_id, trace_spans in spans_by_trace.items():
                    # Create async task to log the spans for this trace
                    task = asyncio.create_task(tracking_store.log_spans(trace_spans))
                    tasks.append((task, len(trace_spans)))

                # Wait for all span logging tasks to complete
                if tasks:
                    results = await asyncio.gather(
                        *[task for task, _ in tasks], return_exceptions=True
                    )
                    for (task, span_count), result in zip(tasks, results):
                        if isinstance(result, Exception):
                            failed_count += span_count
                            error_messages.append(f"Failed to log spans: {result!s}")

    except Exception as e:
        # If there's a general processing error, return 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process trace export request: {e!s}",
        )

    # Build response
    response = OTelExportTraceServiceResponse()

    # If some spans failed, include partial success information
    if failed_count > 0:
        response.partialSuccess = {
            "rejectedSpans": failed_count,
            "errorMessage": "; ".join(error_messages[:5]),  # Limit error messages
        }

    return response


@otel_router.post("/protobuf", response_class=Response)
async def export_traces_protobuf(
    request: Request,
    tracking_store=Depends(get_tracking_store),
) -> Response:
    """
    Export trace data in OpenTelemetry protobuf format.

    This endpoint accepts binary protobuf data and processes it the same way
    as the JSON endpoint.

    Args:
        request: FastAPI request containing binary protobuf data
        tracking_store: MLflow tracking store instance

    Returns:
        Binary protobuf response
    """
    try:
        # Read the binary protobuf data
        body = await request.body()

        # Parse the protobuf request
        proto_request = ExportTraceServiceRequest()
        proto_request.ParseFromString(body)

        # Convert protobuf to dict for processing
        request_dict = MessageToDict(proto_request, preserving_proto_field_name=True)

        # Create pydantic model from dict
        pydantic_request = OTelExportTraceServiceRequest(**request_dict)

        # Process using the same logic as JSON endpoint
        response = await export_traces(pydantic_request, tracking_store)

        # Convert response to protobuf
        proto_response = ExportTraceServiceResponse()
        if response.partialSuccess:
            ParseDict(response.partialSuccess, proto_response.partial_success)

        # Return binary protobuf response
        return Response(
            content=proto_response.SerializeToString(),
            media_type="application/x-protobuf",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process protobuf trace export request: {e!s}",
        )
