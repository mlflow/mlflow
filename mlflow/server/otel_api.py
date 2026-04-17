"""
OpenTelemetry REST API endpoints for MLflow FastAPI server.

This module implements the OpenTelemetry Protocol (OTLP) REST API for ingesting spans
according to the OTel specification:
https://opentelemetry.io/docs/specs/otlp/#otlphttp

Note: This is a minimal implementation that serves as a placeholder for the OTel endpoint.
The actual span ingestion logic would need to properly convert incoming OTel format spans
to MLflow spans, which requires more complex conversion logic.
"""

import json
import logging

from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
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
from mlflow.tracing.utils import dump_span_attribute_value
from mlflow.tracing.utils.otlp import (
    MLFLOW_EXPERIMENT_ID_HEADER,
    OTLP_TRACES_PATH,
    _decode_otel_proto_anyvalue,
    decompress_otlp_body,
)
from mlflow.tracking.request_header.default_request_header_provider import (
    _MLFLOW_PYTHON_CLIENT_USER_AGENT_PREFIX,
    _USER_AGENT,
)

_logger = logging.getLogger(__name__)

# Allowlist of known OTEL client service names.
# Only service names on this list are stored and propagated to root spans.
# This prevents storing arbitrary free-form text from untrusted clients.
_KNOWN_SERVICE_NAMES = frozenset({
    # Claude Code
    "claude-code",
    # Codex CLI (Rust)
    "codex_cli_rs",
    # Codex VS Code extension
    "codex_vscode",
    # Gemini CLI
    "gemini-cli",
    # Qwen Code
    "qwen-code",
})

# Create FastAPI router for OTel endpoints
otel_router = APIRouter(prefix=OTLP_TRACES_PATH, tags=["OpenTelemetry"])


@otel_router.post("", status_code=200)
async def export_traces(
    request: Request,
    x_mlflow_experiment_id: str = Header(..., alias=MLFLOW_EXPERIMENT_ID_HEADER),
    content_type: str | None = Header(default=None),
    content_encoding: str | None = Header(default=None),
    user_agent: str | None = Header(None, alias=_USER_AGENT),
) -> Response:
    """
    Export trace spans to MLflow via the OpenTelemetry protocol.

    This endpoint accepts OTLP/HTTP protobuf trace export requests.
    Protobuf format reference: https://opentelemetry.io/docs/specs/otlp/#binary-protobuf-encoding

    Note: All spans in the batch are persisted in a single log_spans() call. If that
    call fails, the entire batch is rejected (all-or-nothing). Partial-success is not
    supported; clients that need per-trace error isolation should batch by trace.

    Args:
        request: OTel ExportTraceServiceRequest in protobuf format
        x_mlflow_experiment_id: Required header containing the experiment ID
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

    all_spans = []
    completed_trace_ids = set()
    service_names = set()
    for resource_span in parsed_request.resource_spans:
        # Extract service.name from resource attributes for telemetry and root span propagation
        resource_service_name = None
        for attr in resource_span.resource.attributes:
            if attr.key == "service.name":
                value = _decode_otel_proto_anyvalue(attr.value)
                if value is not None and str(value) in _KNOWN_SERVICE_NAMES:
                    resource_service_name = str(value)
                    service_names.add(resource_service_name)
                break

        for scope_span in resource_span.scope_spans:
            for otel_proto_span in scope_span.spans:
                try:
                    mlflow_span = Span.from_otel_proto(otel_proto_span)

                    # Propagate service.name onto root spans so it's visible
                    # in the UI. Per the OTel resource spec, resource attrs
                    # describe the entity producing telemetry:
                    # https://opentelemetry.io/docs/specs/otel/resource/sdk/
                    if mlflow_span.parent_id is None:
                        completed_trace_ids.add(mlflow_span.trace_id)
                        if resource_service_name:
                            mlflow_span._span._attributes["service.name"] = (
                                dump_span_attribute_value(resource_service_name)
                            )

                    all_spans.append(mlflow_span)
                except Exception:
                    raise HTTPException(
                        status_code=422,
                        detail="Cannot convert OpenTelemetry span to MLflow span",
                    )

    if all_spans:
        store = _get_tracking_store()

        try:
            store.log_spans(x_mlflow_experiment_id, all_spans)
        except NotImplementedError:
            store_name = store.__class__.__name__
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"REST OTLP span logging is not supported by {store_name}",
            )
        except MlflowException as e:
            return JSONResponse(
                status_code=e.get_http_status_code(),
                content=json.loads(e.serialize_as_json()),
            )
        except Exception:
            trace_ids = {s.trace_id for s in all_spans}
            _logger.exception("Failed to log OpenTelemetry spans for trace(s): %s", trace_ids)
            raise HTTPException(
                status_code=422,
                detail="Failed to log OpenTelemetry spans",
            )

        if completed_trace_ids:
            if user_agent and user_agent.startswith(_MLFLOW_PYTHON_CLIENT_USER_AGENT_PREFIX):
                trace_source = TraceSource.MLFLOW_PYTHON_CLIENT
            elif service_names:
                trace_source = TraceSource.EXTERNAL_OTEL_CLIENT
            else:
                trace_source = TraceSource.UNKNOWN

            event_params: dict[str, object] = {
                "source": trace_source,
                "count": len(completed_trace_ids),
            }
            if service_names:
                event_params["service_names"] = sorted(service_names)

            _record_event(TracesReceivedByServerEvent, event_params)

    # Return protobuf response as per OTLP specification
    response_message = ExportTraceServiceResponse()
    response_bytes = response_message.SerializeToString()
    return Response(
        content=response_bytes,
        media_type="application/x-protobuf",
        status_code=200,
    )
