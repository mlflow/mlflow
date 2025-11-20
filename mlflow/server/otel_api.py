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
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from pydantic import BaseModel, Field

from mlflow.entities.span import Span
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.server.handlers import _get_tracking_store
from mlflow.server.workspace_helpers import _get_workspace_store, _workspaces_enabled_flag
from mlflow.store.workspace.utils import get_default_workspace_optional
from mlflow.telemetry.events import TraceSource, TracesReceivedByServerEvent
from mlflow.telemetry.track import _record_event
from mlflow.tracing.utils.otlp import (
    MLFLOW_EXPERIMENT_ID_HEADER,
    OTLP_TRACES_PATH,
    OTLP_TRACES_WORKSPACE_PATH,
)
from mlflow.tracking._workspace import context as workspace_context
from mlflow.tracking.request_header.default_request_header_provider import (
    _MLFLOW_PYTHON_CLIENT_USER_AGENT_PREFIX,
    _USER_AGENT,
)

_logger = logging.getLogger(__name__)

# Create FastAPI router for OTel endpoints
otel_router = APIRouter(tags=["OpenTelemetry"])


class OTelExportTraceServiceResponse(BaseModel):
    """
    Pydantic model for the OTLP/HTTP ExportTraceServiceResponse.

    This matches the OpenTelemetry protocol specification for trace export responses.
    Reference: https://opentelemetry.io/docs/specs/otlp/
    """

    partialSuccess: dict[str, Any] | None = Field(
        None, description="Details about partial success of the export operation"
    )


@otel_router.post(OTLP_TRACES_PATH, response_model=OTelExportTraceServiceResponse, status_code=200)
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

    with _workspace_request_context(request, None):
        return await _ingest_traces(request, response, x_mlflow_experiment_id, content_type)


@otel_router.post(
    OTLP_TRACES_WORKSPACE_PATH,
    response_model=OTelExportTraceServiceResponse,
    status_code=200,
)
async def export_traces_for_workspace(
    workspace_name: str,
    request: Request,
    response: Response,
    x_mlflow_experiment_id: str = Header(..., alias=MLFLOW_EXPERIMENT_ID_HEADER),
    content_type: str = Header(None),
) -> OTelExportTraceServiceResponse:
    """
    Workspace-prefixed OTLP ingestion endpoint.
    """

    with _workspace_request_context(request, workspace_name):
        return await _ingest_traces(request, response, x_mlflow_experiment_id, content_type)


async def _ingest_traces(
    request: Request,
    response: Response,
    x_mlflow_experiment_id: str,
    content_type: str | None,
) -> OTelExportTraceServiceResponse:
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
        completed_trace_ids = set()
        for trace_id, trace_spans in spans_by_trace_id.items():
            try:
                store.log_spans(x_mlflow_experiment_id, trace_spans)
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
            user_agent_value = request.headers.get(_USER_AGENT)
            trace_source = (
                TraceSource.MLFLOW_PYTHON_CLIENT
                if user_agent_value
                and user_agent_value.startswith(_MLFLOW_PYTHON_CLIENT_USER_AGENT_PREFIX)
                else TraceSource.UNKNOWN
            )

            _record_event(
                TracesReceivedByServerEvent,
                {
                    "source": trace_source,
                    "count": len(completed_trace_ids),
                },
            )

    return OTelExportTraceServiceResponse()


@contextmanager
def _workspace_request_context(request: Request, workspace_name: str | None):
    if not _workspaces_enabled_flag():
        yield None
        return

    store = _get_workspace_store()
    try:
        if workspace_name:
            workspace = store.get_workspace(workspace_name, request)
        else:
            workspace, _ = get_default_workspace_optional(store, request, logger=_logger)
            if workspace is None:
                raise MlflowException(
                    "Active workspace is required. Prefix the request path with "
                    "'/workspaces/<workspace>' or configure a default workspace.",
                    error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
                )

        with workspace_context.WorkspaceContext(workspace.name):
            yield workspace
    except MlflowException as e:
        # Convert MlflowException to HTTPException for proper HTTP response
        raise HTTPException(status_code=e.get_http_status_code(), detail=e.message)
