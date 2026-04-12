"""
FastAPI application wrapper for MLflow server.

This module provides a FastAPI application that wraps the existing Flask application
using WSGIMiddleware to maintain 100% API compatibility while enabling future migration
to FastAPI endpoints.
"""

import json
import time

from fastapi import FastAPI, Request
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import JSONResponse
from flask import Flask

from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import MLFLOW_GATEWAY_DURATION_HEADER, MLFLOW_GATEWAY_OVERHEAD_HEADER
from mlflow.gateway.providers.utils import provider_call_duration_ms
from mlflow.server import app as flask_app
from mlflow.server.assistant.api import assistant_router
from mlflow.server.fastapi_security import init_fastapi_security
from mlflow.server.gateway_api import gateway_router
from mlflow.server.job_api import job_api_router
from mlflow.server.otel_api import otel_router
from mlflow.server.workspace_helpers import (
    WORKSPACE_HEADER_NAME,
    resolve_workspace_for_request_if_enabled,
)
from mlflow.utils.workspace_context import (
    clear_server_request_workspace,
    set_server_request_workspace,
)
from mlflow.version import VERSION


def add_fastapi_workspace_middleware(fastapi_app: FastAPI) -> None:
    if getattr(fastapi_app.state, "workspace_middleware_added", False):
        return

    @fastapi_app.middleware("http")
    async def workspace_context_middleware(request: Request, call_next):
        try:
            workspace = resolve_workspace_for_request_if_enabled(
                request.url.path,
                request.headers.get(WORKSPACE_HEADER_NAME),
            )
        except MlflowException as e:
            return JSONResponse(
                status_code=e.get_http_status_code(),
                content=json.loads(e.serialize_as_json()),
            )

        set_server_request_workspace(workspace.name if workspace else None)
        try:
            response = await call_next(request)
        finally:
            clear_server_request_workspace()
        return response

    fastapi_app.state.workspace_middleware_added = True


def add_gateway_timing_middleware(fastapi_app: FastAPI) -> None:
    if getattr(fastapi_app.state, "gateway_timing_middleware_added", False):
        return

    @fastapi_app.middleware("http")
    async def gateway_timing_middleware(request: Request, call_next):
        if not request.url.path.startswith("/gateway/"):
            return await call_next(request)

        # Reset the ContextVar so the handler task starts at 0. The handler task
        # inherits a copy of this context (Starlette's call_next uses copy_context),
        # so the reset is visible to send_request inside the handler.
        provider_call_duration_ms.set(0.0)
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)
        # Read provider duration relayed via request.state by _record_gateway_invocation.
        # We can't read the ContextVar directly here because the handler runs in a
        # separate task and ContextVar mutations don't propagate back.
        provider_duration_ms = int(getattr(request.state, "gateway_provider_duration_ms", 0))

        # For non-streaming responses, duration_ms covers the full round-trip.
        # For streaming responses, duration_ms covers only gateway setup time
        # (until the StreamingResponse object is returned, before the stream body
        # is iterated), so it reflects time-to-first-stream rather than total
        # streaming duration.
        response.headers[MLFLOW_GATEWAY_DURATION_HEADER] = str(duration_ms)
        if provider_duration_ms > 0:
            response.headers[MLFLOW_GATEWAY_OVERHEAD_HEADER] = str(
                max(0, duration_ms - provider_duration_ms)
            )
        return response

    fastapi_app.state.gateway_timing_middleware_added = True


def create_fastapi_app(flask_app: Flask = flask_app):
    """
    Create a FastAPI application that wraps the existing Flask app.

    Returns:
        FastAPI application instance with the Flask app mounted via WSGIMiddleware.
    """
    # Create FastAPI app with metadata
    fastapi_app = FastAPI(
        title="MLflow Tracking Server",
        description="MLflow Tracking Server API",
        version=VERSION,
        # TODO: Enable API documentation when we have native FastAPI endpoints
        # For now, disable docs since we only have Flask routes via WSGI
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Initialize security middleware BEFORE adding routes
    init_fastapi_security(fastapi_app)

    add_fastapi_workspace_middleware(fastapi_app)
    add_gateway_timing_middleware(fastapi_app)

    # Include OpenTelemetry API router BEFORE mounting Flask app
    # This ensures FastAPI routes take precedence over the catch-all Flask mount
    fastapi_app.include_router(otel_router)

    fastapi_app.include_router(job_api_router)

    # Include Gateway API router for database-backed endpoints
    # This provides /gateway/{endpoint_name}/mlflow/invocations routes
    fastapi_app.include_router(gateway_router)

    # Include Assistant API router for AI-powered trace analysis
    # This provides /ajax-api/3.0/mlflow/assistant/* endpoints (localhost only)
    fastapi_app.include_router(assistant_router)

    # Mount the entire Flask application at the root path
    # This ensures compatibility with existing APIs
    # NOTE: This must come AFTER include_router to avoid Flask catching all requests
    fastapi_app.mount("/", WSGIMiddleware(flask_app))

    return fastapi_app


# Create the app instance that can be used by ASGI servers
app = create_fastapi_app()
