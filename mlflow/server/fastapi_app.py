"""
FastAPI application wrapper for MLflow server.

This module provides a FastAPI application that wraps the existing Flask application
using WSGIMiddleware to maintain 100% API compatibility while enabling future migration
to FastAPI endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from flask import Flask
from starlette.types import ASGIApp, Receive, Scope, Send

from mlflow.exceptions import MlflowException
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


class WorkspaceContextMiddleware:
    """Pure ASGI middleware for workspace context.

    Unlike @app.middleware("http") which uses Starlette's BaseHTTPMiddleware
    (spawning a background task per request), this passes the ASGI scope/receive/send
    directly — avoiding the task-spawning overhead that causes ~30ms latency under
    concurrent load.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # ASGI headers are list of (name, value) byte pairs, names are lowercase
        workspace_header_key = WORKSPACE_HEADER_NAME.lower().encode()
        workspace_header = None
        for name, value in scope.get("headers", []):
            if name == workspace_header_key:
                workspace_header = value.decode("utf-8")
                break
        path = scope.get("path", "")

        try:
            workspace = resolve_workspace_for_request_if_enabled(path, workspace_header)
        except MlflowException as e:
            status = e.get_http_status_code()
            body = e.serialize_as_json().encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return

        set_server_request_workspace(workspace.name if workspace else None)
        try:
            await self.app(scope, receive, send)
        finally:
            clear_server_request_workspace()


def add_fastapi_workspace_middleware(fastapi_app: FastAPI) -> None:
    if getattr(fastapi_app.state, "workspace_middleware_added", False):
        return
    fastapi_app.add_middleware(WorkspaceContextMiddleware)
    fastapi_app.state.workspace_middleware_added = True


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
