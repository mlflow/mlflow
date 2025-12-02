"""
FastAPI application wrapper for MLflow server.

This module provides a FastAPI application that wraps the existing Flask application
using WSGIMiddleware to maintain 100% API compatibility while enabling future migration
to FastAPI endpoints.
"""

import json

from fastapi import FastAPI, Request
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import JSONResponse
from flask import Flask

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.server import app as flask_app
from mlflow.server.fastapi_security import init_fastapi_security
from mlflow.server.job_api import job_api_router
from mlflow.server.otel_api import otel_router
from mlflow.server.workspace_helpers import WORKSPACE_HEADER_NAME, resolve_workspace_from_header
from mlflow.tracing.utils.otlp import OTLP_TRACES_PATH
from mlflow.utils.workspace_context import reset_workspace, set_current_workspace
from mlflow.version import VERSION

# FastAPI routes that do not go through the Flask WSGI bridge (currently jobs + OTLP).
FASTAPI_NATIVE_PREFIXES = (job_api_router.prefix, OTLP_TRACES_PATH)


def add_fastapi_workspace_middleware(fastapi_app: FastAPI) -> None:
    if getattr(fastapi_app.state, "workspace_middleware_added", False):
        return

    @fastapi_app.middleware("http")
    async def workspace_context_middleware(request: Request, call_next):
        if not MLFLOW_ENABLE_WORKSPACES.get():
            return await call_next(request)

        path = request.url.path
        if not any(path.startswith(prefix) for prefix in FASTAPI_NATIVE_PREFIXES):
            # Skip if it's a Flask route and let the Flask before request handler handle it.
            return await call_next(request)

        try:
            workspace = resolve_workspace_from_header(request.headers.get(WORKSPACE_HEADER_NAME))
            if workspace is None:
                raise MlflowException(
                    f"Active workspace is required. Set the '{WORKSPACE_HEADER_NAME}' request "
                    "header, call mlflow.set_workspace(), or set the MLFLOW_WORKSPACE environment "
                    "variable before making requests.",
                    error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
                )
        except MlflowException as e:
            return JSONResponse(
                status_code=e.get_http_status_code(),
                content=json.loads(e.serialize_as_json()),
            )

        token = set_current_workspace(workspace.name)
        try:
            response = await call_next(request)
        finally:
            reset_workspace(token)
        return response

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

    # Mount the entire Flask application at the root path
    # This ensures compatibility with existing APIs
    # NOTE: This must come AFTER include_router to avoid Flask catching all requests
    fastapi_app.mount("/", WSGIMiddleware(flask_app))

    return fastapi_app


# Create the app instance that can be used by ASGI servers
app = create_fastapi_app()
