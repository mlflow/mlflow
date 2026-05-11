"""
FastAPI application wrapper for MLflow server.

This module provides a FastAPI application that wraps the existing Flask application
using WSGIMiddleware to maintain 100% API compatibility while enabling future migration
to FastAPI endpoints.
"""

import json
import time
import typing

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from flask import Flask
from starlette.middleware.wsgi import WSGIResponder, build_environ
from starlette.types import Receive, Scope, Send

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


class _EfficientWSGIResponder(WSGIResponder):
    """WSGIResponder with O(n) body buffering instead of O(n^2) concatenation.

    Starlette's WSGIMiddleware is deprecated and upstream has declined to fix the
    quadratic body buffering (see https://github.com/Kludex/starlette/pull/2450,
    closed in favor of deprecating the module entirely).

    Ref: https://github.com/Kludex/starlette/blob/0e88e92b592bfa11fd92e331869a8d49ba34b541/starlette/middleware/wsgi.py#L98-L117
    """

    async def __call__(self, receive: Receive, send: Send) -> None:
        # >>> Changed from original: use list + join instead of body += chunk
        chunks: list[bytes] = []
        more_body = True
        while more_body:
            message = await receive()
            if chunk := message.get("body", b""):
                chunks.append(chunk)
            more_body = message.get("more_body", False)
        body = b"".join(chunks)
        del chunks  # Free chunk list before build_environ copies body into BytesIO
        # <<< End of change
        environ = build_environ(self.scope, body)

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(self.sender, send)
            async with self.stream_send:
                await anyio.to_thread.run_sync(self.wsgi, environ, self.start_response)
        if self.exc_info is not None:
            raise self.exc_info[0].with_traceback(self.exc_info[1], self.exc_info[2])


class _EfficientWSGIMiddleware:
    """Drop-in replacement for starlette's WSGIMiddleware that avoids O(n^2) body buffering."""

    def __init__(self, app: typing.Callable[..., typing.Any]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        assert scope["type"] == "http"
        responder = _EfficientWSGIResponder(self.app, scope)
        await responder(receive, send)


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
    fastapi_app.mount("/", _EfficientWSGIMiddleware(flask_app))

    return fastapi_app


# Create the app instance that can be used by ASGI servers
app = create_fastapi_app()
