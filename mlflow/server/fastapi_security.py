"""
FastAPI security middleware configuration using native Starlette/FastAPI middleware.
"""

import logging
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

from mlflow.environment_variables import MLFLOW_ALLOW_INSECURE_CORS, MLFLOW_HOST_HEADER_VALIDATION
from mlflow.server.security_utils import (
    CORS_BLOCKED_MSG,
    HEALTH_ENDPOINTS,
    INVALID_HOST_MSG,
    get_allowed_hosts_from_env,
    get_allowed_origins_from_env,
    get_default_allowed_hosts,
    is_api_endpoint,
    should_block_cors_request,
    validate_host_header,
)

_logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware:
    """Middleware to add security headers to all responses."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"x-content-type-options"] = b"nosniff"
                headers[b"x-frame-options"] = b"SAMEORIGIN"

                if (
                    scope["method"] == "OPTIONS"
                    and message.get("status") == 200
                    and is_api_endpoint(scope["path"])
                ):
                    message["status"] = HTTPStatus.NO_CONTENT

                message["headers"] = list(headers.items())
            await send(message)

        await self.app(scope, receive, send_wrapper)


class HostValidationMiddleware:
    """Middleware to validate Host headers and prevent DNS rebinding attacks."""

    def __init__(self, app: ASGIApp, allowed_hosts: list[str]):
        self.app = app
        self.allowed_hosts = allowed_hosts

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Check if this is a health endpoint that bypasses validation
        path = scope["path"]
        if path in HEALTH_ENDPOINTS:
            return await self.app(scope, receive, send)

        # Extract and validate host header
        headers = dict(scope["headers"])
        host = headers.get(b"host", b"").decode("utf-8")

        if not validate_host_header(self.allowed_hosts, host):
            _logger.warning(f"Rejected request with invalid Host header: {host}")
            await send(
                {
                    "type": "http.response.start",
                    "status": HTTPStatus.FORBIDDEN,
                    "headers": [[b"content-type", b"text/plain"]],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": INVALID_HOST_MSG.encode(),
                }
            )
            return

        await self.app(scope, receive, send)


class CORSBlockingMiddleware:
    """Middleware to actively block cross-origin state-changing requests."""

    def __init__(self, app: ASGIApp, allowed_origins: list[str]):
        self.app = app
        self.allowed_origins = allowed_origins

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        if not is_api_endpoint(scope["path"]):
            return await self.app(scope, receive, send)

        method = scope["method"]
        headers = dict(scope["headers"])
        origin = headers.get(b"origin", b"").decode("utf-8")

        if should_block_cors_request(origin, method, self.allowed_origins):
            _logger.warning(f"Blocked cross-origin request from {origin}")
            await send(
                {
                    "type": "http.response.start",
                    "status": HTTPStatus.FORBIDDEN,
                    "headers": [[b"content-type", b"text/plain"]],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": CORS_BLOCKED_MSG.encode(),
                }
            )
            return

        await self.app(scope, receive, send)


def get_allowed_hosts() -> list[str]:
    """Get list of allowed hosts from environment or defaults."""
    return get_allowed_hosts_from_env() or get_default_allowed_hosts()


def get_allowed_origins() -> list[str]:
    """Get list of allowed CORS origins from environment or defaults."""
    return get_allowed_origins_from_env() or []


def init_fastapi_security(app: FastAPI) -> None:
    """
    Initialize security middleware for FastAPI application.

    This configures:
    - Host header validation (DNS rebinding protection) via custom middleware
    - CORS protection via CORSMiddleware
    - Security headers via custom middleware

    Args:
        app: FastAPI application instance.
    """
    allow_insecure_cors = MLFLOW_ALLOW_INSECURE_CORS.get() == "true"
    enable_host_validation = MLFLOW_HOST_HEADER_VALIDATION.get() != "false"

    allowed_origins = get_allowed_origins()

    app.add_middleware(SecurityHeadersMiddleware)

    if allow_insecure_cors:
        _logger.warning(
            "Running MLflow server with INSECURE CORS mode enabled. "
            "This allows ALL origins and should only be used for development!"
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
    else:
        origins_display = allowed_origins[:5] if allowed_origins else ["localhost (any port)"]
        _logger.info(f"CORS configured with origins: {origins_display}...")
        app.add_middleware(CORSBlockingMiddleware, allowed_origins=allowed_origins)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

    if enable_host_validation:
        allowed_hosts = get_allowed_hosts()
        _logger.info(f"Host validation enabled with hosts: {allowed_hosts[:5]}...")
        app.add_middleware(HostValidationMiddleware, allowed_hosts=allowed_hosts)
    else:
        _logger.warning(
            "Host header validation is DISABLED. "
            "This may leave the server vulnerable to DNS rebinding attacks."
        )

    _logger.info("FastAPI security middleware initialized successfully")
