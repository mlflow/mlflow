"""
FastAPI security middleware configuration using native Starlette/FastAPI middleware.
"""

import logging
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp

from mlflow.environment_variables import MLFLOW_DISABLE_SECURITY_MIDDLEWARE, MLFLOW_X_FRAME_OPTIONS
from mlflow.server.security_utils import (
    CORS_BLOCKED_MSG,
    get_allowed_hosts_from_env,
    get_allowed_origins_from_env,
    get_default_allowed_hosts,
    is_api_endpoint,
    should_block_cors_request,
)

_logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware:
    """Middleware to add security headers to all responses."""

    def __init__(self, app: ASGIApp):
        self.app = app
        self.x_frame_options = MLFLOW_X_FRAME_OPTIONS.get()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"x-content-type-options"] = b"nosniff"

                # Only add X-Frame-Options if not set to "NONE"
                if self.x_frame_options and self.x_frame_options.upper() != "NONE":
                    headers[b"x-frame-options"] = self.x_frame_options.upper().encode()

                if (
                    scope["method"] == "OPTIONS"
                    and message.get("status") == 200
                    and is_api_endpoint(scope["path"])
                ):
                    message["status"] = HTTPStatus.NO_CONTENT

                message["headers"] = list(headers.items())
            await send(message)

        await self.app(scope, receive, send_wrapper)


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
    - Host header validation (DNS rebinding protection) via TrustedHostMiddleware
    - CORS protection via CORSMiddleware
    - Security headers via custom middleware

    Args:
        app: FastAPI application instance.
    """
    # Check if security middleware should be completely disabled
    if MLFLOW_DISABLE_SECURITY_MIDDLEWARE.get() == "true":
        _logger.warning(
            "Security middleware is DISABLED. "
            "This may leave the server vulnerable to various attacks."
        )
        return

    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)

    # Configure CORS
    allowed_origins = get_allowed_origins()

    if allowed_origins and "*" in allowed_origins:
        _logger.warning(
            "Running MLflow server with CORS allowing ALL origins. "
            "This should only be used for development!"
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

    # Configure Host header validation
    allowed_hosts = get_allowed_hosts()

    if allowed_hosts and "*" not in allowed_hosts:
        _logger.info(f"Host validation enabled with hosts: {allowed_hosts[:5]}...")
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    else:
        if "*" in allowed_hosts:
            _logger.warning(
                "Host header validation accepts ALL hosts. "
                "This may leave the server vulnerable to DNS rebinding attacks."
            )

    _logger.info("FastAPI security middleware initialized successfully")
