"""
FastAPI security middleware configuration using native Starlette/FastAPI middleware.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp

from mlflow.environment_variables import (
    MLFLOW_ALLOW_INSECURE_CORS,
    MLFLOW_ALLOWED_HOSTS,
    MLFLOW_CORS_ALLOWED_ORIGINS,
    MLFLOW_HOST_HEADER_VALIDATION,
)

_logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware:
    """
    Middleware to add security headers to all responses.
    """

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
                message["headers"] = list(headers.items())
            await send(message)

        await self.app(scope, receive, send_wrapper)


def get_allowed_hosts() -> list[str]:
    """
    Get list of allowed hosts from environment or defaults.

    Returns:
        List of allowed host patterns.
    """
    if allowed_hosts_env := MLFLOW_ALLOWED_HOSTS.get():
        return [host.strip() for host in allowed_hosts_env.split(",")]

    # Default to localhost variants and private IP ranges
    # Starlette's TrustedHostMiddleware uses ":*" for port wildcards
    return [
        "localhost",
        "localhost:*",
        "127.0.0.1",
        "127.0.0.1:*",
        "[::1]",
        "[::1]:*",
        "0.0.0.0",
        "0.0.0.0:*",
        "192.168.*",
        "10.*",
        *[f"172.{i}.*" for i in range(16, 32)],
        "fc00:*",  # IPv6 private
        "fd00:*",  # IPv6 private
    ]


def get_allowed_origins() -> list[str]:
    """
    Get list of allowed CORS origins from environment or defaults.

    Returns:
        List of allowed origins.
    """
    origins = (
        [origin.strip() for origin in allowed_origins_env.split(",")]
        if (allowed_origins_env := MLFLOW_CORS_ALLOWED_ORIGINS.get())
        else []
    )

    localhost_origins = [
        f"http://{host}:{port}"
        for host in ["localhost", "127.0.0.1", "[::1]"]
        for port in ["3000", "5000", "8000", "8080"]
    ]

    origins.extend(origin for origin in localhost_origins if origin not in origins)
    return origins


def init_fastapi_security(app: FastAPI) -> None:
    """
    Initialize security middleware for FastAPI application using native middleware.

    This configures:
    - Host header validation (DNS rebinding protection) via TrustedHostMiddleware
    - CORS protection via CORSMiddleware
    - Security headers via custom middleware

    Args:
        app: FastAPI application instance.
    """
    allow_insecure_cors = MLFLOW_ALLOW_INSECURE_CORS.get() == "true"
    enable_host_validation = MLFLOW_HOST_HEADER_VALIDATION.get() != "false"

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
        origins = get_allowed_origins()
        _logger.info(f"CORS configured with origins: {origins[:5]}...")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

    if enable_host_validation:
        allowed_hosts = get_allowed_hosts()
        _logger.info(f"Host validation enabled with hosts: {allowed_hosts[:5]}...")
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=allowed_hosts,
        )
    else:
        _logger.warning(
            "Host header validation is DISABLED. "
            "This may leave the server vulnerable to DNS rebinding attacks."
        )

    _logger.info("FastAPI security middleware initialized successfully")
