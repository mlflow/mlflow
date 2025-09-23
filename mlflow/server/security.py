"""
Flask security middleware for MLflow server.

This module provides security middleware for Flask applications to prevent
common web vulnerabilities. For FastAPI applications, use fastapi_security.py
which leverages native Starlette middleware.
"""

import fnmatch
import logging

from flask import Flask, Response, request
from flask_cors import CORS

from mlflow.environment_variables import (
    MLFLOW_ALLOW_INSECURE_CORS,
    MLFLOW_ALLOWED_HOSTS,
    MLFLOW_CORS_ALLOWED_ORIGINS,
    MLFLOW_HOST_HEADER_VALIDATION,
)

_logger = logging.getLogger(__name__)


def get_allowed_hosts() -> list[str]:
    """
    Get list of allowed hosts from environment or defaults.

    Returns:
        List of allowed host patterns.
    """
    if allowed_hosts_env := MLFLOW_ALLOWED_HOSTS.get():
        return [host.strip() for host in allowed_hosts_env.split(",")]

    localhost_variants = ["localhost", "127.0.0.1", "[::1]", "0.0.0.0"]
    common_ports = ["3000", "5000", "8000", "8080"]

    hosts = localhost_variants + [
        f"{host}:{port}" for host in localhost_variants for port in common_ports
    ]

    hosts.extend(
        [
            "192.168.*",
            "10.*",
            *[f"172.{i}.*" for i in range(16, 32)],
        ]
    )

    return hosts


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


def validate_host_header(allowed_hosts: list[str], host: str) -> bool:
    """
    Validate if the host header matches allowed patterns.

    Args:
        allowed_hosts: List of allowed host patterns (supports * wildcard).
        host: Host header value to validate.

    Returns:
        True if host is allowed, False otherwise.
    """
    if not host:
        return False

    return any(
        fnmatch.fnmatch(host, allowed) if "*" in allowed else host == allowed
        for allowed in allowed_hosts
    )


def init_security_middleware(app: Flask) -> None:
    """
    Initialize security middleware for Flask application.

    This configures:
    - Host header validation (DNS rebinding protection)
    - CORS protection via Flask-CORS
    - Security headers

    Args:
        app: Flask application instance.
    """
    allow_insecure_cors = MLFLOW_ALLOW_INSECURE_CORS.get() == "true"
    enable_host_validation = MLFLOW_HOST_HEADER_VALIDATION.get() != "false"

    if allow_insecure_cors:
        _logger.warning(
            "Running MLflow server with INSECURE CORS mode enabled. "
            "This allows ALL origins and should only be used for development!"
        )
        CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
    else:
        origins = get_allowed_origins()
        _logger.info(f"CORS configured with origins: {origins[:5]}...")
        CORS(
            app,
            resources={r"/*": {"origins": origins}},
            supports_credentials=True,
            methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        )

    if enable_host_validation:
        allowed_hosts = get_allowed_hosts()
        _logger.info(f"Host validation enabled with hosts: {allowed_hosts[:5]}...")

        @app.before_request
        def validate_host():
            """Validate Host header to prevent DNS rebinding attacks."""
            if request.path in ["/health", "/version"]:
                return None

            if not validate_host_header(allowed_hosts, host := request.headers.get("Host")):
                _logger.warning(f"Rejected request with invalid Host header: {host}")
                return Response(
                    "Invalid Host header - possible DNS rebinding attack detected",
                    status=403,
                    mimetype="text/plain",
                )
            return None
    else:
        _logger.warning(
            "Host header validation is DISABLED. "
            "This may leave the server vulnerable to DNS rebinding attacks."
        )

    @app.after_request
    def add_security_headers(response: Response) -> Response:
        """Add security headers to all responses."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        return response

    _logger.info("Flask security middleware initialized successfully")
