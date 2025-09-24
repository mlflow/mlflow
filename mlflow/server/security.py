"""
Flask security middleware for MLflow server.

This module provides security middleware for Flask applications to prevent
common web vulnerabilities. For FastAPI applications, use fastapi_security.py
which leverages native Starlette middleware.
"""

import logging
from http import HTTPStatus

from flask import Flask, Response, request
from flask_cors import CORS

from mlflow.environment_variables import MLFLOW_ALLOW_INSECURE_CORS, MLFLOW_HOST_HEADER_VALIDATION
from mlflow.server.security_utils import (
    CORS_BLOCKED_MSG,
    HEALTH_ENDPOINTS,
    INVALID_HOST_MSG,
    LOCALHOST_ORIGIN_PATTERNS,
    get_allowed_hosts_from_env,
    get_allowed_origins_from_env,
    get_default_allowed_hosts,
    is_api_endpoint,
    should_block_cors_request,
    validate_host_header,
)

_logger = logging.getLogger(__name__)


def get_allowed_hosts() -> list[str]:
    """Get list of allowed hosts from environment or defaults."""
    return get_allowed_hosts_from_env() or get_default_allowed_hosts()


def get_allowed_origins() -> list[str]:
    """Get list of allowed CORS origins from environment or defaults."""
    return get_allowed_origins_from_env() or []


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

    allowed_origins = get_allowed_origins()

    if allow_insecure_cors:
        _logger.warning(
            "Running MLflow server with INSECURE CORS mode enabled. "
            "This allows ALL origins and should only be used for development!"
        )
        CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
    else:
        cors_origins = (allowed_origins or []) + LOCALHOST_ORIGIN_PATTERNS
        origins_display = allowed_origins[:5] if allowed_origins else ["localhost (any port)"]
        _logger.info(f"CORS configured with origins: {origins_display}...")
        CORS(
            app,
            resources={r"/*": {"origins": cors_origins}},
            supports_credentials=True,
            methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        )

    if enable_host_validation:
        allowed_hosts = get_allowed_hosts()
        _logger.info(f"Host validation enabled with hosts: {allowed_hosts[:5]}...")

        @app.before_request
        def validate_host():
            if request.path in HEALTH_ENDPOINTS:
                return None

            if not validate_host_header(allowed_hosts, host := request.headers.get("Host")):
                _logger.warning(f"Rejected request with invalid Host header: {host}")
                return Response(
                    INVALID_HOST_MSG, status=HTTPStatus.FORBIDDEN, mimetype="text/plain"
                )
            return None
    else:
        _logger.warning(
            "Host header validation is DISABLED. "
            "This may leave the server vulnerable to DNS rebinding attacks."
        )

    if not allow_insecure_cors:

        @app.before_request
        def block_cross_origin_state_changes():
            if not is_api_endpoint(request.path):
                return None

            origin = request.headers.get("Origin")
            if should_block_cors_request(origin, request.method, allowed_origins):
                _logger.warning(f"Blocked cross-origin request from {origin}")
                return Response(
                    CORS_BLOCKED_MSG, status=HTTPStatus.FORBIDDEN, mimetype="text/plain"
                )
            return None

    @app.after_request
    def add_security_headers(response: Response) -> Response:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"

        if (
            request.method == "OPTIONS"
            and response.status_code == 200
            and is_api_endpoint(request.path)
        ):
            response.status_code = HTTPStatus.NO_CONTENT
            response.data = b""

        return response

    _logger.info("Flask security middleware initialized successfully")
