import logging
from http import HTTPStatus

from flask import Flask, Response, request
from flask_cors import CORS

from mlflow.environment_variables import (
    MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE,
    MLFLOW_SERVER_X_FRAME_OPTIONS,
)
from mlflow.server.security_utils import (
    CORS_BLOCKED_MSG,
    HEALTH_ENDPOINTS,
    INVALID_HOST_MSG,
    LOCALHOST_ORIGIN_PATTERNS,
    get_allowed_hosts_from_env,
    get_allowed_origins_from_env,
    get_default_allowed_hosts,
    is_allowed_host_header,
    is_api_endpoint,
    should_block_cors_request,
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
    if MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE.get() == "true":
        return

    allowed_origins = get_allowed_origins()
    allowed_hosts = get_allowed_hosts()
    x_frame_options = MLFLOW_SERVER_X_FRAME_OPTIONS.get()

    if allowed_origins and "*" in allowed_origins:
        CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
    else:
        cors_origins = (allowed_origins or []) + LOCALHOST_ORIGIN_PATTERNS
        CORS(
            app,
            resources={r"/*": {"origins": cors_origins}},
            supports_credentials=True,
            methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        )

    if allowed_hosts and "*" not in allowed_hosts:

        @app.before_request
        def validate_host():
            if request.path in HEALTH_ENDPOINTS:
                return None

            if not is_allowed_host_header(allowed_hosts, host := request.headers.get("Host")):
                _logger.warning(f"Rejected request with invalid Host header: {host}")
                return Response(
                    INVALID_HOST_MSG, status=HTTPStatus.FORBIDDEN, mimetype="text/plain"
                )
            return None

    if not (allowed_origins and "*" in allowed_origins):

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

        if x_frame_options and x_frame_options.upper() != "NONE":
            response.headers["X-Frame-Options"] = x_frame_options.upper()

        if (
            request.method == "OPTIONS"
            and response.status_code == 200
            and is_api_endpoint(request.path)
        ):
            response.status_code = HTTPStatus.NO_CONTENT
            response.data = b""

        return response
