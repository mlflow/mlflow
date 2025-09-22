"""
FastAPI security middleware for MLflow server.
"""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

from mlflow.environment_variables import (
    MLFLOW_ALLOW_INSECURE_CORS,
    MLFLOW_CORS_ALLOWED_ORIGINS,
    MLFLOW_HOST_HEADER_VALIDATION,
)
from mlflow.server.security import SecurityMiddleware

_logger = logging.getLogger(__name__)


class FastAPISecurityMiddleware(BaseHTTPMiddleware):
    """
    FastAPI/Starlette middleware wrapper for SecurityMiddleware.
    """

    def __init__(self, app, security_middleware: SecurityMiddleware):
        super().__init__(app)
        self.security_middleware = security_middleware

    async def dispatch(self, request: Request, call_next):
        """Process request through security middleware."""

        # Convert FastAPI request to Flask-like request for compatibility
        class RequestAdapter:
            def __init__(self, fastapi_request):
                self._request = fastapi_request
                self.headers = fastapi_request.headers
                self.method = fastapi_request.method
                self.path = str(fastapi_request.url.path)

        adapted_request = RequestAdapter(request)

        # Check if request should be blocked
        validation_response = self.security_middleware.process_request(adapted_request)
        if validation_response:
            return PlainTextResponse(
                content=validation_response.get_data(as_text=True),
                status_code=validation_response.status_code,
            )

        # Process the request
        response = await call_next(request)

        # Add security headers
        class ResponseAdapter:
            def __init__(self, fastapi_response):
                self.headers = {}
                self._response = fastapi_response

        adapted_response = ResponseAdapter(response)
        self.security_middleware.process_response(adapted_response, adapted_request)

        # Apply headers to actual response
        for key, value in adapted_response.headers.items():
            response.headers[key] = value

        return response


def init_fastapi_security(app: FastAPI) -> None:
    """
    Initialize security middleware for FastAPI application.

    Args:
        app: FastAPI application instance.
    """
    # Get configuration from environment variables
    allow_insecure_cors = MLFLOW_ALLOW_INSECURE_CORS.get() == "true"
    enable_host_validation = MLFLOW_HOST_HEADER_VALIDATION.get() != "false"

    # Parse allowed origins from environment
    allowed_origins_env = MLFLOW_CORS_ALLOWED_ORIGINS.get()
    allowed_origins = None
    if allowed_origins_env:
        allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",")]

    # Parse allowed hosts from environment
    allowed_hosts_env = os.environ.get("MLFLOW_ALLOWED_HOSTS")
    allowed_hosts = None
    if allowed_hosts_env:
        allowed_hosts = [host.strip() for host in allowed_hosts_env.split(",")]

    # Create SecurityMiddleware instance
    security_middleware = SecurityMiddleware(
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
        allow_insecure_cors=allow_insecure_cors,
        enable_host_validation=enable_host_validation,
    )

    # Add custom security middleware
    app.add_middleware(FastAPISecurityMiddleware, security_middleware=security_middleware)

    # Configure CORS using FastAPI's built-in CORSMiddleware as a fallback
    # This provides additional CORS handling for FastAPI-specific endpoints
    if allow_insecure_cors:
        _logger.warning(
            "Running MLflow server with INSECURE CORS mode enabled. "
            "This should only be used for development and testing!"
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    elif allowed_origins:
        # Add localhost variants to the allowed origins
        origins = list(allowed_origins)
        localhost_origins = [
            "http://localhost:3000",
            "http://localhost:5000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5000",
        ]
        for origin in localhost_origins:
            if origin not in origins:
                origins.append(origin)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization"],
        )
