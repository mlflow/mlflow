import logging
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

from mlflow.environment_variables import (
    MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE,
    MLFLOW_SERVER_X_FRAME_OPTIONS,
)
from mlflow.server.security_utils import (
    CORS_BLOCKED_MSG,
    HEALTH_ENDPOINTS,
    INVALID_HOST_MSG,
    get_allowed_hosts_from_env,
    get_allowed_origins_from_env,
    get_default_allowed_hosts,
    is_allowed_host_header,
    is_api_endpoint,
    should_block_cors_request,
)
from mlflow.tracing.constant import TRACE_RENDERER_ASSET_PATH

_logger = logging.getLogger(__name__)


class HostValidationMiddleware:
    """Middleware to validate Host headers using fnmatch patterns."""

    def __init__(self, app: ASGIApp, allowed_hosts: list[str]):
        self.app = app
        self.allowed_hosts = allowed_hosts

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        if scope["path"] in HEALTH_ENDPOINTS:
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers", []))
        host = headers.get(b"host", b"").decode("utf-8")

        if not is_allowed_host_header(self.allowed_hosts, host):
            _logger.warning(f"Rejected request with invalid Host header: {host}")

            async def send_403(message):
                if message["type"] == "http.response.start":
                    message["status"] = 403
                    message["headers"] = [(b"content-type", b"text/plain")]
                await send(message)

            await send_403({"type": "http.response.start", "status": 403, "headers": []})
            await send({"type": "http.response.body", "body": INVALID_HOST_MSG.encode()})
            return

        return await self.app(scope, receive, send)


class SecurityHeadersMiddleware:
    """Middleware to add security headers to all responses."""

    def __init__(self, app: ASGIApp):
        self.app = app
        self.x_frame_options = MLFLOW_SERVER_X_FRAME_OPTIONS.get()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"x-content-type-options"] = b"nosniff"

                # Skip X-Frame-Options for notebook renderer to allow iframe embedding in notebooks
                path = scope.get("path", "")
                is_notebook_renderer = path.startswith(TRACE_RENDERER_ASSET_PATH)

                if (
                    self.x_frame_options
                    and self.x_frame_options.upper() != "NONE"
                    and not is_notebook_renderer
                ):
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
    if MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE.get() == "true":
        return

    app.add_middleware(SecurityHeadersMiddleware)

    allowed_origins = get_allowed_origins()

    if allowed_origins and "*" in allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
    else:
        app.add_middleware(CORSBlockingMiddleware, allowed_origins=allowed_origins)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

    allowed_hosts = get_allowed_hosts()

    if allowed_hosts and "*" not in allowed_hosts:
        app.add_middleware(HostValidationMiddleware, allowed_hosts=allowed_hosts)
