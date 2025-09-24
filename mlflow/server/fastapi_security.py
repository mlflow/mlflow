"""
FastAPI security middleware configuration using native Starlette/FastAPI middleware.

Why Custom Middleware Instead of Built-in FastAPI Security?

FastAPI's built-in security middleware has critical limitations that make it unsuitable for
MLflow's tracking server requirements, particularly for private network deployments and development
workflows.

The Core Problem: No Wildcard Support

FastAPI's TrustedHostMiddleware only accepts exact string matches for allowed hosts. This creates
severe operational challenges for MLflow deployments.

Private Network Development: Teams often access MLflow servers via multiple IP addresses
(192.168.1.100, 10.0.0.50, etc.) or hostnames (mlflow-dev, server.local). Without wildcard
support, you'd need to enumerate every possible combination, which is impractical and error-prone.

Dynamic Port Development: Local development often uses random available ports
(localhost:5000, localhost:3001, etc.). FastAPI's middleware would require pre-configuration
of every possible port, breaking the "just works" development experience.

Container/Cloud Deployments: In Kubernetes or cloud environments, services may be accessed
via multiple DNS names or IP addresses that change dynamically. FastAPI's exact-match requirement
would break these deployments.

The Catastrophic Failure Mode

When FastAPI's TrustedHostMiddleware encounters an unrecognized host header, it doesn't just log
a warning - it completely prevents the server from starting. This means:

- A single misconfigured client or unexpected host header kills the entire tracking server
- No graceful degradation or logging-only mode
- Production outages from configuration drift
- Development servers that refuse to start due to network changes

This is unacceptable for a machine learning infrastructure component that needs to be reliable
and developer-friendly.

MLflow's Solution: Fnmatch Pattern Matching with Graceful Defaults

Our custom HostValidationMiddleware provides comprehensive solutions to these problems.

Wildcard Pattern Support: Uses Python's fnmatch for patterns like *.company.com (all subdomains),
localhost:* (any localhost port), and 192.168.*.* (entire private subnet).

Graceful Request Handling: Invalid hosts get 403 responses with clear error messages,
but the server continues running and serving valid requests.

Intelligent Defaults: Automatically allows RFC 1918 private IP ranges and localhost
without configuration, enabling zero-config development while maintaining security.

Development-Friendly Logging: Clear warnings about rejected requests help debug
configuration issues without breaking the server.

Why Not Just Use allow_origins=["*"]?

While FastAPI's CORSMiddleware supports wildcards, using allow_origins=["*"] creates different
security issues.

Credentials + Wildcard Forbidden: The CORS spec prohibits Access-Control-Allow-Credentials: true
with Access-Control-Allow-Origin: *, breaking authenticated requests.

No Selective Blocking: You can't allow some origins while blocking others - it's all or nothing.

State-Changing Request Vulnerability: Wildcard CORS allows any website to make state-changing
requests (POST, PUT, DELETE) to your MLflow server, enabling CSRF attacks.

The Layered Security Approach

Our custom middleware provides defense in depth.

HostValidationMiddleware: Prevents DNS rebinding attacks via Host header validation.

CORSBlockingMiddleware: Actively blocks cross-origin state-changing requests from
non-allowlisted origins, even when the underlying CORSMiddleware uses wildcards.

SecurityHeadersMiddleware: Adds security headers like X-Frame-Options and X-Content-Type-Options.

Native CORSMiddleware: Handles standard CORS preflight and response headers.

This combination provides the security benefits of strict configuration while maintaining the
flexibility and reliability that MLflow deployments require.

When You Might Disable This Middleware

The only scenarios where you should consider MLFLOW_DISABLE_SECURITY_MIDDLEWARE=true:

Reverse Proxy Handles Everything: Your nginx/Apache/Traefik already implements all
host validation, CORS, and security headers, making MLflow's middleware redundant.

Fully Isolated Environment: Running in a completely trusted network with no external
access and no web browser usage (pure API-only deployments).

Custom Security Requirements: You have specific security needs that require different
middleware implementations.

For all other cases, this custom middleware provides the right balance of security and usability
that makes MLflow suitable for both development and production environments.
"""

import logging
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

from mlflow.environment_variables import MLFLOW_DISABLE_SECURITY_MIDDLEWARE, MLFLOW_X_FRAME_OPTIONS
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

        if not validate_host_header(self.allowed_hosts, host):
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
        self.x_frame_options = MLFLOW_X_FRAME_OPTIONS.get()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"x-content-type-options"] = b"nosniff"

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
    if MLFLOW_DISABLE_SECURITY_MIDDLEWARE.get() == "true":
        _logger.warning(
            "Security middleware is DISABLED. "
            "This may leave the server vulnerable to various attacks."
        )
        return

    app.add_middleware(SecurityHeadersMiddleware)

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

    allowed_hosts = get_allowed_hosts()

    if allowed_hosts and "*" in allowed_hosts:
        _logger.warning(
            "Host header validation accepts ALL hosts. "
            "This may leave the server vulnerable to DNS rebinding attacks."
        )
    elif allowed_hosts:
        _logger.info(f"Host validation enabled with hosts: {allowed_hosts[:5]}...")
        app.add_middleware(HostValidationMiddleware, allowed_hosts=allowed_hosts)

    _logger.info("FastAPI security middleware initialized successfully")
