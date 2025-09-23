"""
Security middleware for MLflow server to prevent CORS attacks, DNS rebinding,
and other vulnerabilities.
"""

import ipaddress
import logging
import socket
from urllib.parse import urlparse

from flask import Flask, Request, Response, request

from mlflow.environment_variables import (
    MLFLOW_ALLOW_INSECURE_CORS,
    MLFLOW_ALLOWED_HOSTS,
    MLFLOW_CORS_ALLOWED_ORIGINS,
    MLFLOW_HOST_HEADER_VALIDATION,
)

_logger = logging.getLogger(__name__)


class SecurityMiddleware:
    """
    Security middleware to protect MLflow server from various attacks including:
    - DNS rebinding attacks
    - CORS attacks
    - Host header injection
    """

    def __init__(
        self,
        allowed_hosts: list[str] | None = None,
        allowed_origins: list[str] | None = None,
        allow_insecure_cors: bool = False,
        enable_host_validation: bool = True,
    ):
        """
        Initialize the security middleware.

        Args:
            allowed_hosts: List of allowed host headers. If None, defaults to localhost variants.
            allowed_origins: List of allowed CORS origins. If None, no CORS headers are set.
            allow_insecure_cors: If True, allows all origins (dangerous, only for development).
            enable_host_validation: If True, validates Host header to prevent DNS rebinding.
        """
        self.allow_insecure_cors = allow_insecure_cors
        self.enable_host_validation = enable_host_validation

        self.allowed_hosts = (
            self._get_default_allowed_hosts() if allowed_hosts is None else set(allowed_hosts)
        )
        self.allowed_origins = set() if allowed_origins is None else set(allowed_origins)

        if not allow_insecure_cors:
            self._add_localhost_origins()

        _logger.info(f"Security middleware initialized with allowed_hosts: {self.allowed_hosts}")

    def _get_default_allowed_hosts(self) -> set[str]:
        """Get default allowed hosts for local development."""
        hosts = {
            "localhost",
            "127.0.0.1",
            "[::1]",  # IPv6 localhost
            "0.0.0.0",
        }

        common_ports = ["5000", "3000", "8080", "8000"]
        localhost_variants = ["localhost", "127.0.0.1", "[::1]", "0.0.0.0"]

        hosts.update(f"{host}:{port}" for host in localhost_variants for port in common_ports)

        try:
            if hostname := socket.gethostname():
                hosts.update([hostname, f"{hostname}:5000", f"{hostname}:3000"])

                if (fqdn := socket.getfqdn()) != hostname:
                    hosts.update([fqdn, f"{fqdn}:5000", f"{fqdn}:3000"])
        except Exception as e:
            _logger.debug(f"Could not determine hostname: {e}")

        return hosts

    def _add_localhost_origins(self):
        """Add common localhost origins for development."""
        localhost_origins = [
            "http://localhost:3000",
            "http://localhost:5000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5000",
            "http://[::1]:3000",
            "http://[::1]:5000",
        ]
        self.allowed_origins.update(localhost_origins)

    def _is_private_ip(self, hostname: str) -> bool:
        """Check if a hostname resolves to a private IP address."""
        try:
            if ":" in hostname and not hostname.startswith("["):
                # NB: IPv6 addresses have multiple colons, IPv4 has max one
                hostname = hostname if hostname.count(":") > 1 else hostname.split(":")[0]
            elif hostname.startswith("[") and "]:" in hostname:
                hostname = hostname[1 : hostname.index("]")]
            elif hostname.startswith("[") and hostname.endswith("]"):
                hostname = hostname[1:-1]

            try:
                ip = ipaddress.ip_address(hostname)
                return ip.is_private or ip.is_loopback or ip.is_link_local
            except ValueError:
                pass

            return any(
                (ip := ipaddress.ip_address(addr[4][0])).is_private
                or ip.is_loopback
                or ip.is_link_local
                for addr in socket.getaddrinfo(hostname, None)
            )
        except Exception as e:
            _logger.debug(f"Could not determine if {hostname} is private: {e}")
            return False  # NB: Conservative denial if determination fails

    def _validate_host_header(self, request: Request) -> bool:
        """
        Validate the Host header to prevent DNS rebinding attacks.

        Returns:
            True if the host is valid, False otherwise.
        """
        if not self.enable_host_validation:
            return True

        if not (host := request.headers.get("Host")):
            _logger.warning("Request missing Host header")
            return False

        if host in self.allowed_hosts:
            return True

        if self._is_private_ip(host):
            _logger.debug(f"Allowing private host: {host}")
            return True

        _logger.warning(f"Rejected request with untrusted Host header: {host}")
        return False

    def _validate_origin(self, request: Request) -> bool:
        """
        Validate the Origin header to prevent CORS attacks.

        Returns:
            True if the origin is valid or not present, False otherwise.
        """
        if not (origin := request.headers.get("Origin")):
            return True

        if self.allow_insecure_cors:
            _logger.debug("Allowing all origins due to insecure CORS mode")
            return True

        if origin in self.allowed_origins:
            return True

        try:
            if (parsed := urlparse(origin)).hostname and self._is_private_ip(parsed.hostname):
                _logger.debug(f"Allowing private origin: {origin}")
                return True
        except Exception as e:
            _logger.debug(f"Could not parse origin {origin}: {e}")

        _logger.warning(f"Rejected request with untrusted Origin header: {origin}")
        return False

    def _add_security_headers(self, response: Response, request: Request) -> None:
        """Add security headers to the response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"

        if origin := request.headers.get("Origin"):
            if self.allow_insecure_cors:
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            elif origin in self.allowed_origins or self._is_origin_allowed_private(origin):
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Vary"] = "Origin"

    def _is_origin_allowed_private(self, origin: str) -> bool:
        """Check if an origin is from an allowed private address."""
        try:
            parsed = urlparse(origin)
            return parsed.hostname and self._is_private_ip(parsed.hostname)
        except Exception:
            return False

    def process_request(self, request: Request) -> Response | None:
        """
        Process incoming request and validate security headers.

        Returns:
            Response object if request should be blocked, None otherwise.
        """
        if request.path in ["/health", "/version"]:
            return None

        if not self._validate_host_header(request):
            return Response(
                "Invalid Host header - possible DNS rebinding attack detected",
                status=403,
                mimetype="text/plain",
            )

        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            if not self._validate_origin(request):
                return Response(
                    "Cross-origin request blocked",
                    status=403,
                    mimetype="text/plain",
                )

        if request.method == "OPTIONS" and (origin := request.headers.get("Origin")):
            if (
                self.allow_insecure_cors
                or origin in self.allowed_origins
                or self._is_origin_allowed_private(origin)
            ):
                response = Response(status=204)
                self._add_security_headers(response, request)
                return response

        return None

    def process_response(self, response: Response, request: Request) -> Response:
        """
        Process outgoing response and add security headers.

        Returns:
            Modified response with security headers.
        """
        self._add_security_headers(response, request)
        return response


def init_security_middleware(app: Flask) -> SecurityMiddleware:
    """
    Initialize and configure security middleware for the Flask app.

    Args:
        app: Flask application instance.

    Returns:
        Configured SecurityMiddleware instance.
    """
    allow_insecure_cors = MLFLOW_ALLOW_INSECURE_CORS.get() == "true"
    enable_host_validation = MLFLOW_HOST_HEADER_VALIDATION.get() != "false"

    allowed_origins = (
        [origin.strip() for origin in origins.split(",")]
        if (origins := MLFLOW_CORS_ALLOWED_ORIGINS.get())
        else None
    )

    allowed_hosts = (
        [host.strip() for host in hosts.split(",")]
        if (hosts := MLFLOW_ALLOWED_HOSTS.get())
        else None
    )

    middleware = SecurityMiddleware(
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
        allow_insecure_cors=allow_insecure_cors,
        enable_host_validation=enable_host_validation,
    )

    @app.before_request
    def before_request():
        """Process request through security middleware."""
        if response := middleware.process_request(request):
            return response

    @app.after_request
    def after_request(response):
        """Process response through security middleware."""
        return middleware.process_response(response, request)

    if allow_insecure_cors:
        _logger.warning(
            "Running MLflow server with INSECURE CORS mode enabled. "
            "This should only be used for development and testing."
        )

    return middleware
