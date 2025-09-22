"""
Security middleware for MLflow server to prevent CORS attacks, DNS rebinding,
and other vulnerabilities.
"""

import ipaddress
import logging
import os
import socket
from urllib.parse import urlparse

from flask import Flask, Request, Response, request

from mlflow.environment_variables import (
    MLFLOW_ALLOW_INSECURE_CORS,
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

        # Initialize allowed hosts
        if allowed_hosts is None:
            # Default to localhost and common local addresses
            self.allowed_hosts = self._get_default_allowed_hosts()
        else:
            self.allowed_hosts = set(allowed_hosts)

        # Initialize allowed origins
        if allowed_origins is None:
            self.allowed_origins = set()
        else:
            self.allowed_origins = set(allowed_origins)

        # Add localhost variants to allowed origins if not in insecure mode
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

        # Add common port variations
        common_ports = ["5000", "3000", "8080", "8000"]
        localhost_variants = ["localhost", "127.0.0.1", "[::1]", "0.0.0.0"]

        for host in localhost_variants:
            for port in common_ports:
                if host.startswith("["):  # IPv6
                    hosts.add(f"{host}:{port}")
                else:
                    hosts.add(f"{host}:{port}")

        # Try to add the actual hostname
        try:
            hostname = socket.gethostname()
            hosts.add(hostname)
            hosts.add(f"{hostname}:5000")
            hosts.add(f"{hostname}:3000")

            # Also add the FQDN
            fqdn = socket.getfqdn()
            if fqdn != hostname:
                hosts.add(fqdn)
                hosts.add(f"{fqdn}:5000")
                hosts.add(f"{fqdn}:3000")
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
            # Remove port if present
            if ":" in hostname and not hostname.startswith("["):
                # Check if this might be an IPv6 address without brackets
                # IPv6 addresses have multiple colons
                if hostname.count(":") > 1:
                    # This is likely an IPv6 address
                    pass
                else:
                    # IPv4 with port
                    hostname = hostname.split(":")[0]
            elif hostname.startswith("[") and "]:" in hostname:
                # IPv6 with port
                hostname = hostname[1 : hostname.index("]")]
            elif hostname.startswith("[") and hostname.endswith("]"):
                # IPv6 without port but with brackets
                hostname = hostname[1:-1]

            # Try to parse as IP address directly
            try:
                ip = ipaddress.ip_address(hostname)
                return ip.is_private or ip.is_loopback or ip.is_link_local
            except ValueError:
                # Not an IP address, try to resolve
                pass

            # Resolve hostname to IP
            addr_info = socket.getaddrinfo(hostname, None)
            for addr in addr_info:
                ip_str = addr[4][0]
                ip = ipaddress.ip_address(ip_str)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return True
            return False
        except Exception as e:
            _logger.debug(f"Could not determine if {hostname} is private: {e}")
            # Be conservative and deny if we can't determine
            return False

    def _validate_host_header(self, request: Request) -> bool:
        """
        Validate the Host header to prevent DNS rebinding attacks.

        Returns:
            True if the host is valid, False otherwise.
        """
        if not self.enable_host_validation:
            return True

        host = request.headers.get("Host")
        if not host:
            _logger.warning("Request missing Host header")
            return False

        # Check against allowed hosts
        if host in self.allowed_hosts:
            return True

        # Check if it's a private IP/hostname
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
        origin = request.headers.get("Origin")
        if not origin:
            # No Origin header means it's likely a same-origin request or direct API call
            return True

        if self.allow_insecure_cors:
            _logger.debug("Allowing all origins due to insecure CORS mode")
            return True

        # Check against allowed origins
        if origin in self.allowed_origins:
            return True

        # Check if origin is from a private/local address
        try:
            parsed = urlparse(origin)
            if parsed.hostname and self._is_private_ip(parsed.hostname):
                _logger.debug(f"Allowing private origin: {origin}")
                return True
        except Exception as e:
            _logger.debug(f"Could not parse origin {origin}: {e}")

        _logger.warning(f"Rejected request with untrusted Origin header: {origin}")
        return False

    def _add_security_headers(self, response: Response, request: Request) -> None:
        """Add security headers to the response."""
        # Add X-Content-Type-Options to prevent MIME sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Add X-Frame-Options to prevent clickjacking
        response.headers["X-Frame-Options"] = "SAMEORIGIN"

        # Handle CORS headers
        origin = request.headers.get("Origin")
        if origin:
            if self.allow_insecure_cors:
                # Insecure mode - allow all origins (only for development!)
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            elif origin in self.allowed_origins or (self._is_origin_allowed_private(origin)):
                # Set specific origin instead of wildcard for security
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
        # Skip validation for health check and version endpoints
        if request.path in ["/health", "/version"]:
            return None

        # Validate Host header
        if not self._validate_host_header(request):
            return Response(
                "Invalid Host header - possible DNS rebinding attack detected",
                status=403,
                mimetype="text/plain",
            )

        # Validate Origin header for state-changing requests
        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            if not self._validate_origin(request):
                return Response(
                    "Cross-origin request blocked",
                    status=403,
                    mimetype="text/plain",
                )

        # Handle preflight OPTIONS requests
        if request.method == "OPTIONS":
            origin = request.headers.get("Origin")
            if origin and (
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

    # Create middleware instance
    middleware = SecurityMiddleware(
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
        allow_insecure_cors=allow_insecure_cors,
        enable_host_validation=enable_host_validation,
    )

    # Register middleware with Flask
    @app.before_request
    def before_request():
        """Process request through security middleware."""
        response = middleware.process_request(request)
        if response:
            return response

    @app.after_request
    def after_request(response):
        """Process response through security middleware."""
        return middleware.process_response(response, request)

    if allow_insecure_cors:
        _logger.warning(
            "Running MLflow server with INSECURE CORS mode enabled. "
            "This should only be used for development and testing!"
        )

    return middleware
