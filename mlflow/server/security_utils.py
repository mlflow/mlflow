"""
Shared security utilities for MLflow server middleware.

This module contains common functions used by both Flask and FastAPI
security implementations.
"""

import fnmatch
from urllib.parse import urlparse

from mlflow.environment_variables import (
    MLFLOW_SERVER_ALLOWED_HOSTS,
    MLFLOW_SERVER_CORS_ALLOWED_ORIGINS,
)

# Security response messages
INVALID_HOST_MSG = "Invalid Host header - possible DNS rebinding attack detected"
CORS_BLOCKED_MSG = "Cross-origin request blocked"

# HTTP methods that modify state
STATE_CHANGING_METHODS = ["POST", "PUT", "DELETE", "PATCH"]

# Paths exempt from host validation
HEALTH_ENDPOINTS = ["/health", "/version"]

# API path prefix for MLflow endpoints
API_PATH_PREFIX = "/api/"

# Test-only endpoints that should not have CORS blocking
TEST_ENDPOINTS = ["/test", "/api/test"]

# Localhost addresses
LOCALHOST_VARIANTS = ["localhost", "127.0.0.1", "[::1]", "0.0.0.0"]
CORS_LOCALHOST_HOSTS = ["localhost", "127.0.0.1", "[::1]", "::1"]

# Private IP range start values for 172.16.0.0/12
PRIVATE_172_RANGE_START = 16
PRIVATE_172_RANGE_END = 32

# Regex patterns for localhost origins
LOCALHOST_ORIGIN_PATTERNS = [
    r"^http://localhost(:[0-9]+)?$",
    r"^http://127\.0\.0\.1(:[0-9]+)?$",
    r"^http://\[::1\](:[0-9]+)?$",
]


def get_localhost_addresses() -> list[str]:
    """Get localhost/loopback addresses."""
    return LOCALHOST_VARIANTS


def get_private_ip_patterns() -> list[str]:
    """
    Generate wildcard patterns for private IP ranges.

    These are the standard RFC-defined private address ranges:
    - RFC 1918 (IPv4): 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
      https://datatracker.ietf.org/doc/html/rfc1918
    - RFC 4193 (IPv6): fc00::/7
      https://datatracker.ietf.org/doc/html/rfc4193

    Additional references:
    - IANA IPv4 Special-Purpose Address Registry:
      https://www.iana.org/assignments/iana-ipv4-special-registry/
    - IANA IPv6 Special-Purpose Address Registry:
      https://www.iana.org/assignments/iana-ipv6-special-registry/
    """
    return [
        "192.168.*",
        "10.*",
        *[f"172.{i}.*" for i in range(PRIVATE_172_RANGE_START, PRIVATE_172_RANGE_END)],
        "fc00:*",
        "fd00:*",
    ]


def get_allowed_hosts_from_env() -> list[str] | None:
    """Get allowed hosts from environment variable."""
    if allowed_hosts_env := MLFLOW_SERVER_ALLOWED_HOSTS.get():
        return [host.strip() for host in allowed_hosts_env.split(",")]
    return None


def get_allowed_origins_from_env() -> list[str] | None:
    """Get allowed CORS origins from environment variable."""
    if allowed_origins_env := MLFLOW_SERVER_CORS_ALLOWED_ORIGINS.get():
        return [origin.strip() for origin in allowed_origins_env.split(",")]
    return None


def is_localhost_origin(origin: str) -> bool:
    """Check if an origin is from localhost."""
    if not origin:
        return False

    try:
        parsed = urlparse(origin)
        hostname = parsed.hostname
        return hostname in CORS_LOCALHOST_HOSTS
    except Exception:
        return False


def should_block_cors_request(origin: str, method: str, allowed_origins: list[str] | None) -> bool:
    """Determine if a CORS request should be blocked."""
    if not origin or method not in STATE_CHANGING_METHODS:
        return False

    if is_localhost_origin(origin):
        return False

    if allowed_origins:
        # If wildcard "*" is in the list, allow all origins
        if "*" in allowed_origins:
            return False
        if origin in allowed_origins:
            return False

    return True


def is_api_endpoint(path: str) -> bool:
    """Check if a path is an API endpoint that should have CORS/OPTIONS handling."""
    return path.startswith(API_PATH_PREFIX) and path not in TEST_ENDPOINTS


def is_allowed_host_header(allowed_hosts: list[str], host: str) -> bool:
    """Validate if the host header matches allowed patterns."""
    if not host:
        return False

    # If wildcard "*" is in the list, allow all hosts
    if "*" in allowed_hosts:
        return True

    return any(
        fnmatch.fnmatch(host, allowed) if "*" in allowed else host == allowed
        for allowed in allowed_hosts
    )


def get_default_allowed_hosts() -> list[str]:
    """Get default allowed hosts patterns."""
    wildcard_hosts = []
    for host in get_localhost_addresses():
        if host.startswith("["):
            # IPv6: escape opening bracket for fnmatch
            escaped = host.replace("[", "[[]", 1)
            wildcard_hosts.append(f"{escaped}:*")
        else:
            wildcard_hosts.append(f"{host}:*")

    return get_localhost_addresses() + wildcard_hosts + get_private_ip_patterns()
