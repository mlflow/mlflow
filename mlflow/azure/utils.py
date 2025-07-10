"""Utility functions for Azure authentication."""

from __future__ import annotations

import time
import logging
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


def is_token_expired(token_expiry: Optional[datetime], buffer_seconds: int = 300) -> bool:
    """Check if a token is expired or will expire within the buffer time.

    Args:
        token_expiry: Token expiration datetime
        buffer_seconds: Buffer time in seconds before actual expiry

    Returns:
        True if token is expired or will expire soon
    """
    if not token_expiry:
        return True

    buffer_time = timedelta(seconds=buffer_seconds)
    return datetime.utcnow() + buffer_time >= token_expiry


def parse_connection_string(connection_string: str) -> dict[str, Any]:
    """Parse a PostgreSQL connection string and extract components.

    Args:
        connection_string: PostgreSQL connection string

    Returns:
        Dictionary with connection components
    """
    # Handle azure-postgres:// scheme by converting to postgresql:// for parsing
    original_scheme = connection_string.split("://")[0] if "://" in connection_string else "unknown"
    if connection_string.startswith("azure-postgres://"):
        connection_string = connection_string.replace("azure-postgres://", "postgresql://", 1)

    parsed = urlparse(connection_string)

    # Parse query parameters
    query_params = {}
    if parsed.query:
        query_params = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(parsed.query).items()}

    return {
        "scheme": original_scheme,  # Use original scheme (azure-postgres or postgresql)
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/") if parsed.path else "",
        "params": query_params,
    }


def build_connection_string(
    username: str,
    password: str,
    hostname: str,
    port: int = 5432,
    database: str = "postgres",
    **params: Any,
) -> str:
    """Build a PostgreSQL connection string.

    Args:
        username: Database username
        password: Database password
        hostname: Database hostname
        port: Database port
        database: Database name
        **params: Additional connection parameters

    Returns:
        PostgreSQL connection string
    """
    param_string = ""
    if params:
        param_string = "?" + "&".join(f"{k}={v}" for k, v in params.items())

    return f"postgresql://{username}:{password}@{hostname}:{port}/{database}{param_string}"


def sanitize_connection_string_for_logging(connection_string: str) -> str:
    """Sanitize connection string for safe logging by removing sensitive information.

    Args:
        connection_string: Original connection string

    Returns:
        Sanitized connection string with password masked
    """
    try:
        parsed = parse_connection_string(connection_string)

        # Mask password
        masked_password = "***" if parsed.get("password") else None

        # Rebuild without sensitive info
        safe_parts = []
        safe_parts.append(f"scheme={parsed.get('scheme', 'unknown')}")
        safe_parts.append(f"host={parsed.get('hostname', 'unknown')}")
        safe_parts.append(f"port={parsed.get('port', 5432)}")
        safe_parts.append(f"database={parsed.get('database', 'unknown')}")
        safe_parts.append(f"user={parsed.get('username', 'unknown')}")

        if masked_password:
            safe_parts.append("password=***")

        return " | ".join(safe_parts)

    except Exception as e:
        logger.warning("Failed to sanitize connection string", extra={"error": str(e)})
        return "connection_string=<unable_to_parse>"


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff delay.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Delay in seconds
    """
    delay = base_delay * (2**attempt)
    return min(delay, max_delay)


def retry_on_exception(
    func: callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        base_delay: Base delay between retries
        exceptions: Exception types to catch and retry on

    Returns:
        Function result

    Raises:
        Last exception if all retries failed
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(
                    "Function failed after all retries",
                    extra={"attempt": attempt, "max_retries": max_retries, "error": str(e)},
                )
                raise

            delay = exponential_backoff(attempt, base_delay)
            logger.warning(
                "Function failed, retrying",
                extra={"attempt": attempt, "delay": delay, "error": str(e)},
            )

            time.sleep(delay)

    # This should never be reached
    raise last_exception or Exception("Unexpected error in retry logic")