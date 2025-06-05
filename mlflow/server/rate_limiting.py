"""
Rate limiting for MLflow tracking server using Flask-Limiter.

This module provides rate limiting functionality to control and mitigate high volumes of
incoming requests from clients, prevent abuse, and improve system reliability and scalability.
"""

import logging
from typing import Optional

from flask import Flask, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from mlflow.environment_variables import (
    MLFLOW_TRACKING_SERVER_RATE_LIMITING_ARTIFACT_LIMITS,
    MLFLOW_TRACKING_SERVER_RATE_LIMITING_DEFAULT_LIMITS,
    MLFLOW_TRACKING_SERVER_RATE_LIMITING_ENABLED,
    MLFLOW_TRACKING_SERVER_RATE_LIMITING_EXPERIMENT_LIMITS,
    MLFLOW_TRACKING_SERVER_RATE_LIMITING_KEY_FUNC,
    MLFLOW_TRACKING_SERVER_RATE_LIMITING_LOGGING_LIMITS,
    MLFLOW_TRACKING_SERVER_RATE_LIMITING_RUN_LIMITS,
    MLFLOW_TRACKING_SERVER_RATE_LIMITING_SEARCH_LIMITS,
    MLFLOW_TRACKING_SERVER_RATE_LIMITING_STORAGE_URI,
)

_logger = logging.getLogger(__name__)

# Global limiter instance
_limiter: Optional[Limiter] = None


def get_client_id():
    """
    Get client identifier for rate limiting based on configured key function.
    """
    key_func = MLFLOW_TRACKING_SERVER_RATE_LIMITING_KEY_FUNC.get()

    if key_func == "ip":
        return get_remote_address()
    elif key_func == "user":
        # Try to get user from request headers (for authenticated requests)
        return request.headers.get("X-User-ID") or request.headers.get("Authorization", "anonymous")
    elif key_func == "session":
        # Use session ID if available, fallback to IP
        return request.headers.get("X-Session-ID") or get_remote_address()
    else:
        # Default to IP-based rate limiting
        return get_remote_address()


def _parse_limits(limits_str: str) -> list[str]:
    """
    Parse comma-separated rate limit strings into a list.

    Args:
        limits_str: Comma-separated string of rate limits (e.g., "100 per hour, 10 per minute")

    Returns:
        List of rate limit strings
    """
    if not limits_str:
        return []
    return [limit.strip() for limit in limits_str.split(",") if limit.strip()]


def init_rate_limiting(app: Flask) -> Optional[Limiter]:
    """
    Initialize rate limiting for the Flask app.

    Args:
        app: Flask application instance

    Returns:
        Limiter instance if rate limiting is enabled, None otherwise
    """
    global _limiter

    if not MLFLOW_TRACKING_SERVER_RATE_LIMITING_ENABLED.get():
        _logger.info("Rate limiting is disabled")
        return None

    storage_uri = MLFLOW_TRACKING_SERVER_RATE_LIMITING_STORAGE_URI.get()
    default_limits = _parse_limits(MLFLOW_TRACKING_SERVER_RATE_LIMITING_DEFAULT_LIMITS.get())

    _logger.info(f"Initializing rate limiting with storage: {storage_uri}")
    _logger.info(f"Default rate limits: {default_limits}")

    try:
        _limiter = Limiter(
            get_client_id,
            app=app,
            storage_uri=storage_uri,
            default_limits=default_limits,
            fail_on_first_breach=True,
        )

        _logger.info("Rate limiting initialized successfully")
        return _limiter

    except Exception as e:
        _logger.error(f"Failed to initialize rate limiting: {e}")
        _logger.warning("Continuing without rate limiting")
        return None


def get_limiter() -> Optional[Limiter]:
    """
    Get the global limiter instance.

    Returns:
        Limiter instance if initialized, None otherwise
    """
    return _limiter


def get_rate_limit_status():
    """
    Get the current rate limiting status and configuration.

    Returns:
        Dictionary with rate limiting status information
    """
    if not _limiter:
        return {"enabled": False}

    return {
        "enabled": True,
        "storage_uri": MLFLOW_TRACKING_SERVER_RATE_LIMITING_STORAGE_URI.get(),
        "key_function": MLFLOW_TRACKING_SERVER_RATE_LIMITING_KEY_FUNC.get(),
        "default_limits": _parse_limits(MLFLOW_TRACKING_SERVER_RATE_LIMITING_DEFAULT_LIMITS.get()),
        "experiment_limits": _parse_limits(
            MLFLOW_TRACKING_SERVER_RATE_LIMITING_EXPERIMENT_LIMITS.get()
        ),
        "run_limits": _parse_limits(MLFLOW_TRACKING_SERVER_RATE_LIMITING_RUN_LIMITS.get()),
        "logging_limits": _parse_limits(MLFLOW_TRACKING_SERVER_RATE_LIMITING_LOGGING_LIMITS.get()),
        "search_limits": _parse_limits(MLFLOW_TRACKING_SERVER_RATE_LIMITING_SEARCH_LIMITS.get()),
        "artifact_limits": _parse_limits(
            MLFLOW_TRACKING_SERVER_RATE_LIMITING_ARTIFACT_LIMITS.get()
        ),
    }


# Decorators for applying specific rate limits to endpoints
def experiment_limit(func):
    """Decorator to apply experiment-specific rate limits."""
    if _limiter:
        limits = _parse_limits(MLFLOW_TRACKING_SERVER_RATE_LIMITING_EXPERIMENT_LIMITS.get())
        if limits:
            return _limiter.limit(",".join(limits), override_defaults=False)(func)
    return func


def run_limit(func):
    """Decorator to apply run-specific rate limits."""
    if _limiter:
        limits = _parse_limits(MLFLOW_TRACKING_SERVER_RATE_LIMITING_RUN_LIMITS.get())
        if limits:
            return _limiter.limit(",".join(limits), override_defaults=False)(func)
    return func


def logging_limit(func):
    """Decorator to apply logging-specific rate limits."""
    if _limiter:
        limits = _parse_limits(MLFLOW_TRACKING_SERVER_RATE_LIMITING_LOGGING_LIMITS.get())
        if limits:
            return _limiter.limit(",".join(limits), override_defaults=False)(func)
    return func


def search_limit(func):
    """Decorator to apply search-specific rate limits."""
    if _limiter:
        limits = _parse_limits(MLFLOW_TRACKING_SERVER_RATE_LIMITING_SEARCH_LIMITS.get())
        if limits:
            return _limiter.limit(",".join(limits), override_defaults=False)(func)
    return func


def artifact_limit(func):
    """Decorator to apply artifact-specific rate limits."""
    if _limiter:
        limits = _parse_limits(MLFLOW_TRACKING_SERVER_RATE_LIMITING_ARTIFACT_LIMITS.get())
        if limits:
            return _limiter.limit(",".join(limits), override_defaults=False)(func)
    return func
