"""
Rate limiting for MLflow AI Gateway.

Provides per-endpoint sliding-window rate limiting (calls per minute).
Enforces the limit at the gateway layer before the request reaches the LLM.
Returns HTTP 429 with a Retry-After header when the limit is exceeded.

This module mirrors the budget-check pattern in mlflow/gateway/budget.py:
    check_rate_limit(endpoint_config)
called in every route handler in mlflow/server/gateway_api.py, right after
check_budget_limit().
"""

import threading
import time
from collections import deque

from fastapi import HTTPException

from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig

_WINDOW_SECONDS = 60


class _SlidingWindow:
    """Thread-safe 60-second sliding window request counter."""

    __slots__ = ("_lock", "_timestamps")

    def __init__(self):
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def record_and_check(self, limit: int) -> tuple[bool, int]:
        """
        Record the incoming request and check if the limit is exceeded.

        Returns:
            (allowed, retry_after_seconds)
        """
        now = time.monotonic()
        cutoff = now - _WINDOW_SECONDS

        with self._lock:
            # Evict timestamps outside the window
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()

            if len(self._timestamps) >= limit:
                # How many seconds until the oldest request ages out
                retry_after = int(_WINDOW_SECONDS - (now - self._timestamps[0])) + 1
                return False, retry_after

            self._timestamps.append(now)
            return True, 0


class _RateLimiterRegistry:
    """One SlidingWindow per endpoint, created lazily."""

    def __init__(self):
        self._windows: dict[str, _SlidingWindow] = {}
        self._lock = threading.Lock()

    def get_or_create(self, endpoint_name: str) -> _SlidingWindow:
        try:
            return self._windows[endpoint_name]
        except KeyError:
            with self._lock:
                if endpoint_name not in self._windows:
                    self._windows[endpoint_name] = _SlidingWindow()
                return self._windows[endpoint_name]

    def remove(self, endpoint_name: str) -> None:
        """Call this when an endpoint is deleted so the window is GC'd."""
        with self._lock:
            self._windows.pop(endpoint_name, None)


# Module-level singleton — one per gateway worker process
_registry = _RateLimiterRegistry()


def check_rate_limit(endpoint_config: GatewayEndpointConfig) -> None:
    """
    Enforce the rate limit for the given endpoint.

    Call this in every gateway route handler immediately after
    check_budget_limit(), e.g.:

        check_budget_limit(store, endpoint_config, workspace=workspace)
        check_rate_limit(endpoint_config)

    Args:
        endpoint_config: The resolved endpoint configuration. If
            ``calls_per_minute`` is None or 0, this is a no-op.

    Raises:
        HTTPException(429): When the per-minute call limit is exceeded,
            with a ``Retry-After`` header indicating when the client
            may retry.
    """
    limit = endpoint_config.calls_per_minute
    if not limit:
        return

    window = _registry.get_or_create(endpoint_config.endpoint_name)
    allowed, retry_after = window.record_and_check(limit)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit of {limit} calls/minute exceeded for endpoint "
                f"'{endpoint_config.endpoint_name}'. "
                f"Retry after {retry_after} second(s)."
            ),
            headers={"Retry-After": str(retry_after)},
        )
