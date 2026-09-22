"""Per-endpoint request rate limiting for the MLflow AI Gateway.

The legacy gateway exposed a per-route ``limit`` with ``calls`` and ``renewal_period``
keys. The new gateway stores the equivalent control as a single
``calls_per_minute`` integer on the endpoint, i.e. the legacy
``{"calls": N, "renewal_period": "minute"}`` configuration. A coarser or finer
renewal period can be added later without changing the enforcement point.

The limit is request based, not token based: a rejection has to happen before the
request reaches the provider, and token counts are only known from the provider
response.
"""

import math
import threading
import time
from collections import deque

from fastapi import HTTPException

from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig

RATE_LIMIT_WINDOW_SECONDS = 60


class _SlidingWindow:
    """Sliding window counter for a single endpoint.

    A ``threading.Lock`` guards the deque because the gateway is served by
    uvicorn, which dispatches to a thread pool in addition to the event loop.
    The critical section performs no I/O and contains no ``await``, so holding
    the lock from an async handler never blocks the event loop for a
    measurable amount of time.
    """

    __slots__ = ("_lock", "_timestamps")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def acquire(self, limit: int) -> int | None:
        """Record a request if the window has room.

        Returns:
            ``None`` if the request is allowed, otherwise the number of seconds
            the caller should wait before retrying (always >= 1).
        """
        now = time.monotonic()
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        with self._lock:
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()

            if len(self._timestamps) >= limit:
                oldest = self._timestamps[0]
                return max(1, math.ceil(oldest - cutoff))

            self._timestamps.append(now)
            return None


class _RateLimiterRegistry:
    """Lazily created sliding windows, one per endpoint name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: dict[str, _SlidingWindow] = {}

    def get(self, key: str) -> _SlidingWindow:
        if window := self._windows.get(key):
            return window
        with self._lock:
            return self._windows.setdefault(key, _SlidingWindow())

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()


# Windows are per gateway worker process, so a limit of N with W workers admits up
# to N * W calls per minute. This matches the legacy gateway's in-process limiter.
_registry = _RateLimiterRegistry()


def reset_rate_limiters() -> None:
    """Drop all in-memory windows. Intended for tests."""
    _registry.reset()


def check_rate_limit(endpoint_config: GatewayEndpointConfig) -> None:
    """Enforce the endpoint's calls-per-minute limit.

    Args:
        endpoint_config: Resolved endpoint configuration. A ``calls_per_minute``
            of ``None`` or 0 disables rate limiting for the endpoint.

    Raises:
        HTTPException: 429 with a ``Retry-After`` header when the window is full.
    """
    if not (limit := endpoint_config.calls_per_minute):
        return

    if (retry_after := _registry.get(endpoint_config.endpoint_name).acquire(limit)) is None:
        return

    raise HTTPException(
        status_code=429,
        detail=(
            f"Rate limit exceeded for endpoint '{endpoint_config.endpoint_name}'. "
            f"Limit: {limit} calls per minute. Retry after {retry_after} second(s)."
        ),
        headers={"Retry-After": str(retry_after)},
    )
