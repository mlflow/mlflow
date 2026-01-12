"""
Rate limiter for Gateway API endpoints.

This module provides rate limiting functionality for gateway endpoints,
supporting both default endpoint limits and per-user limits.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock

from mlflow.store.tracking.abstract_store import AbstractStore

_logger = logging.getLogger(__name__)


@dataclass
class RateLimitWindow:
    """Tracks request counts within a sliding window."""

    requests: list[float] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

    def count_requests_in_window(self, window_start: float) -> int:
        """Count requests within the time window, cleaning up old entries."""
        with self.lock:
            # Remove expired requests
            self.requests = [ts for ts in self.requests if ts >= window_start]
            return len(self.requests)

    def add_request(self, timestamp: float) -> None:
        """Record a new request."""
        with self.lock:
            self.requests.append(timestamp)


class GatewayRateLimiter:
    """
    In-memory rate limiter for gateway endpoints.

    Uses a sliding window algorithm to track request counts per endpoint/user
    combination. Rate limit configurations are fetched from the database.

    Note: This implementation is suitable for single-server deployments.
    For distributed deployments, consider using Redis or another distributed
    rate limiting solution.
    """

    # Window size in seconds (1 minute for QPM - queries per minute)
    WINDOW_SIZE = 60.0

    def __init__(self):
        # Key: (endpoint_id, username or None for default)
        self._windows: dict[tuple[str, str | None], RateLimitWindow] = defaultdict(
            RateLimitWindow
        )
        self._global_lock = Lock()

    def _get_window_key(self, endpoint_id: str, username: str | None) -> tuple[str, str | None]:
        """Get the cache key for a rate limit window."""
        return (endpoint_id, username)

    def _get_window(self, endpoint_id: str, username: str | None) -> RateLimitWindow:
        """Get or create a rate limit window."""
        key = self._get_window_key(endpoint_id, username)
        with self._global_lock:
            if key not in self._windows:
                self._windows[key] = RateLimitWindow()
            return self._windows[key]

    def check_rate_limit(
        self,
        store: AbstractStore,
        endpoint_id: str,
        username: str | None = None,
    ) -> tuple[bool, int | None, int | None]:
        """
        Check if a request is allowed under the rate limit.

        Args:
            store: The tracking store to fetch rate limit configurations.
            endpoint_id: The ID of the endpoint being called.
            username: The username making the request (if available).

        Returns:
            Tuple of (allowed, limit, remaining):
            - allowed: True if the request is allowed, False if rate limited
            - limit: The QPM limit (None if no limit configured)
            - remaining: Remaining requests in the current window (None if no limit)
        """
        # Get effective rate limit for this user/endpoint
        rate_limit = store.get_gateway_rate_limit_for_user(endpoint_id, username)

        if rate_limit is None:
            # No rate limit configured - allow the request
            return True, None, None

        limit = rate_limit.queries_per_minute
        now = time.time()
        window_start = now - self.WINDOW_SIZE

        # Get window for the specific user, or default if no user-specific limit
        effective_username = rate_limit.username  # This is the actual limit's username
        window = self._get_window(endpoint_id, effective_username)

        # Count current requests in the window
        current_count = window.count_requests_in_window(window_start)

        if current_count >= limit:
            # Rate limit exceeded
            remaining = 0
            _logger.warning(
                f"Rate limit exceeded for endpoint={endpoint_id}, user={username}, "
                f"limit={limit} QPM"
            )
            return False, limit, remaining

        # Allow the request and record it
        window.add_request(now)
        remaining = limit - current_count - 1
        return True, limit, remaining

    def record_request(
        self,
        store: AbstractStore,
        endpoint_id: str,
        username: str | None = None,
    ) -> None:
        """
        Record a request for rate limiting purposes.

        This is called after check_rate_limit returns True, but can also
        be called separately if needed for tracking purposes.

        Note: check_rate_limit already records the request, so this is
        typically only needed if you skipped check_rate_limit.
        """
        rate_limit = store.get_gateway_rate_limit_for_user(endpoint_id, username)
        if rate_limit is None:
            return

        effective_username = rate_limit.username
        window = self._get_window(endpoint_id, effective_username)
        window.add_request(time.time())

    def clear(self) -> None:
        """Clear all rate limit windows. Useful for testing."""
        with self._global_lock:
            self._windows.clear()


# Global rate limiter instance
_rate_limiter: GatewayRateLimiter | None = None
_rate_limiter_lock = Lock()


def get_rate_limiter() -> GatewayRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    with _rate_limiter_lock:
        if _rate_limiter is None:
            _rate_limiter = GatewayRateLimiter()
        return _rate_limiter


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self,
        endpoint_id: str,
        limit: int,
        username: str | None = None,
        retry_after: int = 60,
    ):
        self.endpoint_id = endpoint_id
        self.limit = limit
        self.username = username
        self.retry_after = retry_after
        message = f"Rate limit exceeded: {limit} queries per minute"
        if username:
            message += f" for user '{username}'"
        super().__init__(message)
