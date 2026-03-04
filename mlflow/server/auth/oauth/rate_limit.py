import time
from collections import defaultdict
from threading import Lock

from flask import Response, make_response, request


class RateLimiter:
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def _get_client_key(self) -> str:
        return request.remote_addr or "unknown"

    def _cleanup(self, key: str, now: float):
        cutoff = now - self._window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

    def is_rate_limited(self) -> Response | None:
        key = self._get_client_key()
        now = time.time()

        with self._lock:
            self._cleanup(key, now)

            if len(self._requests[key]) >= self._max_requests:
                resp = make_response("Too many requests. Please try again later.", 429)
                resp.headers["Retry-After"] = str(self._window_seconds)
                return resp

            self._requests[key].append(now)
            return None


# Default rate limiter for auth endpoints: 30 requests per minute per IP
auth_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)
