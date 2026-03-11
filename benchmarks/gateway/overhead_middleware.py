"""
ASGI middleware that measures gateway request processing overhead.

Adds an X-MLflow-Gateway-Overhead-Ms response header with the total
time spent inside the gateway (wall-clock, including upstream provider call).
Only used during benchmarking — no production code changes required.
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class OverheadMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-MLflow-Gateway-Overhead-Ms"] = f"{elapsed_ms:.2f}"
        return response
