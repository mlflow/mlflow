from __future__ import annotations

from mlflow.server.security_utils import strip_static_prefix


def is_unprotected_route(path: str, static_prefix: str | None = None) -> bool:
    """
    Determine whether a request path is unprotected by auth.

    Unprotected:
    - Static assets
    - Favicon
    - Health/version endpoints
    """
    normalized_path = strip_static_prefix(path, static_prefix)

    return normalized_path.startswith(
        (
            "/static/",
            "/favicon.ico",
            "/health",
            "/version",
        )
    )
