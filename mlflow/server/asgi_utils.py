from __future__ import annotations

import os

from starlette.requests import Request as StarletteRequest


def get_routed_asgi_path(request: StarletteRequest) -> str:
    """Return the routed ASGI path for a FastAPI request.

    Prefer ``request.scope["path"]`` because Starlette reconstructs
    ``request.url.path`` from the Host header, which can diverge from the actual
    routed path when the Host header is malformed. Fall back to ``request.url``
    only if the ASGI scope path is unavailable.
    """

    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        scope_path = scope.get("path")
        if isinstance(scope_path, str) and scope_path:
            return scope_path

    return str(getattr(getattr(request, "url", None), "path", "") or "")


def get_server_base_url(request: StarletteRequest) -> str:
    """Base URL for self-calls to this server (``request.base_url`` + ``--static-prefix``)."""
    # Imported lazily to keep this module's own import graph light.
    from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR

    static_prefix = os.environ.get(STATIC_PREFIX_ENV_VAR, "").rstrip("/")
    return str(request.base_url).rstrip("/") + static_prefix
