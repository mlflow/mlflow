from __future__ import annotations

import ipaddress
import os

from starlette.requests import Request as StarletteRequest

from mlflow.environment_variables import _MLFLOW_SERVER_TLS_LISTENER


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
    """Base URL of this server as the client addressed it, plus ``--static-prefix``.

    ``request.base_url`` comes from the client-supplied ``Host`` header, so this is the URL to
    show to and hand back to the caller, but never a destination for server-internal
    credentials: a caller can point it anywhere. Self-calls that carry such credentials use
    :func:`get_server_self_url`.
    """
    # Imported lazily to keep this module's own import graph light.
    from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR

    static_prefix = os.environ.get(STATIC_PREFIX_ENV_VAR, "").rstrip("/")
    return str(request.base_url).rstrip("/") + static_prefix


def _self_call_host(host: str) -> str:
    """The host part of a self-call URL for the local address ``host``.

    A wildcard bind address is reachable only via loopback, an IPv4-mapped IPv6 address is the
    IPv4 address it wraps, and an IPv6 literal needs brackets in a URL. A non-IP value (a
    hostname reported by the ASGI server, e.g. Starlette's test client) is used as is.
    """
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host
    if ip.is_unspecified:
        ip = ipaddress.ip_address("::1" if ip.version == 6 else "127.0.0.1")
    elif ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return f"[{ip}]" if ip.version == 6 else str(ip)


def get_server_self_url(request: StarletteRequest) -> str | None:
    """URL at which this server reaches itself: the local address the request arrived on, plus
    ``--static-prefix``. ``None`` when the ASGI server reports no address (e.g. a unix socket).

    The address is the ASGI ``server`` entry (the server's own end of the accepted connection),
    never the client-supplied ``Host`` header, so a self-call that carries server-internal
    credentials (the MLflow Assistant's gateway provider authenticates with the internal gateway
    token) cannot be pointed at a host of the caller's choosing. The call goes to the server
    socket directly, bypassing any TLS-terminating proxy in front of the server: plain HTTP, or
    HTTPS when uvicorn terminates TLS itself (``_MLFLOW_SERVER_TLS_LISTENER``, set by
    ``mlflow server`` from its ``--uvicorn-opts``).
    """
    # Imported lazily to keep this module's own import graph light.
    from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR

    scope = getattr(request, "scope", None)
    server = scope.get("server") if isinstance(scope, dict) else None
    if not (isinstance(server, (tuple, list)) and len(server) == 2 and server[0] and server[1]):
        return None
    host, port = server
    scheme = "https" if _MLFLOW_SERVER_TLS_LISTENER.get() else "http"
    static_prefix = os.environ.get(STATIC_PREFIX_ENV_VAR, "").rstrip("/")
    return f"{scheme}://{_self_call_host(str(host))}:{int(port)}{static_prefix}"
