"""
Optional server-side login sessions for the MLflow basic-auth app.

By default, the auth app's ``authorization_function`` is
``authenticate_request_basic_auth`` (see ``mlflow.server.auth`` and
``config.py``), which re-validates the client's username/password on every
single request. That's correct on its own — any instance behind a load
balancer can independently verify the same credentials against the shared
auth database — but it means a browser session has nothing that survives
across requests server-side, and re-prompts for credentials more than users
expect from a normal "login."

This module is a drop-in alternative, enabled by pointing the auth ini
file's ``authorization_function`` at
``mlflow.server.auth.session:authenticate_request_session`` instead. Once
enabled: a request that authenticates via Basic Auth is issued a random,
opaque session id (via the ``mlflow_session`` cookie), recorded in the same
auth database as users/roles (``SqlSession`` in
``mlflow.server.auth.db.models``). A later request presenting a still-valid
session cookie is authenticated by looking up that row — no password
re-check — and, because the row lives in the shared database rather than in
one worker's memory, this works identically no matter which MLflow instance
behind a load balancer receives the request. That cross-instance gap is
exactly https://github.com/mlflow/mlflow/issues/13643.

Deliberately out of scope for this first cut (see the issue thread for the
fuller design discussion):
  * a signed-cookie-only backend that needs no database row at all — a
    reasonable fast-follow once this one is in use, sharing the same
    ``authorization_function`` extension point.
  * an explicit ``/logout`` endpoint — until one exists, a session simply
    expires after ``session_ttl_seconds``, or an admin can rotate
    ``MLFLOW_FLASK_SERVER_SECRET_KEY`` (invalidates the CSRF/flash cookie,
    not this session cookie) or delete the user's rows in the ``sessions``
    table directly.
  * routes served by MLflow's native FastAPI routers (jobs, gateway,
    assistant, otel, artifact streaming, MCP) minting a session cookie
    end-to-end. Presenting an *existing* session cookie authenticates
    correctly there too, since FastAPI's permission middleware bridges any
    custom ``authorization_function`` through a synthetic Flask request
    context (see ``_authenticate_custom_for_fastapi`` in
    ``mlflow.server.auth``). But a fresh Basic Auth login on one of those
    routes still creates a session row in the database (this module has no
    way to tell it's running inside that bridge rather than a real request) -
    it just never reaches the client, because the bridge's synthetic Flask
    response is discarded rather than translated into a ``Set-Cookie`` on the
    real Starlette response. That row isn't reused by anything and is
    harmless beyond the wasted row - it expires and is swept like any other -
    but it means a client whose *first* authenticated request happens to land
    on one of those routes won't get a reusable cookie from it. Given
    ``authenticate_request_basic_auth`` accepts credentials on every request
    regardless, that request still succeeds; it just isn't the one that
    established the session. Wiring a ``Set-Cookie`` through that bridge is
    left as a follow-up (it needs ``add_fastapi_permission_middleware`` in
    ``mlflow.server.auth`` to inspect the outcome of the bridge call, not
    just this module).
"""

from __future__ import annotations

import logging

from flask import Response, g, request
from werkzeug.datastructures import Authorization

from mlflow.server.auth import authenticate_request_basic_auth, store
from mlflow.server.auth.config import read_auth_config

_logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "mlflow_session"

# Loaded lazily (not at import time) so importing this module never requires
# a fully configured auth environment, and so a config change on disk is
# picked up by a server restart the same way the rest of AuthConfig is.
_session_ttl_seconds: int | None = None


def _get_session_ttl_seconds() -> int:
    global _session_ttl_seconds
    if _session_ttl_seconds is None:
        _session_ttl_seconds = read_auth_config().session_ttl_seconds
    return _session_ttl_seconds


def authenticate_request_session() -> Authorization | Response:
    """
    Authorization function (see ``AuthConfig.authorization_function``) that
    layers a server-side session on top of ``authenticate_request_basic_auth``.

    Returns the same shapes that function does (a ``werkzeug`` ``Authorization``
    on success, or a 401 ``Response`` prompting for Basic Auth), so it's a
    transparent replacement from the caller's (``_before_request``'s)
    perspective.

    ``mlflow.server.auth`` calls the configured authorization function more
    than once per request by design - once from ``_before_request``, and
    again from inside whichever per-route validator runs next, each time it
    needs the caller's username for a permission check. That's free for the
    default, stateless ``authenticate_request_basic_auth``, but this function
    has a side effect (minting a session row) that must happen at most once
    per request, so the result is memoized on ``g`` for the request's
    duration.
    """
    if (cached := getattr(g, "_mlflow_session_auth_result", None)) is not None:
        return cached

    if session_id := request.cookies.get(SESSION_COOKIE_NAME):
        if username := store.get_session_username(session_id):
            result = Authorization(auth_type="basic", data={"username": username})
            g._mlflow_session_auth_result = result
            return result
        # Cookie didn't resolve to a live session (expired, revoked, or from a
        # database this server no longer points at) - tell the client to drop
        # it once we've re-authenticated below, rather than presenting it
        # forever.
        g.mlflow_session_stale = True

    result = authenticate_request_basic_auth()
    if isinstance(result, Authorization) and result.username:
        # Fresh, successful Basic Auth login: mint a session so the *next*
        # request - on this instance or any other sharing this auth database
        # - is recognized without re-sending credentials.
        g.mlflow_new_session_id = store.create_session(
            result.username, ttl_seconds=_get_session_ttl_seconds()
        )
    g._mlflow_session_auth_result = result
    return result


def apply_session_cookie(resp: Response) -> Response:
    """
    Registered as a Flask ``after_request`` hook by ``create_app`` (always,
    not just when sessions are enabled - it's a cheap no-op unless
    ``authenticate_request_session`` set something on ``g`` for this request).
    """
    if session_id := getattr(g, "mlflow_new_session_id", None):
        resp.set_cookie(
            SESSION_COOKIE_NAME,
            session_id,
            max_age=_get_session_ttl_seconds(),
            httponly=True,
            samesite="Lax",
            secure=request.is_secure,
        )
    elif getattr(g, "mlflow_session_stale", False):
        resp.delete_cookie(SESSION_COOKIE_NAME)
    return resp
