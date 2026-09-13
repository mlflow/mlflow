"""
Optional server-side login sessions for the MLflow basic-auth app.

The default authorization_function (``authenticate_request_basic_auth``)
re-validates the client's username/password on every request, so nothing
survives across requests server-side. Point ``authorization_function`` at
``mlflow.server.auth.session:authenticate_request_session`` instead to issue
a session cookie on login, backed by the ``sessions`` table (see
``mlflow.server.auth.db.models.SqlSession``). Because that table lives in
the shared auth database rather than one worker's memory, a session is
recognized by whichever MLflow instance behind a load balancer gets the next
request. See https://github.com/mlflow/mlflow/issues/13643.

Not yet handled, left as follow-ups:
  * a signed-cookie-only backend needing no DB row (fast-follow via the same
    authorization_function extension point).
  * an explicit ``/logout`` endpoint - sessions just expire on TTL for now.
  * minting a *reusable* cookie for a first login that lands on one of
    MLflow's native FastAPI routes (gateway, jobs, assistant, ...). Auth
    still succeeds there and a session row is still created, but the
    Set-Cookie can't reach the client through that bridge yet, so the row
    goes unused and expires normally.
"""

from __future__ import annotations

import logging

from flask import Response, g, request
from werkzeug.datastructures import Authorization

from mlflow.server.auth import authenticate_request_basic_auth, store
from mlflow.server.auth.config import read_auth_config

_logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "mlflow_session"

_session_ttl_seconds: int | None = None


def _get_session_ttl_seconds() -> int:
    global _session_ttl_seconds
    if _session_ttl_seconds is None:
        _session_ttl_seconds = read_auth_config().session_ttl_seconds
    return _session_ttl_seconds


def authenticate_request_session() -> Authorization | Response:
    """
    Drop-in ``authorization_function`` that layers a session on top of
    ``authenticate_request_basic_auth``. Memoized on ``g`` because
    mlflow.server.auth calls the configured authorization function more than
    once per request (``_before_request``, then again per validator), and
    this one has a side effect (minting a session row) that must happen once.
    """
    if (cached := getattr(g, "_mlflow_session_auth_result", None)) is not None:
        return cached

    # Explicit Basic credentials always take precedence over an existing
    # session cookie - otherwise a client that reuses a cookie jar (or a
    # browser) while sending fresh Authorization for a *different* user
    # would silently stay authenticated as whoever the cookie belongs to.
    # Only consult the cookie when no credentials were presented this
    # request.
    if request.authorization is None and (session_id := request.cookies.get(SESSION_COOKIE_NAME)):
        if username := store.get_session_username(session_id):
            result = Authorization(auth_type="basic", data={"username": username})
            g._mlflow_session_auth_result = result
            return result
        # Stale cookie: drop it once we've re-authenticated below.
        g.mlflow_session_stale = True

    result = authenticate_request_basic_auth()
    if isinstance(result, Authorization) and result.username:
        g.mlflow_new_session_id = store.create_session(
            result.username, ttl_seconds=_get_session_ttl_seconds()
        )
    g._mlflow_session_auth_result = result
    return result


def apply_session_cookie(resp: Response) -> Response:
    """Flask ``after_request`` hook; no-op unless ``g`` has a pending
    session cookie to set or clear (see ``authenticate_request_session``)."""
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
