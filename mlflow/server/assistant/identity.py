"""Authenticated-user identity for the MLflow Assistant's FastAPI routes.

The Assistant API is a FastAPI-native router mounted alongside the MLflow Flask app. MLflow's
auth plugin authenticates Flask routes with per-endpoint before-request handlers, and its FastAPI
permission middleware only authenticates native routes that resolve an authorization validator --
the Assistant routes do not, so they have no authenticated user of their own. This module resolves
the caller's identity by delegating to the auth plugin's public FastAPI authentication entry point
(``authenticate_fastapi_request_user``), so the Assistant keys on the same username the rest of the
server would and reuses its credential cache and custom-``authorization_function`` handling.

When the auth plugin is not active (a no-auth server), there is no identity to establish: the
resolver returns ``None`` and the Assistant keeps its existing single-user behavior.
"""

import sys

from fastapi import Request
from starlette.responses import Response as StarletteResponse

# Sent when the auth plugin is active but the request has no valid credentials, so a browser or
# client knows to present basic-auth (matching the Flask auth plugin's challenge).
BASIC_AUTH_CHALLENGE_HEADERS = {"WWW-Authenticate": 'Basic realm="mlflow"'}


def auth_plugin_active() -> bool:
    """Whether the MLflow auth plugin is running and initialized.

    ``is_auth_enabled()`` is the plugin's authoritative signal: it is set only when the auth app
    factory runs (``mlflow server --app-name basic-auth``). Importing ``mlflow.server.auth`` is
    NOT sufficient -- other code paths (e.g. the GraphQL middleware) import that module even on a
    no-auth server, so a bare ``sys.modules`` check would wrongly demand credentials there. Read
    the flag via ``sys.modules.get`` so a no-auth server never imports the module (which binds to
    the auth database on import). Mirrors ``_is_server_auth_enabled`` in ``mlflow.server.handlers``.
    """
    auth_module = sys.modules.get("mlflow.server.auth")
    return bool(auth_module and auth_module.is_auth_enabled())


class AssistantAuthError(Exception):
    """Raised when the auth plugin is active but the request has no valid credentials.

    The route layer translates this into a 401 with a basic-auth challenge. Kept as a plain
    exception (not ``HTTPException``) so this module does not depend on how the router responds.
    """


def resolve_authenticated_username(request: Request) -> str | None:
    """Return the caller's authenticated username.

    Returns ``None`` when the auth plugin is not active (no-auth server). When the plugin is
    active, delegates to its FastAPI authentication entry point and returns the authenticated
    username, or raises :class:`AssistantAuthError` when credentials are missing or invalid (the
    plugin returns ``None``, or a ``Response`` from a custom auth function) so the Assistant cannot
    be driven anonymously on an authenticated deployment.
    """
    if not auth_plugin_active():
        return None

    # Imported lazily and only when the plugin is active, so a no-auth server never pulls in the
    # auth module (which binds to the auth database on import).
    from mlflow.server.auth import authenticate_fastapi_request_user

    result = authenticate_fastapi_request_user(request)
    if result is None or isinstance(result, StarletteResponse):
        raise AssistantAuthError(
            "Valid MLflow credentials are required to use the MLflow Assistant."
        )
    return result.username
