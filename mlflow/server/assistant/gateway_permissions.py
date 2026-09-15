"""Grant Assistant users access to the Assistant's managed AI Gateway endpoints.

The Assistant creates one gateway endpoint per vendor (``ensure_gateway_connection``) directly
through the tracking store, not through the gateway's HTTP create-endpoint route. So the normal
"the creator gets MANAGE, everyone else gets nothing" grant never runs, and on an auth-enabled
server a non-admin caller is denied (403) at the gateway's USE check even though they are allowed
to use the Assistant. This module grants the caller USE on those managed endpoints so their own
Assistant turns can reach the gateway. It treats an Assistant-managed endpoint as shared server
infrastructure: any user permitted to use the Assistant may use it.
"""

import logging

from mlflow.assistant.gateway_connection import managed_gateway_endpoint_names
from mlflow.server.assistant.identity import auth_plugin_active

_logger = logging.getLogger(__name__)


def ensure_assistant_gateway_use_permission(username: str | None) -> None:
    """Grant ``username`` USE on the Assistant's managed gateway endpoints.

    No-op when the auth plugin is not active (there are no permissions to manage) or the caller
    is anonymous. Matches the exact set of names the Assistant creates (see
    ``managed_gateway_endpoint_names``), so an operator's own gateway endpoints are out of scope.
    Idempotent: endpoints the user can already use are skipped, so a steady-state turn performs
    no writes.

    Best-effort by design: any failure here (a store without gateway support, a workspace or
    lookup error, a transient auth-store error) must not break the turn. The gateway's own USE
    check stays the authoritative gate and surfaces a clear 403 if the user still lacks access.

    Known limitation: with ``MLFLOW_ENABLE_WORKSPACES`` on, the auth store resolves the active
    workspace from request context, which this background thread does not carry, so the auth-store
    calls raise and are caught here -- the grant is skipped (the caller keeps getting the gateway's
    403) rather than landing in the wrong workspace. Because the very first auth-store read raises,
    no partial grant is written and no write repeats per turn. Granting in the endpoint's workspace
    is a follow-up; workspaces are off by default.
    """
    if not username or not auth_plugin_active():
        return

    # Imported lazily and only when the plugin is active, so a no-auth server never pulls in the
    # auth module (which binds to the auth database on import).
    from mlflow.server.auth import store as auth_store
    from mlflow.server.auth.permissions import USE, get_permission
    from mlflow.tracking._tracking_service.utils import _get_store

    try:
        try:
            endpoints = _get_store().list_gateway_endpoints()
        except (AttributeError, NotImplementedError):
            # A tracking store without gateway support has no Assistant endpoints to authorize;
            # this is expected, so return quietly (no warning). Any other error from this lookup
            # falls through to the fail-open handler below.
            return
        managed_names = managed_gateway_endpoint_names()
        managed = [e for e in endpoints if e.name in managed_names]
        if not managed:
            return
        already_usable = {
            p.endpoint_id
            for p in auth_store.list_gateway_endpoint_permissions(username)
            if get_permission(p.permission).can_use
        }
        for endpoint in managed:
            if endpoint.endpoint_id not in already_usable:
                auth_store.grant_user_permission(
                    username, "gateway_endpoint", endpoint.endpoint_id, USE.name
                )
    except Exception:
        # Any unexpected failure (an endpoint listing error, a permission-store read/write error, a
        # workspace resolution error) must not break the turn: the gateway's own USE check stays
        # the authoritative gate and returns a clear 403 if the user still lacks access.
        _logger.warning(
            "Could not ensure gateway USE permission for %s; the gateway will enforce access",
            username,
            exc_info=True,
        )
