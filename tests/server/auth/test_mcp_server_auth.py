from typing import Any

import pytest

from mlflow.server.auth.__init__ import _is_mcp_sub_resource_path, _permission_to_allowed_actions
from mlflow.server.auth.permissions import EDIT, MANAGE, NO_PERMISSIONS, READ, USE

# ---------------------------------------------------------------------------
# _permission_to_allowed_actions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("permission", "expected"),
    [
        (READ, []),
        (USE, ["USE"]),
        (EDIT, ["USE", "UPDATE"]),
        (MANAGE, ["USE", "UPDATE", "DELETE", "MANAGE"]),
        (NO_PERMISSIONS, []),
    ],
)
def test_permission_to_allowed_actions(permission, expected):
    assert _permission_to_allowed_actions(permission) == expected


# ---------------------------------------------------------------------------
# Backend visibility: the backend returns every server the user can read.
# ---------------------------------------------------------------------------


def _is_visible(server: dict[str, Any], permission) -> bool:
    return permission.can_read


@pytest.mark.parametrize(
    ("server", "permission", "expected_visible"),
    [
        # Active server with endpoints — visible to all readable permissions
        ({"name": "s1", "status": "active", "access_endpoints": [{"endpoint_id": 1}]}, READ, True),
        ({"name": "s1", "status": "active", "access_endpoints": [{"endpoint_id": 1}]}, USE, True),
        ({"name": "s1", "status": "active", "access_endpoints": [{"endpoint_id": 1}]}, EDIT, True),
        (
            {"name": "s1", "status": "active", "access_endpoints": [{"endpoint_id": 1}]},
            MANAGE,
            True,
        ),
        # No endpoints — still visible
        ({"name": "s1", "status": "active", "access_endpoints": []}, READ, True),
        ({"name": "s1", "status": "active", "access_endpoints": []}, USE, True),
        ({"name": "s1", "status": "active", "access_endpoints": []}, EDIT, True),
        ({"name": "s1", "status": "active", "access_endpoints": []}, MANAGE, True),
        # Draft server with endpoints — still visible
        ({"name": "s1", "status": "draft", "access_endpoints": [{"endpoint_id": 1}]}, EDIT, True),
        ({"name": "s1", "status": "draft", "access_endpoints": [{"endpoint_id": 1}]}, MANAGE, True),
        # Deprecated with endpoints — still visible
        (
            {"name": "s1", "status": "deprecated", "access_endpoints": [{"endpoint_id": 1}]},
            READ,
            True,
        ),
        (
            {"name": "s1", "status": "deprecated", "access_endpoints": [{"endpoint_id": 1}]},
            MANAGE,
            True,
        ),
        # NO_PERMISSIONS — never visible regardless of server state
        (
            {"name": "s1", "status": "active", "access_endpoints": [{"endpoint_id": 1}]},
            NO_PERMISSIONS,
            False,
        ),
        (
            {"name": "s1", "status": "active", "access_endpoints": []},
            NO_PERMISSIONS,
            False,
        ),
        (
            {"name": "s1", "status": "draft", "access_endpoints": [{"endpoint_id": 1}]},
            NO_PERMISSIONS,
            False,
        ),
    ],
)
def test_visibility(server, permission, expected_visible):
    assert _is_visible(server, permission) == expected_visible


def test_stamp_enriches_dict():
    server = {"name": "s1", "status": "active", "access_endpoints": [{"endpoint_id": 1}]}
    actions = _permission_to_allowed_actions(EDIT)
    server["allowed_actions"] = actions
    assert server["allowed_actions"] == ["USE", "UPDATE"]
    assert "DELETE" not in server["allowed_actions"]
    assert "MANAGE" not in server["allowed_actions"]


# ---------------------------------------------------------------------------
# connect_options permission: EDIT (can_update) now covers connect_options
# PATCH — the old MANAGE gate was removed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("permission", "update_allowed"),
    [
        (READ, False),
        (USE, False),
        (EDIT, True),
        (MANAGE, True),
        (NO_PERMISSIONS, False),
    ],
)
def test_connect_options_requires_update(permission, update_allowed):
    actions = _permission_to_allowed_actions(permission)
    assert ("UPDATE" in actions) == update_allowed


# ---------------------------------------------------------------------------
# _is_mcp_sub_resource_path: structural parsing after {namespace}/{slug}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Server detail — NOT a sub-resource
        ("/ajax-api/3.0/mlflow/mcp-servers/default/my-server", False),
        ("/api/3.0/mlflow/mcp-servers/default/my-server", False),
        # Sub-resource paths — IS a sub-resource
        ("/ajax-api/3.0/mlflow/mcp-servers/default/my-server/versions/3", True),
        ("/ajax-api/3.0/mlflow/mcp-servers/default/my-server/aliases/latest", True),
        ("/ajax-api/3.0/mlflow/mcp-servers/default/my-server/tags", True),
        ("/ajax-api/3.0/mlflow/mcp-servers/default/my-server/endpoints", True),
        # Reserved word as namespace — NOT a sub-resource (this was the bug)
        ("/ajax-api/3.0/mlflow/mcp-servers/tags/my-server", False),
        ("/ajax-api/3.0/mlflow/mcp-servers/versions/my-server", False),
        ("/ajax-api/3.0/mlflow/mcp-servers/endpoints/my-server", False),
        ("/ajax-api/3.0/mlflow/mcp-servers/aliases/my-server", False),
        # Reserved word as namespace WITH sub-resource — IS a sub-resource
        ("/ajax-api/3.0/mlflow/mcp-servers/tags/my-server/versions/1", True),
        ("/ajax-api/3.0/mlflow/mcp-servers/versions/foo/aliases/latest", True),
        # Bare prefix — NOT a sub-resource
        ("/ajax-api/3.0/mlflow/mcp-servers", False),
        ("/ajax-api/3.0/mlflow/mcp-servers/", False),
    ],
)
def test_is_mcp_sub_resource_path(path, expected):
    assert _is_mcp_sub_resource_path(path) == expected
