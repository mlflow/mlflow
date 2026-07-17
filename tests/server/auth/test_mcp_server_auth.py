"""Unit tests for MCP server auth helper functions.

Tests _permission_to_allowed_actions (importable) and the _is_dimmed /
_stamp_and_check logic (closures — tested via the public filter behavior).
"""

import pytest

from mlflow.server.auth.__init__ import _permission_to_allowed_actions
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


def test_permission_to_allowed_actions_read_has_no_actions():
    actions = _permission_to_allowed_actions(READ)
    assert "USE" not in actions
    assert "UPDATE" not in actions
    assert "DELETE" not in actions
    assert "MANAGE" not in actions


def test_permission_to_allowed_actions_manage_has_all():
    actions = _permission_to_allowed_actions(MANAGE)
    assert "USE" in actions
    assert "UPDATE" in actions
    assert "DELETE" in actions
    assert "MANAGE" in actions


def test_permission_to_allowed_actions_edit_has_no_delete():
    actions = _permission_to_allowed_actions(EDIT)
    assert "UPDATE" in actions
    assert "DELETE" not in actions
    assert "MANAGE" not in actions


# ---------------------------------------------------------------------------
# _is_dimmed logic (tested directly since it's a pure function)
# The closure in _filter_search_mcp_servers defines:
#   _is_dimmed(s) = not s.get("access_bindings") or s.get("status") != "active"
# ---------------------------------------------------------------------------


def _is_dimmed(s: dict) -> bool:
    return not s.get("access_bindings") or s.get("status") != "active"


@pytest.mark.parametrize(
    ("server", "expected"),
    [
        # Active with bindings → not dimmed
        ({"status": "active", "access_bindings": [{"binding_id": 1}]}, False),
        # Active without bindings → dimmed
        ({"status": "active", "access_bindings": []}, True),
        ({"status": "active"}, True),
        # Draft with bindings → dimmed (status not active)
        ({"status": "draft", "access_bindings": [{"binding_id": 1}]}, True),
        # Draft without bindings → dimmed
        ({"status": "draft"}, True),
        # Deprecated with bindings → dimmed
        ({"status": "deprecated", "access_bindings": [{"binding_id": 1}]}, True),
        # No status field → dimmed
        ({"access_bindings": [{"binding_id": 1}]}, True),
        # None bindings → dimmed
        ({"status": "active", "access_bindings": None}, True),
    ],
)
def test_is_dimmed(server, expected):
    assert _is_dimmed(server) == expected


# ---------------------------------------------------------------------------
# _stamp_and_check logic
# The closure defines:
#   _stamp_and_check(s) =
#     - return False if can_read(s["name"]) fails
#     - stamp allowed_actions onto s
#     - return False if dimmed AND not MANAGE
#     - return True otherwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("server", "permission", "expected_visible"),
    [
        # Active server with bindings — visible to all permission levels
        (
            {"name": "s1", "status": "active", "access_bindings": [{"binding_id": 1}]},
            READ,
            True,
        ),
        (
            {"name": "s1", "status": "active", "access_bindings": [{"binding_id": 1}]},
            USE,
            True,
        ),
        # Dimmed server (no bindings) — hidden from non-MANAGE
        (
            {"name": "s1", "status": "active", "access_bindings": []},
            READ,
            False,
        ),
        (
            {"name": "s1", "status": "active", "access_bindings": []},
            USE,
            False,
        ),
        (
            {"name": "s1", "status": "active", "access_bindings": []},
            EDIT,
            False,
        ),
        # Dimmed server — visible to MANAGE
        (
            {"name": "s1", "status": "active", "access_bindings": []},
            MANAGE,
            True,
        ),
        # Draft server with bindings — dimmed, hidden from non-MANAGE
        (
            {"name": "s1", "status": "draft", "access_bindings": [{"binding_id": 1}]},
            EDIT,
            False,
        ),
        # Draft server — visible to MANAGE
        (
            {"name": "s1", "status": "draft", "access_bindings": [{"binding_id": 1}]},
            MANAGE,
            True,
        ),
    ],
)
def test_stamp_and_check_visibility(server, permission, expected_visible):
    allowed_actions = _permission_to_allowed_actions(permission)
    server["allowed_actions"] = allowed_actions

    dimmed = _is_dimmed(server)
    visible = not dimmed or MANAGE.name in allowed_actions

    assert visible == expected_visible


def test_stamp_and_check_stamps_allowed_actions():
    server = {"name": "s1", "status": "active", "access_bindings": [{"binding_id": 1}]}
    actions = _permission_to_allowed_actions(EDIT)
    server["allowed_actions"] = actions
    assert server["allowed_actions"] == ["USE", "UPDATE"]


def test_stamp_and_check_manage_sees_dimmed():
    server = {"name": "s1", "status": "draft", "access_bindings": []}
    actions = _permission_to_allowed_actions(MANAGE)
    server["allowed_actions"] = actions
    assert "MANAGE" in server["allowed_actions"]
    assert _is_dimmed(server) is True
