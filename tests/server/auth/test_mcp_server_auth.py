"""Unit tests for MCP server auth helper functions.

Tests _permission_to_allowed_actions (importable) and the visibility
logic that the response filter applies.
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


# ---------------------------------------------------------------------------
# Visibility logic: _is_dimmed + stamp-and-check
#
# The response filter in _filter_search_mcp_servers uses this combined rule:
#   dimmed = not server.get("access_endpoints") or server.get("status") != "active"
#   visible = not dimmed or "MANAGE" in allowed_actions
#
# These tests verify the logic against _permission_to_allowed_actions output,
# ensuring the two halves compose correctly.
# ---------------------------------------------------------------------------


def _is_dimmed(s: dict) -> bool:
    """Mirror of the closure in _filter_search_mcp_servers."""
    return not s.get("access_endpoints") or s.get("status") != "active"


def _is_visible(server: dict, permission) -> bool:
    """Combine dimming + permission check as the production filter does."""
    allowed_actions = _permission_to_allowed_actions(permission)
    dimmed = _is_dimmed(server)
    return not dimmed or "MANAGE" in allowed_actions


@pytest.mark.parametrize(
    ("server", "expected"),
    [
        ({"status": "active", "access_endpoints": [{"endpoint_id": 1}]}, False),
        ({"status": "active", "access_endpoints": []}, True),
        ({"status": "active"}, True),
        ({"status": "draft", "access_endpoints": [{"endpoint_id": 1}]}, True),
        ({"status": "draft"}, True),
        ({"status": "deprecated", "access_endpoints": [{"endpoint_id": 1}]}, True),
        ({"access_endpoints": [{"endpoint_id": 1}]}, True),
        ({"status": "active", "access_endpoints": None}, True),
    ],
)
def test_is_dimmed(server, expected):
    assert _is_dimmed(server) == expected


@pytest.mark.parametrize(
    ("server", "permission", "expected_visible"),
    [
        # Active server with endpoints — visible to all permission levels
        ({"name": "s1", "status": "active", "access_endpoints": [{"endpoint_id": 1}]}, READ, True),
        ({"name": "s1", "status": "active", "access_endpoints": [{"endpoint_id": 1}]}, USE, True),
        ({"name": "s1", "status": "active", "access_endpoints": [{"endpoint_id": 1}]}, EDIT, True),
        (
            {"name": "s1", "status": "active", "access_endpoints": [{"endpoint_id": 1}]},
            MANAGE,
            True,
        ),
        # Dimmed server (no endpoints) — hidden from non-MANAGE
        ({"name": "s1", "status": "active", "access_endpoints": []}, READ, False),
        ({"name": "s1", "status": "active", "access_endpoints": []}, USE, False),
        ({"name": "s1", "status": "active", "access_endpoints": []}, EDIT, False),
        ({"name": "s1", "status": "active", "access_endpoints": []}, MANAGE, True),
        # Draft server with endpoints — dimmed, hidden from non-MANAGE
        ({"name": "s1", "status": "draft", "access_endpoints": [{"endpoint_id": 1}]}, EDIT, False),
        ({"name": "s1", "status": "draft", "access_endpoints": [{"endpoint_id": 1}]}, MANAGE, True),
        # Deprecated with endpoints — dimmed, hidden from non-MANAGE
        (
            {"name": "s1", "status": "deprecated", "access_endpoints": [{"endpoint_id": 1}]},
            READ,
            False,
        ),
        (
            {"name": "s1", "status": "deprecated", "access_endpoints": [{"endpoint_id": 1}]},
            MANAGE,
            True,
        ),
    ],
)
def test_visibility(server, permission, expected_visible):
    assert _is_visible(server, permission) == expected_visible


def test_stamp_enriches_dict():
    """Verify that _permission_to_allowed_actions produces the right shape for stamping."""
    server = {"name": "s1", "status": "active", "access_endpoints": [{"endpoint_id": 1}]}
    actions = _permission_to_allowed_actions(EDIT)
    server["allowed_actions"] = actions
    assert server["allowed_actions"] == ["USE", "UPDATE"]
    assert "DELETE" not in server["allowed_actions"]
    assert "MANAGE" not in server["allowed_actions"]
