import pytest

from mlflow.exceptions import MlflowException
from mlflow.server.auth.permissions import (
    ALL_PERMISSIONS,
    EDIT,
    MANAGE,
    NO_PERMISSIONS,
    PERMISSION_PRIORITY,
    READ,
    USE,
    VALID_RESOURCE_TYPES,
    _validate_permission,
    _validate_resource_type,
    get_permission,
    max_permission,
)

# ---- Permission hierarchy ---------------------------------------------------

# The canonical ordering the rest of the auth layer relies on.
_EXPECTED_ORDER = [NO_PERMISSIONS, READ, USE, EDIT, MANAGE]


def test_permission_priority_is_total_order():
    priorities = [PERMISSION_PRIORITY[p.name] for p in _EXPECTED_ORDER]
    assert priorities == sorted(priorities)
    assert len(set(priorities)) == len(priorities)


@pytest.mark.parametrize(
    ("permission", "can_read", "can_use", "can_update", "can_delete", "can_manage"),
    [
        (NO_PERMISSIONS, False, False, False, False, False),
        (READ, True, False, False, False, False),
        (USE, True, True, False, False, False),
        (EDIT, True, True, True, False, False),
        (MANAGE, True, True, True, True, True),
    ],
)
def test_permission_capability_matrix(
    permission, can_read, can_use, can_update, can_delete, can_manage
):
    """Each permission level exposes exactly the capabilities it should.

    This pins the capability semantics: upgrading READ → USE adds can_use,
    USE → EDIT adds can_update, EDIT → MANAGE adds can_delete AND can_manage.
    If a capability bit shifts without the corresponding test update, this
    catches it.
    """
    assert permission.can_read is can_read
    assert permission.can_use is can_use
    assert permission.can_update is can_update
    assert permission.can_delete is can_delete
    assert permission.can_manage is can_manage


@pytest.mark.parametrize("permission", _EXPECTED_ORDER)
def test_get_permission_roundtrip(permission):
    assert get_permission(permission.name) is permission


def test_all_permissions_dict_is_complete():
    assert set(ALL_PERMISSIONS) == {p.name for p in _EXPECTED_ORDER}


# ---- max_permission ---------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        # Identical inputs return the same value.
        ("READ", "READ", "READ"),
        ("MANAGE", "MANAGE", "MANAGE"),
        # Higher on the right wins.
        ("READ", "USE", "USE"),
        ("USE", "EDIT", "EDIT"),
        ("EDIT", "MANAGE", "MANAGE"),
        # Higher on the left wins.
        ("USE", "READ", "USE"),
        ("EDIT", "USE", "EDIT"),
        ("MANAGE", "EDIT", "MANAGE"),
        # NO_PERMISSIONS loses to everything — a user with any active grant
        # beats an explicit deny.
        ("NO_PERMISSIONS", "READ", "READ"),
        ("READ", "NO_PERMISSIONS", "READ"),
        ("NO_PERMISSIONS", "MANAGE", "MANAGE"),
        ("NO_PERMISSIONS", "NO_PERMISSIONS", "NO_PERMISSIONS"),
    ],
)
def test_max_permission(a, b, expected):
    assert max_permission(a, b) == expected


def test_max_permission_is_symmetric():
    for a_perm in _EXPECTED_ORDER:
        for b_perm in _EXPECTED_ORDER:
            forward = max_permission(a_perm.name, b_perm.name)
            reverse = max_permission(b_perm.name, a_perm.name)
            assert forward == reverse


def test_max_permission_treats_unknown_as_lowest():
    # Defensive behavior: an unknown permission name should not silently win.
    assert max_permission("UNKNOWN", "READ") == "READ"
    assert max_permission("READ", "UNKNOWN") == "READ"
    # Two unknowns → the first wins (fallback to priority 0 for both, ``>=`` breaks the tie).
    assert max_permission("UNKNOWN_A", "UNKNOWN_B") == "UNKNOWN_A"


# ---- Validators -------------------------------------------------------------


@pytest.mark.parametrize("permission", [p.name for p in _EXPECTED_ORDER])
def test_validate_permission_accepts_known(permission):
    _validate_permission(permission)


@pytest.mark.parametrize("bogus", ["", "read", "ADMIN", "Manage", "NONE"])
def test_validate_permission_rejects_unknown(bogus):
    with pytest.raises(MlflowException, match="Invalid permission"):
        _validate_permission(bogus)


@pytest.mark.parametrize("resource_type", sorted(VALID_RESOURCE_TYPES))
def test_validate_resource_type_accepts_known(resource_type):
    _validate_resource_type(resource_type)


@pytest.mark.parametrize("bogus", ["", "Experiment", "workspaces", "run", "trace"])
def test_validate_resource_type_rejects_unknown(bogus):
    with pytest.raises(MlflowException, match="Invalid resource type"):
        _validate_resource_type(bogus)


def test_workspace_is_a_valid_resource_type():
    """Regression guard: the role-based permission model depends on
    ``workspace`` being a first-class resource type for the workspace-wide
    grant shape ``(workspace, *, PERMISSION)``. If it's ever removed from
    VALID_RESOURCE_TYPES, the UI and the role-resolution logic both break.
    """
    assert "workspace" in VALID_RESOURCE_TYPES
