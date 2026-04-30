from dataclasses import dataclass

from mlflow import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


@dataclass
class Permission:
    name: str
    can_read: bool
    can_use: bool
    can_update: bool
    can_delete: bool
    can_manage: bool


READ = Permission(
    name="READ",
    can_read=True,
    can_use=False,
    can_update=False,
    can_delete=False,
    can_manage=False,
)

USE = Permission(
    name="USE",
    can_read=True,
    can_use=True,
    can_update=False,
    can_delete=False,
    can_manage=False,
)

EDIT = Permission(
    name="EDIT",
    can_read=True,
    can_use=True,
    can_update=True,
    can_delete=False,
    can_manage=False,
)

MANAGE = Permission(
    name="MANAGE",
    can_read=True,
    can_use=True,
    can_update=True,
    can_delete=True,
    can_manage=True,
)

NO_PERMISSIONS = Permission(
    name="NO_PERMISSIONS",
    can_read=False,
    can_use=False,
    can_update=False,
    can_delete=False,
    can_manage=False,
)

ALL_PERMISSIONS = {
    READ.name: READ,
    USE.name: USE,
    EDIT.name: EDIT,
    MANAGE.name: MANAGE,
    NO_PERMISSIONS.name: NO_PERMISSIONS,
}


def get_permission(permission: str) -> Permission:
    return ALL_PERMISSIONS[permission]


PERMISSION_PRIORITY = {
    NO_PERMISSIONS.name: 0,
    READ.name: 1,
    USE.name: 2,
    EDIT.name: 3,
    MANAGE.name: 4,
}

VALID_RESOURCE_TYPES = frozenset({
    "experiment",
    "registered_model",
    "scorer",
    "gateway_secret",
    "gateway_endpoint",
    "gateway_model_definition",
    # Special: workspace-wide permissions that apply to all resources in the role's
    # workspace. resource_pattern must be "*". permission=MANAGE additionally grants
    # role/user management capabilities within the workspace (workspace-admin).
    "workspace",
    # Special: a "default" permission on every resource of every type in the role's
    # workspace. resource_pattern must be "*". Used for the per-user workspace-wide
    # grant that replaces the legacy `workspace_permissions` table.
    #
    # Nominally distinct from `resource_type="workspace"`, but the workspace-admin
    # helpers (``is_workspace_admin`` / ``list_workspace_admin_workspaces``) treat
    # ``permission=MANAGE`` on ``resource_type="*"`` as admin-equivalent too, to
    # preserve the legacy semantic where a MANAGE entry in ``workspace_permissions``
    # conferred workspace-admin rights. Lower permission levels (READ/USE/EDIT) on
    # ``resource_type="*"`` are a bare resource-access grant and do *not* imply admin.
    "*",
})


# Permissions grantable at workspace scope (``resource_type='*'``). USE confers
# workspace access + resource creation + receipt of ``default_permission`` for new
# resources; MANAGE additionally grants role/user management within the workspace.
# READ/EDIT are intentionally excluded: we do not support a "see workspace but cannot
# create" tier, and EDIT collapses into USE for workspace-wide grants.
WORKSPACE_GRANTABLE_PERMISSIONS = frozenset({USE.name, MANAGE.name})

# Permissions grantable on a concrete resource (experiment, registered_model, scorer,
# gateway_*). NO_PERMISSIONS is intentionally excluded: an absent grant combined with
# the configured ``default_permission`` already expresses "no access"; an explicit
# NO_PERMISSIONS grant on a resource is no longer supported.
RESOURCE_GRANTABLE_PERMISSIONS = frozenset({READ.name, USE.name, EDIT.name, MANAGE.name})


def _validate_permission(permission: str):
    if permission not in ALL_PERMISSIONS:
        raise MlflowException(
            f"Invalid permission '{permission}'. Valid permissions are: {tuple(ALL_PERMISSIONS)}",
            INVALID_PARAMETER_VALUE,
        )


def _validate_resource_type(resource_type: str):
    if resource_type not in VALID_RESOURCE_TYPES:
        raise MlflowException(
            f"Invalid resource type '{resource_type}'. "
            f"Valid resource types are: {tuple(sorted(VALID_RESOURCE_TYPES))}",
            INVALID_PARAMETER_VALUE,
        )


def _validate_permission_for_resource_type(permission: str, resource_type: str) -> None:
    """Validate that ``permission`` is a legal grant on a row of ``resource_type``.

    - ``resource_type='workspace'`` accepts only ``MANAGE`` — the workspace-admin grant.
    - ``resource_type='*'`` accepts ``USE`` or ``MANAGE`` — a workspace-wide grant
      conferring access+create or full admin.
    - Concrete resource types accept any of ``READ`` / ``USE`` / ``EDIT`` / ``MANAGE``.
      ``NO_PERMISSIONS`` is rejected: an absent grant combined with the configured
      ``default_permission`` already expresses "no access".
    """
    _validate_permission(permission)
    _validate_resource_type(resource_type)
    if resource_type == "workspace":
        if permission != MANAGE.name:
            raise MlflowException(
                f"Invalid permission '{permission}' for resource_type='workspace'. "
                f"Workspace-admin grants accept only '{MANAGE.name}'.",
                INVALID_PARAMETER_VALUE,
            )
        return
    if resource_type == "*":
        if permission not in WORKSPACE_GRANTABLE_PERMISSIONS:
            raise MlflowException(
                f"Invalid permission '{permission}' for resource_type='*'. "
                f"Workspace-wide grants accept only: "
                f"{tuple(sorted(WORKSPACE_GRANTABLE_PERMISSIONS))}.",
                INVALID_PARAMETER_VALUE,
            )
        return
    if permission not in RESOURCE_GRANTABLE_PERMISSIONS:
        raise MlflowException(
            f"Invalid permission '{permission}' for resource_type='{resource_type}'. "
            f"Resource-level grants accept only: "
            f"{tuple(sorted(RESOURCE_GRANTABLE_PERMISSIONS))}.",
            INVALID_PARAMETER_VALUE,
        )


def max_permission(a: str, b: str) -> str:
    return a if PERMISSION_PRIORITY.get(a, 0) >= PERMISSION_PRIORITY.get(b, 0) else b
