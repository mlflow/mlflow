from dataclasses import dataclass

from mlflow import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


@dataclass
class Permission:
    name: str
    can_read: bool
    can_use: bool
    can_create: bool
    can_update: bool
    can_delete: bool
    can_manage: bool


READ = Permission(
    name="READ",
    can_read=True,
    can_use=False,
    can_create=False,
    can_update=False,
    can_delete=False,
    can_manage=False,
)

USE = Permission(
    name="USE",
    can_read=True,
    can_use=True,
    can_create=False,
    can_update=False,
    can_delete=False,
    can_manage=False,
)

# Strict superset of USE (adds can_create), strict subset of EDIT (lacks
# can_update). Pairs with creator-as-owner: a user with workspace-wide
# CONTRIBUTE can create new resources, automatically owns them via the
# AFTER_REQUEST_PATH_HANDLERS that grant the creator MANAGE on the new row,
# and cannot touch other users' resources because CONTRIBUTE.can_update is
# False.
CONTRIBUTE = Permission(
    name="CONTRIBUTE",
    can_read=True,
    can_use=True,
    can_create=True,
    can_update=False,
    can_delete=False,
    can_manage=False,
)

EDIT = Permission(
    name="EDIT",
    can_read=True,
    can_use=True,
    can_create=True,
    can_update=True,
    can_delete=False,
    can_manage=False,
)

MANAGE = Permission(
    name="MANAGE",
    can_read=True,
    can_use=True,
    can_create=True,
    can_update=True,
    can_delete=True,
    can_manage=True,
)

NO_PERMISSIONS = Permission(
    name="NO_PERMISSIONS",
    can_read=False,
    can_use=False,
    can_create=False,
    can_update=False,
    can_delete=False,
    can_manage=False,
)

ALL_PERMISSIONS = {
    READ.name: READ,
    USE.name: USE,
    CONTRIBUTE.name: CONTRIBUTE,
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
    CONTRIBUTE.name: 3,
    EDIT.name: 4,
    MANAGE.name: 5,
}

VALID_RESOURCE_TYPES = frozenset({
    "experiment",
    "registered_model",
    "scorer",
    "gateway_secret",
    "gateway_endpoint",
    "gateway_model_definition",
    # Special: workspace-wide permissions that apply to all resources in the role's workspace.
    # resource_pattern must be "*". permission=MANAGE additionally grants role/user
    # management capabilities within the workspace.
    "workspace",
})


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


def max_permission(a: str, b: str) -> str:
    return a if PERMISSION_PRIORITY.get(a, 0) >= PERMISSION_PRIORITY.get(b, 0) else b
