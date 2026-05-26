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

# Resource-type discriminants for `role_permissions.resource_type`. Use these
# constants instead of string literals so call sites are searchable and typo-safe.
#
# Concrete resource types — grants apply to a single named resource:
RESOURCE_TYPE_EXPERIMENT = "experiment"
RESOURCE_TYPE_REGISTERED_MODEL = "registered_model"
RESOURCE_TYPE_PROMPT = "prompt"
RESOURCE_TYPE_SCORER = "scorer"
RESOURCE_TYPE_GATEWAY_SECRET = "gateway_secret"
RESOURCE_TYPE_GATEWAY_ENDPOINT = "gateway_endpoint"
RESOURCE_TYPE_GATEWAY_MODEL_DEFINITION = "gateway_model_definition"

# Workspace-wide permissions slot. ``resource_pattern`` must be ``"*"``. The
# permission level distinguishes member from admin:
#
# - ``USE`` — workspace access + resource creation + receive ``default_permission``
#   for resources without an explicit grant.
# - ``MANAGE`` — everything ``USE`` grants, plus role/user administration within
#   the workspace (workspace-admin).
#
# ``READ`` and ``EDIT`` are intentionally excluded: pre-simplification "see
# workspace but cannot create" (READ) and "edit everything" (EDIT) are no longer
# expressible at workspace scope. The ``e5f6a7b8c9d0`` migration rewrites legacy
# ``READ`` rows to ``USE`` and fans ``EDIT`` rows out to per-type EDIT grants
# anchored on ``('workspace', '*', USE)``.
RESOURCE_TYPE_WORKSPACE = "workspace"

VALID_RESOURCE_TYPES = frozenset({
    RESOURCE_TYPE_EXPERIMENT,
    RESOURCE_TYPE_REGISTERED_MODEL,
    RESOURCE_TYPE_PROMPT,
    RESOURCE_TYPE_SCORER,
    RESOURCE_TYPE_GATEWAY_SECRET,
    RESOURCE_TYPE_GATEWAY_ENDPOINT,
    RESOURCE_TYPE_GATEWAY_MODEL_DEFINITION,
    RESOURCE_TYPE_WORKSPACE,
})


# Permissions grantable at workspace scope (``resource_type='workspace'``). USE
# is the regular member tier (access + create + receive ``default_permission``);
# MANAGE additionally grants role/user administration. READ/EDIT are intentionally
# excluded: see ``RESOURCE_TYPE_WORKSPACE`` docstring.
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

    - ``resource_type='workspace'`` accepts ``USE`` or ``MANAGE`` — the workspace-wide
      grant slot. ``USE`` is the regular member tier; ``MANAGE`` additionally grants
      role/user administration.
    - Concrete resource types accept any of ``READ`` / ``USE`` / ``EDIT`` / ``MANAGE``.
      ``NO_PERMISSIONS`` is rejected: an absent grant combined with the configured
      ``default_permission`` already expresses "no access".
    """
    _validate_permission(permission)
    _validate_resource_type(resource_type)
    if resource_type == RESOURCE_TYPE_WORKSPACE:
        if permission not in WORKSPACE_GRANTABLE_PERMISSIONS:
            raise MlflowException(
                f"Invalid permission '{permission}' for resource_type='{RESOURCE_TYPE_WORKSPACE}'. "
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
