from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

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

    @property
    def denied(self) -> bool:
        """An explicit ``DENY``, distinct from merely lacking a capability.

        A low positive grant and the legacy ``NO_PERMISSIONS`` sentinel also lack
        capabilities, but neither is an operator's opt-in veto.
        """
        return self.name == DENY.name


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

DENY = Permission(
    name="DENY",
    can_read=False,
    can_use=False,
    can_update=False,
    can_delete=False,
    can_manage=False,
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
    DENY.name: DENY,
}


def get_permission(permission: str) -> Permission:
    return ALL_PERMISSIONS[permission]


PERMISSION_PRIORITY = {
    # DENY is negative so it can never win a ``max`` fold. The fold is expected to
    # short-circuit on it first; the ordering is a second line of defence.
    DENY.name: -1,
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
RESOURCE_TYPE_MCP_SERVER = "mcp_server"

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
# Sub-resources (RFC 0000). Each is independently grantable because it has a use case
# for a permission that differs from its parent's; all are wildcard-only grain (see TYPE).
RESOURCE_TYPE_RUN = "run"
RESOURCE_TYPE_TRACE = "trace"
RESOURCE_TYPE_ASSESSMENT = "assessment"
RESOURCE_TYPE_LOGGED_MODEL = "logged_model"
RESOURCE_TYPE_REVIEW_QUEUE = "review_queue"
RESOURCE_TYPE_REGISTERED_MODEL_VERSION = "registered_model_version"
RESOURCE_TYPE_PROMPT_VERSION = "prompt_version"
RESOURCE_TYPE_SCORER_VERSION = "scorer_version"
RESOURCE_TYPE_MCP_SERVER_VERSION = "mcp_server_version"

RESOURCE_TYPE_WORKSPACE = "workspace"


class PatternKind(Enum):
    """The grain a grant's ``resource_pattern`` may take for a resource type."""

    WILDCARD = auto()  # "*" — any resource of the type
    ID = auto()  # an exact resource id


WILDCARD_AND_ID = frozenset({PatternKind.WILDCARD, PatternKind.ID})
WILDCARD_ONLY = frozenset({PatternKind.WILDCARD})

# Resource type -> the grain its grants may use. Top-level types keep per-id grants
# (today's behaviour); sub-resources are wildcard-only, because a per-id child grant
# cannot be enforced in list/search paths until filter push-down lands, and a grant that
# holds on a point route but not in search is worse than no grant at all.
TYPE: dict[str, frozenset[PatternKind]] = {
    # The workspace slot is a single wildcard grant per workspace ("am I a member /
    # an admin here?"), never per-id: the workspace itself is named by the role's
    # ``workspace`` column, not by the pattern.
    RESOURCE_TYPE_WORKSPACE: WILDCARD_ONLY,
    RESOURCE_TYPE_EXPERIMENT: WILDCARD_AND_ID,
    RESOURCE_TYPE_REGISTERED_MODEL: WILDCARD_AND_ID,
    RESOURCE_TYPE_PROMPT: WILDCARD_AND_ID,
    RESOURCE_TYPE_SCORER: WILDCARD_AND_ID,
    RESOURCE_TYPE_GATEWAY_SECRET: WILDCARD_AND_ID,
    RESOURCE_TYPE_GATEWAY_ENDPOINT: WILDCARD_AND_ID,
    RESOURCE_TYPE_GATEWAY_MODEL_DEFINITION: WILDCARD_AND_ID,
    RESOURCE_TYPE_MCP_SERVER: WILDCARD_AND_ID,
    RESOURCE_TYPE_RUN: WILDCARD_ONLY,
    RESOURCE_TYPE_TRACE: WILDCARD_ONLY,
    RESOURCE_TYPE_ASSESSMENT: WILDCARD_ONLY,
    RESOURCE_TYPE_LOGGED_MODEL: WILDCARD_ONLY,
    RESOURCE_TYPE_REVIEW_QUEUE: WILDCARD_ONLY,
    RESOURCE_TYPE_REGISTERED_MODEL_VERSION: WILDCARD_ONLY,
    RESOURCE_TYPE_PROMPT_VERSION: WILDCARD_ONLY,
    RESOURCE_TYPE_SCORER_VERSION: WILDCARD_ONLY,
    RESOURCE_TYPE_MCP_SERVER_VERSION: WILDCARD_ONLY,
}

# Derived, so the set of valid types is declared exactly once.
VALID_RESOURCE_TYPES = frozenset(TYPE.keys())


class GrantLoadKey(NamedTuple):
    """The address a grant lookup is made at: a resource type and an id (or ``"*"``).

    Deliberately flat. It carries no parent, because a resource's relationships are
    declared per operation on the requirement that needs them, not globally.
    """

    resource_type: str
    resource_id: str


# Permissions grantable at workspace scope (``resource_type='workspace'``). USE
# is the regular member tier (access + create + receive ``default_permission``);
# MANAGE additionally grants role/user administration. READ/EDIT are intentionally
# excluded: see ``RESOURCE_TYPE_WORKSPACE`` docstring.
WORKSPACE_GRANTABLE_PERMISSIONS = frozenset({USE.name, MANAGE.name})

# Permissions grantable on a concrete resource (experiment, registered_model, scorer,
# gateway_*). NO_PERMISSIONS is intentionally excluded: an absent grant combined with
# the configured ``default_permission`` already expresses "no access"; an explicit
# NO_PERMISSIONS grant on a resource is no longer supported.
# ``DENY`` is the sub-resource RFC's absolute deny: it denies within its own tier even
# where an inherited tier would allow. ``NO_PERMISSIONS`` stays excluded — an absent grant
# plus ``default_permission`` already expresses "no access".
RESOURCE_GRANTABLE_PERMISSIONS = frozenset({
    READ.name,
    USE.name,
    EDIT.name,
    MANAGE.name,
    DENY.name,
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


def matches(resource_pattern: str, resource_type: str, resource_id: str | None) -> bool:
    """Does a grant's ``resource_pattern`` on ``resource_type`` apply to ``resource_id``?

    Honours the grain the type declares: a wildcard-only sub-resource matches only ``"*"``,
    a top-level type matches ``"*"`` or its exact id. An unknown type matches nothing.
    """
    patterns = TYPE.get(resource_type)
    if patterns is None:
        return False
    return (PatternKind.WILDCARD in patterns and resource_pattern == "*") or (
        PatternKind.ID in patterns and resource_pattern == resource_id
    )


def _validate_resource_pattern(resource_pattern: str, resource_type: str) -> None:
    """Reject a grant whose pattern is a grain the type does not declare.

    This is what keeps a per-id sub-resource grant (e.g. ``(run, <run_id>, EDIT)``) from
    being written at all, rather than being written and then silently ignored by the fold.
    """
    _validate_resource_type(resource_type)
    if resource_pattern == "*":
        return
    if PatternKind.ID not in TYPE[resource_type]:
        raise MlflowException(
            f"Invalid resource_pattern '{resource_pattern}' for "
            f"resource_type='{resource_type}'. This resource type supports only "
            f"wildcard ('*') grants.",
            INVALID_PARAMETER_VALUE,
        )


def max_permission(a: str, b: str) -> str:
    return a if PERMISSION_PRIORITY.get(a, 0) >= PERMISSION_PRIORITY.get(b, 0) else b
