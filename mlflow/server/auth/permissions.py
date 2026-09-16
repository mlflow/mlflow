from dataclasses import dataclass
from enum import Enum, auto

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

# DENY is the sub-resource permissions RFC's absolute-deny grant level. It is
# distinct from NO_PERMISSIONS: NO_PERMISSIONS remains the inert, ungrantable
# sentinel used for "no presence / absent grant" (preserving backward compatibility),
# whereas DENY is a first-class *grantable* level that denies its resource even where
# a parent grant would allow. DENY is evaluated ahead of the ``max`` fold within its
# tier; the workspace-admin bypass still takes precedence over it. Like
# NO_PERMISSIONS, every ``can_*`` is False, so a validator reading ``.can_update``
# denies exactly as an absent grant would.
DENY = Permission(
    name="DENY",
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
RESOURCE_TYPE_WORKSPACE = "workspace"

# Sub-resource (child) types made independently grantable by the sub-resource
# permissions RFC. A child is resolved on its own tier when the caller holds any
# grant on the child type; otherwise it falls back to its parent (see
# ``get_role_permission_for_resource``). All child grants are wildcard-grain only
# in this RFC (per-id grain is deferred pending search-filter push-down).
#
# experiment children:
RESOURCE_TYPE_RUN = "run"
RESOURCE_TYPE_TRACE = "trace"
RESOURCE_TYPE_ASSESSMENT = "assessment"
RESOURCE_TYPE_LOGGED_MODEL = "logged_model"
RESOURCE_TYPE_REVIEW_QUEUE = "review_queue"
# version children (mirror registered_model_version):
RESOURCE_TYPE_REGISTERED_MODEL_VERSION = "registered_model_version"
RESOURCE_TYPE_PROMPT_VERSION = "prompt_version"
RESOURCE_TYPE_SCORER_VERSION = "scorer_version"
RESOURCE_TYPE_MCP_SERVER_VERSION = "mcp_server_version"


class PatternKind(Enum):
    """The grain a ``resource_pattern`` may take for a resource type."""

    WILDCARD = auto()  # "*" — any resource of the type
    ID = auto()  # an exact resource id
    # REGEX = auto()   # future — not in this RFC


# Parents may be granted at wildcard or per-id grain (today's behavior). New
# children are wildcard-only until request-level search-filter push-down lands.
WILDCARD_AND_ID = frozenset({PatternKind.WILDCARD, PatternKind.ID})
WILDCARD_ONLY = frozenset({PatternKind.WILDCARD})

# Resource-type registry: name -> allowed grain. Replaces the bare
# ``VALID_RESOURCE_TYPES`` frozenset; ``VALID_RESOURCE_TYPES`` is derived from
# ``TYPE.keys()`` so the valid-types set is defined once, not maintained twice.
TYPE: dict[str, frozenset[str]] = {
    # Top-level (unchanged grain):
    RESOURCE_TYPE_WORKSPACE: WILDCARD_AND_ID,
    RESOURCE_TYPE_EXPERIMENT: WILDCARD_AND_ID,
    RESOURCE_TYPE_REGISTERED_MODEL: WILDCARD_AND_ID,
    RESOURCE_TYPE_PROMPT: WILDCARD_AND_ID,
    RESOURCE_TYPE_SCORER: WILDCARD_AND_ID,
    RESOURCE_TYPE_GATEWAY_SECRET: WILDCARD_AND_ID,
    RESOURCE_TYPE_GATEWAY_ENDPOINT: WILDCARD_AND_ID,
    RESOURCE_TYPE_GATEWAY_MODEL_DEFINITION: WILDCARD_AND_ID,
    RESOURCE_TYPE_MCP_SERVER: WILDCARD_AND_ID,
    # Sub-resources added by this RFC (wildcard-only grain):
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

VALID_RESOURCE_TYPES = frozenset(TYPE.keys())


# Permissions grantable at workspace scope (``resource_type='workspace'``). USE
# is the regular member tier (access + create + receive ``default_permission``);
# MANAGE additionally grants role/user administration. READ/EDIT are intentionally
# excluded: see ``RESOURCE_TYPE_WORKSPACE`` docstring.
WORKSPACE_GRANTABLE_PERMISSIONS = frozenset({USE.name, MANAGE.name})

# Permissions grantable on a concrete resource (experiment, registered_model, scorer,
# gateway_*, and the sub-resource child types). ``NO_PERMISSIONS`` remains excluded (an
# absent grant plus the configured ``default_permission`` already expresses "no access").
# ``DENY`` is the sub-resource RFC's absolute-deny level: a ``(child, *, DENY)`` grant
# denies the child even where the parent would allow (restriction). It is evaluated ahead
# of the ``max`` fold within its tier; the workspace-admin bypass still takes precedence.
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


def matches(resource_pattern: str, resource_type: str, resource_id: str | None) -> bool:
    """Grain-aware match key: does a grant's ``resource_pattern`` on ``resource_type``
    apply to ``resource_id``?

    Honors the grain the type declares in ``TYPE``: a wildcard-only child matches only
    ``"*"``; a wildcard-and-id parent matches ``"*"`` or the exact id. A type not in
    ``TYPE`` matches nothing (defensive; grant validation rejects such types earlier).
    """
    patterns = TYPE.get(resource_type)
    if patterns is None:
        return False
    return (PatternKind.WILDCARD in patterns and resource_pattern == "*") or (
        PatternKind.ID in patterns and resource_pattern == resource_id
    )


def _validate_resource_pattern(resource_pattern: str, resource_type: str) -> None:
    """Validate that ``resource_pattern`` is a grain the type declares.

    Wildcard (`"*"`) is always allowed. A concrete id is allowed only for types whose
    grain includes ``PatternKind.ID`` (parents today). This rejects a concrete-id child
    grant (e.g. ``(run, <run_id>, ...)``) at the source, enforcing the RFC's
    wildcard-only child grain rather than silently dropping it in the fold.
    """
    _validate_resource_type(resource_type)
    patterns = TYPE[resource_type]
    if resource_pattern == "*":
        return
    if PatternKind.ID not in patterns:
        raise MlflowException(
            f"Invalid resource_pattern '{resource_pattern}' for "
            f"resource_type='{resource_type}'. This resource type only supports "
            f"wildcard ('*') grants; per-id grants are not supported.",
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
