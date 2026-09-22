"""What an operation requires, and how a set of requirements is resolved.

``permissions`` defines what a grant IS; this module defines what an operation NEEDS and
how grants answer it. Everything here is pure -- it takes grant rows and returns
permissions -- so the precedence rules, which are the security-critical part, are testable
without a store, a session, or a request.

The impure half lives in ``mlflow.server.auth``: loading the rows, resolving the anchor's
workspace, and deciding what an absent grant means all need the server's wiring.

Two folds, and they are different operations:

* **within a key** (:func:`fold_grants_for_key`) -- a caller can hold several roles, so
  several rows can match one key. ``DENY`` among them wins; otherwise the highest.
* **across keys** (:func:`governing_permission`) -- a requirement's own key, then each
  fallback level. The first key holding ANY grant decides and the rest are not consulted
  (RFC 0000 tier override), so a ``DENY`` is never rescued by a more permissive ancestor.
"""

from typing import TYPE_CHECKING, NamedTuple

from mlflow.server.auth.permissions import (
    DENY,
    MANAGE,
    NO_PERMISSIONS,
    RESOURCE_TYPE_WORKSPACE,
    GrantLoadKey,
    Permission,
    get_permission,
    matches,
    max_permission,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mlflow.server.auth.sqlalchemy_store import RoleGrantRow

# A veto-only requirement: the key must not be denied, and nothing positive is required of
# it. Not an RFC 0008 verb -- that catalogue has no veto -- so it is a local addition. It
# cannot appear in a grant: grants carry permission LEVELS (uppercase), and this is an
# action.
#
# It exists because the fold is tier override, so a present grant overrides an inherited one
# downward as well as upward. A positive requirement on a key the pre-existing check never
# consulted would therefore deny a caller who merely holds a narrower grant there.
ACTION_NOT_DENIED = "not_denied"

# ``create`` has no ``can_create``: it gates on the workspace grant's ``can_use``.
_ACTION_CAPABILITY = {
    "read": "can_read",
    "use": "can_use",
    "update": "can_update",
    "delete": "can_delete",
    "manage": "can_manage",
    "create": "can_use",
}


class Requirement(NamedTuple):
    """One thing an operation requires: ``action`` on a resource.

    ``fallback_if_no_grant`` lists keys to consult, in order, ONLY when the caller holds no
    grant on this resource's own type -- inheritance declared per operation rather than in a
    global parent map.

    Two shapes are used in practice:

    * an operation on an EXISTING sub-resource states one requirement whose chain ends where
      the pre-existing check looked, so an absent child grant inherits and a present one can
      escalate;
    * a CREATE states a positive requirement on the container plus an ``ACTION_NOT_DENIED``
      requirement on the type being created. A create cannot be authorized by a grant on the
      resource itself (it does not exist), and sub-resource grants are wildcard-only grain,
      so a positive requirement there would let one grant confer creation in every container
      in the workspace.
    """

    resource_type: str
    resource_id: str | None  # None for create-in-workspace (RFC 0008 convention)
    action: str  # read | use | update | delete | manage | create | not_denied
    fallback_if_no_grant: "tuple[tuple[str, str], ...]" = ()


def requirement_to_grant_load_keys(requirement: Requirement) -> list[GrantLoadKey]:
    """The keys that could decide ``requirement``: its own first, then each fallback.

    Keys are flat addresses, never (child, parent) pairs, so each is answered independently
    and ``None`` means exactly "no grant on this type" -- the distinction a pre-folded view
    destroys by substituting ``default_permission`` for an absent grant.
    """
    return [
        GrantLoadKey(requirement.resource_type, requirement.resource_id or "*"),
        *(GrantLoadKey(t, i) for t, i in requirement.fallback_if_no_grant),
    ]


def requirements_to_grant_load_keys(
    requirements: "Sequence[Requirement]",
) -> list[GrantLoadKey]:
    """Every key that could decide any requirement, deduplicated -> ONE query."""
    keys: list[GrantLoadKey] = []
    for requirement in requirements:
        keys.extend(requirement_to_grant_load_keys(requirement))
    return list(dict.fromkeys(keys))


def is_workspace_admin_grant(grant: "RoleGrantRow") -> bool:
    return (
        grant.resource_type == RESOURCE_TYPE_WORKSPACE
        and grant.resource_pattern == "*"
        and grant.permission == MANAGE.name
    )


def fold_grants_for_key(grants: "Sequence[RoleGrantRow]", key: GrantLoadKey) -> Permission | None:
    # None means SILENT, not "no access": only the fold across keys knows whether a
    # fallback key or the default should speak next.
    denied = False
    best: str | None = None
    for grant in grants:
        if grant.resource_type != key.resource_type:
            continue
        if not matches(grant.resource_pattern, key.resource_type, key.resource_id):
            continue
        if grant.permission == DENY.name:
            denied = True
        else:
            best = grant.permission if best is None else max_permission(best, grant.permission)
    if denied:
        return DENY
    return get_permission(best) if best is not None else None


def floor_positive_permission(perm: Permission, default_permission: str) -> Permission:
    # A positive grant never resolves below default_permission. DENY and the legacy
    # NO_PERMISSIONS sentinel are exempt: flooring a veto to the default would void it.
    if perm.name in (NO_PERMISSIONS.name, DENY.name):
        return perm
    return get_permission(max_permission(perm.name, default_permission))


def governing_permission(
    requirement: Requirement,
    grants: "dict[GrantLoadKey, Permission | None]",
    default_permission: str,
    absent: Permission,
) -> Permission:
    """Which key's grant governs ``requirement`` -- RFC 0000's tier override.

    The first key holding ANY grant decides; the keys behind it are not consulted ("not a
    cross-tier max"). So a ``DENY`` is never rescued by a more permissive ancestor, and
    equally a narrower positive grant overrides a broader one.

    ``absent`` is what no grant anywhere resolves to, which the caller supplies because it
    depends on the workspace. No capability comparison happens here, so a route whose
    decision is not a plain conjunction can apply its own logic to the result.
    """
    for key in requirement_to_grant_load_keys(requirement):
        grant = grants[key]
        if grant is not None:
            return grant if grant.denied else floor_positive_permission(grant, default_permission)
    return floor_positive_permission(absent, default_permission)


def requirement_met(requirement: Requirement, permission: Permission) -> bool:
    if requirement.action == ACTION_NOT_DENIED:
        return not permission.denied
    return bool(getattr(permission, _ACTION_CAPABILITY[requirement.action]))
