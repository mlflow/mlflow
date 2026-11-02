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
  (tier override), so a ``DENY`` is never rescued by a more permissive ancestor.
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
# it. Not one of the standard action verbs -- that catalogue has no veto -- so it is a local
# addition. It
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

    A fallback entry is either a bare ``(resource_type, resource_id)`` key, which inherits this
    requirement's own ``action``, or a ``Requirement`` naming its own ``action`` for that rung.
    The second form exists because a destructive operation on a sub-resource can require
    ``delete`` of a grant on the tier itself while a caller holding no tier grant inherits the
    weaker level the pre-existing check used -- two cases that differ by WHICH key governs, not
    by the permission it yields. A fallback ``Requirement`` may not itself carry
    ``fallback_if_no_grant``: a chain is declared in one place, flat and readable.

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
    resource_id: str | None  # None for create-in-workspace
    action: str  # read | use | update | delete | manage | create | not_denied
    # Each entry: a bare (type, id) key inheriting ``action``, or a Requirement with its own.
    fallback_if_no_grant: "tuple[tuple[str, str] | Requirement, ...]" = ()


def requirement_rungs(requirement: Requirement) -> "list[tuple[GrantLoadKey, str]]":
    """Each key that could decide ``requirement`` paired with the action it must satisfy.

    Own key first, then each fallback in order. A bare ``(type, id)`` fallback inherits
    ``requirement.action``; a ``Requirement`` fallback supplies its own.
    """
    rungs = [
        (
            GrantLoadKey(requirement.resource_type, requirement.resource_id or "*"),
            requirement.action,
        )
    ]
    for entry in requirement.fallback_if_no_grant:
        if isinstance(entry, Requirement):
            if entry.fallback_if_no_grant:
                raise ValueError(
                    f"Fallback requirement {entry.resource_type!r} may not declare its own "
                    "fallback_if_no_grant; list every rung on the outermost requirement."
                )
            rungs.append((
                GrantLoadKey(entry.resource_type, entry.resource_id or "*"),
                entry.action,
            ))
        else:
            resource_type, resource_id = entry
            rungs.append((GrantLoadKey(resource_type, resource_id), requirement.action))
    return rungs


def requirement_to_grant_load_keys(requirement: Requirement) -> list[GrantLoadKey]:
    """The keys that could decide ``requirement``: its own first, then each fallback."""
    return [key for key, _action in requirement_rungs(requirement)]


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
    """Which key's grant governs ``requirement`` -- the tier override."""
    return governing_permission_and_action(requirement, grants, default_permission, absent)[0]


def governing_permission_and_action(
    requirement: Requirement,
    grants: "dict[GrantLoadKey, Permission | None]",
    default_permission: str,
    absent: Permission,
) -> "tuple[Permission, str]":
    """The governing permission AND the action that rung requires.

    They are resolved together because a fallback rung may carry a different action from the
    requirement's own key, so the permission alone no longer determines what must be satisfied.
    """
    rungs = requirement_rungs(requirement)
    for key, action in rungs:
        grant = grants[key]
        if grant is not None:
            permission = (
                grant if grant.denied else floor_positive_permission(grant, default_permission)
            )
            return permission, action
    # No rung held a grant: the last rung's action is the inherited level, which is the one a
    # caller with no grant anywhere has to clear.
    return floor_positive_permission(absent, default_permission), rungs[-1][1]


def action_met(action: str, permission: Permission) -> bool:
    if action == ACTION_NOT_DENIED:
        return not permission.denied
    return bool(getattr(permission, _ACTION_CAPABILITY[action]))


def requirement_met(requirement: Requirement, permission: Permission) -> bool:
    return action_met(requirement.action, permission)


def requirement_satisfied(
    requirement: Requirement,
    grants: "dict[GrantLoadKey, Permission | None]",
    default_permission: str,
    absent: Permission,
) -> bool:
    """Whether ``requirement`` is satisfied, honoring a fallback rung's own action."""
    permission, action = governing_permission_and_action(
        requirement, grants, default_permission, absent
    )
    return action_met(action, permission)
