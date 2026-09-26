# Unit tests for the requirement model's two folds. These are pure: a grant row is a plain tuple, so
# both folds are exercised without a store, a session, or a Flask request. That is the point of
# resolving permissions outside the store -- the precedence rules are the security-critical part and
# they should be testable directly.

import pytest

from mlflow.server.auth.permissions import (
    DENY,
    EDIT,
    MANAGE,
    READ,
    RESOURCE_TYPE_EXPERIMENT,
    RESOURCE_TYPE_GATEWAY_ENDPOINT,
    RESOURCE_TYPE_RUN,
    RESOURCE_TYPE_SCORER,
    RESOURCE_TYPE_SCORER_VERSION,
    RESOURCE_TYPE_TRACE,
    RESOURCE_TYPE_WORKSPACE,
    USE,
    GrantLoadKey,
)
from mlflow.server.auth.requirements import (
    ACTION_NOT_DENIED,
    Requirement,
    fold_grants_for_key,
    governing_permission,
    is_workspace_admin_grant,
    requirement_met,
    requirement_to_grant_load_keys,
    requirements_to_grant_load_keys,
)
from mlflow.server.auth.sqlalchemy_store import RoleGrantRow

EXPERIMENT_ID = "5"
SCORER_KEY = "5/my_scorer"


# The fold takes ``default_permission`` and ``absent`` as arguments, so these tests need no
# config patching: what an absent grant means is the caller's decision, not the fold's.
DEFAULT_PERMISSION = READ.name
ABSENT = READ


def grant(resource_type, pattern, permission):
    return RoleGrantRow(resource_type, pattern, permission)


def decide(requirements, rows):
    """Resolve requirements against grant rows, as ``_authorize`` will."""
    keys = requirements_to_grant_load_keys(requirements)
    if any(is_workspace_admin_grant(row) for row in rows):
        permissions = dict.fromkeys(keys, MANAGE)
    else:
        permissions = {key: fold_grants_for_key(rows, key) for key in keys}
    return all(
        requirement_met(
            requirement,
            governing_permission(requirement, permissions, DEFAULT_PERMISSION, ABSENT),
        )
        for requirement in requirements
    )


# --------------------------------------------------------------- load keys


def test_keys_are_own_type_then_each_fallback_in_order():
    requirement = Requirement(
        RESOURCE_TYPE_SCORER_VERSION,
        "*",
        "update",
        fallback_if_no_grant=((RESOURCE_TYPE_SCORER, SCORER_KEY), (RESOURCE_TYPE_EXPERIMENT, "5")),
    )
    assert requirement_to_grant_load_keys(requirement) == [
        GrantLoadKey(RESOURCE_TYPE_SCORER_VERSION, "*"),
        GrantLoadKey(RESOURCE_TYPE_SCORER, SCORER_KEY),
        GrantLoadKey(RESOURCE_TYPE_EXPERIMENT, "5"),
    ]


def test_action_is_not_part_of_the_key():
    # A veto and a positive requirement on the same resource load once, not twice.
    veto = Requirement(RESOURCE_TYPE_RUN, "*", ACTION_NOT_DENIED)
    positive = Requirement(RESOURCE_TYPE_RUN, "*", "update")
    assert requirement_to_grant_load_keys(veto) == requirement_to_grant_load_keys(positive)
    assert requirements_to_grant_load_keys([veto, positive]) == [
        GrantLoadKey(RESOURCE_TYPE_RUN, "*")
    ]


def test_keys_are_deduplicated_across_requirements_sharing_a_fallback():
    fallback = ((RESOURCE_TYPE_EXPERIMENT, "5"),)
    keys = requirements_to_grant_load_keys([
        Requirement(RESOURCE_TYPE_RUN, "*", "update", fallback),
        Requirement(RESOURCE_TYPE_SCORER_VERSION, "*", "update", fallback),
    ])
    assert keys.count(GrantLoadKey(RESOURCE_TYPE_EXPERIMENT, "5")) == 1
    assert len(keys) == 3


# ----------------------------------------------------- the fold WITHIN a key


def test_within_a_key_the_highest_of_several_role_grants_wins():
    rows = [
        grant(RESOURCE_TYPE_EXPERIMENT, "5", READ.name),
        grant(RESOURCE_TYPE_EXPERIMENT, "*", EDIT.name),
    ]
    assert fold_grants_for_key(rows, GrantLoadKey(RESOURCE_TYPE_EXPERIMENT, "5")) == EDIT


def test_within_a_key_deny_beats_the_highest_positive():
    rows = [
        grant(RESOURCE_TYPE_EXPERIMENT, "5", MANAGE.name),
        grant(RESOURCE_TYPE_EXPERIMENT, "5", DENY.name),
    ]
    assert fold_grants_for_key(rows, GrantLoadKey(RESOURCE_TYPE_EXPERIMENT, "5")) == DENY


def test_a_grant_on_a_different_resource_of_the_same_type_does_not_fold_in():
    rows = [grant(RESOURCE_TYPE_EXPERIMENT, "7", MANAGE.name)]
    assert fold_grants_for_key(rows, GrantLoadKey(RESOURCE_TYPE_EXPERIMENT, "5")) is None


def test_a_per_id_grant_never_matches_a_wildcard_only_type():
    # Grant validation rejects these at the source; the fold must ignore them too.
    rows = [grant(RESOURCE_TYPE_RUN, "some-run-id", MANAGE.name)]
    assert fold_grants_for_key(rows, GrantLoadKey(RESOURCE_TYPE_RUN, "*")) is None


def test_silence_is_distinguishable_from_no_access():
    """``None`` must not be collapsed into a permission: only the across-keys fold knows
    whether a fallback key may still speak.
    """
    assert fold_grants_for_key([], GrantLoadKey(RESOURCE_TYPE_RUN, "*")) is None


# ----------------------------------------------------- the fold ACROSS keys


@pytest.mark.parametrize(
    ("label", "rows", "expected"),
    [
        (
            "absent child grant inherits the parent",
            [grant(RESOURCE_TYPE_EXPERIMENT, "5", EDIT.name)],
            True,
        ),
        (
            "parent below the action denies",
            [grant(RESOURCE_TYPE_EXPERIMENT, "5", READ.name)],
            False,
        ),
        ("a positive child grant escalates", [grant(RESOURCE_TYPE_RUN, "*", EDIT.name)], True),
        (
            "a lower child grant overrides DOWNWARD",
            [
                grant(RESOURCE_TYPE_RUN, "*", READ.name),
                grant(RESOURCE_TYPE_EXPERIMENT, "5", EDIT.name),
            ],
            False,
        ),
        (
            "a child DENY is never rescued by the parent",
            [
                grant(RESOURCE_TYPE_RUN, "*", DENY.name),
                grant(RESOURCE_TYPE_EXPERIMENT, "5", MANAGE.name),
            ],
            False,
        ),
        ("no grant anywhere falls back to default_permission", [], False),
    ],
)
def test_tier_override_across_keys(label, rows, expected):
    requirement = Requirement(
        RESOURCE_TYPE_RUN, "*", "update", ((RESOURCE_TYPE_EXPERIMENT, EXPERIMENT_ID),)
    )
    assert decide([requirement], rows) is expected, label


def test_a_parent_deny_does_not_override_a_present_child_grant():
    """RFC 0000: a parent NONE denies a child only via fallback, never over a present
    child grant.
    """
    requirement = Requirement(
        RESOURCE_TYPE_SCORER_VERSION,
        "*",
        "update",
        ((RESOURCE_TYPE_SCORER, SCORER_KEY), (RESOURCE_TYPE_EXPERIMENT, EXPERIMENT_ID)),
    )
    parent_deny = grant(RESOURCE_TYPE_SCORER, SCORER_KEY, DENY.name)
    experiment = grant(RESOURCE_TYPE_EXPERIMENT, EXPERIMENT_ID, EDIT.name)

    assert decide([requirement], [parent_deny, experiment]) is False
    assert (
        decide(
            [requirement],
            [parent_deny, experiment, grant(RESOURCE_TYPE_SCORER_VERSION, "*", EDIT.name)],
        )
        is True
    )


# ------------------------------------------------------------------- vetoes


@pytest.mark.parametrize("level", [READ, USE, EDIT, MANAGE])
def test_a_veto_never_adds_a_positive_requirement(level):
    """Any positive level on a vetoed key passes: the veto reads only ``denied``. This is
    what keeps a composite route from denying callers the pre-existing check allowed.
    """
    requirements = [
        Requirement(RESOURCE_TYPE_EXPERIMENT, EXPERIMENT_ID, "update"),
        Requirement(RESOURCE_TYPE_SCORER, SCORER_KEY, ACTION_NOT_DENIED),
    ]
    rows = [
        grant(RESOURCE_TYPE_EXPERIMENT, EXPERIMENT_ID, EDIT.name),
        grant(RESOURCE_TYPE_SCORER, SCORER_KEY, level.name),
    ]
    assert decide(requirements, rows) is True


def test_a_veto_blocks_only_on_deny():
    requirements = [
        Requirement(RESOURCE_TYPE_EXPERIMENT, EXPERIMENT_ID, "update"),
        Requirement(RESOURCE_TYPE_SCORER, SCORER_KEY, ACTION_NOT_DENIED),
    ]
    rows = [
        grant(RESOURCE_TYPE_EXPERIMENT, EXPERIMENT_ID, EDIT.name),
        grant(RESOURCE_TYPE_SCORER, SCORER_KEY, DENY.name),
    ]
    assert decide(requirements, rows) is False


def test_a_veto_requires_no_grant_to_be_present():
    requirements = [
        Requirement(RESOURCE_TYPE_EXPERIMENT, EXPERIMENT_ID, "update"),
        Requirement(RESOURCE_TYPE_RUN, "*", ACTION_NOT_DENIED),
    ]
    rows = [grant(RESOURCE_TYPE_EXPERIMENT, EXPERIMENT_ID, EDIT.name)]
    assert decide(requirements, rows) is True


# -------------------------------------------------------------------- create


def test_create_gates_on_the_workspace_grant():
    requirement = Requirement(
        RESOURCE_TYPE_EXPERIMENT, "*", "create", ((RESOURCE_TYPE_WORKSPACE, "*"),)
    )
    assert decide([requirement], [grant(RESOURCE_TYPE_WORKSPACE, "*", USE.name)]) is True
    assert decide([requirement], []) is False


def test_a_resource_specific_grant_confers_no_create_right():
    requirement = Requirement(
        RESOURCE_TYPE_EXPERIMENT, "*", "create", ((RESOURCE_TYPE_WORKSPACE, "*"),)
    )
    assert decide([requirement], [grant(RESOURCE_TYPE_EXPERIMENT, "5", MANAGE.name)]) is False


def test_a_type_level_deny_vetoes_create():
    requirement = Requirement(
        RESOURCE_TYPE_EXPERIMENT, "*", "create", ((RESOURCE_TYPE_WORKSPACE, "*"),)
    )
    rows = [
        grant(RESOURCE_TYPE_WORKSPACE, "*", USE.name),
        grant(RESOURCE_TYPE_EXPERIMENT, "*", DENY.name),
    ]
    assert decide([requirement], rows) is False


# ----------------------------------------------------------- admin bypass


def test_workspace_admin_is_not_restrictable_by_a_deny():
    requirements = [
        Requirement(RESOURCE_TYPE_EXPERIMENT, EXPERIMENT_ID, "update"),
        Requirement(RESOURCE_TYPE_RUN, "*", ACTION_NOT_DENIED),
    ]
    rows = [
        grant(RESOURCE_TYPE_WORKSPACE, "*", MANAGE.name),
        grant(RESOURCE_TYPE_RUN, "*", DENY.name),
    ]
    assert decide(requirements, rows) is True


def test_workspace_use_alone_confers_no_resource_access():
    # Pre-RFC behaviour: workspace USE is membership, not resource access.
    rows = [grant(RESOURCE_TYPE_WORKSPACE, "*", USE.name)]
    assert not any(is_workspace_admin_grant(row) for row in rows)
    assert fold_grants_for_key(rows, GrantLoadKey(RESOURCE_TYPE_EXPERIMENT, "5")) is None


# ------------------------------------------------------------------- floor


def test_a_positive_grant_never_resolves_below_the_default():
    requirement = Requirement(RESOURCE_TYPE_EXPERIMENT, EXPERIMENT_ID, "update")
    permissions = {GrantLoadKey(RESOURCE_TYPE_EXPERIMENT, EXPERIMENT_ID): READ}
    governing = governing_permission(requirement, permissions, EDIT.name, EDIT)
    assert requirement_met(requirement, governing) is True


def test_a_deny_is_not_floored_up_to_the_default():
    requirement = Requirement(RESOURCE_TYPE_EXPERIMENT, EXPERIMENT_ID, "update")
    permissions = {GrantLoadKey(RESOURCE_TYPE_EXPERIMENT, EXPERIMENT_ID): DENY}
    governing = governing_permission(requirement, permissions, MANAGE.name, MANAGE)
    assert requirement_met(requirement, governing) is False


# ---------------------------------------------------------------------------
# Composite routes: one operation naming several resources
# ---------------------------------------------------------------------------


def _prompt_optimization_requirements(experiment_id, scorer_pattern, endpoint=None):
    """The shape validate_can_create_prompt_optimization_job builds."""
    requirements = [
        Requirement(RESOURCE_TYPE_EXPERIMENT, experiment_id, "update"),
        Requirement(RESOURCE_TYPE_RUN, "*", ACTION_NOT_DENIED),
        Requirement(RESOURCE_TYPE_SCORER_VERSION, "*", ACTION_NOT_DENIED),
        Requirement(RESOURCE_TYPE_SCORER, scorer_pattern, ACTION_NOT_DENIED),
    ]
    if endpoint is not None:
        requirements.append(
            Requirement(RESOURCE_TYPE_GATEWAY_ENDPOINT, endpoint, ACTION_NOT_DENIED)
        )
    return requirements


SCORER_PATTERN = f"{EXPERIMENT_ID}/myscorer"


def test_prompt_optimization_needs_only_the_experiment_positively():
    # The worker runs with no caller identity, so every named resource is checked at submit
    # time -- but only the experiment is a positive requirement, exactly as before.
    requirements = _prompt_optimization_requirements(EXPERIMENT_ID, SCORER_PATTERN)
    rows = [grant(RESOURCE_TYPE_EXPERIMENT, "*", EDIT.name)]
    assert decide(requirements, rows) is True


@pytest.mark.parametrize(
    ("denied_type", "denied_pattern"),
    [
        (RESOURCE_TYPE_RUN, "*"),
        (RESOURCE_TYPE_SCORER_VERSION, "*"),
        (RESOURCE_TYPE_SCORER, SCORER_PATTERN),
    ],
)
def test_prompt_optimization_is_vetoed_by_any_named_resource(denied_type, denied_pattern):
    requirements = _prompt_optimization_requirements(EXPERIMENT_ID, SCORER_PATTERN)
    rows = [
        grant(RESOURCE_TYPE_EXPERIMENT, "*", EDIT.name),
        grant(denied_type, denied_pattern, DENY.name),
    ]
    assert decide(requirements, rows) is False


def test_prompt_optimization_ignores_a_deny_on_a_scorer_it_does_not_name():
    requirements = _prompt_optimization_requirements(EXPERIMENT_ID, SCORER_PATTERN)
    rows = [
        grant(RESOURCE_TYPE_EXPERIMENT, "*", EDIT.name),
        grant(RESOURCE_TYPE_SCORER, f"{EXPERIMENT_ID}/other", DENY.name),
    ]
    assert decide(requirements, rows) is True


def test_prompt_optimization_gates_the_gateway_endpoint_the_config_names():
    # reflection_model reaches the worker verbatim and a "gateway:/" URI dispatches to that
    # endpoint, so caller-authored config cannot reach an endpoint the operator denied.
    requirements = _prompt_optimization_requirements(EXPERIMENT_ID, SCORER_PATTERN, "ep1")
    rows = [
        grant(RESOURCE_TYPE_EXPERIMENT, "*", EDIT.name),
        grant(RESOURCE_TYPE_GATEWAY_ENDPOINT, "ep1", DENY.name),
    ]
    assert decide(requirements, rows) is False


def _invoke_scorer_requirements(experiment_id, scorer_pattern):
    """The shape validate_can_invoke_scorer builds for a named registered scorer."""
    return [
        Requirement(RESOURCE_TYPE_EXPERIMENT, experiment_id, "update"),
        Requirement(RESOURCE_TYPE_TRACE, "*", ACTION_NOT_DENIED),
        Requirement(RESOURCE_TYPE_SCORER, scorer_pattern, ACTION_NOT_DENIED),
        Requirement(
            RESOURCE_TYPE_SCORER_VERSION,
            "*",
            ACTION_NOT_DENIED,
            fallback_if_no_grant=((RESOURCE_TYPE_SCORER, scorer_pattern),),
        ),
    ]


def test_invoke_scorer_is_vetoed_by_a_deny_on_the_named_scorer():
    requirements = _invoke_scorer_requirements(EXPERIMENT_ID, SCORER_PATTERN)
    rows = [
        grant(RESOURCE_TYPE_EXPERIMENT, "*", EDIT.name),
        grant(RESOURCE_TYPE_SCORER, SCORER_PATTERN, DENY.name),
    ]
    assert decide(requirements, rows) is False


def test_invoke_scorer_version_grant_does_not_rescue_a_scorer_deny():
    # Tier override decides WHICH tier supplies the action, and the version's grant does win
    # that contest -- but the scorer also carries its own veto as a separate requirement, so a
    # DENY on it is not rescued. Same shape as a run grant failing under an experiment DENY:
    # folding the parent veto into the fallback chain instead would let the child grant
    # short-circuit past it.
    requirements = _invoke_scorer_requirements(EXPERIMENT_ID, SCORER_PATTERN)
    rows = [
        grant(RESOURCE_TYPE_EXPERIMENT, "*", EDIT.name),
        grant(RESOURCE_TYPE_SCORER, SCORER_PATTERN, DENY.name),
        grant(RESOURCE_TYPE_SCORER_VERSION, "*", READ.name),
    ]
    assert decide(requirements, rows) is False
