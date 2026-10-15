# Store tests for conditions (condition-based access control).
#
# Covers CRUD, the partial-update contract, write-time validation, cascade, and the
# runtime loader -- including that per-user conditions are picked up with no
# special-casing (D10).

import uuid

import pytest
from sqlalchemy import event

from mlflow.exceptions import MlflowException
from mlflow.server.auth.conditions import (
    CONDITION_TYPE_TARGET,
    CONDITION_TYPE_VALUE,
    RESOURCE_VALUES_SHAPES,
)
from mlflow.server.auth.sqlalchemy_store import (
    _REGISTRY_RENAME_VERSION_TYPES,
    SqlAlchemyStore,
)

_PASSWORD = "password1234"
_WORKSPACE = "default"


@pytest.fixture
def store(tmp_path):
    s = SqlAlchemyStore()
    s.init_db(f"sqlite:///{tmp_path}/basic_auth.db")
    return s


@pytest.fixture
def user(store):
    return store.create_user("alice", _PASSWORD)


@pytest.fixture
def role(store):
    return store.create_role("dev", _WORKSPACE, "dev role")


# ---- CRUD ------------------------------------------------------------------


# Two purposes is two rows now, each with its own type and filter.
def test_add_and_get(store, role):
    value = store.add_condition(
        role.id,
        "registered_model",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'lifecycle'",
    )
    target = store.add_condition(
        role.id,
        "registered_model",
        condition_type=CONDITION_TYPE_TARGET,
        filter="tags.lifecycle = 'dev'",
    )
    assert value.resource_type == "registered_model"
    assert (value.type, value.filter) == (
        CONDITION_TYPE_VALUE,
        "tag_key != 'lifecycle'",
    )
    assert (target.type, target.filter) == (
        CONDITION_TYPE_TARGET,
        "tags.lifecycle = 'dev'",
    )

    fetched = store.get_condition(value.id)
    assert fetched.to_json() == value.to_json()


def test_each_type_has_its_own_slot_space(store, role):
    """A slot numbers a condition WITHIN its type, so the first of each is slot 1.

    Sharing one sequence would make the slots of one type depend on how many of the
    other exist, and the slot is what a denial message names.
    """
    value = store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'"
    )
    target = store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_TARGET, filter="tags.a = '1'"
    )
    second_value = store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'b'"
    )
    assert (value.condition_slot, target.condition_slot) == (1, 1)
    assert second_value.condition_slot == 2


def test_update_sets_the_filter(store, role):
    created = store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'"
    )
    updated = store.update_condition(created.id, filter="tag_key != 'z'")
    assert updated.filter == "tag_key != 'z'"
    assert updated.type == CONDITION_TYPE_VALUE, "the type is not rewritten"


def test_update_leaves_a_sibling_condition_unchanged(store, role):
    """The two filters were one row, so an update touching one could clear the other.

    They are separate rows now, which removes that failure mode by construction. What
    is worth pinning instead is that the rows really are independent.
    """
    value = store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'"
    )
    target = store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_TARGET, filter="tags.b = '1'"
    )
    store.update_condition(value.id, filter="tag_key != 'z'")
    assert store.get_condition(target.id).filter == "tags.b = '1'"


def test_update_cannot_clear_the_filter(store, role):
    """Clearing is refused rather than emptying the row or deleting it.

    A condition IS its filter, so an empty one is not a weaker restriction but an
    unconditional pass -- and because conditions of a type are alternatives, it would
    nullify every sibling of its type rather than merely relaxing itself. Removing the
    condition is the operation that says that, and it is already covered by remove.
    """
    created = store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'"
    )
    with pytest.raises(MlflowException, match="filter"):
        store.update_condition(created.id, filter=None)
    # Refused on the way in: the stored row is untouched, not half-updated or removed.
    assert store.get_condition(created.id).filter == "tag_key != 'a'"


def test_update_cannot_change_the_type(store, role):
    """The type selects which half of a request the filter is read against, so retyping
    a row silently repurposes a filter written for the other half -- a request filter
    would start being matched against resource state, or the reverse.
    """
    created = store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'"
    )
    with pytest.raises(TypeError, match="condition_type"):
        store.update_condition(created.id, condition_type=CONDITION_TYPE_TARGET)


def test_update_missing_raises(store, role):
    with pytest.raises(MlflowException, match="not found"):
        store.update_condition(99999, filter="tag_key != 'a'")


# ---- Write-time validation -------------------------------------------------


@pytest.mark.parametrize(
    "resource_type", ["scorer", "workspace", "assessment", "review_queue", "bogus"]
)
def test_unsupported_resource_type_rejected(store, role, resource_type):
    """Rejected at authoring time rather than silently never matching: a condition on
    an unsupported type is worse than none, because the admin believes a restriction
    is in force.
    """
    with pytest.raises(MlflowException, match="not supported for resource type"):
        store.add_condition(
            role.id, resource_type, condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'"
        )


@pytest.mark.parametrize(
    "resource_type",
    [
        "experiment",
        "run",
        "trace",
        "logged_model",
        "registered_model",
        "registered_model_version",
        "prompt",
        "prompt_version",
    ],
)
def test_supported_resource_types_accepted(store, role, resource_type):
    # Includes prompt and prompt_version at full parity with the model types (D2).
    assert store.add_condition(
        role.id, resource_type, condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'"
    )


def test_a_keyed_value_condition_is_accepted(store, role):
    """`tags.<key>` as a VALUE condition must survive write-time validation.

    The store parses a value condition with the request namespace, so this is the same
    authoring path as `parse_condition` -- but it is the path an admin actually uses, and
    it used to reject this syntax outright.
    """
    condition = store.add_condition(
        role.id,
        "run",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tags.a IN ('x','y','z') AND tag_key IN ('a')",
    )
    assert condition.filter == "tags.a IN ('x','y','z') AND tag_key IN ('a')"


def test_a_keyed_value_condition_still_refuses_a_reserved_key(store, role):
    # D4 is enforced on the KEY wherever `tags` is the identifier.
    with pytest.raises(MlflowException, match=r"reserved tag keys"):
        store.add_condition(
            role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tags.mlflow.runName = 'x'"
        )


@pytest.mark.parametrize(
    ("condition_type", "filter", "expected"),
    [
        (CONDITION_TYPE_VALUE, "tag_key = 'a' OR tag_key = 'b'", "OR is not supported"),
        # `tags.<key>` is accepted as a value condition (keyed request clause), so the
        # rejected resource identifier here is the alias form, which has no request
        # reading. See test_a_keyed_value_condition_is_accepted below.
        (CONDITION_TYPE_VALUE, "aliases.x = '1'", "resource-condition identifier"),
        (CONDITION_TYPE_TARGET, "tag_key = 'a'", "request-condition identifier"),
        (CONDITION_TYPE_VALUE, "tag_key > 'a'", "not supported in conditions"),
        (CONDITION_TYPE_VALUE, "tag_key = 'mlflow.runName'", "reserved tag keys"),
        (CONDITION_TYPE_VALUE, "garbage", "Invalid clause"),
    ],
)
def test_malformed_condition_rejected(store, role, condition_type, filter, expected):
    """A condition that failed to parse at evaluation time would have to fail open
    (unsafe) or deny every mutation (an outage), so it must be caught on the way in.

    The type now selects which namespace the filter is read against, so the two
    identifier-rejection cases are the same assertion from opposite sides: a resource
    identifier in a VALUE filter and a request identifier in a TARGET one.
    """
    with pytest.raises(MlflowException, match=expected):
        store.add_condition(role.id, "run", condition_type=condition_type, filter=filter)


def test_malformed_condition_rejected_on_update(store, role):
    created = store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'"
    )
    with pytest.raises(MlflowException, match="OR is not supported"):
        store.update_condition(created.id, filter="tag_key = 'a' OR tag_key = 'b'")
    # The stored row is untouched by the rejected update.
    assert store.get_condition(created.id).filter == "tag_key != 'a'"


@pytest.mark.parametrize(
    ("resource_type", "condition_type", "filter", "expected"),
    [
        ("run", CONDITION_TYPE_VALUE, "alias = 'champion'", "does not carry aliases"),
        ("experiment", CONDITION_TYPE_VALUE, "alias LIKE 'dev-%'", "does not carry aliases"),
        ("trace", CONDITION_TYPE_TARGET, "aliases.champion = '3'", "does not carry aliases"),
        ("logged_model", CONDITION_TYPE_TARGET, "aliases.x = '1'", "does not carry aliases"),
        (
            "registered_model_version",
            CONDITION_TYPE_VALUE,
            "alias = 'champion'",
            "condition the 'registered_model' resource type instead",
        ),
        (
            "prompt_version",
            CONDITION_TYPE_TARGET,
            "aliases.champion = '3'",
            "condition the 'prompt' resource type instead",
        ),
    ],
)
def test_alias_condition_rejected_for_a_type_without_aliases(
    store, role, resource_type, condition_type, filter, expected
):
    """Parsing is not enough. An alias clause on a type that carries no aliases is
    vacuous on the request side -- so the admin sees a saved restriction that can
    never fire -- and denies every mutation on the resource side. Both are caught on
    the way in, and nothing is persisted.
    """
    with pytest.raises(MlflowException, match=expected):
        store.add_condition(
            role.id,
            resource_type,
            condition_type=condition_type,
            filter=filter,
        )
    # Nothing was persisted. Asserted over the role's listing rather than by id,
    # because a rejected add never produced one.
    assert store.list_conditions(role.id) == []


def test_alias_condition_rejected_for_a_type_without_aliases_on_update(store, role):
    """The update path validates too: an admin editing a legitimate tag condition into
    an alias one must not slip past the check the add path applies.
    """
    created = store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'"
    )
    with pytest.raises(MlflowException, match="does not carry aliases"):
        store.update_condition(created.id, filter="alias = 'champion'")
    assert store.get_condition(created.id).filter == "tag_key != 'a'"


def test_alias_condition_accepted_for_the_registry_entry_types(store, role):
    for resource_type in ("registered_model", "prompt"):
        created = store.add_condition(
            role.id,
            resource_type,
            condition_type=CONDITION_TYPE_VALUE,
            filter="alias = 'champion'",
        )
        # The resource-side spelling is accepted on the same types, as its own row.
        store.add_condition(
            role.id,
            resource_type,
            condition_type=CONDITION_TYPE_TARGET,
            filter="aliases.champion = '3'",
        )
        assert created.filter == "alias = 'champion'"


def test_reserved_key_rejected_in_target_condition_too(store, role):
    # D4 is symmetric at the store boundary as well as the parser's.
    with pytest.raises(MlflowException, match="reserved tag keys"):
        store.add_condition(
            role.id,
            "registered_model",
            condition_type=CONDITION_TYPE_TARGET,
            filter="tags.`mlflow.prompt.is_prompt` = 'true'",
        )


# ---- Cascade ---------------------------------------------------------------


def test_delete_role_cascades_conditions(store, user, role):
    store.assign_role_to_user(user.id, role.id)
    store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'"
    )
    store.delete_role(role.id)
    assert store.list_conditions_for_user(user.id, _WORKSPACE, ["run"]) == []


# ---- Loader ----------------------------------------------------------------


def test_loader_returns_rows_from_every_role(store, user):
    a = store.create_role("a", _WORKSPACE, None)
    b = store.create_role("b", _WORKSPACE, None)
    store.assign_role_to_user(user.id, a.id)
    store.assign_role_to_user(user.id, b.id)
    store.add_condition(a.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'")
    store.add_condition(b.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'b'")

    rows = store.list_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert {r.filter for r in rows} == {"tag_key != 'a'", "tag_key != 'b'"}


def test_loader_returns_only_unscoped_conditions_when_no_parent_is_in_play(store, user, role):
    """A type with no parent in play matches only unscoped conditions.

    Not a fail-open: a scoped condition governs children of a *named* parent, so a
    request that names none is outside every scope. The refusal for a child whose
    parent could not be resolved lives at context construction, where the caller
    still knows it had a child to govern.
    """
    store.assign_role_to_user(user.id, role.id)
    store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'global'"
    )
    store.add_condition(
        role.id,
        "run",
        container_resource_type="experiment",
        container_resource_pattern="7",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'exp7'",
    )

    rows = store.list_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert {r.filter for r in rows} == {"tag_key != 'global'"}


def test_loader_adds_a_scoped_condition_only_for_its_own_parent(store, user, role):
    """The whole point of the bound being a *storage* bound: a role may hold many
    scoped conditions, and a request pays only for the ones that apply.
    """
    store.assign_role_to_user(user.id, role.id)
    store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'global'"
    )
    for parent in ("7", "9"):
        store.add_condition(
            role.id,
            "run",
            container_resource_type="experiment",
            container_resource_pattern=parent,
            condition_type=CONDITION_TYPE_VALUE,
            filter=f"tag_key != 'exp{parent}'",
        )

    def load(parent):
        rows = store.list_conditions_for_user(user.id, _WORKSPACE, ["run"], {"run": [parent]})
        return {r.filter for r in rows}

    assert load("7") == {"tag_key != 'global'", "tag_key != 'exp7'"}
    assert load("9") == {"tag_key != 'global'", "tag_key != 'exp9'"}
    # An experiment no condition names is governed only by the unscoped one.
    assert load("99") == {"tag_key != 'global'"}


def test_loader_does_not_let_a_parent_id_satisfy_another_type_scope(store, user, role):
    """Ids are only meaningful against their own type.

    Pins the per-type predicate build: under one shared parent filter across all the
    types queried, a request touching runs of experiment 7 would also pull in a
    version condition scoped to registered model 7, because the bare id would satisfy
    a scope belonging to another type. Mutation-tested against exactly that shape.
    """
    store.assign_role_to_user(user.id, role.id)
    store.add_condition(
        role.id,
        "run",
        container_resource_type="experiment",
        container_resource_pattern="7",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'runscope'",
    )
    store.add_condition(
        role.id,
        "registered_model_version",
        container_resource_type="registered_model",
        container_resource_pattern="7",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'modelscope'",
    )

    # Only the run's parent is in play, though both types are queried.
    rows = store.list_conditions_for_user(
        user.id, _WORKSPACE, ["run", "registered_model_version"], {"run": ["7"]}
    )
    assert {r.filter for r in rows} == {"tag_key != 'runscope'"}

    rows = store.list_conditions_for_user(
        user.id,
        _WORKSPACE,
        ["run", "registered_model_version"],
        {"run": ["7"], "registered_model_version": ["7"]},
    )
    assert {r.filter for r in rows} == {
        "tag_key != 'runscope'",
        "tag_key != 'modelscope'",
    }


def test_loader_excludes_roles_the_user_does_not_hold(store, user):
    mine = store.create_role("mine", _WORKSPACE, None)
    theirs = store.create_role("theirs", _WORKSPACE, None)
    store.assign_role_to_user(user.id, mine.id)
    store.add_condition(
        mine.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'mine'"
    )
    store.add_condition(
        theirs.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'theirs'"
    )

    rows = store.list_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert [r.filter for r in rows] == ["tag_key != 'mine'"]


def test_loader_filters_by_resource_type(store, user, role):
    store.assign_role_to_user(user.id, role.id)
    store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'run'"
    )
    store.add_condition(
        role.id, "trace", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'trace'"
    )

    rows = store.list_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert [r.resource_type for r in rows] == ["run"]


def test_loader_scopes_by_workspace(store, user):
    here = store.create_role("here", _WORKSPACE, None)
    store.assign_role_to_user(user.id, here.id)
    store.add_condition(
        here.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'here'"
    )

    assert store.list_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert store.list_conditions_for_user(user.id, "elsewhere", ["run"]) == []


def test_loader_empty_types_returns_empty_without_querying(store, user, role):
    store.assign_role_to_user(user.id, role.id)
    store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'"
    )
    assert store.list_conditions_for_user(user.id, _WORKSPACE, []) == []


def test_loader_rejects_unsupported_type(store, user):
    with pytest.raises(MlflowException, match="not supported for resource type"):
        store.list_conditions_for_user(user.id, _WORKSPACE, ["scorer"])


def test_loader_issues_one_query_for_many_types(store, user, role):
    """The loader is on the hot path for every mutation, so its cost must not scale
    with the number of types a validator declares.
    """
    store.assign_role_to_user(user.id, role.id)
    store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'"
    )
    store.add_condition(
        role.id, "trace", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'b'"
    )

    engine = store.engine
    statements = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        rows = store.list_conditions_for_user(
            user.id,
            _WORKSPACE,
            [
                "run",
                "trace",
                "experiment",
                "logged_model",
                "registered_model",
                "prompt",
            ],
        )
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert len(rows) == 2
    assert len(statements) == 1, f"expected one SELECT, got {len(statements)}: {statements}"


def test_loader_picks_up_per_user_conditions_with_no_special_casing(store, user):
    """D10. Per-user grants live on a synthetic ``__user_<id>__`` role that is a real
    ``roles`` row, so the ordinary join finds its conditions too -- which is why the
    loader needs no union and no second query.
    """
    store.grant_user_permission("alice", "run", "*", "READ")
    synthetic = next(r for r in store.list_roles([_WORKSPACE]) if r.name == f"__user_{user.id}__")
    store.add_condition(
        synthetic.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'peruser'"
    )

    rows = store.list_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert [r.filter for r in rows] == ["tag_key != 'peruser'"]


def test_loader_aggregates_named_and_per_user_conditions(store, user, role):
    """Both arrive from the same query, so the caller ANDs them without knowing
    which kind of role each came from.
    """
    store.assign_role_to_user(user.id, role.id)
    store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'fromrole'"
    )

    store.grant_user_permission("alice", "run", "*", "READ")
    synthetic = next(r for r in store.list_roles([_WORKSPACE]) if r.name == f"__user_{user.id}__")
    store.add_condition(
        synthetic.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'fromuser'"
    )

    rows = store.list_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert {r.filter for r in rows} == {
        "tag_key != 'fromrole'",
        "tag_key != 'fromuser'",
    }


def test_loader_rows_are_detached_plain_tuples(store, user, role):
    """Evaluation runs outside the store, so nothing it receives should be a live
    SQLAlchemy instance.
    """
    store.assign_role_to_user(user.id, role.id)
    store.add_condition(
        role.id, "run", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'"
    )
    store.add_condition(role.id, "run", condition_type=CONDITION_TYPE_TARGET, filter="tags.b = '1'")
    rows = store.list_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert all(isinstance(row, tuple) for row in rows)
    # One row per condition, each carrying its own type -- and the scope axes travel
    # with it, because the gate matches the container per context and the store cannot
    # do that for it.
    assert set(rows) == {
        ("run", CONDITION_TYPE_VALUE, "tag_key != 'a'", "*", "workspace", "*"),
        ("run", CONDITION_TYPE_TARGET, "tags.b = '1'", "*", "workspace", "*"),
    }


# ---- The user-addressed add ------------------------------------------------
#
# The counterpart of ``grant_user_resource_permission``: a caller naming a user should
# not have to know that per-user access is stored on a hidden ``__user_<id>__`` role,
# nor have to create that role first. Requiring a direct grant before a direct condition
# would be an ordering constraint with no model behind it.


def _synthetic_role(store, user_id):
    name = store._synthetic_user_role_name(user_id)
    return next((r for r in store.list_roles() if r.name == name), None)


def test_creates_the_synthetic_role_on_demand(store, user):
    # The headline: no direct grant has to exist first.
    assert _synthetic_role(store, user.id) is None

    mc = store.add_user_condition(
        "alice", "run", condition_type=CONDITION_TYPE_TARGET, filter="tags.lifecycle != 'prod'"
    )

    role = _synthetic_role(store, user.id)
    assert role is not None, "the synthetic role must be created on demand"
    assert mc.role_id == role.id
    assert mc.filter == "tags.lifecycle != 'prod'"


def test_reuses_the_role_a_direct_grant_already_created(store, user):
    # It must land on the SAME role the direct grants use, not a second one.
    store.grant_user_resource_permission("alice", "experiment", "*", "EDIT")
    role = _synthetic_role(store, user.id)
    assert role is not None

    mc = store.add_user_condition(
        "alice", "run", condition_type=CONDITION_TYPE_TARGET, filter="tags.x = 'y'"
    )
    assert mc.role_id == role.id


def test_the_condition_is_visible_through_the_role_addressed_list(store, user):
    # Remove and list stay role-addressed, so the two paths have to agree.
    mc = store.add_user_condition(
        "alice", "run", condition_type=CONDITION_TYPE_TARGET, filter="tags.x = 'y'"
    )
    role = _synthetic_role(store, user.id)
    assert [c.id for c in store.list_conditions(role.id)] == [mc.id]

    store.remove_condition(mc.id)
    assert store.list_conditions(role.id) == []


def test_allocates_distinct_slots_like_the_role_path(store, user):
    a = store.add_user_condition(
        "alice", "run", condition_type=CONDITION_TYPE_TARGET, filter="tags.a = '1'"
    )
    b = store.add_user_condition(
        "alice", "run", condition_type=CONDITION_TYPE_TARGET, filter="tags.b = '2'"
    )
    assert {a.condition_slot, b.condition_slot} == {1, 2}


def test_cannot_build_an_object_with_no_filter(store, user):
    """There is no "neither filter" case to refuse any more: condition_type and filter
    are required arguments, so the call cannot be made at all. That is a stronger
    guarantee than the rejection it replaces, which could only fire after the request
    had been assembled.
    """
    with pytest.raises(TypeError, match="condition_type|filter"):
        store.add_user_condition("alice", "run")


def test_refuses_an_unparsable_filter_without_creating_the_role(store, user):
    """Validation runs BEFORE the role is created.

    Otherwise a rejected request leaves behind a role the caller never asked for,
    and the next read reports the user as having a synthetic role with nothing on it.
    """
    with pytest.raises(MlflowException, match="comparator"):
        store.add_user_condition(
            "alice", "run", condition_type=CONDITION_TYPE_TARGET, filter="tags.x >= 'y'"
        )
    assert _synthetic_role(store, user.id) is None, (
        "a refused condition must not leave a role behind"
    )


def test_refuses_a_reserved_key_like_the_role_path(store, user):
    with pytest.raises(MlflowException, match="mlflow\\."):
        store.add_user_condition(
            "alice", "run", condition_type=CONDITION_TYPE_TARGET, filter="tags.mlflow.runName = 'x'"
        )


def test_refuses_a_container_the_type_cannot_have(store):
    store.create_user("u-cont", "password1234")
    with pytest.raises(MlflowException, match="no container other than"):
        store.add_user_condition(
            "u-cont",
            "experiment",
            container_resource_type="experiment",
            container_resource_pattern="5",
            condition_type=CONDITION_TYPE_TARGET,
            filter="tags.a = 'b'",
        )


def test_missing_user_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.add_user_condition(
            "nobody", "run", condition_type=CONDITION_TYPE_TARGET, filter="tags.x = 'y'"
        )


def test_the_condition_reaches_the_runtime_loader(store, user):
    # D10: a per-user condition is picked up with no special-casing.
    store.add_user_condition(
        "alice", "run", condition_type=CONDITION_TYPE_TARGET, filter="tags.x = 'y'"
    )
    rows = store.list_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert [r.filter for r in rows] == ["tags.x = 'y'"]


# ---- The single-resource scope ---------------------------------------------


# A condition is addressed like a grant -- type plus pattern -- and says which
#     container it applies within. Each type narrows on one axis, because the grain map
#     makes the choice for it.
#


def test_stores_and_returns_a_resource_pattern(store, role):
    created = store.add_condition(
        role.id,
        "experiment",
        resource_pattern="5",
        condition_type=CONDITION_TYPE_TARGET,
        filter="tags.env = 'dev'",
    )
    assert created.resource_pattern == "5"
    assert created.container_resource_type == "workspace"
    assert store.get_condition(created.id).resource_pattern == "5"


def test_stores_and_returns_a_container(store, role):
    created = store.add_condition(
        role.id,
        "run",
        container_resource_type="experiment",
        container_resource_pattern="3",
        condition_type=CONDITION_TYPE_TARGET,
        filter="tags.env = 'dev'",
    )
    assert created.resource_pattern == "*", "a sub-resource is wildcard-only"
    assert (created.container_resource_type, created.container_resource_pattern) == (
        "experiment",
        "3",
    )


def test_omitting_everything_is_the_whole_workspace(store, role):
    created = store.add_condition(
        role.id, "experiment", condition_type=CONDITION_TYPE_TARGET, filter="tags.env = 'dev'"
    )
    assert (
        created.resource_pattern,
        created.container_resource_type,
        created.container_resource_pattern,
    ) == ("*", "workspace", "*"), "the pre-scope shape must stay the default"


def test_a_wildcard_container_collapses_to_the_workspace(store, role):
    # One stored form per meaning, so the loader's SQL needs no wildcard branch.
    created = store.add_condition(
        role.id,
        "run",
        container_resource_type="experiment",
        container_resource_pattern="*",
        condition_type=CONDITION_TYPE_TARGET,
        filter="tags.a = 'b'",
    )
    assert created.container_resource_type == "workspace"


def test_a_sub_resource_cannot_be_narrowed_per_id(store, role):
    """Grants refuse a per-id child grant because it cannot be enforced in list and
    search paths. A condition inherits that rule rather than restating it, so the two
    are addressable at exactly the same grain.
    """
    with pytest.raises(MlflowException, match="wildcard"):
        store.add_condition(
            role.id,
            "run",
            resource_pattern="abc",
            condition_type=CONDITION_TYPE_TARGET,
            filter="tags.a = 'b'",
        )


def test_refuses_an_empty_or_padded_pattern(store, role):
    for bad in ("", "   "):
        with pytest.raises(MlflowException, match="non-empty"):
            store.add_condition(
                role.id,
                "experiment",
                resource_pattern=bad,
                condition_type=CONDITION_TYPE_TARGET,
                filter="tags.a = 'b'",
            )
    with pytest.raises(MlflowException, match="whitespace"):
        store.add_condition(
            role.id,
            "experiment",
            resource_pattern=" 5 ",
            condition_type=CONDITION_TYPE_TARGET,
            filter="tags.a = 'b'",
        )


def test_a_top_level_type_narrows_by_pattern_not_container(store, role):
    """The asymmetry this model fixes: an experiment condition had nothing between
    "every experiment" and nothing, because it has no container to name.
    """
    with pytest.raises(MlflowException, match="no container other than"):
        store.add_condition(
            role.id,
            "experiment",
            container_resource_type="experiment",
            container_resource_pattern="5",
            condition_type=CONDITION_TYPE_TARGET,
            filter="tags.a = 'b'",
        )
    created = store.add_condition(
        role.id,
        "experiment",
        resource_pattern="5",
        condition_type=CONDITION_TYPE_TARGET,
        filter="tags.a = 'b'",
    )
    assert created.resource_pattern == "5"


def test_the_runtime_loader_carries_the_pattern(store, role):
    """The gate matches the pattern in Python, so the loader MUST return it.

    Filtering that axis in SQL would be a fail-open: a cascade's child ids are unknown
    when the query runs, so a condition naming one of those children would be dropped.
    """
    store.create_user("alice-scope", "password1234")
    user = store.get_user("alice-scope")
    store.assign_role_to_user(user.id, role.id)
    store.add_condition(
        role.id,
        "experiment",
        resource_pattern="5",
        condition_type=CONDITION_TYPE_TARGET,
        filter="tags.a = 'b'",
    )
    rows = store.list_conditions_for_user(user.id, _WORKSPACE, ["experiment"])
    assert [r.resource_pattern for r in rows] == ["5"]


def test_the_loader_filters_the_container_in_sql(store, role):
    # The container axis *is* resolved before the query, so it is filtered there.
    store.create_user("bob-scope", "password1234")
    user = store.get_user("bob-scope")
    store.assign_role_to_user(user.id, role.id)
    store.add_condition(
        role.id,
        "run",
        container_resource_type="experiment",
        container_resource_pattern="3",
        condition_type=CONDITION_TYPE_TARGET,
        filter="tags.a = 'b'",
    )
    in_play = store.list_conditions_for_user(user.id, _WORKSPACE, ["run"], {"run": ["3"]})
    assert len(in_play) == 1
    other = store.list_conditions_for_user(user.id, _WORKSPACE, ["run"], {"run": ["9"]})
    assert other == [], "a condition contained by another experiment must not load"


def test_update_replaces_the_scope_as_a_whole(store, role):
    created = store.add_condition(
        role.id, "experiment", condition_type=CONDITION_TYPE_TARGET, filter="tags.a = 'b'"
    )
    # `update_filter` defaults to True, so a scope-only update must say it is not
    # touching the filter -- otherwise the filter would be read as cleared, which is
    # now refused outright rather than deleting the row.
    untouched = {"update_filter": False}
    narrowed = store.update_condition(
        created.id, resource_pattern="7", update_scope=True, **untouched
    )
    assert narrowed.resource_pattern == "7"
    widened = store.update_condition(created.id, update_scope=True, **untouched)
    assert widened.resource_pattern == "*", "an omitted axis takes its widest default"


def test_update_without_the_flag_leaves_the_scope_alone(store, role):
    # A client echoing the object back must not widen a scope it never touched.
    created = store.add_condition(
        role.id,
        "experiment",
        resource_pattern="7",
        condition_type=CONDITION_TYPE_TARGET,
        filter="tags.a = 'b'",
    )
    same = store.update_condition(created.id, filter="tags.a = 'c'")
    assert same.resource_pattern == "7"


# ---- Registry rename -------------------------------------------------------


def test_rename_moves_only_the_rows_addressed_by_the_old_name(store, role):
    """A rename rewrites the two axes that carry the name, and nothing else.

    The isolation half of the rename: a row scoped to a *sibling* name, a row that is
    workspace-wide, and a version row whose container names a different model must all
    come back unchanged, or the rename would retarget restrictions at resources it never
    touched -- the fail-open direction in a different disguise.
    """
    scoped = store.add_condition(
        role.id,
        "registered_model",
        resource_pattern="old",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'a'",
    )
    sibling = store.add_condition(
        role.id,
        "registered_model",
        resource_pattern="other",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'b'",
    )
    workspace_wide = store.add_condition(
        role.id, "registered_model", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'c'"
    )
    version = store.add_condition(
        role.id,
        "registered_model_version",
        container_resource_type="registered_model",
        container_resource_pattern="old",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'd'",
    )
    other_version = store.add_condition(
        role.id,
        "registered_model_version",
        container_resource_type="registered_model",
        container_resource_pattern="other",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'e'",
    )

    store.rename_conditions_for_registry_resource("old", "new", ("registered_model",))

    assert store.get_condition(scoped.id).resource_pattern == "new"
    assert store.get_condition(version.id).container_resource_pattern == "new"
    # Untouched.
    assert store.get_condition(sibling.id).resource_pattern == "other"
    assert store.get_condition(workspace_wide.id).resource_pattern == "*"
    assert store.get_condition(other_version.id).container_resource_pattern == "other"


def test_rename_does_not_cross_registry_families(store, role):
    """A rename touches only the family that was renamed, parent rows and version rows.

    The case the old unconditional sweep got wrong. A condition does not require its
    resource to exist, so a ``prompt`` row naming the string a ``registered_model``
    currently holds is a policy pre-created for a prompt nobody has created yet -- not
    debris from this model. Moving it with the model's rename leaves that prompt
    unrestricted the moment it is created, which is fail-open and reachable by anyone who
    can rename. Names being unique across the registry makes "one family matches" true of
    the resource, not of the policy.

    Pinned on both axes, because they are filtered separately: the parent rows by
    ``resource_type``, the version rows by their paired ``container_resource_type``.
    """
    model = store.add_condition(
        role.id,
        "registered_model",
        resource_pattern="shared",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'a'",
    )
    prompt = store.add_condition(
        role.id,
        "prompt",
        resource_pattern="shared",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'b'",
    )
    model_version = store.add_condition(
        role.id,
        "registered_model_version",
        container_resource_type="registered_model",
        container_resource_pattern="shared",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'c'",
    )
    prompt_version = store.add_condition(
        role.id,
        "prompt_version",
        container_resource_type="prompt",
        container_resource_pattern="shared",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'd'",
    )

    store.rename_conditions_for_registry_resource("shared", "renamed", ("registered_model",))

    # The renamed family moves, parent and version alike.
    assert store.get_condition(model.id).resource_pattern == "renamed"
    assert store.get_condition(model_version.id).container_resource_pattern == "renamed"
    # The prompt family is left alone -- it was never what got renamed.
    assert store.get_condition(prompt.id).resource_pattern == "shared"
    assert store.get_condition(prompt_version.id).container_resource_pattern == "shared"
    # And a rename never rewrites a container *type*, only its pattern.
    assert store.get_condition(prompt_version.id).container_resource_type == "prompt"
    assert store.get_condition(model_version.id).container_resource_type == "registered_model"


def test_renaming_a_prompt_moves_the_prompt_family_and_not_the_model_family(store, role):
    # The mirror of the test above: ``RenameRegisteredModel`` renames prompts too, so the
    # isolation has to hold in both directions.
    model = store.add_condition(
        role.id,
        "registered_model",
        resource_pattern="shared",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'a'",
    )
    prompt = store.add_condition(
        role.id,
        "prompt",
        resource_pattern="shared",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'b'",
    )
    prompt_version = store.add_condition(
        role.id,
        "prompt_version",
        container_resource_type="prompt",
        container_resource_pattern="shared",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'c'",
    )

    store.rename_conditions_for_registry_resource("shared", "renamed", ("prompt",))

    assert store.get_condition(prompt.id).resource_pattern == "renamed"
    assert store.get_condition(prompt_version.id).container_resource_pattern == "renamed"
    assert store.get_condition(model.id).resource_pattern == "shared"


@pytest.mark.parametrize("families", [(), ("experiment",), ("registered_model", "experiment")])
def test_rename_refuses_a_family_that_is_not_a_registry_family(store, role, families):
    """Refused, not silently ignored.

    A no-op would drop every restriction on the renamed resource while leaving its grants
    intact, which is the exact fail-open this sweep exists to close, so an unusable
    selection has to be loud rather than default to something.
    """
    scoped = store.add_condition(
        role.id,
        "registered_model",
        resource_pattern="old",
        condition_type=CONDITION_TYPE_VALUE,
        filter="tag_key != 'a'",
    )
    with pytest.raises(MlflowException, match="resource_families"):
        store.rename_conditions_for_registry_resource("old", "new", families)
    assert store.get_condition(scoped.id).resource_pattern == "old"


def test_every_registry_family_pairs_with_its_own_version_type():
    """Derived from the map so the two halves of a family cannot drift apart.

    Each key is a parent type whose ``resource_pattern`` is the name; each value is the
    version type that names that parent as its ``container_resource_type``. Asserted
    against the condition vocabulary rather than restated, so adding a registry family to
    one and not the other fails here.
    """
    assert set(_REGISTRY_RENAME_VERSION_TYPES) == {
        "registered_model",
        "prompt",
    }
    for parent, version in _REGISTRY_RENAME_VERSION_TYPES.items():
        assert parent in RESOURCE_VALUES_SHAPES
        assert version in RESOURCE_VALUES_SHAPES
        assert version == f"{parent}_version"


def test_rename_is_a_no_op_when_nothing_is_scoped_to_the_name(store, role):
    unscoped = store.add_condition(
        role.id, "registered_model", condition_type=CONDITION_TYPE_VALUE, filter="tag_key != 'a'"
    )
    store.rename_conditions_for_registry_resource("absent", "new", ("registered_model",))
    assert store.get_condition(unscoped.id).resource_pattern == "*"


# F-0043. ``parse_condition`` treats "" and None identically -- both constrain
#     nothing -- so a stored blank is a row that holds a condition slot and reads as a
#     configured restriction while refusing no request. That is the worst failure shape for
#     a security control, so a blank is normalized to absent at the boundary.
#


def test_a_blank_filter_is_refused(store):
    role = store.create_role("default", f"blank-{uuid.uuid4().hex[:8]}", "t")
    with pytest.raises(MlflowException, match="needs a 'filter'"):
        store.add_condition(role.id, "trace", condition_type=CONDITION_TYPE_VALUE, filter="")


def test_whitespace_is_not_a_filter_either(store):
    role = store.create_role("default", f"blank-{uuid.uuid4().hex[:8]}", "t")
    with pytest.raises(MlflowException, match="needs a 'filter'"):
        store.add_condition(role.id, "trace", condition_type=CONDITION_TYPE_VALUE, filter="   ")


def test_a_blank_filter_is_refused_rather_than_stored_as_absent(store):
    """There is no longer a half to be absent: a blank filter is refused on the way in.

    When the two filters shared a row, blanking one left the other restricting and the
    blank was normalised to NULL so nothing downstream had to decide what "" meant. A
    condition now IS its filter, so there is no surviving half -- and an empty filter
    would be an unconditional pass that nullifies every sibling of its type, since
    conditions of a type are alternatives.
    """
    role = store.create_role("default", f"blank-{uuid.uuid4().hex[:8]}", "t")
    with pytest.raises(MlflowException, match="needs a 'filter'"):
        store.add_condition(role.id, "trace", condition_type=CONDITION_TYPE_VALUE, filter="")
    assert store.list_conditions(role.id) == [], "nothing was persisted"


def test_updating_a_filter_to_blank_is_refused(store):
    """Blanking by update is refused for the same reason as blanking by add, and the
    row is left intact rather than deleted -- remove is how a condition goes away.
    """
    role = store.create_role("default", f"blank-{uuid.uuid4().hex[:8]}", "t")
    condition = store.add_condition(
        role.id, "trace", condition_type=CONDITION_TYPE_VALUE, filter="tag_key = 'a'"
    )
    with pytest.raises(MlflowException, match="filter"):
        store.update_condition(condition.id, filter="", update_filter=True)
    assert store.get_condition(condition.id).filter == "tag_key = 'a'"
