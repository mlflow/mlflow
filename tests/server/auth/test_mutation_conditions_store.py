# Store tests for mutation conditions (condition-based access control).
#
# Covers CRUD, the partial-update contract, write-time validation, cascade, and the
# runtime loader -- including that per-user conditions are picked up with no
# special-casing (D10).

import uuid

import pytest
from sqlalchemy import event

from mlflow.exceptions import MlflowException
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore

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


def test_add_and_get(store, role):
    created = store.add_mutation_condition(
        role.id,
        "registered_model",
        value_condition="tag_key != 'lifecycle'",
        target_condition="tags.lifecycle = 'dev'",
    )
    assert created.resource_type == "registered_model"
    assert created.value_condition == "tag_key != 'lifecycle'"
    assert created.target_condition == "tags.lifecycle = 'dev'"

    fetched = store.get_mutation_condition(created.id)
    assert fetched.to_json() == created.to_json()


def test_add_with_only_one_condition(store, role):
    # Either may be absent -- a role may restrict values, targets, or both.
    value_only = store.add_mutation_condition(
        role.id, "run", value_condition="tag_key != 'a'", target_condition=None
    )
    assert value_only.target_condition is None

    target_only = store.add_mutation_condition(
        role.id, "trace", value_condition=None, target_condition="tags.a = '1'"
    )
    assert target_only.value_condition is None


def test_several_conditions_per_resource_type_get_distinct_slots(store, role):
    """A role may hold several conditions for one type -- that is what makes
    per-parent scoping expressible, one object per governed parent.

    They **AND**: both must pass. Several narrow objects are not alternatives.
    """
    first = store.add_mutation_condition(
        role.id, "registered_model", value_condition="tag_key != 'a'"
    )
    second = store.add_mutation_condition(
        role.id, "registered_model", value_condition="tag_key != 'b'"
    )
    assert {first.condition_slot, second.condition_slot} == {1, 2}
    assert len(store.list_mutation_conditions(role.id)) == 2


def test_slot_is_reused_after_a_removal(store, role):
    """Lowest-free rather than max-plus-one. Otherwise a role that repeatedly adds and
    removes would exhaust the range while holding almost nothing.
    """
    first = store.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")
    assert first.condition_slot == 1
    store.remove_mutation_condition(first.id)
    again = store.add_mutation_condition(role.id, "run", value_condition="tag_key != 'b'")
    assert again.condition_slot == 1


def test_add_refuses_an_object_with_neither_filter(store, role):
    """An object with no filter restricts nothing while occupying a slot and reading
    as a configured restriction.
    """
    with pytest.raises(MlflowException, match="at least one"):
        store.add_mutation_condition(role.id, "run")


def test_same_resource_type_on_different_roles_is_fine(store):
    a = store.create_role("a", _WORKSPACE, None)
    b = store.create_role("b", _WORKSPACE, None)
    store.add_mutation_condition(a.id, "run", value_condition="tag_key != 'x'")
    store.add_mutation_condition(b.id, "run", value_condition="tag_key != 'y'")
    assert len(store.list_mutation_conditions(a.id)) == 1
    assert len(store.list_mutation_conditions(b.id)) == 1


def test_get_missing_raises(store, role):
    with pytest.raises(MlflowException, match="not found"):
        store.get_mutation_condition(99999)


def test_add_to_missing_role_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.add_mutation_condition(99999, "run", value_condition="tag_key != 'a'")


def test_remove(store, role):
    created = store.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")
    store.remove_mutation_condition(created.id)
    with pytest.raises(MlflowException, match="not found"):
        store.get_mutation_condition(created.id)


def test_remove_missing_raises(store, role):
    with pytest.raises(MlflowException, match="not found"):
        store.remove_mutation_condition(99999)


def test_list_for_role(store, role):
    store.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")
    store.add_mutation_condition(
        role.id, "trace", value_condition=None, target_condition="tags.b = '1'"
    )
    listed = store.list_mutation_conditions(role.id)
    assert {m.resource_type for m in listed} == {"run", "trace"}


def test_list_for_role_empty(store, role):
    assert store.list_mutation_conditions(role.id) == []


# ---- Partial update --------------------------------------------------------


def test_update_sets_both(store, role):
    created = store.add_mutation_condition(
        role.id, "run", value_condition="tag_key != 'a'", target_condition="tags.b = '1'"
    )
    updated = store.update_mutation_condition(
        created.id, value_condition="tag_key != 'z'", target_condition="tags.c = '2'"
    )
    assert updated.value_condition == "tag_key != 'z'"
    assert updated.target_condition == "tags.c = '2'"


def test_update_leaves_omitted_field_unchanged(store, role):
    """ "Leave unchanged" must be distinguishable from "clear" -- otherwise an update
    that only touches one field silently removes the other restriction.
    """
    created = store.add_mutation_condition(
        role.id, "run", value_condition="tag_key != 'a'", target_condition="tags.b = '1'"
    )
    updated = store.update_mutation_condition(
        created.id, value_condition="tag_key != 'z'", update_target_condition=False
    )
    assert updated.value_condition == "tag_key != 'z'"
    assert updated.target_condition == "tags.b = '1'"


def test_update_clears_with_explicit_none(store, role):
    created = store.add_mutation_condition(
        role.id, "run", value_condition="tag_key != 'a'", target_condition="tags.b = '1'"
    )
    updated = store.update_mutation_condition(
        created.id, target_condition=None, update_value_condition=False
    )
    assert updated.value_condition == "tag_key != 'a'"
    assert updated.target_condition is None


def test_update_clearing_both_deletes_the_object(store, role):
    """Clearing both filters removes the row and returns ``None``.

    The alternative -- keeping an object with neither filter -- would hold a slot
    against the per-type limit while restricting nothing, and would list as a
    configured restriction that cannot fire. ``add`` refuses such an object for the
    same reason, so ``update`` must not be able to manufacture one.
    """
    created = store.add_mutation_condition(
        role.id, "run", value_condition="tag_key != 'a'", target_condition="tags.b = '1'"
    )
    assert (
        store.update_mutation_condition(created.id, value_condition=None, target_condition=None)
        is None
    )
    with pytest.raises(MlflowException, match="not found"):
        store.get_mutation_condition(created.id)


def test_update_missing_raises(store, role):
    with pytest.raises(MlflowException, match="not found"):
        store.update_mutation_condition(99999, value_condition="tag_key != 'a'")


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
        store.add_mutation_condition(role.id, resource_type, value_condition="tag_key != 'a'")


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
    assert store.add_mutation_condition(role.id, resource_type, value_condition="tag_key != 'a'")


def test_a_keyed_value_condition_is_accepted(store, role):
    """`tags.<key>` as a VALUE condition must survive write-time validation.

    The store parses a value condition with the request namespace, so this is the same
    authoring path as `parse_condition` -- but it is the path an admin actually uses, and
    it used to reject this syntax outright.
    """
    condition = store.add_mutation_condition(
        role.id,
        "run",
        value_condition="tags.a IN ('x','y','z') AND tag_key IN ('a')",
    )
    assert condition.value_condition == "tags.a IN ('x','y','z') AND tag_key IN ('a')"


def test_a_keyed_value_condition_still_refuses_a_reserved_key(store, role):
    """D4 is enforced on the KEY wherever `tags` is the identifier."""
    with pytest.raises(MlflowException, match=r"reserved tag keys"):
        store.add_mutation_condition(role.id, "run", value_condition="tags.mlflow.runName = 'x'")


@pytest.mark.parametrize(
    ("value_condition", "target_condition", "expected"),
    [
        ("tag_key = 'a' OR tag_key = 'b'", None, "OR is not supported"),
        # `tags.<key>` is accepted as a value condition (keyed request clause), so the
        # rejected resource identifier here is the alias form, which has no request
        # reading. See test_a_keyed_value_condition_is_accepted below.
        ("aliases.x = '1'", None, "resource-condition identifier"),
        (None, "tag_key = 'a'", "request-condition identifier"),
        ("tag_key > 'a'", None, "not supported in conditions"),
        ("tag_key = 'mlflow.runName'", None, "reserved tag keys"),
        ("garbage", None, "Invalid clause"),
    ],
)
def test_malformed_condition_rejected(store, role, value_condition, target_condition, expected):
    """A condition that failed to parse at evaluation time would have to fail open
    (unsafe) or deny every mutation (an outage), so it must be caught on the way in.
    """
    with pytest.raises(MlflowException, match=expected):
        store.add_mutation_condition(
            role.id, "run", value_condition=value_condition, target_condition=target_condition
        )


def test_malformed_condition_rejected_on_update(store, role):
    created = store.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")
    with pytest.raises(MlflowException, match="OR is not supported"):
        store.update_mutation_condition(
            created.id, value_condition="tag_key = 'a' OR tag_key = 'b'"
        )
    # The stored row is untouched by the rejected update.
    assert store.get_mutation_condition(created.id).value_condition == "tag_key != 'a'"


@pytest.mark.parametrize(
    ("resource_type", "value_condition", "target_condition", "expected"),
    [
        ("run", "alias = 'champion'", None, "does not carry aliases"),
        ("experiment", "alias LIKE 'dev-%'", None, "does not carry aliases"),
        ("trace", None, "aliases.champion = '3'", "does not carry aliases"),
        ("logged_model", None, "aliases.x = '1'", "does not carry aliases"),
        (
            "registered_model_version",
            "alias = 'champion'",
            None,
            "condition the 'registered_model' resource type instead",
        ),
        (
            "prompt_version",
            None,
            "aliases.champion = '3'",
            "condition the 'prompt' resource type instead",
        ),
    ],
)
def test_alias_condition_rejected_for_a_type_without_aliases(
    store, role, resource_type, value_condition, target_condition, expected
):
    """Parsing is not enough. An alias clause on a type that carries no aliases is
    vacuous on the request side -- so the admin sees a saved restriction that can
    never fire -- and denies every mutation on the resource side. Both are caught on
    the way in, and nothing is persisted.
    """
    with pytest.raises(MlflowException, match=expected):
        store.add_mutation_condition(
            role.id,
            resource_type,
            value_condition=value_condition,
            target_condition=target_condition,
        )
    # Nothing was persisted. Asserted over the role's listing rather than by id,
    # because a rejected add never produced one.
    assert store.list_mutation_conditions(role.id) == []


def test_alias_condition_rejected_for_a_type_without_aliases_on_update(store, role):
    """The update path validates too: an admin editing a legitimate tag condition into
    an alias one must not slip past the check the add path applies.
    """
    created = store.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")
    with pytest.raises(MlflowException, match="does not carry aliases"):
        store.update_mutation_condition(created.id, value_condition="alias = 'champion'")
    assert store.get_mutation_condition(created.id).value_condition == "tag_key != 'a'"


def test_alias_condition_accepted_for_the_registry_entry_types(store, role):
    for resource_type in ("registered_model", "prompt"):
        created = store.add_mutation_condition(
            role.id,
            resource_type,
            value_condition="alias = 'champion'",
            target_condition="aliases.champion = '3'",
        )
        assert created.value_condition == "alias = 'champion'"


def test_reserved_key_rejected_in_target_condition_too(store, role):
    """D4 is symmetric at the store boundary as well as the parser's."""
    with pytest.raises(MlflowException, match="reserved tag keys"):
        store.add_mutation_condition(
            role.id,
            "registered_model",
            value_condition=None,
            target_condition="tags.`mlflow.prompt.is_prompt` = 'true'",
        )


# ---- Cascade ---------------------------------------------------------------


def test_delete_role_cascades_conditions(store, user, role):
    store.assign_role_to_user(user.id, role.id)
    store.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")
    store.delete_role(role.id)
    assert store.list_mutation_conditions_for_user(user.id, _WORKSPACE, ["run"]) == []


# ---- Loader ----------------------------------------------------------------


def test_loader_returns_rows_from_every_role(store, user):
    a = store.create_role("a", _WORKSPACE, None)
    b = store.create_role("b", _WORKSPACE, None)
    store.assign_role_to_user(user.id, a.id)
    store.assign_role_to_user(user.id, b.id)
    store.add_mutation_condition(a.id, "run", value_condition="tag_key != 'a'")
    store.add_mutation_condition(b.id, "run", value_condition="tag_key != 'b'")

    rows = store.list_mutation_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert {r.value_condition for r in rows} == {"tag_key != 'a'", "tag_key != 'b'"}


def test_loader_returns_only_unscoped_conditions_when_no_parent_is_in_play(store, user, role):
    """A type with no parent in play matches only unscoped conditions.

    Not a fail-open: a scoped condition governs children of a *named* parent, so a
    request that names none is outside every scope. The refusal for a child whose
    parent could not be resolved lives at context construction, where the caller
    still knows it had a child to govern.
    """
    store.assign_role_to_user(user.id, role.id)
    store.add_mutation_condition(role.id, "run", value_condition="tag_key != 'global'")
    store.add_mutation_condition(
        role.id,
        "run",
        container_resource_type="experiment",
        container_resource_pattern="7",
        value_condition="tag_key != 'exp7'",
    )

    rows = store.list_mutation_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert {r.value_condition for r in rows} == {"tag_key != 'global'"}


def test_loader_adds_a_scoped_condition_only_for_its_own_parent(store, user, role):
    """The whole point of the bound being a *storage* bound: a role may hold many
    scoped conditions, and a request pays only for the ones that apply.
    """
    store.assign_role_to_user(user.id, role.id)
    store.add_mutation_condition(role.id, "run", value_condition="tag_key != 'global'")
    for parent in ("7", "9"):
        store.add_mutation_condition(
            role.id,
            "run",
            container_resource_type="experiment",
            container_resource_pattern=parent,
            value_condition=f"tag_key != 'exp{parent}'",
        )

    def load(parent):
        rows = store.list_mutation_conditions_for_user(
            user.id, _WORKSPACE, ["run"], {"run": [parent]}
        )
        return {r.value_condition for r in rows}

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
    store.add_mutation_condition(
        role.id,
        "run",
        container_resource_type="experiment",
        container_resource_pattern="7",
        value_condition="tag_key != 'runscope'",
    )
    store.add_mutation_condition(
        role.id,
        "registered_model_version",
        container_resource_type="registered_model",
        container_resource_pattern="7",
        value_condition="tag_key != 'modelscope'",
    )

    # Only the run's parent is in play, though both types are queried.
    rows = store.list_mutation_conditions_for_user(
        user.id, _WORKSPACE, ["run", "registered_model_version"], {"run": ["7"]}
    )
    assert {r.value_condition for r in rows} == {"tag_key != 'runscope'"}

    rows = store.list_mutation_conditions_for_user(
        user.id,
        _WORKSPACE,
        ["run", "registered_model_version"],
        {"run": ["7"], "registered_model_version": ["7"]},
    )
    assert {r.value_condition for r in rows} == {
        "tag_key != 'runscope'",
        "tag_key != 'modelscope'",
    }


def test_loader_excludes_roles_the_user_does_not_hold(store, user):
    mine = store.create_role("mine", _WORKSPACE, None)
    theirs = store.create_role("theirs", _WORKSPACE, None)
    store.assign_role_to_user(user.id, mine.id)
    store.add_mutation_condition(mine.id, "run", value_condition="tag_key != 'mine'")
    store.add_mutation_condition(theirs.id, "run", value_condition="tag_key != 'theirs'")

    rows = store.list_mutation_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert [r.value_condition for r in rows] == ["tag_key != 'mine'"]


def test_loader_filters_by_resource_type(store, user, role):
    store.assign_role_to_user(user.id, role.id)
    store.add_mutation_condition(role.id, "run", value_condition="tag_key != 'run'")
    store.add_mutation_condition(role.id, "trace", value_condition="tag_key != 'trace'")

    rows = store.list_mutation_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert [r.resource_type for r in rows] == ["run"]


def test_loader_scopes_by_workspace(store, user):
    here = store.create_role("here", _WORKSPACE, None)
    store.assign_role_to_user(user.id, here.id)
    store.add_mutation_condition(here.id, "run", value_condition="tag_key != 'here'")

    assert store.list_mutation_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert store.list_mutation_conditions_for_user(user.id, "elsewhere", ["run"]) == []


def test_loader_empty_types_returns_empty_without_querying(store, user, role):
    store.assign_role_to_user(user.id, role.id)
    store.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")
    assert store.list_mutation_conditions_for_user(user.id, _WORKSPACE, []) == []


def test_loader_rejects_unsupported_type(store, user):
    with pytest.raises(MlflowException, match="not supported for resource type"):
        store.list_mutation_conditions_for_user(user.id, _WORKSPACE, ["scorer"])


def test_loader_issues_one_query_for_many_types(store, user, role):
    """The loader is on the hot path for every mutation, so its cost must not scale
    with the number of types a validator declares.
    """
    store.assign_role_to_user(user.id, role.id)
    store.add_mutation_condition(role.id, "run", value_condition="tag_key != 'a'")
    store.add_mutation_condition(role.id, "trace", value_condition="tag_key != 'b'")

    engine = store.engine
    statements = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        rows = store.list_mutation_conditions_for_user(
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
    store.add_mutation_condition(synthetic.id, "run", value_condition="tag_key != 'peruser'")

    rows = store.list_mutation_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert [r.value_condition for r in rows] == ["tag_key != 'peruser'"]


def test_loader_aggregates_named_and_per_user_conditions(store, user, role):
    """Both arrive from the same query, so the caller ANDs them without knowing
    which kind of role each came from.
    """
    store.assign_role_to_user(user.id, role.id)
    store.add_mutation_condition(role.id, "run", value_condition="tag_key != 'fromrole'")

    store.grant_user_permission("alice", "run", "*", "READ")
    synthetic = next(r for r in store.list_roles([_WORKSPACE]) if r.name == f"__user_{user.id}__")
    store.add_mutation_condition(synthetic.id, "run", value_condition="tag_key != 'fromuser'")

    rows = store.list_mutation_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert {r.value_condition for r in rows} == {
        "tag_key != 'fromrole'",
        "tag_key != 'fromuser'",
    }


def test_loader_rows_are_detached_plain_tuples(store, user, role):
    """Evaluation runs outside the store, so nothing it receives should be a live
    SQLAlchemy instance.
    """
    store.assign_role_to_user(user.id, role.id)
    store.add_mutation_condition(
        role.id, "run", value_condition="tag_key != 'a'", target_condition="tags.b = '1'"
    )
    (row,) = store.list_mutation_conditions_for_user(user.id, _WORKSPACE, ["run"])
    assert isinstance(row, tuple)
    # The scope axes travel with the row: the gate matches the container per context,
    # which the store cannot do for it.
    assert row == ("run", "tag_key != 'a'", "tags.b = '1'", "*", "workspace", "*")


# ---- The user-addressed add ------------------------------------------------
#
# The counterpart of ``grant_user_resource_permission``: a caller naming a user should
# not have to know that per-user access is stored on a hidden ``__user_<id>__`` role,
# nor have to create that role first. Requiring a direct grant before a direct condition
# would be an ordering constraint with no model behind it.


def _synthetic_role(store, user_id):
    name = store._synthetic_user_role_name(user_id)
    return next((r for r in store.list_roles() if r.name == name), None)


class TestTheUserAddressedAdd:
    def test_creates_the_synthetic_role_on_demand(self, store, user):
        """The headline: no direct grant has to exist first."""
        assert _synthetic_role(store, user.id) is None

        mc = store.add_user_mutation_condition(
            "alice", "run", target_condition="tags.lifecycle != 'prod'"
        )

        role = _synthetic_role(store, user.id)
        assert role is not None, "the synthetic role must be created on demand"
        assert mc.role_id == role.id
        assert mc.target_condition == "tags.lifecycle != 'prod'"

    def test_reuses_the_role_a_direct_grant_already_created(self, store, user):
        """It must land on the SAME role the direct grants use, not a second one."""
        store.grant_user_resource_permission("alice", "experiment", "*", "EDIT")
        role = _synthetic_role(store, user.id)
        assert role is not None

        mc = store.add_user_mutation_condition("alice", "run", target_condition="tags.x = 'y'")
        assert mc.role_id == role.id

    def test_the_condition_is_visible_through_the_role_addressed_list(self, store, user):
        """Remove and list stay role-addressed, so the two paths have to agree."""
        mc = store.add_user_mutation_condition("alice", "run", target_condition="tags.x = 'y'")
        role = _synthetic_role(store, user.id)
        assert [c.id for c in store.list_mutation_conditions(role.id)] == [mc.id]

        store.remove_mutation_condition(mc.id)
        assert store.list_mutation_conditions(role.id) == []

    def test_allocates_distinct_slots_like_the_role_path(self, store, user):
        a = store.add_user_mutation_condition("alice", "run", target_condition="tags.a = '1'")
        b = store.add_user_mutation_condition("alice", "run", target_condition="tags.b = '2'")
        assert {a.condition_slot, b.condition_slot} == {1, 2}

    def test_refuses_an_object_with_neither_filter(self, store, user):
        with pytest.raises(MlflowException, match="at least one of"):
            store.add_user_mutation_condition("alice", "run")

    def test_refuses_an_unparseable_filter_without_creating_the_role(self, store, user):
        """Validation runs BEFORE the role is created.

        Otherwise a rejected request leaves behind a role the caller never asked for,
        and the next read reports the user as having a synthetic role with nothing on it.
        """
        with pytest.raises(MlflowException, match="comparator"):
            store.add_user_mutation_condition("alice", "run", target_condition="tags.x >= 'y'")
        assert _synthetic_role(store, user.id) is None, (
            "a refused condition must not leave a role behind"
        )

    def test_refuses_a_reserved_key_like_the_role_path(self, store, user):
        with pytest.raises(MlflowException, match="mlflow\\."):
            store.add_user_mutation_condition(
                "alice", "run", target_condition="tags.mlflow.runName = 'x'"
            )

    def test_refuses_a_container_the_type_cannot_have(self, store):
        store.create_user("u-cont", "password1234")
        with pytest.raises(MlflowException, match="no container other than"):
            store.add_user_mutation_condition(
                "u-cont",
                "experiment",
                container_resource_type="experiment",
                container_resource_pattern="5",
                target_condition="tags.a = 'b'",
            )

    def test_missing_user_raises(self, store):
        with pytest.raises(MlflowException, match="not found"):
            store.add_user_mutation_condition("nobody", "run", target_condition="tags.x = 'y'")

    def test_the_condition_reaches_the_runtime_loader(self, store, user):
        """D10: a per-user condition is picked up with no special-casing."""
        store.add_user_mutation_condition("alice", "run", target_condition="tags.x = 'y'")
        rows = store.list_mutation_conditions_for_user(user.id, _WORKSPACE, ["run"])
        assert [r.target_condition for r in rows] == ["tags.x = 'y'"]


# ---- The single-resource scope ---------------------------------------------


class TestTheTwoScopeAxes:
    """A condition is addressed like a grant -- type plus pattern -- and says which
    container it applies within. Each type narrows on one axis, because the grain map
    makes the choice for it.
    """

    def test_stores_and_returns_a_resource_pattern(self, store, role):
        created = store.add_mutation_condition(
            role.id, "experiment", resource_pattern="5", target_condition="tags.env = 'dev'"
        )
        assert created.resource_pattern == "5"
        assert created.container_resource_type == "workspace"
        assert store.get_mutation_condition(created.id).resource_pattern == "5"

    def test_stores_and_returns_a_container(self, store, role):
        created = store.add_mutation_condition(
            role.id,
            "run",
            container_resource_type="experiment",
            container_resource_pattern="3",
            target_condition="tags.env = 'dev'",
        )
        assert created.resource_pattern == "*", "a sub-resource is wildcard-only"
        assert (created.container_resource_type, created.container_resource_pattern) == (
            "experiment",
            "3",
        )

    def test_omitting_everything_is_the_whole_workspace(self, store, role):
        created = store.add_mutation_condition(
            role.id, "experiment", target_condition="tags.env = 'dev'"
        )
        assert (
            created.resource_pattern,
            created.container_resource_type,
            created.container_resource_pattern,
        ) == ("*", "workspace", "*"), "the pre-scope shape must stay the default"

    def test_a_wildcard_container_collapses_to_the_workspace(self, store, role):
        """One stored form per meaning, so the loader's SQL needs no wildcard branch."""
        created = store.add_mutation_condition(
            role.id,
            "run",
            container_resource_type="experiment",
            container_resource_pattern="*",
            target_condition="tags.a = 'b'",
        )
        assert created.container_resource_type == "workspace"

    def test_a_sub_resource_cannot_be_narrowed_per_id(self, store, role):
        """Grants refuse a per-id child grant because it cannot be enforced in list and
        search paths. A condition inherits that rule rather than restating it, so the two
        are addressable at exactly the same grain.
        """
        with pytest.raises(MlflowException, match="wildcard"):
            store.add_mutation_condition(
                role.id, "run", resource_pattern="abc", target_condition="tags.a = 'b'"
            )

    def test_refuses_an_empty_or_padded_pattern(self, store, role):
        for bad in ("", "   "):
            with pytest.raises(MlflowException, match="non-empty"):
                store.add_mutation_condition(
                    role.id, "experiment", resource_pattern=bad, target_condition="tags.a = 'b'"
                )
        with pytest.raises(MlflowException, match="whitespace"):
            store.add_mutation_condition(
                role.id, "experiment", resource_pattern=" 5 ", target_condition="tags.a = 'b'"
            )

    def test_a_top_level_type_narrows_by_pattern_not_container(self, store, role):
        """The asymmetry this model fixes: an experiment condition had nothing between
        "every experiment" and nothing, because it has no container to name.
        """
        with pytest.raises(MlflowException, match="no container other than"):
            store.add_mutation_condition(
                role.id,
                "experiment",
                container_resource_type="experiment",
                container_resource_pattern="5",
                target_condition="tags.a = 'b'",
            )
        created = store.add_mutation_condition(
            role.id, "experiment", resource_pattern="5", target_condition="tags.a = 'b'"
        )
        assert created.resource_pattern == "5"

    def test_the_runtime_loader_carries_the_pattern(self, store, role):
        """The gate matches the pattern in Python, so the loader MUST return it.

        Filtering that axis in SQL would be a fail-open: a cascade's child ids are unknown
        when the query runs, so a condition naming one of those children would be dropped.
        """
        store.create_user("alice-scope", "password1234")
        user = store.get_user("alice-scope")
        store.assign_role_to_user(user.id, role.id)
        store.add_mutation_condition(
            role.id, "experiment", resource_pattern="5", target_condition="tags.a = 'b'"
        )
        rows = store.list_mutation_conditions_for_user(user.id, _WORKSPACE, ["experiment"])
        assert [r.resource_pattern for r in rows] == ["5"]

    def test_the_loader_filters_the_container_in_sql(self, store, role):
        """The container axis *is* resolved before the query, so it is filtered there."""
        store.create_user("bob-scope", "password1234")
        user = store.get_user("bob-scope")
        store.assign_role_to_user(user.id, role.id)
        store.add_mutation_condition(
            role.id,
            "run",
            container_resource_type="experiment",
            container_resource_pattern="3",
            target_condition="tags.a = 'b'",
        )
        in_play = store.list_mutation_conditions_for_user(
            user.id, _WORKSPACE, ["run"], {"run": ["3"]}
        )
        assert len(in_play) == 1
        other = store.list_mutation_conditions_for_user(
            user.id, _WORKSPACE, ["run"], {"run": ["9"]}
        )
        assert other == [], "a condition contained by another experiment must not load"

    def test_update_replaces_the_scope_as_a_whole(self, store, role):
        created = store.add_mutation_condition(
            role.id, "experiment", target_condition="tags.a = 'b'"
        )
        # The filter flags default to True, so a scope-only update must say it is not
        # touching them -- otherwise both filters clear and the object is deleted.
        untouched = {"update_value_condition": False, "update_target_condition": False}
        narrowed = store.update_mutation_condition(
            created.id, resource_pattern="7", update_scope=True, **untouched
        )
        assert narrowed.resource_pattern == "7"
        widened = store.update_mutation_condition(created.id, update_scope=True, **untouched)
        assert widened.resource_pattern == "*", "an omitted axis takes its widest default"

    def test_update_without_the_flag_leaves_the_scope_alone(self, store, role):
        """A client echoing the object back must not widen a scope it never touched."""
        created = store.add_mutation_condition(
            role.id, "experiment", resource_pattern="7", target_condition="tags.a = 'b'"
        )
        same = store.update_mutation_condition(created.id, target_condition="tags.a = 'c'")
        assert same.resource_pattern == "7"


# ---- Registry rename -------------------------------------------------------


def test_rename_moves_only_the_rows_addressed_by_the_old_name(store, role):
    """A rename rewrites the two axes that carry the name, and nothing else.

    The isolation half of the rename: a row scoped to a *sibling* name, a row that is
    workspace-wide, and a version row whose container names a different model must all
    come back unchanged, or the rename would retarget restrictions at resources it never
    touched -- the fail-open direction in a different disguise.
    """
    scoped = store.add_mutation_condition(
        role.id, "registered_model", resource_pattern="old", value_condition="tag_key != 'a'"
    )
    sibling = store.add_mutation_condition(
        role.id, "registered_model", resource_pattern="other", value_condition="tag_key != 'b'"
    )
    workspace_wide = store.add_mutation_condition(
        role.id, "registered_model", value_condition="tag_key != 'c'"
    )
    version = store.add_mutation_condition(
        role.id,
        "registered_model_version",
        container_resource_type="registered_model",
        container_resource_pattern="old",
        value_condition="tag_key != 'd'",
    )
    other_version = store.add_mutation_condition(
        role.id,
        "registered_model_version",
        container_resource_type="registered_model",
        container_resource_pattern="other",
        value_condition="tag_key != 'e'",
    )

    store.rename_conditions_for_registry_resource("old", "new")

    assert store.get_mutation_condition(scoped.id).resource_pattern == "new"
    assert store.get_mutation_condition(version.id).container_resource_pattern == "new"
    # Untouched.
    assert store.get_mutation_condition(sibling.id).resource_pattern == "other"
    assert store.get_mutation_condition(workspace_wide.id).resource_pattern == "*"
    assert store.get_mutation_condition(other_version.id).container_resource_pattern == "other"


def test_rename_does_not_cross_registry_families(store, role):
    """Each version type keeps its own container type across a rename.

    Both families are swept unconditionally -- names are unique across the registry, so
    one matches and the other is a no-op. The isolation comes from the per-type
    ``resource_type ==`` filter, which already partitions the rows; the paired
    ``container_resource_type`` is defence in depth, the same relationship
    ``_scope_predicates`` documents. What this pins is the outcome: a rename moves the
    container *pattern* and never the container *type*, so no row can end up addressed by
    the other family's container.
    """
    prompt_version = store.add_mutation_condition(
        role.id,
        "prompt_version",
        container_resource_type="prompt",
        container_resource_pattern="shared",
        value_condition="tag_key != 'a'",
    )
    model_version = store.add_mutation_condition(
        role.id,
        "registered_model_version",
        container_resource_type="registered_model",
        container_resource_pattern="shared",
        value_condition="tag_key != 'b'",
    )

    store.rename_conditions_for_registry_resource("shared", "renamed")

    # Both move, because each matched its OWN family's container type -- and the sweep
    # covers both families. What must not happen is one row taking the other's container.
    assert store.get_mutation_condition(prompt_version.id).container_resource_type == "prompt"
    assert (
        store.get_mutation_condition(model_version.id).container_resource_type == "registered_model"
    )
    assert store.get_mutation_condition(prompt_version.id).container_resource_pattern == "renamed"
    assert store.get_mutation_condition(model_version.id).container_resource_pattern == "renamed"


def test_rename_is_a_no_op_when_nothing_is_scoped_to_the_name(store, role):
    unscoped = store.add_mutation_condition(
        role.id, "registered_model", value_condition="tag_key != 'a'"
    )
    store.rename_conditions_for_registry_resource("absent", "new")
    assert store.get_mutation_condition(unscoped.id).resource_pattern == "*"


class TestABlankFilterIsNotARestriction:
    """F-0043. ``parse_condition`` treats "" and None identically -- both constrain
    nothing -- so a stored blank is a row that holds a condition slot and reads as a
    configured restriction while refusing no request. That is the worst failure shape for
    a security control, so a blank is normalized to absent at the boundary.
    """

    def test_a_blank_value_filter_alone_is_refused(self, store):
        role = store.create_role("default", f"blank-{uuid.uuid4().hex[:8]}", "t")
        with pytest.raises(MlflowException, match="at least one of"):
            store.add_mutation_condition(role.id, "trace", value_condition="")

    def test_whitespace_is_not_a_filter_either(self, store):
        role = store.create_role("default", f"blank-{uuid.uuid4().hex[:8]}", "t")
        with pytest.raises(MlflowException, match="at least one of"):
            store.add_mutation_condition(
                role.id, "trace", value_condition="   ", target_condition="\t"
            )

    def test_a_blank_half_is_stored_as_absent(self, store):
        role = store.create_role("default", f"blank-{uuid.uuid4().hex[:8]}", "t")
        condition = store.add_mutation_condition(
            role.id, "trace", value_condition="", target_condition="tags.gate = 'open'"
        )
        # Not "" -- absent, so nothing downstream has to decide what a blank means.
        assert condition.value_condition is None
        assert condition.target_condition == "tags.gate = 'open'"

    def test_updating_a_half_to_blank_clears_it(self, store):
        role = store.create_role("default", f"blank-{uuid.uuid4().hex[:8]}", "t")
        condition = store.add_mutation_condition(
            role.id,
            "trace",
            value_condition="tag_key = 'a'",
            target_condition="tags.gate = 'open'",
        )
        # update_target_condition defaults to True, so it has to be switched OFF to leave
        # the other half alone -- otherwise both clear and the object is deleted.
        updated = store.update_mutation_condition(
            condition.id,
            value_condition="",
            update_value_condition=True,
            update_target_condition=False,
        )
        assert updated is not None
        assert updated.value_condition is None
        assert updated.target_condition == "tags.gate = 'open'"

    def test_blanking_both_halves_deletes_the_object(self, store):
        role = store.create_role("default", f"blank-{uuid.uuid4().hex[:8]}", "t")
        condition = store.add_mutation_condition(role.id, "trace", value_condition="tag_key = 'a'")
        # Blanking the only filter leaves nothing to restrict, so the row must go rather
        # than linger as an empty restriction.
        assert (
            store.update_mutation_condition(
                condition.id, value_condition="", update_value_condition=True
            )
            is None
        )
        with pytest.raises(MlflowException, match="not found"):
            store.get_mutation_condition(condition.id)
