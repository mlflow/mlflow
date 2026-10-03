# Store tests for mutation conditions (condition-based access control).
#
# Covers CRUD, the partial-update contract, write-time validation, cascade, and the
# runtime loader -- including that per-user conditions are picked up with no
# special-casing (D10).

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


@pytest.mark.parametrize(
    ("value_condition", "target_condition", "expected"),
    [
        ("tag_key = 'a' OR tag_key = 'b'", None, "OR is not supported"),
        ("tags.x = '1'", None, "resource-condition identifier"),
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
    assert row == ("run", "tag_key != 'a'", "tags.b = '1'")
