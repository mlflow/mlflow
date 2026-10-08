# Parity between pushed-down and in-memory evaluation of a target condition.
#
# ``AbstractStore.find_failing_resource`` lets a store answer "is there a resource
# here that fails this tag predicate" without loading the resources. That makes
# **two** implementations of one semantic: the store's SQL predicate and the
# auth layer's :func:`evaluate_resource`. Nothing in the type system forces them
# to agree, and a disagreement is not symmetric -- a pushdown that accepts a
# resource the matcher would reject **grants a mutation the condition forbids**.
#
# So every comparator is checked against every tag state here, and the state that
# matters most is the third one: a resource with *no* such tag. D20 makes absence
# fail on the target side, so ``!=`` and ``NOT IN`` must EXCLUDE an untagged
# resource. That is the reading a hand-written SQL predicate gets wrong by
# default, because ``NOT (value = 'x')`` over a join quietly drops to "no row
# matched" and lets the untagged resource through.

import tempfile
from types import SimpleNamespace

import pytest
from sqlalchemy import sql

from mlflow.entities import RunTag, ViewType
from mlflow.server import auth as auth_module
from mlflow.server.auth import resources as auth_resources
from mlflow.server.auth.conditions import (
    NAMESPACE_RESOURCE,
    Clause,
    MutationConditionSpec,
    evaluate_resource,
    parse_condition,
)
from mlflow.server.auth.resources import version_resource_id
from mlflow.store.model_registry.sqlalchemy_store import (
    SqlAlchemyStore as RegistrySqlAlchemyStore,
)
from mlflow.store.tracking.dbmodels.models import SqlTag
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

# Both halves of the refusal: a SQL store says "cannot express a target condition on
# <entity>", the abstract default says "cannot answer a target condition". Matching either
# keeps these assertions from passing on an unrelated NotImplementedError.
_CANNOT_ANSWER = r"cannot (express|answer) a target condition"

TAG_KEY = "lifecycle"

# MCP names are reverse-DNS and so contain "/" -- the separator
# ``version_resource_id`` percent-encodes. Using realistic names here is what makes
# the composite-id tests meaningful: a store that parsed a joined id would split
# these in the wrong place.
PROD_SERVER = "com.example/svc-prod"
DEV_SERVER = "com.example/svc-dev"
BARE_SERVER = "com.example/svc-bare"

# (comparator, pushdown value, the equivalent condition filter text)
COMPARATORS = [
    ("=", "prod", f"tags.{TAG_KEY} = 'prod'"),
    ("!=", "prod", f"tags.{TAG_KEY} != 'prod'"),
    ("LIKE", "pro%", f"tags.{TAG_KEY} LIKE 'pro%'"),
    ("ILIKE", "PRO%", f"tags.{TAG_KEY} ILIKE 'PRO%'"),
    ("IN", ("prod", "staging"), f"tags.{TAG_KEY} IN ('prod','staging')"),
    ("NOT IN", ("prod", "staging"), f"tags.{TAG_KEY} NOT IN ('prod','staging')"),
]


@pytest.fixture
def store_with_runs(monkeypatch):
    """Three runs covering the tag states a condition can meet.

    ``untagged`` is the one that distinguishes the two readings of a negative
    comparator, so it is not an edge case here -- it is the point.
    """
    d = tempfile.mkdtemp()
    store = SqlAlchemyStore(f"sqlite:///{d}/mlflow.db", d)
    monkeypatch.setattr(auth_resources, "_tracking_store", lambda: store)
    experiment_id = store.create_experiment("parity")
    ids = {}
    for label, tag in (("matching", "prod"), ("other", "dev"), ("untagged", None)):
        run_id = store.create_run(experiment_id, "u", 0, [], label).info.run_id
        ids[label] = run_id
        if tag is not None:
            store.set_tag(run_id, RunTag(TAG_KEY, tag))
    return store, experiment_id, ids


def _store_with(monkeypatch, runs):
    """A fresh store holding one experiment with the given ``(label, tag)`` runs."""
    d = tempfile.mkdtemp()
    store = SqlAlchemyStore(f"sqlite:///{d}/mlflow.db", d)
    monkeypatch.setattr(auth_resources, "_tracking_store", lambda: store)
    experiment_id = store.create_experiment("cascade")
    for label, tag in runs:
        run_id = store.create_run(experiment_id, "u", 0, [], label).info.run_id
        if tag is not None:
            store.set_tag(run_id, RunTag(TAG_KEY, tag))
    return store, experiment_id


def _run_ids(store, experiment_id):
    """Every run id under the experiment, matching the enumerator's ViewType.ALL."""
    return [
        run.info.run_id
        for run in store.search_runs([experiment_id], None, ViewType.ALL, max_results=500)
    ]


def _in_memory(filter_text, ids):
    """Evaluate the condition the way the gate does when it cannot push down."""
    clauses = parse_condition(filter_text, NAMESPACE_RESOURCE)
    matched = set()
    for label, run_id in ids.items():
        auth_resources.clear_cache()
        if evaluate_resource(clauses, auth_resources.attrs_for("run", run_id)):
            matched.add(label)
    return matched


def _satisfying(store, entity, ids, clauses):
    """The subset of ``ids`` satisfying every clause, asked one id at a time.

    ``find_failing_resource`` answers "is there a failure here", which is all the gate
    ever asks of it. These parity tests are about agreement with
    :func:`evaluate_resource` *per resource*, so they ask per resource: a batch answer
    naming one failing id cannot distinguish "this one failed" from "these three
    failed", and that distinction is what parity means.
    """
    return {i for i in ids if store.find_failing_resource(entity, clauses, ids=[i]) is None}


@pytest.mark.parametrize(("comparator", "value", "filter_text"), COMPARATORS)
def test_pushdown_agrees_with_in_memory_evaluation(store_with_runs, comparator, value, filter_text):
    store, _, ids = store_with_runs
    by_id = {run_id: label for label, run_id in ids.items()}
    clauses = [("tags", TAG_KEY, comparator, value)]

    # A store that cannot push this down RAISES, so returning at all is the assertion:
    # there is no fallback that would quietly load every resource instead.
    store.find_failing_resource("run", clauses, ids=list(by_id))

    assert {by_id[i] for i in _satisfying(store, "run", list(by_id), clauses)} == _in_memory(
        filter_text, ids
    )


@pytest.mark.parametrize(("comparator", "value", "_filter_text"), COMPARATORS)
def test_an_untagged_resource_satisfies_no_comparator(
    store_with_runs, comparator, value, _filter_text
):
    """D20, stated directly rather than inferred from the parity above.

    Parity alone would still pass if *both* implementations wrongly included the
    untagged resource, so the absence rule is asserted on its own.
    """
    store, _, ids = store_with_runs
    failing = store.find_failing_resource(
        "run", [("tags", TAG_KEY, comparator, value)], ids=[ids["untagged"]]
    )
    assert failing == ids["untagged"], (
        f"{comparator} accepted a resource with no {TAG_KEY!r} tag; on the target side "
        "an absent tag must fail every comparator"
    )


def test_every_clause_must_hold(store_with_runs):
    # Clauses are conjunctive, matching ``combine``'s AND-only semantics.
    store, _, ids = store_with_runs
    clauses = [("tags", TAG_KEY, "=", "prod"), ("tags", TAG_KEY, "=", "dev")]
    assert _satisfying(store, "run", list(ids.values()), clauses) == set(), (
        "no run can hold two different values for one tag key"
    )


def test_neither_empty_input_can_fail(store_with_runs):
    """Nothing to judge, or nothing to judge it against, both mean nothing failed.

    Neither may be confused with "I cannot evaluate the clauses", which is a raise. The
    caller has no fallback, so anything returned means a verdict -- and ``None`` is the
    permissive one, which is why being unable to answer must not be expressible as a
    return value at all.
    """
    store, _, ids = store_with_runs
    assert store.find_failing_resource("run", [], ids=list(ids.values())) is None
    assert store.find_failing_resource("run", [("tags", TAG_KEY, "=", "prod")], ids=[]) is None


def test_an_id_list_larger_than_the_sql_parameter_cap_still_works(store_with_runs):
    """Ids bind one SQL parameter each, and backends cap how many a statement carries.

    Unchunked, SQLite raises "too many SQL variables" above ~32k -- and a caller has no
    way to know the cap, so the failure would surface as an opaque 500 on a large bulk
    delete rather than a refusal. Verified to fail before chunking was added.
    """
    store, _, ids = store_with_runs
    padded = list(ids.values()) + [f"absent-{i}" for i in range(60_000)]
    failing = store.find_failing_resource("run", [("tags", TAG_KEY, "!=", "prod")], ids=padded)
    # Unchunked this builds one statement with 60,003 bind parameters and raises before
    # returning anything, so the assertion that matters is that a verdict comes back at
    # all. An absent id fails like any other resource with no such tag, so one is found.
    assert failing is not None
    assert failing != ids["other"], "the dev-tagged run satisfies != 'prod' and must not fail"


# Every query in both stores is built via ``_get_query`` -- the hook a
#     workspace-aware subclass overrides to enforce tenant isolation. The registry
#     store filters ``model.workspace`` there directly; the open-source tracking
#     store returns a bare query and documents that "workspace-aware subclasses
#     override this to enforce scoping". Every existing MCP query honours it.
#
#     A pushdown that builds its own ``session.query`` therefore bypasses the one
#     place isolation is enforced, and could match rows no other query in the
#     process would see. The tag tables carry no ``workspace`` column today, so
#     nothing leaks yet -- but the MCP and registry tables do, and a subclass that
#     scopes a tag table by joining would be silently skipped here.
#


def test_the_filter_honours_an_overridden_get_query(store_with_runs, monkeypatch):
    store, _, ids = store_with_runs
    original = store._get_query
    # Patched only after setup, so creating the runs still works. A subclass
    # excluding everything stands in for one scoping to another workspace:
    # if the pushdown routes through the hook it must now match nothing.
    monkeypatch.setattr(
        store, "_get_query", lambda session, model: original(session, model).filter(sql.false())
    )
    # With every tag row hidden nothing satisfies ``= 'prod'``, so every id fails.
    assert store.find_failing_resource(
        "run", [("tags", TAG_KEY, "=", "prod")], ids=list(ids.values())
    ) in set(ids.values()), (
        "the pushdown must build its query through _get_query; bypassing it ignores "
        "whatever scoping a workspace-aware subclass applies"
    )


def test_the_cascade_honours_an_overridden_get_query(store_with_runs, monkeypatch):
    store, experiment_id, _ = store_with_runs
    original = store._get_query
    monkeypatch.setattr(
        store, "_get_query", lambda session, model: original(session, model).filter(sql.false())
    )
    # With every row hidden the cascade sees no children, which is vacuously
    # permitted -- the same answer as a genuinely empty experiment. Asserting
    # False here is asserting the hook was consulted, not that nothing failed.
    assert (
        store.find_failing_resource(
            "run", [("tags", TAG_KEY, "!=", "prod")], parent_id=experiment_id
        )
        is None
    )


def test_the_cascade_scopes_the_satisfying_set_too(monkeypatch):
    """Scoping only the outer half is the fail-OPEN direction, so it needs its own test.

    The all-hidden test above cannot catch it: with children hidden the outer
    query returns nothing and the cascade permits, whatever the subquery did.
    Here the children stay visible and only tag rows are hidden, which splits
    the two readings -- and the correct one denies.
    """
    store, experiment_id = _store_with(monkeypatch, [("a", "dev"), ("b", "dev")])
    original = store._get_query

    def scoped(session, model):
        query = original(session, model)
        return query.filter(sql.false()) if model is SqlTag else query

    monkeypatch.setattr(store, "_get_query", scoped)
    # Honoured: no tag row is visible, so no child satisfies ``!= 'prod'`` and
    # every child fails -> deny. Bypassed: the dev tags are read out of scope,
    # every child satisfies -> permit, granting a write on another tenant's state.
    failing = store.find_failing_resource(
        "run", [("tags", TAG_KEY, "!=", "prod")], parent_id=experiment_id
    )
    assert failing is not None


def test_an_unknown_entity_raises_rather_than_answering(store_with_runs):
    """An unmapped entity must raise, never return a verdict.

    Returning an id would deny a mutation it never judged; returning ``None`` would
    permit every one. With no fallback left, the second is the dangerous direction -- so
    being unable to answer is an exception rather than any value a caller might read as
    a pass.
    """
    store, _, ids = store_with_runs
    with pytest.raises(NotImplementedError, match=_CANNOT_ANSWER):
        store.find_failing_resource(
            "not_an_entity", [("tags", TAG_KEY, "=", "prod")], ids=list(ids.values())
        )


def _matches_in_memory(value, comparator, operand):
    """The in-memory semantic the pushdown must agree with, stated independently.

    Deliberately not a call into the condition evaluator: if both sides of a
    parity test route through the same code, the test proves only that the code
    equals itself. Absence returns ``False`` for every comparator (D20).
    """
    if value is None:
        return False
    if comparator == "=":
        return value == operand
    if comparator == "!=":
        return value != operand
    if comparator == "LIKE":
        return value.startswith(operand.rstrip("%"))
    if comparator == "ILIKE":
        return value.lower().startswith(operand.rstrip("%").lower())
    if comparator == "IN":
        return value in operand
    if comparator == "NOT IN":
        return value not in operand
    raise AssertionError(f"unhandled comparator {comparator}")


# Deleting a parent must judge every child, without enumerating them.
#
#     The expensive shape this replaces: enumerate every child id, fetch each
#     child, project its tags, evaluate. That is what forced a cap on how many
#     children a conditioned delete could consider at all -- an experiment with
#     more children than the cap could not be deleted once any condition existed
#     on those types, even one scoped elsewhere, because the refusal was about
#     enumerability rather than about matching.
#
#     The question is only ever "does ANY child fail", so it needs no enumeration
#     and no tag values: one ``LIMIT 1`` query over a ``NOT IN (satisfies)``
#     subquery. Cost stops depending on child count.
#


def _answer(store, experiment_id, comparator="!=", value="prod"):
    """The failing child's id, or ``None`` if every child satisfies."""
    return store.find_failing_resource(
        "run", [("tags", TAG_KEY, comparator, value)], parent_id=experiment_id
    )


def test_a_failing_child_is_found(store_with_runs):
    """The fixture holds a prod run and an untagged one, both failing.

    The id comes back rather than a bare ``True``, which is what lets a cascade
    denial say which child blocked it.
    """
    store, experiment_id, ids = store_with_runs
    assert _answer(store, experiment_id) in {ids["matching"], ids["untagged"]}


def test_all_satisfying_children_pass(monkeypatch):
    store, experiment_id = _store_with(monkeypatch, [("a", "dev"), ("b", "dev")])
    assert _answer(store, experiment_id) is None


def test_an_untagged_child_fails(monkeypatch):
    """D20 again, and the case a complement-based query gets wrong.

    Searching for children that *violate* the filter cannot work, because
    the complement of ``!= 'prod'`` is not ``= 'prod'`` when absence fails
    on both. An untagged child satisfies neither and must still deny.
    """
    store, experiment_id = _store_with(monkeypatch, [("a", "dev"), ("b", None)])
    assert _answer(store, experiment_id) is not None


def test_a_parent_with_no_children_passes(monkeypatch):
    # Vacuous, and distinct from "could not enumerate", which denied.
    store, experiment_id = _store_with(monkeypatch, [])
    assert _answer(store, experiment_id) is None


def test_no_clauses_cannot_fail(store_with_runs):
    store, experiment_id, _ = store_with_runs
    assert store.find_failing_resource("run", [], parent_id=experiment_id) is None


def test_a_child_failing_only_the_second_clause_is_found(monkeypatch):
    # Clauses are conjunctive, so failing any one of them fails the child.
    store, experiment_id = _store_with(monkeypatch, [("a", "dev")])
    assert (
        store.find_failing_resource(
            "run",
            [("tags", TAG_KEY, "!=", "prod"), ("tags", TAG_KEY, "=", "prod")],
            parent_id=experiment_id,
        )
        is not None
    )


def test_a_sibling_parents_children_are_not_considered(monkeypatch):
    """The query must be scoped to the parent, or one experiment's
    protected run would block deleting an unrelated experiment.
    """
    store, experiment_id = _store_with(monkeypatch, [("a", "dev")])
    other = store.create_experiment("other")
    run_id = store.create_run(other, "u", 0, [], "p").info.run_id
    store.set_tag(run_id, RunTag(TAG_KEY, "prod"))
    assert _answer(store, experiment_id) is None
    assert _answer(store, other) == run_id


def test_an_unknown_entity_declines(store_with_runs):
    store, experiment_id, _ = store_with_runs
    with pytest.raises(NotImplementedError, match=_CANNOT_ANSWER):
        store.find_failing_resource(
            "not_an_entity", [("tags", TAG_KEY, "=", "prod")], parent_id=experiment_id
        )


@pytest.mark.parametrize(("comparator", "value", "filter_text"), COMPARATORS)
def test_the_answer_matches_judging_each_child_individually(
    monkeypatch, comparator, value, filter_text
):
    """Parity with the enumerate-and-evaluate path this replaces.

    Parametrized over every comparator, which it previously was not: the id
    selector's parity test covered all six while this covered only ``!=``, leaving
    the cascade's SQL translation of the other five unchecked against the evaluator
    it must agree with. That asymmetry matters more now, since the cascade is the
    only selector whose comparator semantics live in SQL at all.
    """
    clauses = parse_condition(filter_text, NAMESPACE_RESOURCE)
    for runs in (
        [("a", "dev"), ("b", "dev")],
        [("a", "dev"), ("b", "prod")],
        [("a", "dev"), ("b", None)],
        [("a", None)],
        [],
    ):
        store, experiment_id = _store_with(monkeypatch, runs)
        individually = False
        for run_id in _run_ids(store, experiment_id):
            auth_resources.clear_cache()
            if not evaluate_resource(clauses, auth_resources.attrs_for("run", run_id)):
                individually = True
                break
        pushed = _answer(store, experiment_id, comparator, value)
        assert (pushed is not None) is individually, (runs, comparator)


# The gate must prefer pushdown, not merely tolerate it.
#
#     The store is the only evaluator now, so what is pinned here is narrower than it was
#     but no less important: the gate ASKS, and trusts the reply. Deleting the pushdown call
#     would once have been masked by the enumeration fallback answering every case; now it
#     would fail loudly, which is the point of having removed the fallback.
#


@staticmethod
def _gate(monkeypatch, pushdown_answer):
    from mlflow.server import auth as auth_module
    from mlflow.server.auth.conditions import ConditionScope, context_for

    condition = f"tags.{TAG_KEY} != 'prod'"

    class Store:
        def get_user(self, username):
            return SimpleNamespace(id=1, username=username, is_admin=False)

        def is_workspace_admin(self, user_id, workspace):
            return False

        def list_mutation_conditions_for_user(
            self, user_id, workspace, resource_types, parents=None
        ):
            return [
                MutationConditionSpec(
                    resource_type="run",
                    value_condition=None,
                    target_condition=condition,
                    resource_pattern="*",
                )
            ]

    monkeypatch.setattr(auth_module, "store", Store())
    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(find_failing_resource=lambda *a, **k: pushdown_answer),
    )

    # No ids and a parent IS the cascade shape; there is no resolver to supply and
    # nothing the gate could enumerate even if it wanted to.
    context = context_for(
        "run",
        None,
        ConditionScope.MUTATE,
        parent_resource_id="e-1",
    )
    return auth_module.authorize_on_conditions("alice", "w", [context])


def test_a_reported_failure_denies(monkeypatch):
    assert _gate(monkeypatch, "child-9") is False


def test_a_reported_pass_permits(monkeypatch):
    assert _gate(monkeypatch, None) is True


# A clause the store cannot express must take the whole row in memory.
#
#     Both resource namespaces now push, so the conversion is expected to succeed
#     for ``tags.*`` and ``aliases.*`` alike -- the earlier version of this class
#     asserted that an alias clause *declined*, which was true only while pushdown
#     was tag-only.
#
#     The invariant it protects is unchanged, and is the reason the helper is
#     asserted directly rather than end to end: a clause that cannot be pushed must
#     decline the **whole row**, never have its understood half pushed, because a
#     conjunction judged on a subset of itself is the fail-open direction -- the
#     dropped clause is exactly the one that would have denied.
#


@staticmethod
def _pushed(filter_text):
    from mlflow.server.auth import _pushable_clauses

    return _pushable_clauses(parse_condition(filter_text, NAMESPACE_RESOURCE))


def test_tag_clauses_convert():
    assert _pushed(f"tags.{TAG_KEY} != 'prod'") == [("tags", TAG_KEY, "!=", "prod")]


def test_several_tag_clauses_convert_in_order():
    pushed = _pushed(f"tags.{TAG_KEY} != 'prod' AND tags.team = 'ml'")
    assert pushed == [("tags", TAG_KEY, "!=", "prod"), ("tags", "team", "=", "ml")]


def test_an_alias_clause_converts():
    assert _pushed("aliases.production = 'yes'") == [("aliases", "production", "=", "yes")]


def test_a_row_mixing_tags_and_aliases_converts_whole():
    # Both halves in one call, so the conjunction is never split.
    pushed = _pushed(f"tags.{TAG_KEY} != 'prod' AND aliases.production = 'yes'")
    assert pushed == [
        ("tags", TAG_KEY, "!=", "prod"),
        ("aliases", "production", "=", "yes"),
    ]


def test_an_unrecognised_identifier_declines_the_whole_row():
    """The guard that outlives any particular namespace list.

    Constructed directly rather than parsed, because the parser rejects an
    unknown prefix at authoring time -- so this pins the helper's own refusal
    rather than relying on the parser never changing.
    """
    from mlflow.server.auth import _pushable_clauses

    rows = [
        Clause(identifier="tags", key=TAG_KEY, comparator="!=", value="prod"),
        Clause(identifier="something_new", key="x", comparator="=", value="y"),
    ]
    assert _pushable_clauses(rows) is None


def test_a_clause_with_no_key_declines():
    # A flat request-style clause has ``key is None`` and names no column.
    from mlflow.server.auth import _pushable_clauses

    rows = [Clause(identifier="tags", key=None, comparator="=", value="v")]
    assert _pushable_clauses(rows) is None


# MCP tags, the alias namespace, and a composite-keyed version id.
#
#     These are the cases the first pass could not answer: it covered four tracking
#     tag tables and declined everything else, so an alias condition or an MCP
#     condition silently took the in-memory path. Coverage is the point of the
#     unified clause shape, so each namespace is exercised against a real store.
#


@pytest.fixture
def mcp_store(monkeypatch):
    d = tempfile.mkdtemp()
    store = SqlAlchemyStore(f"sqlite:///{d}/mlflow.db", d)
    monkeypatch.setattr(auth_resources, "_tracking_store", lambda: store)
    for name, tag in (
        (PROD_SERVER, "prod"),
        (DEV_SERVER, "dev"),
        (BARE_SERVER, None),
    ):
        store.create_mcp_server(name)
        store.create_mcp_server_version({"name": name, "version": "1.0.0"})
        if tag is not None:
            store.set_mcp_server_tag(name, TAG_KEY, tag)
            store.set_mcp_server_version_tag(name, "1.0.0", TAG_KEY, tag)
    store.set_mcp_server_alias(PROD_SERVER, "champion", "1.0.0")
    return store


ALL_SERVERS = [PROD_SERVER, DEV_SERVER, BARE_SERVER]


def test_mcp_server_tags_push_down(mcp_store):
    assert _satisfying(mcp_store, "mcp_server", ALL_SERVERS, [("tags", TAG_KEY, "=", "prod")]) == {
        PROD_SERVER
    }


def test_an_untagged_mcp_server_satisfies_no_negative_comparator(mcp_store):
    # D20 again, on a type whose id is a name rather than a uuid.
    assert _satisfying(mcp_store, "mcp_server", ALL_SERVERS, [("tags", TAG_KEY, "!=", "prod")]) == {
        DEV_SERVER
    }


def test_mcp_server_aliases_push_down(mcp_store):
    """The alias namespace, which the first pass declined outright.

    An alias row is ``(name, alias, version)``, so the clause key is the alias
    name and the compared value is the version it points at.
    """
    assert _satisfying(
        mcp_store, "mcp_server", ALL_SERVERS, [("aliases", "champion", "=", "1.0.0")]
    ) == {PROD_SERVER}


def test_an_absent_alias_satisfies_nothing(mcp_store):
    assert _satisfying(
        mcp_store, "mcp_server", ALL_SERVERS, [("aliases", "champion", "!=", "9.9.9")]
    ) == {PROD_SERVER}


def test_tags_and_aliases_are_conjunctive_in_one_call(mcp_store):
    # The reason both namespaces share a call rather than two methods.
    both_hold = [("tags", TAG_KEY, "=", "prod"), ("aliases", "champion", "=", "1.0.0")]
    assert _satisfying(mcp_store, "mcp_server", ALL_SERVERS, both_hold) == {PROD_SERVER}
    one_fails = [("tags", TAG_KEY, "=", "dev"), ("aliases", "champion", "=", "1.0.0")]
    assert _satisfying(mcp_store, "mcp_server", ALL_SERVERS, one_fails) == set()


def test_an_mcp_version_is_matched_by_its_decomposed_id(mcp_store):
    # A composite id arrives as parts, so the store never parses ``name/version``.
    ids = [(name, "1.0.0") for name in ALL_SERVERS]
    assert _satisfying(mcp_store, "mcp_server_version", ids, [("tags", TAG_KEY, "=", "prod")]) == {
        (PROD_SERVER, "1.0.0")
    }


def test_an_mcp_version_id_must_match_both_parts(mcp_store):
    # The half-match a plain ``IN`` on the name column would wrongly accept.
    assert (
        _satisfying(
            mcp_store,
            "mcp_server_version",
            [(PROD_SERVER, "2.0.0")],
            [("tags", TAG_KEY, "=", "prod")],
        )
        == set()
    )


def test_an_alias_clause_on_an_mcp_version_declines(mcp_store):
    # D18: a version's aliases live on its parent, so it exposes no alias table.
    with pytest.raises(NotImplementedError, match=_CANNOT_ANSWER):
        mcp_store.find_failing_resource(
            "mcp_server_version",
            [("aliases", "champion", "=", "1.0.0")],
            ids=[(PROD_SERVER, "1.0.0")],
        )


# Asserted on the store method, because nothing reaches this end to end.
#
#     Cascade entities are experiment children -- runs, traces, logged models --
#     and none of those own aliases (D18), so no authored condition puts an alias
#     clause on a cascading delete today.
#
#     It still needs a test. Widening ``_pushable_clauses`` to convert the alias
#     namespace means the cascade now *receives* a clause shape it never saw while
#     pushdown was tag-only, and the guard that refuses it was confirmed unprotected
#     by mutation: replacing its ``return None`` with ``continue`` left all 45 tests
#     passing. Silently dropping the clause is the fail-open direction.
#


def test_an_alias_clause_declines_rather_than_being_dropped(store_with_runs):
    store, experiment_id, _ = store_with_runs
    with pytest.raises(NotImplementedError, match=_CANNOT_ANSWER):
        # An unexpressible clause must refuse the call, not be skipped: a skipped
        # clause is a conjunction judged on a subset of itself.
        store.find_failing_resource(
            "run", [("aliases", "champion", "=", "1.0.0")], parent_id=experiment_id
        )


def test_a_mixed_row_declines_whole_rather_than_pushing_its_tag_half(store_with_runs):
    """The dangerous case: the tag half alone would answer, and answer wrongly.

    Every run here is dev-tagged, so the tag clause alone permits. Dropping the
    alias clause would turn a conjunction nothing can satisfy into a pass.
    """
    store, experiment_id, _ = store_with_runs
    with pytest.raises(NotImplementedError, match=_CANNOT_ANSWER):
        store.find_failing_resource(
            "run",
            [("tags", TAG_KEY, "!=", "nothing"), ("aliases", "champion", "=", "x")],
            parent_id=experiment_id,
        )


REGISTRY_COMPARATORS = [
    ("=", "prod"),
    ("!=", "prod"),
    ("LIKE", "pro%"),
    ("ILIKE", "PRO%"),
    ("IN", ("prod", "staging")),
    ("NOT IN", ("prod", "staging")),
]


# The second store. Parity is re-proved here rather than inherited.
#
#     The pushdown_registry is a different store with its own tables, and the run-tag
#     projection bug (`5d2ea6472`) is the standing example of a projection silently
#     returning less than it appears to. So every comparator is checked against the
#     in-memory evaluator again rather than assumed from the tracking side.
#
#     Two pushdown_registry-specific hazards the tracking tables do not have:
#
#     - ``version`` is an ``INTEGER`` here, in both the alias table and the version
#       tag table, while every condition value is a string. Uncast, ``1`` would be
#       compared against ``'1'`` -- matching nothing, which on the resource side
#       denies every mutation of the type.
#     - A prompt *is* a registered model (T12.9): there are no prompt tables, so the
#       ``prompt`` and ``registered_model`` types resolve to the same rows.
#


@pytest.fixture
def pushdown_registry(monkeypatch):
    from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
    from mlflow.store.model_registry.sqlalchemy_store import (
        SqlAlchemyStore as RegistrySqlAlchemyStore,
    )

    d = tempfile.mkdtemp()
    store = RegistrySqlAlchemyStore(f"sqlite:///{d}/pushdown_registry.db")
    for name, tag in (("m-prod", "prod"), ("m-dev", "dev"), ("m-bare", None)):
        store.create_registered_model(name)
        store.create_model_version(name, source="s")
        if tag is not None:
            store.set_registered_model_tag(name, RegisteredModelTag(TAG_KEY, tag))
            store.set_model_version_tag(name, "1", ModelVersionTag(TAG_KEY, tag))
    store.set_registered_model_alias("m-prod", "champion", "1")
    return store


ALL_MODELS = ["m-prod", "m-dev", "m-bare"]


@pytest.mark.parametrize(("comparator", "value"), REGISTRY_COMPARATORS)
def test_every_comparator_agrees_with_in_memory_on_registry_pushdown(
    pushdown_registry, comparator, value
):
    # The parity contract, re-proved on the pushdown_registry's own tables.
    clauses = [("tags", TAG_KEY, comparator, value)]
    # The pushdown_registry store must push this down; it would RAISE if it could not.
    pushdown_registry.find_failing_resource("registered_model", clauses, ids=ALL_MODELS)
    pushed = _satisfying(pushdown_registry, "registered_model", ALL_MODELS, clauses)

    tags = {"m-prod": {TAG_KEY: "prod"}, "m-dev": {TAG_KEY: "dev"}, "m-bare": {}}
    expected = {
        name for name, t in tags.items() if _matches_in_memory(t.get(TAG_KEY), comparator, value)
    }
    assert pushed == expected


def test_an_untagged_model_satisfies_no_negative_comparator(pushdown_registry):
    # D20 on the pushdown_registry side.
    assert _satisfying(
        pushdown_registry, "registered_model", ALL_MODELS, [("tags", TAG_KEY, "!=", "prod")]
    ) == {"m-dev"}


def test_an_integer_version_compares_as_the_string_it_was_written_as(pushdown_registry):
    # The cast. Uncast this returns nothing and denies every mutation.
    assert _satisfying(
        pushdown_registry, "registered_model", ALL_MODELS, [("aliases", "champion", "=", "1")]
    ) == {"m-prod"}


def test_an_integer_version_supports_the_text_comparators_too(pushdown_registry):
    # ``LIKE`` on an INTEGER column only works because the cast makes it text.
    assert _satisfying(
        pushdown_registry, "registered_model", ALL_MODELS, [("aliases", "champion", "LIKE", "1%")]
    ) == {"m-prod"}


def test_a_prompt_resolves_to_the_same_rows_as_a_registered_model(pushdown_registry):
    """T12.9: there are no prompt tables, so both types read the same rows.

    That is a storage fact and nothing more. It does NOT mean both types\' conditions
    apply to one entry: ``_authorize_registry_entry`` declares exactly ONE resource
    type per request, chosen by reading ``mlflow.prompt.is_prompt`` off the persisted
    entity, so only the declared type\'s conditions are ever loaded. A
    ``registered_model`` condition never governs a request classified as a prompt --
    verified black-box in the conformance suite, with a positive control. Shared
    storage is a pushdown-layer detail the gate never lets matter.
    """
    clause = [("tags", TAG_KEY, "=", "prod")]
    assert _satisfying(pushdown_registry, "prompt", ALL_MODELS, clause) == _satisfying(
        pushdown_registry, "registered_model", ALL_MODELS, clause
    )


def test_a_model_version_is_matched_by_its_decomposed_id(pushdown_registry):
    ids = [(name, "1") for name in ALL_MODELS]
    assert _satisfying(
        pushdown_registry, "registered_model_version", ids, [("tags", TAG_KEY, "=", "prod")]
    ) == {("m-prod", "1")}


def test_a_model_version_id_must_match_both_parts(pushdown_registry):
    assert (
        _satisfying(
            pushdown_registry,
            "registered_model_version",
            [("m-prod", "2")],
            [("tags", TAG_KEY, "=", "prod")],
        )
        == set()
    )


def test_an_alias_clause_on_a_model_version_declines(pushdown_registry):
    # D18: a version's aliases belong to its parent, so it exposes no alias table.
    with pytest.raises(NotImplementedError, match=_CANNOT_ANSWER):
        pushdown_registry.find_failing_resource(
            "registered_model_version",
            [("aliases", "champion", "=", "1")],
            ids=[("m-prod", "1")],
        )


def test_an_unmapped_entity_declines(pushdown_registry):
    # A tracking type must not be answered from pushdown_registry tables.
    with pytest.raises(NotImplementedError, match=_CANNOT_ANSWER):
        pushdown_registry.find_failing_resource("run", [("tags", TAG_KEY, "=", "x")], ids=["r1"])


# The cast must be asserted on the SQL, because SQLite cannot catch its absence.
#
#     SQLite is dynamically typed and happily evaluates ``1 = '1'`` as true, so every
#     behavioural test in this file passes with or without the cast -- confirmed by
#     mutation: deleting it left all 61 tests green. PostgreSQL does not coerce, so
#     the same code would compare an ``INTEGER`` column against a string, match
#     nothing, and -- because absence fails on the target side (D20) -- deny every
#     mutation of the type.
#
#     That failure mode is invisible to a correctness test on the development
#     backend and would only appear in production, so the guarantee is pinned on the
#     generated SQL instead. Same category as the cost assertions: the thing that
#     matters is not observable in the result.
#


def test_an_integer_column_is_cast_to_text():
    from mlflow.store import condition_pushdown
    from mlflow.store.model_registry.dbmodels.models import SqlRegisteredModelAlias

    compiled = str(condition_pushdown.comparable(SqlRegisteredModelAlias.version))
    assert "CAST" in compiled.upper(), (
        "an INTEGER version must be compared as text, or a strictly-typed "
        "backend matches nothing and denies every mutation of the type"
    )


def test_a_text_column_is_left_alone():
    # No gratuitous cast: it would defeat an index for no benefit.
    from mlflow.store import condition_pushdown
    from mlflow.store.model_registry.dbmodels.models import SqlRegisteredModelTag

    assert condition_pushdown.comparable(SqlRegisteredModelTag.value) is (
        SqlRegisteredModelTag.value
    )


def test_a_composite_id_predicate_casts_its_integer_part():
    # The id side needs it too, not just the compared value.
    from mlflow.store import condition_pushdown
    from mlflow.store.model_registry.dbmodels.models import SqlModelVersionTag

    predicate = condition_pushdown.id_predicate(
        (SqlModelVersionTag.name, SqlModelVersionTag.version), [("m", "1")]
    )
    assert "CAST" in str(predicate).upper()


def test_the_tracking_tag_tables_need_no_cast():
    # Every tracking tag value is already text, so nothing is wrapped there.
    from mlflow.store import condition_pushdown
    from mlflow.store.tracking.dbmodels.models import SqlTag

    assert condition_pushdown.comparable(SqlTag.value) is SqlTag.value


# The batch path: the request names its ids, so the store filters them.
#
#     Same reasoning as the cascade tests: the gate must ASK the store and trust the reply.
#     There is no longer a bulk loader to fall back to -- deleting the pushdown call would
#     now fail loudly rather than being masked by it -- so "without loading" is structural
#     rather than something these tests have to police.
#

CONDITION = f"tags.{TAG_KEY} != 'prod'"


@staticmethod
def _gate_for_explicit_ids(
    monkeypatch, *, fails, resource_type="run", ids=("r-1", "r-2"), rows=None
):
    from mlflow.server import auth as auth_module
    from mlflow.server.auth.conditions import ConditionContext, ConditionScope

    target_rows = rows or [CONDITION]

    class Store:
        def get_user(self, username):
            return SimpleNamespace(id=1, username=username, is_admin=False)

        def is_workspace_admin(self, user_id, workspace):
            return False

        def list_mutation_conditions_for_user(self, user_id, workspace, types, parents=None):
            return [
                MutationConditionSpec(
                    resource_type=resource_type,
                    value_condition=None,
                    target_condition=c,
                    resource_pattern="*",
                )
                for c in target_rows
            ]

    monkeypatch.setattr(auth_module, "store", Store())
    seen = {}

    def _filter(entity, clauses, *, ids=None, parent_id=None):
        seen["entity"] = entity
        seen["ids"] = list(ids or ())
        seen["clauses"] = list(clauses)
        return fails

    fake = SimpleNamespace(find_failing_resource=_filter)
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: fake)
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: fake, raising=False)

    context = ConditionContext(
        resource_type=resource_type,
        scope=ConditionScope.MUTATE,
        request=None,
        resource_ids=tuple(ids),
        parent_resource_id="e-1",
    )
    return auth_module.authorize_on_conditions("alice", "w", [context]), seen


def test_a_fully_satisfied_set_permits_without_loading(monkeypatch):
    allowed, seen = _gate_for_explicit_ids(monkeypatch, fails=None)
    assert allowed is True
    assert seen["ids"] == ["r-1", "r-2"]


def test_an_unsatisfied_id_denies_without_loading(monkeypatch):
    allowed, _ = _gate_for_explicit_ids(monkeypatch, fails="r-2")
    assert allowed is False, "an id the store reported as failing must deny"


def test_an_absent_id_denies_like_a_failed_clause(monkeypatch):
    # Indistinguishable by design: a 404 would reveal which ids exist.
    allowed, _ = _gate_for_explicit_ids(monkeypatch, fails="r-1")
    assert allowed is False


def test_a_composite_id_is_pushed_as_parts(monkeypatch):
    """The store must never receive the auth layer's joined ``name/version``.

    The composite format is this layer's invention -- it percent-encodes the
    name because a name may itself contain ``/`` -- so the store is handed the
    decomposed parts and the gate recomposes to compare.
    """
    joined = version_resource_id("com.example/svc", "1.0.0")
    allowed, seen = _gate_for_explicit_ids(
        monkeypatch,
        fails=None,
        resource_type="registered_model_version",
        ids=(joined,),
    )
    assert allowed is True
    assert seen["ids"] == [("com.example/svc", "1.0.0")], (
        "a composite id must reach the store as parts, not as a joined string"
    )


def test_one_unpushable_row_falls_back_for_the_whole_context(monkeypatch):
    # A conjunction must not be answered from the half the store understood.
    from mlflow.server import auth as auth_module

    pushed = {"called": False}

    def _filter(*a, **k):
        pushed["called"] = True
        return None

    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(find_failing_resource=_filter),
    )
    from mlflow.server.auth import _pushable_clauses
    from mlflow.server.auth.conditions import Clause

    assert (
        _pushable_clauses([Clause(identifier="bogus", key="k", comparator="=", value="v")]) is None
    )


# A registry type must be asked of the registry store, not the tracking one.
#
#     Found by mutation: collapsing ``_condition_store`` to always return the
#     tracking store changed nothing, because the harness above points both getters
#     at one fake. Asking the wrong store is not a correctness hole -- an unmapped
#     entity declines rather than answering from the wrong table -- but it silently
#     loses the pushdown, which is the kind of regression that shows up as a
#     performance report months later rather than as a failure.
#


@staticmethod
def _which_store_was_asked(monkeypatch, resource_type, resource_id):
    from mlflow.server import auth as auth_module
    from mlflow.server.auth.conditions import ConditionContext, ConditionScope

    class Store:
        def get_user(self, username):
            return SimpleNamespace(id=1, username=username, is_admin=False)

        def is_workspace_admin(self, user_id, workspace):
            return False

        def list_mutation_conditions_for_user(self, user_id, workspace, types, parents=None):
            return [
                MutationConditionSpec(
                    resource_type=resource_type,
                    value_condition=None,
                    target_condition=f"tags.{TAG_KEY} != 'prod'",
                    resource_pattern="*",
                )
            ]

    monkeypatch.setattr(auth_module, "store", Store())
    asked = []

    def _store(label):
        return SimpleNamespace(
            find_failing_resource=lambda entity, clauses, **k: asked.append(label),
        )

    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: _store("tracking"))
    monkeypatch.setattr(
        auth_module, "_get_model_registry_store", lambda: _store("registry"), raising=False
    )
    context = ConditionContext(
        resource_type=resource_type,
        scope=ConditionScope.MUTATE,
        request=None,
        resource_ids=(resource_id,),
        parent_resource_id="p-1",
    )
    assert auth_module.authorize_on_conditions("alice", "w", [context]) is True
    return asked


def test_a_registry_entry_asks_the_registry_store(monkeypatch):
    assert _which_store_was_asked(monkeypatch, "registered_model", "m-1") == ["registry"]


def test_a_prompt_asks_the_registry_store(monkeypatch):
    # A prompt is stored as a registered model, so it is the registry's to answer.
    assert _which_store_was_asked(monkeypatch, "prompt", "p-1") == ["registry"]


def test_a_run_asks_the_tracking_store(monkeypatch):
    assert _which_store_was_asked(monkeypatch, "run", "r-1") == ["tracking"]


def test_an_mcp_server_asks_the_tracking_store(monkeypatch):
    # MCP lives in the tracking store despite being a registry in name.
    assert _which_store_was_asked(monkeypatch, "mcp_server", "com.example/s") == ["tracking"]


# One row the converter cannot push must refuse the WHOLE context.
#
#     Found by mutation: turning the row loop's early exit into ``continue`` changed
#     nothing, because every authored condition converts -- the parser rejects an unknown
#     identifier at authoring time, so this is unreachable from a stored condition. The
#     converter is therefore stubbed to refuse one row, which is the only way to exercise
#     the loop's own contract.
#
#     It used to fall back and evaluate every row in memory. With no fallback left it must
#     raise instead -- and critically, no row may be pushed first: the rows are a
#     conjunction, and pushing the half the store understood would judge that conjunction
#     against a subset of itself, which is the fail-open direction.
#


def test_an_unpushable_row_refuses_before_pushing_any(monkeypatch):
    from mlflow.server import auth as auth_module
    from mlflow.server.auth.conditions import ConditionContext, ConditionScope

    pushable = f"tags.{TAG_KEY} != 'prod'"
    unpushable = "tags.team = 'ml'"

    class Store:
        def get_user(self, username):
            return SimpleNamespace(id=1, username=username, is_admin=False)

        def is_workspace_admin(self, user_id, workspace):
            return False

        def list_mutation_conditions_for_user(self, user_id, workspace, types, parents=None):
            return [
                MutationConditionSpec(
                    resource_type="run",
                    value_condition=None,
                    target_condition=c,
                    resource_pattern="*",
                )
                for c in (pushable, unpushable)
            ]

    monkeypatch.setattr(auth_module, "store", Store())
    real = auth_module._pushable_clauses

    def _selective(clauses):
        if any(c.key == "team" for c in clauses):
            return None
        return real(clauses)

    monkeypatch.setattr(auth_module, "_pushable_clauses", _selective)

    filtered = []
    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(
            find_failing_resource=lambda e, c, **k: filtered.append(c),
        ),
    )
    context = ConditionContext(
        resource_type="run",
        scope=ConditionScope.MUTATE,
        request=None,
        resource_ids=("r-1",),
        parent_resource_id="e-1",
    )
    with pytest.raises(NotImplementedError, match=r"cannot be pushed down"):
        auth_module.authorize_on_conditions("alice", "w", [context])
    assert filtered == [], "no row may be pushed once one of them cannot be"


# The two scope guarantees that no behavioural test reaches end to end.
#
#     Both are asserted directly on the helpers. A behavioural test cannot see either: the
#     fallback path answers identically to the pushdown, and an over-broad scope match only
#     shows up as a *denial* on a resource the admin never named -- which no wired route
#     exercises, because every route in the suite names a single resource whose pattern
#     matches. Mutation testing found both escaping, which is what put them here.
#


@staticmethod
def _row(resource_pattern):
    return MutationConditionSpec(
        resource_type="run",
        value_condition=None,
        target_condition=f"tags.{TAG_KEY} != 'prod'",
        resource_pattern=resource_pattern,
    )


def test_an_id_pattern_governs_only_the_resource_it_names():
    """Charging it against a sibling would deny a mutation on a resource the admin
    never pointed the condition at -- over-restriction rather than a hole, but still
    not what was written.
    """
    row = _row("abc")
    assert auth_module._row_governs(row, "abc") is True
    assert auth_module._row_governs(row, "xyz") is False
    # Nothing to judge against: a create, or a cascade before resolution.
    assert auth_module._row_governs(row, None) is False


def test_a_wildcard_pattern_governs_every_resource_including_unnamed():
    # The pre-scope behaviour, and what makes restricting a create possible at all.
    row = _row("*")
    for asked in ("abc", "xyz", None):
        assert auth_module._row_governs(row, asked) is True


@staticmethod
def _cascade_pushdown(monkeypatch, rows):
    """Ask the cascade selector with these rows, recording whether the store was asked."""
    from mlflow.server.auth.conditions import ConditionScope, context_for

    asked = []
    fake = SimpleNamespace(
        find_failing_resource=lambda *a, **k: asked.append(k) or None,
    )
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: fake)
    context = context_for("run", None, ConditionScope.MUTATE, None, parent_resource_id="e-1")

    def ask():
        return auth_module._cascade_target_pushdown(context, rows, parent_id="e-1")

    # `asked` is returned separately from the call so a test can still inspect it when
    # the call raises -- an unpushable row must not reach the store at all.
    return ask, asked


def test_an_id_scoped_row_refuses_the_cascade_pushdown(monkeypatch):
    """The cascade sends ONE clause set covering ALL of a parent\'s children, which a row
    naming a single resource breaks: its clauses apply to one child and must not be
    charged against its siblings. There is no fallback to judge each child against the
    rows that govern it, so this must RAISE rather than quietly skip the row.

    Unreachable in practice: every cascade-reachable type (run, trace, logged model, and
    the three version types) is wildcard-only, so ``normalize_condition_scope`` refuses an
    id pattern for all six and such a row cannot be stored. Kept because a future tier
    with id-grain patterns would otherwise fail silently, in the fail-open direction.
    """
    ask, asked = _cascade_pushdown(monkeypatch, [_row("abc")])
    with pytest.raises(NotImplementedError, match=r"scoped"):
        ask()
    assert asked == [], "a scoped row must not be pushed at all, not pushed and ignored"


def test_an_unscoped_cascade_row_is_still_pushed(monkeypatch):
    """The positive control: the refusal must be keyed on the pattern, not on having rows.

    Without this, raising unconditionally would pass the test above while breaking
    every cascade.
    """
    ask, asked = _cascade_pushdown(monkeypatch, [_row("*")])
    assert ask() is None
    assert len(asked) == 1, "an unscoped row must reach the store"


def test_one_scoped_row_among_unscoped_ones_still_refuses(monkeypatch):
    # Rows are conjunctive, so one unpushable row refuses the whole context.
    ask, _ = _cascade_pushdown(monkeypatch, [_row("*"), _row("abc")])
    with pytest.raises(NotImplementedError, match=r"scoped"):
        ask()


# Every pushdown mapping must name columns that exist on its model.
#
#     Found by conformance testing, not by these unit tests: ``logged_model`` declared the tag
#     key/value columns as ``key``/``value``, but ``SqlLoggedModelTag`` calls them ``tag_key``
#     and ``tag_value``. The result was an ``AttributeError`` inside the gate, surfacing as a
#     **500 on every logged-model tag write** as soon as any logged-model condition existed --
#     a fail-closed-by-crash, not a denial.
#
#     The existing pushdown tests all build their own clauses against ``SqlTag``, whose columns
#     really are ``key``/``value``, so the table itself was never exercised per entity. These two
#     tests check the table rather than one path through it, so a future entry with a mistyped
#     column fails here instead of in production.
#


@pytest.mark.parametrize(
    "store",
    [SqlAlchemyStore, RegistrySqlAlchemyStore],
    ids=["tracking", "registry"],
)
def test_every_namespace_mapping_names_real_columns(store):
    problems = []
    for entity, namespaces in store._PUSHDOWN_NAMESPACES.items():
        for namespace, (model_name, id_names, key_name, value_name) in namespaces.items():
            model = store._PUSHDOWN_MODELS[model_name]
            for column in (*id_names, key_name, value_name):
                if not hasattr(model, column):
                    actual = [c.name for c in model.__table__.columns]
                    problems.append(
                        f"{entity}.{namespace}: {model_name} has no {column!r} "
                        f"(actual columns: {actual})"
                    )
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "store",
    [SqlAlchemyStore, RegistrySqlAlchemyStore],
    ids=["tracking", "registry"],
)
def test_every_cascade_entity_names_real_columns(store):
    """The cascade path does not consult ``_PUSHDOWN_NAMESPACES``.

    It has its own table, so fixing the namespace mapping alone left this path still
    broken for the same entity. Both are checked, independently, on both stores.
    """
    problems = []
    for entity, mapping in store._CASCADE_PUSHDOWN_ENTITIES.items():
        (
            child_model,
            child_id_names,
            parent_name,
            tag_model,
            tag_id_names,
            tag_key_name,
            tag_value_name,
            tag_parent_name,
        ) = mapping
        columns = [
            *((child_model, name) for name in child_id_names),
            (child_model, parent_name),
            *((tag_model, name) for name in tag_id_names),
            (tag_model, tag_key_name),
            (tag_model, tag_value_name),
        ]
        if tag_parent_name is not None:
            columns.append((tag_model, tag_parent_name))
        for model, column in columns:
            if not hasattr(model, column):
                actual = [c.name for c in model.__table__.columns]
                problems.append(
                    f"{entity}: {model.__name__} has no {column!r} (actual columns: {actual})"
                )
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "store",
    [SqlAlchemyStore, RegistrySqlAlchemyStore],
    ids=["tracking", "registry"],
)
def test_every_cascade_discriminator_is_well_formed(store):
    """The id columns minus the parent must leave a usable discriminator.

    ``find_failing_child`` derives the membership test from exactly this
    subtraction rather than from a declared column, so a mapping that subtracts
    to nothing -- or to a different width on the two sides -- would build a
    predicate that compares the wrong things. An empty discriminator is the worse
    case: it would compare nothing and acquit every child.
    """
    problems = []
    for entity, mapping in store._CASCADE_PUSHDOWN_ENTITIES.items():
        (_, child_id_names, parent_name, _, tag_id_names, _, _, tag_parent_name) = mapping
        child = [name for name in child_id_names if name != parent_name]
        tag = [name for name in tag_id_names if name != tag_parent_name]
        if not child or not tag:
            problems.append(
                f"{entity}: subtracting the parent leaves no discriminator "
                f"(child {child_id_names}-{parent_name!r}, "
                f"tag {tag_id_names}-{tag_parent_name!r})"
            )
        elif len(child) != len(tag):
            problems.append(f"{entity}: discriminator widths differ -- child {child}, tag {tag}")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "store",
    [SqlAlchemyStore, RegistrySqlAlchemyStore],
    ids=["tracking", "registry"],
)
def test_a_composite_keyed_child_scopes_its_tag_subquery(store):
    """A child whose id is not globally unique MUST declare the tag parent column.

    This is the invariant the whole version cascade turns on. Without it the
    satisfying set holds bare discriminators -- version numbers -- and a sibling
    parent's satisfying version acquits this parent's failing one, which is the
    fail-open direction. Derived from the mapping rather than trusted, so a future
    composite-keyed entity cannot be added without it.
    """
    problems = []
    for entity, mapping in store._CASCADE_PUSHDOWN_ENTITIES.items():
        (_, child_id_names, parent_name, _, tag_id_names, _, _, tag_parent_name) = mapping
        composite = len(child_id_names) > 1
        if composite and tag_parent_name is None:
            problems.append(
                f"{entity}: child id {child_id_names} is composite, so its "
                f"discriminator repeats across parents and the tag subquery must be "
                f"scoped -- but tag_parent_name is None"
            )
        if not composite and parent_name in child_id_names:
            problems.append(
                f"{entity}: single-column child id {child_id_names} must not be the "
                f"parent column {parent_name!r}, or the discriminator is empty"
            )
    assert not problems, "\n".join(problems)


# Exactly one selector, enforced loudly.
#
#     Neither selector is a safe default. Defaulting to ``ids=[]`` would answer "nothing
#     failed" for a cascade whose parent was never passed, permitting the whole cascade;
#     defaulting to a parent would judge the wrong population. A programming error here
#     is not a runtime condition, so it raises rather than declining.
#


@pytest.mark.parametrize("kwargs", [{}, {"ids": ["a"], "parent_id": "1"}], ids=["neither", "both"])
def test_exactly_one_selector_is_required(store_with_runs, kwargs):
    store, _, _ = store_with_runs
    with pytest.raises(ValueError, match="exactly one"):
        store.find_failing_resource("run", [], **kwargs)


def test_the_registry_enforces_it_too(tmp_path):
    registry = RegistrySqlAlchemyStore(f"sqlite:///{tmp_path}/registry.db")
    with pytest.raises(ValueError, match="exactly one"):
        registry.find_failing_resource("registered_model", [], ids=["m"], parent_id="p")


# A version tier's cascade is answered in SQL, like every other cascade.
#
#     These three tiers were the last to enumerate instead of pushing down, which is
#     why they alone could hit ``MAX_CASCADE_CHILDREN`` and refuse a delete outright
#     past 2000 versions. What kept them out was not the join but *identity*: the
#     satisfying subquery selects the child id, and one column names a child only
#     when it is globally unique. ``run_uuid``, ``request_id`` and ``model_id`` are;
#     a version is not -- ``name`` matches every version of the model and ``version``
#     matches version 3 of any model.
#
#     Scoping the subquery to the parent restores it, because ``version`` is unique
#     within one ``name``. That makes the parent filter on the *inner* half
#     load-bearing rather than redundant, which is what
#     ``test_the_cascade_judges_only_its_own_parents_versions`` exists to prove.
#


@pytest.fixture
def cascade_registry():
    from mlflow.entities.model_registry import ModelVersionTag

    d = tempfile.mkdtemp()
    store = RegistrySqlAlchemyStore(f"sqlite:///{d}/cascade_registry.db")
    # Version NUMBERS collide across models on purpose: every model here has a
    # version 1, and ``m-prod``'s disagrees with the others'. That collision is
    # what makes the subquery's parent filter observable.
    for name, tags in (
        ("m-prod", ["prod"]),
        ("m-dev", ["dev"]),
        ("m-mixed", ["dev", "prod"]),
        ("m-bare", [None]),
    ):
        store.create_registered_model(name)
        for version, tag in enumerate(tags, start=1):
            store.create_model_version(name, source="s")
            if tag is not None:
                store.set_model_version_tag(name, str(version), ModelVersionTag(TAG_KEY, tag))
    return store


PARENTS = {
    "m-prod": ["prod"],
    "m-dev": ["dev"],
    "m-mixed": ["dev", "prod"],
    "m-bare": [None],
}

SATISFY_DEV = [("tags", TAG_KEY, "=", "dev")]


@pytest.mark.parametrize("entity", ["registered_model_version", "prompt_version"])
def test_a_failing_version_is_named_rather_than_enumerated(cascade_registry, entity):
    """The cascade answers, and the answer identifies the version that failed.

    Asked of the two-version model so the failing version is the *second* one --
    a query that simply returned the parent's first version would pass a
    single-version fixture by accident.

    A prompt *is* a registered model (T12.9) -- no prompt tables exist -- so both
    entity names resolve to the same rows and both must be mapped.
    """
    failing = cascade_registry.find_failing_resource(entity, SATISFY_DEV, parent_id="m-mixed")
    assert failing == ("m-mixed", "2"), "version 2 is the prod-tagged one"


def test_a_wholly_satisfying_parent_permits(cascade_registry):
    assert (
        cascade_registry.find_failing_resource(
            "registered_model_version", SATISFY_DEV, parent_id="m-dev"
        )
        is None
    )


def test_the_cascade_judges_only_its_own_parents_versions(cascade_registry):
    """The fail-OPEN test: an unscoped subquery acquits via a sibling model.

    ``m-prod`` holds exactly one version, numbered 1 and tagged ``prod``, so it
    fails ``= 'dev'`` and the delete must be refused. But ``m-dev`` and
    ``m-mixed`` each have a *dev-tagged version 1* too. Unscoped, the satisfying
    set is the bare number ``{1}``, ``m-prod``'s version 1 tests as a member of
    it, and the cascade permits -- granting a delete the condition forbids on the
    strength of another model's tag.

    Scoped to the parent the satisfying set is empty, and the refusal is correct.
    """
    failing = cascade_registry.find_failing_resource(
        "registered_model_version", SATISFY_DEV, parent_id="m-prod"
    )
    assert failing == ("m-prod", "1"), (
        "permitting here means the satisfying set leaked other models' version "
        "numbers across the parent boundary"
    )


def test_an_untagged_version_still_fails(cascade_registry):
    # D20 on the cascade path: absence satisfies nothing.
    failing = cascade_registry.find_failing_resource(
        "registered_model_version", SATISFY_DEV, parent_id="m-bare"
    )
    assert failing == ("m-bare", "1")


@pytest.mark.parametrize(("comparator", "value"), REGISTRY_COMPARATORS)
def test_every_comparator_agrees_with_in_memory_on_a_version_cascade(
    cascade_registry, comparator, value
):
    """Parity on the cascade selector, asked per parent.

    Per parent rather than in bulk because a cascade answers one parent's question;
    and every parent is asked because the comparators differ in how they treat the
    untagged version.
    """
    clauses = [("tags", TAG_KEY, comparator, value)]
    for parent, versions in PARENTS.items():
        pushed = cascade_registry.find_failing_resource(
            "registered_model_version", clauses, parent_id=parent
        )
        any_fails = any(not _matches_in_memory(tag, comparator, value) for tag in versions)
        assert (pushed is not None) is any_fails, (
            f"{parent}: SQL and the in-memory matcher disagree on {comparator} {value!r}"
        )


def test_the_registry_cascade_scopes_the_satisfying_set_too(cascade_registry, monkeypatch):
    """The cascade_registry twin of the tracking test of the same name.

    Needed separately because the cascade_registry's ``_get_query`` scopes by
    ``workspace`` over tables keyed by *name*, which is not unique across
    workspaces -- so scoping only the outer half reads children in scope and
    judges them against another tenant's tag rows.
    """
    from mlflow.store.model_registry.dbmodels.models import SqlModelVersionTag

    original = cascade_registry._get_query

    def scoped(session, model):
        query = original(session, model)
        return query.filter(sql.false()) if model is SqlModelVersionTag else query

    monkeypatch.setattr(cascade_registry, "_get_query", scoped)
    # Honoured: no tag row is visible, so no version satisfies and every version
    # fails -> deny. Bypassed: the dev tags are read out of scope, every version
    # satisfies -> permit.
    failing = cascade_registry.find_failing_resource(
        "registered_model_version", SATISFY_DEV, parent_id="m-dev"
    )
    assert failing is not None


def test_an_mcp_server_version_cascade_is_answered(monkeypatch, tmp_path):
    # The third tier, on the tracking store, whose version column is a VARCHAR.
    store = SqlAlchemyStore(f"sqlite:///{tmp_path}/mcp.db", str(tmp_path))
    store.create_mcp_server("demo/gateway")
    for version, tag in (("1.0.0", "prod"), ("2.0.0", "dev")):
        store.create_mcp_server_version({"name": "demo/gateway", "version": version})
        store.set_mcp_server_version_tag("demo/gateway", version, TAG_KEY, tag)
    failing = store.find_failing_resource(
        "mcp_server_version", SATISFY_DEV, parent_id="demo/gateway"
    )
    assert failing == ("demo/gateway", "1.0.0")


def test_a_failing_chunk_stops_the_later_chunks(monkeypatch):
    """Exit-on-first-failure is per CHUNK, not merely per clause.

    ``DeleteTraces`` caps nothing -- the handler validates only that ``request_ids`` is
    an array of strings -- so an id list is unbounded and chunked. Querying every chunk
    after one has already answered the question is wasted round trips on exactly the
    request shape that motivated the pushdown.

    Every run here fails, so the first chunk settles it whatever order the ids sort in;
    the test does not depend on where the failing id lands.
    """
    store, experiment_id = _store_with(monkeypatch, [("a", "prod"), ("b", "prod"), ("c", "prod")])
    ids = _run_ids(store, experiment_id)
    assert len(ids) == 3
    monkeypatch.setattr(type(store), "_TAG_PUSHDOWN_ID_CHUNK", 3)  # 1 id per chunk after the
    seen = []  # clause's own 2 parameters
    original = SqlAlchemyStore._get_query

    def counting(self, session, model):
        seen.append(model)
        return original(self, session, model)

    monkeypatch.setattr(SqlAlchemyStore, "_get_query", counting)
    failing = store.find_failing_resource("run", [("tags", TAG_KEY, "!=", "prod")], ids=ids)
    assert failing in ids
    assert len([m for m in seen if m is SqlTag]) == 1, (
        "the first failing chunk must settle it; later chunks cannot change the verdict"
    )


def test_the_chunk_budget_leaves_room_for_the_clause_parameters(monkeypatch):
    """An ``IN`` clause binds one parameter per listed value, on top of the key.

    Sizing chunks by the id cap alone overflows the backend's limit once the clause
    itself is wide -- a 900-id chunk plus a 100-value ``IN`` list is 1001 parameters
    against SQLite's 999. The cap must stay a property of the statement rather than a
    limit on what a caller may ask.
    """
    store, experiment_id = _store_with(monkeypatch, [("a", "dev")])
    wide = tuple(f"v{i}" for i in range(200))
    ids = [f"absent-{i}" for i in range(2_000)]
    answer = store.find_failing_resource("run", [("tags", TAG_KEY, "IN", wide)], ids=ids)
    assert answer is not None, "absent ids satisfy nothing, so one must be reported"


# One stage, not three.
#
#     A target condition used to have a fallback: a store that could not express the predicate
#     returned ``DECLINED``, the gate enumerated the resources and evaluated the clauses in
#     Python. That fallback is gone. Every conditionable type has an id-selector mapping and
#     all six cascade tiers have a cascade mapping, so no SQL store can decline for anything
#     authorable -- the fallback existed only for a backend the auth plugin is not meant to be
#     paired with, and keeping it meant two evaluators of one semantic where a drift between
#     them is a difference in who may write what.
#
#     So a store that cannot answer must RAISE. The auth plugin already mandates SQL for its
#     own grants (``database_uri`` is a required config field with no default), which makes a
#     filesystem tracking backend a half-configured server rather than a supported one, and a
#     loud 500 in the log beats a silently slow path. A value condition is pure and keeps
#     working on any backend -- only the target half needs a store.
#


def test_the_abstract_tracking_store_raises():
    from mlflow.store.tracking.abstract_store import AbstractStore

    with pytest.raises(NotImplementedError, match=_CANNOT_ANSWER):
        AbstractStore.find_failing_resource(
            SimpleNamespace(), "run", (("tags", "a", "=", "b"),), ids=["r1"]
        )


def test_the_abstract_registry_store_raises():
    from mlflow.store.model_registry.abstract_store import AbstractStore

    with pytest.raises(NotImplementedError, match=_CANNOT_ANSWER):
        AbstractStore.find_failing_resource(
            SimpleNamespace(), "registered_model", (("tags", "a", "=", "b"),), ids=["m1"]
        )


def test_the_raise_names_the_store_that_could_not_answer():
    """The message is for whoever reads the log, so it has to say which backend and that
    the unsupported half is TARGET conditions -- not that a database is missing, since the
    auth plugin always has one.
    """
    from mlflow.store.tracking.abstract_store import AbstractStore

    with pytest.raises(NotImplementedError, match=r"target condition"):
        AbstractStore.find_failing_resource(
            SimpleNamespace(), "run", (("tags", "a", "=", "b"),), ids=["r1"]
        )


@pytest.mark.parametrize("selector", ["ids", "parent"])
def test_an_unmapped_entity_raises_rather_than_declining(monkeypatch, selector):
    """A forgotten mapping on a future eleventh conditionable type must be loud. With no
    fallback left, declining would mean permitting.
    """
    store, experiment_id = _store_with(monkeypatch, [])
    kwargs = {"ids": ["x1"]} if selector == "ids" else {"parent_id": experiment_id}
    with pytest.raises(NotImplementedError, match=_CANNOT_ANSWER):
        store.find_failing_resource("not_a_real_entity", (("tags", "a", "=", "b"),), **kwargs)


def test_a_namespace_the_entity_lacks_raises(monkeypatch):
    """``aliases.*`` on a run: the clause is well-formed but no run table answers it.
    Authoring rejects this, so it is unreachable -- which is exactly why it must not be
    silent if a future type reaches it.
    """
    store, _ = _store_with(monkeypatch, [])
    with pytest.raises(NotImplementedError, match=_CANNOT_ANSWER):
        store.find_failing_resource("run", (("aliases", "prod", "=", "1"),), ids=["r1"])


def test_an_alias_clause_on_a_cascade_raises(monkeypatch):
    """No cascade child owns aliases (D18) -- a version's aliases live on its parent -- so
    there is no table to answer from. Ignoring the clause would judge a conjunction against
    a subset of itself, which is the fail-open direction.
    """
    store, experiment_id = _store_with(monkeypatch, [])
    with pytest.raises(NotImplementedError, match=_CANNOT_ANSWER):
        store.find_failing_resource(
            "run", (("aliases", "prod", "=", "1"),), parent_id=experiment_id
        )


def test_the_sentinel_is_gone():
    # Nothing may reintroduce a third verdict: the method answers or raises.
    import mlflow.store.condition_pushdown as cp

    assert not hasattr(cp, "DECLINED")
    assert not hasattr(cp, "Declined")
