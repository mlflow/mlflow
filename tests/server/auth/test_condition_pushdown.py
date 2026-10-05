"""Parity between pushed-down and in-memory evaluation of a target condition.

``AbstractStore.find_failing_resource`` lets a store answer "is there a resource
here that fails this tag predicate" without loading the resources. That makes
**two** implementations of one semantic: the store's SQL predicate and the
auth layer's :func:`evaluate_resource`. Nothing in the type system forces them
to agree, and a disagreement is not symmetric -- a pushdown that accepts a
resource the matcher would reject **grants a mutation the condition forbids**.

So every comparator is checked against every tag state here, and the state that
matters most is the third one: a resource with *no* such tag. D20 makes absence
fail on the target side, so ``!=`` and ``NOT IN`` must EXCLUDE an untagged
resource. That is the reading a hand-written SQL predicate gets wrong by
default, because ``NOT (value = 'x')`` over a join quietly drops to "no row
matched" and lets the untagged resource through.
"""

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
    evaluate_resource,
    parse_condition,
)
from mlflow.server.auth.resources import version_resource_id
from mlflow.store.condition_pushdown import DECLINED
from mlflow.store.model_registry.sqlalchemy_store import (
    SqlAlchemyStore as RegistrySqlAlchemyStore,
)
from mlflow.store.tracking.dbmodels.models import SqlTag
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

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

    assert store.find_failing_resource("run", clauses, ids=list(by_id)) is not DECLINED, (
        f"SqlAlchemyStore must push {comparator} down; DECLINED falls back to loading "
        "every resource, which is the cost this exists to avoid"
    )

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
    """Clauses are conjunctive, matching ``combine``'s AND-only semantics."""
    store, _, ids = store_with_runs
    clauses = [("tags", TAG_KEY, "=", "prod"), ("tags", TAG_KEY, "=", "dev")]
    assert _satisfying(store, "run", list(ids.values()), clauses) == set(), (
        "no run can hold two different values for one tag key"
    )


def test_neither_empty_input_can_fail(store_with_runs):
    """Nothing to judge, or nothing to judge it against, both mean nothing failed.

    Neither may be confused with ``DECLINED``: "no clause can fail" and "I cannot
    evaluate the clauses" look the same to a caller that tests truthiness, and only one
    of them is safe to treat as a pass.
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
    assert failing is not DECLINED
    assert failing != ids["other"], "the dev-tagged run satisfies != 'prod' and must not fail"


class TestPushdownGoesThroughTheWorkspaceHook:
    """Every query in both stores is built via ``_get_query`` -- the hook a
    workspace-aware subclass overrides to enforce tenant isolation. The registry
    store filters ``model.workspace`` there directly; the open-source tracking
    store returns a bare query and documents that "workspace-aware subclasses
    override this to enforce scoping". Every existing MCP query honours it.

    A pushdown that builds its own ``session.query`` therefore bypasses the one
    place isolation is enforced, and could match rows no other query in the
    process would see. The tag tables carry no ``workspace`` column today, so
    nothing leaks yet -- but the MCP and registry tables do, and a subclass that
    scopes a tag table by joining would be silently skipped here.
    """

    def test_the_filter_honours_an_overridden_get_query(self, store_with_runs, monkeypatch):
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

    def test_the_cascade_honours_an_overridden_get_query(self, store_with_runs, monkeypatch):
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

    def test_the_cascade_scopes_the_satisfying_set_too(self, monkeypatch):
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
        assert failing is not DECLINED


def test_an_unknown_entity_declines_rather_than_answering(store_with_runs):
    """An unmapped entity must return ``DECLINED``, never a verdict.

    Returning an id would deny a mutation it never judged; returning ``None`` would
    permit every one. Only ``DECLINED`` routes the caller to in-memory evaluation, and
    it is a distinct value precisely so neither mistake is expressible.
    """
    store, _, ids = store_with_runs
    assert (
        store.find_failing_resource(
            "not_an_entity", [("tags", TAG_KEY, "=", "prod")], ids=list(ids.values())
        )
        is DECLINED
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


class TestCascadePushdown:
    """Deleting a parent must judge every child, without enumerating them.

    The expensive shape this replaces: enumerate every child id, fetch each
    child, project its tags, evaluate. That is what forced a cap on how many
    children a conditioned delete could consider at all -- an experiment with
    more children than the cap could not be deleted once any condition existed
    on those types, even one scoped elsewhere, because the refusal was about
    enumerability rather than about matching.

    The question is only ever "does ANY child fail", so it needs no enumeration
    and no tag values: one ``LIMIT 1`` query over a ``NOT IN (satisfies)``
    subquery. Cost stops depending on child count.
    """

    def _answer(self, store, experiment_id, comparator="!=", value="prod"):
        """The failing child's id, or ``None`` if every child satisfies."""
        return store.find_failing_resource(
            "run", [("tags", TAG_KEY, comparator, value)], parent_id=experiment_id
        )

    def test_a_failing_child_is_found(self, store_with_runs):
        """The fixture holds a prod run and an untagged one, both failing.

        The id comes back rather than a bare ``True``, which is what lets a cascade
        denial say which child blocked it.
        """
        store, experiment_id, ids = store_with_runs
        assert self._answer(store, experiment_id) in {ids["matching"], ids["untagged"]}

    def test_all_satisfying_children_pass(self, monkeypatch):
        store, experiment_id = _store_with(monkeypatch, [("a", "dev"), ("b", "dev")])
        assert self._answer(store, experiment_id) is None

    def test_an_untagged_child_fails(self, monkeypatch):
        """D20 again, and the case a complement-based query gets wrong.

        Searching for children that *violate* the filter cannot work, because
        the complement of ``!= 'prod'`` is not ``= 'prod'`` when absence fails
        on both. An untagged child satisfies neither and must still deny.
        """
        store, experiment_id = _store_with(monkeypatch, [("a", "dev"), ("b", None)])
        assert self._answer(store, experiment_id) is not None

    def test_a_parent_with_no_children_passes(self, monkeypatch):
        """Vacuous, and distinct from "could not enumerate", which denied."""
        store, experiment_id = _store_with(monkeypatch, [])
        assert self._answer(store, experiment_id) is None

    def test_no_clauses_cannot_fail(self, store_with_runs):
        store, experiment_id, _ = store_with_runs
        assert store.find_failing_resource("run", [], parent_id=experiment_id) is None

    def test_a_child_failing_only_the_second_clause_is_found(self, monkeypatch):
        """Clauses are conjunctive, so failing any one of them fails the child."""
        store, experiment_id = _store_with(monkeypatch, [("a", "dev")])
        assert (
            store.find_failing_resource(
                "run",
                [("tags", TAG_KEY, "!=", "prod"), ("tags", TAG_KEY, "=", "prod")],
                parent_id=experiment_id,
            )
            is not None
        )

    def test_a_sibling_parents_children_are_not_considered(self, monkeypatch):
        """The query must be scoped to the parent, or one experiment's
        protected run would block deleting an unrelated experiment.
        """
        store, experiment_id = _store_with(monkeypatch, [("a", "dev")])
        other = store.create_experiment("other")
        run_id = store.create_run(other, "u", 0, [], "p").info.run_id
        store.set_tag(run_id, RunTag(TAG_KEY, "prod"))
        assert self._answer(store, experiment_id) is None
        assert self._answer(store, other) == run_id

    def test_an_unknown_entity_declines(self, store_with_runs):
        store, experiment_id, _ = store_with_runs
        assert (
            store.find_failing_resource(
                "not_an_entity", [("tags", TAG_KEY, "=", "prod")], parent_id=experiment_id
            )
            is DECLINED
        )

    @pytest.mark.parametrize(("comparator", "value", "filter_text"), COMPARATORS)
    def test_the_answer_matches_judging_each_child_individually(
        self, monkeypatch, comparator, value, filter_text
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
            pushed = self._answer(store, experiment_id, comparator, value)
            assert pushed is not DECLINED, f"{comparator} must push down"
            assert (pushed is not None) is individually, (runs, comparator)


class TestTheGateConsultsPushdown:
    """The gate must prefer pushdown, not merely tolerate it.

    Without these, deleting the pushdown call from the gate would fail nothing:
    the enumeration fallback would quietly answer every case and the whole suite
    would stay green. So the enumerator here *raises* -- reaching it is the
    failure. The answers themselves are covered above; what is pinned here is
    that the gate asks the store before it enumerates, and trusts the reply.
    """

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
                    SimpleNamespace(
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

        def _must_not_enumerate(_experiment_id):
            raise AssertionError(
                "the gate enumerated children although the store answered the "
                "predicate; pushdown exists precisely to avoid this"
            )

        monkeypatch.setattr(auth_resources, "runs_of_experiment", _must_not_enumerate)
        context = context_for(
            "run",
            None,
            ConditionScope.MUTATE,
            resource_id_resolver=lambda: _must_not_enumerate("e-1"),
            parent_resource_id="e-1",
        )
        return auth_module.authorize_on_conditions("alice", "w", [context])

    def test_a_reported_failure_denies_without_enumerating(self, monkeypatch):
        assert self._gate(monkeypatch, "child-9") is False

    def test_a_reported_pass_permits_without_enumerating(self, monkeypatch):
        assert self._gate(monkeypatch, None) is True

    def test_a_decline_falls_back_to_enumeration(self, monkeypatch):
        """And the fallback is reached, proving the decline is honoured.

        ``DECLINED`` rather than ``None``: ``None`` now means "nothing failed", so if
        the two were the same value a store that could not answer would permit every
        cascade silently.
        """
        with pytest.raises(AssertionError, match="the gate enumerated children"):
            self._gate(monkeypatch, DECLINED)


class TestWhichClausesArePushed:
    """A clause the store cannot express must take the whole row in memory.

    Both resource namespaces now push, so the conversion is expected to succeed
    for ``tags.*`` and ``aliases.*`` alike -- the earlier version of this class
    asserted that an alias clause *declined*, which was true only while pushdown
    was tag-only.

    The invariant it protects is unchanged, and is the reason the helper is
    asserted directly rather than end to end: a clause that cannot be pushed must
    decline the **whole row**, never have its understood half pushed, because a
    conjunction judged on a subset of itself is the fail-open direction -- the
    dropped clause is exactly the one that would have denied.
    """

    @staticmethod
    def _pushed(filter_text):
        from mlflow.server.auth import _pushable_clauses

        return _pushable_clauses(parse_condition(filter_text, NAMESPACE_RESOURCE))

    def test_tag_clauses_convert(self):
        assert self._pushed(f"tags.{TAG_KEY} != 'prod'") == [("tags", TAG_KEY, "!=", "prod")]

    def test_several_tag_clauses_convert_in_order(self):
        pushed = self._pushed(f"tags.{TAG_KEY} != 'prod' AND tags.team = 'ml'")
        assert pushed == [("tags", TAG_KEY, "!=", "prod"), ("tags", "team", "=", "ml")]

    def test_an_alias_clause_converts(self):
        assert self._pushed("aliases.production = 'yes'") == [("aliases", "production", "=", "yes")]

    def test_a_row_mixing_tags_and_aliases_converts_whole(self):
        """Both halves in one call, so the conjunction is never split."""
        pushed = self._pushed(f"tags.{TAG_KEY} != 'prod' AND aliases.production = 'yes'")
        assert pushed == [
            ("tags", TAG_KEY, "!=", "prod"),
            ("aliases", "production", "=", "yes"),
        ]

    def test_an_unrecognised_identifier_declines_the_whole_row(self):
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

    def test_a_clause_with_no_key_declines(self):
        """A flat request-style clause has ``key is None`` and names no column."""
        from mlflow.server.auth import _pushable_clauses

        rows = [Clause(identifier="tags", key=None, comparator="=", value="v")]
        assert _pushable_clauses(rows) is None


class TestTheNewlyCoveredTypes:
    """MCP tags, the alias namespace, and a composite-keyed version id.

    These are the cases the first pass could not answer: it covered four tracking
    tag tables and declined everything else, so an alias condition or an MCP
    condition silently took the in-memory path. Coverage is the point of the
    unified clause shape, so each namespace is exercised against a real store.
    """

    @pytest.fixture
    def mcp_store(self, monkeypatch):
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

    def test_mcp_server_tags_push_down(self, mcp_store):
        assert _satisfying(
            mcp_store, "mcp_server", self.ALL_SERVERS, [("tags", TAG_KEY, "=", "prod")]
        ) == {PROD_SERVER}

    def test_an_untagged_mcp_server_satisfies_no_negative_comparator(self, mcp_store):
        """D20 again, on a type whose id is a name rather than a uuid."""
        assert _satisfying(
            mcp_store, "mcp_server", self.ALL_SERVERS, [("tags", TAG_KEY, "!=", "prod")]
        ) == {DEV_SERVER}

    def test_mcp_server_aliases_push_down(self, mcp_store):
        """The alias namespace, which the first pass declined outright.

        An alias row is ``(name, alias, version)``, so the clause key is the alias
        name and the compared value is the version it points at.
        """
        assert _satisfying(
            mcp_store, "mcp_server", self.ALL_SERVERS, [("aliases", "champion", "=", "1.0.0")]
        ) == {PROD_SERVER}

    def test_an_absent_alias_satisfies_nothing(self, mcp_store):
        assert _satisfying(
            mcp_store, "mcp_server", self.ALL_SERVERS, [("aliases", "champion", "!=", "9.9.9")]
        ) == {PROD_SERVER}

    def test_tags_and_aliases_are_conjunctive_in_one_call(self, mcp_store):
        """The reason both namespaces share a call rather than two methods."""
        both_hold = [("tags", TAG_KEY, "=", "prod"), ("aliases", "champion", "=", "1.0.0")]
        assert _satisfying(mcp_store, "mcp_server", self.ALL_SERVERS, both_hold) == {PROD_SERVER}
        one_fails = [("tags", TAG_KEY, "=", "dev"), ("aliases", "champion", "=", "1.0.0")]
        assert _satisfying(mcp_store, "mcp_server", self.ALL_SERVERS, one_fails) == set()

    def test_a_version_is_matched_by_its_decomposed_id(self, mcp_store):
        """A composite id arrives as parts, so the store never parses ``name/version``."""
        ids = [(name, "1.0.0") for name in self.ALL_SERVERS]
        assert _satisfying(
            mcp_store, "mcp_server_version", ids, [("tags", TAG_KEY, "=", "prod")]
        ) == {(PROD_SERVER, "1.0.0")}

    def test_a_version_id_must_match_both_parts(self, mcp_store):
        """The half-match a plain ``IN`` on the name column would wrongly accept."""
        assert (
            _satisfying(
                mcp_store,
                "mcp_server_version",
                [(PROD_SERVER, "2.0.0")],
                [("tags", TAG_KEY, "=", "prod")],
            )
            == set()
        )

    def test_an_alias_clause_on_a_version_declines(self, mcp_store):
        """D18: a version's aliases live on its parent, so it exposes no alias table."""
        assert (
            mcp_store.find_failing_resource(
                "mcp_server_version",
                [("aliases", "champion", "=", "1.0.0")],
                ids=[(PROD_SERVER, "1.0.0")],
            )
            is DECLINED
        )


class TestTheCascadeDeclinesWhatItCannotExpress:
    """Asserted on the store method, because nothing reaches this end to end.

    Cascade entities are experiment children -- runs, traces, logged models --
    and none of those own aliases (D18), so no authored condition puts an alias
    clause on a cascading delete today.

    It still needs a test. Widening ``_pushable_clauses`` to convert the alias
    namespace means the cascade now *receives* a clause shape it never saw while
    pushdown was tag-only, and the guard that refuses it was confirmed unprotected
    by mutation: replacing its ``return None`` with ``continue`` left all 45 tests
    passing. Silently dropping the clause is the fail-open direction.
    """

    def test_an_alias_clause_declines_rather_than_being_dropped(self, store_with_runs):
        store, experiment_id, _ = store_with_runs
        assert (
            store.find_failing_resource(
                "run", [("aliases", "champion", "=", "1.0.0")], parent_id=experiment_id
            )
            is DECLINED
        ), "an unexpressible clause must decline the call, not be skipped"

    def test_a_mixed_row_declines_whole_rather_than_pushing_its_tag_half(self, store_with_runs):
        """The dangerous case: the tag half alone would answer, and answer wrongly.

        Every run here is dev-tagged, so the tag clause alone permits. Dropping the
        alias clause would turn a conjunction nothing can satisfy into a pass.
        """
        store, experiment_id, _ = store_with_runs
        assert (
            store.find_failing_resource(
                "run",
                [("tags", TAG_KEY, "!=", "nothing"), ("aliases", "champion", "=", "x")],
                parent_id=experiment_id,
            )
            is DECLINED
        )


REGISTRY_COMPARATORS = [
    ("=", "prod"),
    ("!=", "prod"),
    ("LIKE", "pro%"),
    ("ILIKE", "PRO%"),
    ("IN", ("prod", "staging")),
    ("NOT IN", ("prod", "staging")),
]


class TestRegistryPushdown:
    """The second store. Parity is re-proved here rather than inherited.

    The registry is a different store with its own tables, and the run-tag
    projection bug (`5d2ea6472`) is the standing example of a projection silently
    returning less than it appears to. So every comparator is checked against the
    in-memory evaluator again rather than assumed from the tracking side.

    Two registry-specific hazards the tracking tables do not have:

    - ``version`` is an ``INTEGER`` here, in both the alias table and the version
      tag table, while every condition value is a string. Uncast, ``1`` would be
      compared against ``'1'`` -- matching nothing, which on the resource side
      denies every mutation of the type.
    - A prompt *is* a registered model (T12.9): there are no prompt tables, so the
      ``prompt`` and ``registered_model`` types resolve to the same rows.
    """

    @pytest.fixture
    def registry(self, monkeypatch):
        from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
        from mlflow.store.model_registry.sqlalchemy_store import (
            SqlAlchemyStore as RegistrySqlAlchemyStore,
        )

        d = tempfile.mkdtemp()
        store = RegistrySqlAlchemyStore(f"sqlite:///{d}/registry.db")
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
    def test_every_comparator_agrees_with_in_memory(self, registry, comparator, value):
        """The parity contract, re-proved on the registry's own tables."""
        clauses = [("tags", TAG_KEY, comparator, value)]
        assert (
            registry.find_failing_resource("registered_model", clauses, ids=self.ALL_MODELS)
            is not DECLINED
        ), "the registry store must push the predicate down"
        pushed = _satisfying(registry, "registered_model", self.ALL_MODELS, clauses)

        tags = {"m-prod": {TAG_KEY: "prod"}, "m-dev": {TAG_KEY: "dev"}, "m-bare": {}}
        expected = {
            name
            for name, t in tags.items()
            if _matches_in_memory(t.get(TAG_KEY), comparator, value)
        }
        assert pushed == expected

    def test_an_untagged_model_satisfies_no_negative_comparator(self, registry):
        """D20 on the registry side."""
        assert _satisfying(
            registry, "registered_model", self.ALL_MODELS, [("tags", TAG_KEY, "!=", "prod")]
        ) == {"m-dev"}

    def test_an_integer_version_compares_as_the_string_it_was_written_as(self, registry):
        """The cast. Uncast this returns nothing and denies every mutation."""
        assert _satisfying(
            registry, "registered_model", self.ALL_MODELS, [("aliases", "champion", "=", "1")]
        ) == {"m-prod"}

    def test_an_integer_version_supports_the_text_comparators_too(self, registry):
        """``LIKE`` on an INTEGER column only works because the cast makes it text."""
        assert _satisfying(
            registry, "registered_model", self.ALL_MODELS, [("aliases", "champion", "LIKE", "1%")]
        ) == {"m-prod"}

    def test_a_prompt_resolves_to_the_same_rows_as_a_registered_model(self, registry):
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
        assert _satisfying(registry, "prompt", self.ALL_MODELS, clause) == _satisfying(
            registry, "registered_model", self.ALL_MODELS, clause
        )

    def test_a_version_is_matched_by_its_decomposed_id(self, registry):
        ids = [(name, "1") for name in self.ALL_MODELS]
        assert _satisfying(
            registry, "registered_model_version", ids, [("tags", TAG_KEY, "=", "prod")]
        ) == {("m-prod", "1")}

    def test_a_version_id_must_match_both_parts(self, registry):
        assert (
            _satisfying(
                registry,
                "registered_model_version",
                [("m-prod", "2")],
                [("tags", TAG_KEY, "=", "prod")],
            )
            == set()
        )

    def test_an_alias_clause_on_a_version_declines(self, registry):
        """D18: a version's aliases belong to its parent, so it exposes no alias table."""
        assert (
            registry.find_failing_resource(
                "registered_model_version",
                [("aliases", "champion", "=", "1")],
                ids=[("m-prod", "1")],
            )
            is DECLINED
        )

    def test_an_unmapped_entity_declines(self, registry):
        """A tracking type must not be answered from registry tables."""
        assert (
            registry.find_failing_resource("run", [("tags", TAG_KEY, "=", "x")], ids=["r1"])
            is DECLINED
        )


class TestTheIntegerCastIsStructural:
    """The cast must be asserted on the SQL, because SQLite cannot catch its absence.

    SQLite is dynamically typed and happily evaluates ``1 = '1'`` as true, so every
    behavioural test in this file passes with or without the cast -- confirmed by
    mutation: deleting it left all 61 tests green. PostgreSQL does not coerce, so
    the same code would compare an ``INTEGER`` column against a string, match
    nothing, and -- because absence fails on the target side (D20) -- deny every
    mutation of the type.

    That failure mode is invisible to a correctness test on the development
    backend and would only appear in production, so the guarantee is pinned on the
    generated SQL instead. Same category as the cost assertions: the thing that
    matters is not observable in the result.
    """

    def test_an_integer_column_is_cast_to_text(self):
        from mlflow.store import condition_pushdown
        from mlflow.store.model_registry.dbmodels.models import SqlRegisteredModelAlias

        compiled = str(condition_pushdown.comparable(SqlRegisteredModelAlias.version))
        assert "CAST" in compiled.upper(), (
            "an INTEGER version must be compared as text, or a strictly-typed "
            "backend matches nothing and denies every mutation of the type"
        )

    def test_a_text_column_is_left_alone(self):
        """No gratuitous cast: it would defeat an index for no benefit."""
        from mlflow.store import condition_pushdown
        from mlflow.store.model_registry.dbmodels.models import SqlRegisteredModelTag

        assert condition_pushdown.comparable(SqlRegisteredModelTag.value) is (
            SqlRegisteredModelTag.value
        )

    def test_a_composite_id_predicate_casts_its_integer_part(self):
        """The id side needs it too, not just the compared value."""
        from mlflow.store import condition_pushdown
        from mlflow.store.model_registry.dbmodels.models import SqlModelVersionTag

        predicate = condition_pushdown.id_predicate(
            (SqlModelVersionTag.name, SqlModelVersionTag.version), [("m", "1")]
        )
        assert "CAST" in str(predicate).upper()

    def test_the_tracking_tag_tables_need_no_cast(self):
        """Every tracking tag value is already text, so nothing is wrapped there."""
        from mlflow.store import condition_pushdown
        from mlflow.store.tracking.dbmodels.models import SqlTag

        assert condition_pushdown.comparable(SqlTag.value) is SqlTag.value


class TestTheGateConsultsPushdownForExplicitIds:
    """The batch path: the request names its ids, so the store filters them.

    Same reasoning as the cascade tests -- the loader here *raises*, because
    deleting the pushdown call would otherwise fail nothing: ``attrs_for_bulk``
    would quietly answer every case and the suite would stay green.

    ``attrs_for_bulk`` is not replaced. It remains the path for any store that
    declines, which is every non-SQL backend, and for traces it is barely worse
    than pushdown anyway (``batch_get_trace_infos`` is already one call for N
    ids, so pushdown only avoids moving the tag payload).
    """

    CONDITION = f"tags.{TAG_KEY} != 'prod'"

    @staticmethod
    def _gate(monkeypatch, *, fails, resource_type="run", ids=("r-1", "r-2"), rows=None):
        from mlflow.server import auth as auth_module
        from mlflow.server.auth.conditions import ConditionContext, ConditionScope

        target_rows = rows or [TestTheGateConsultsPushdownForExplicitIds.CONDITION]

        class Store:
            def get_user(self, username):
                return SimpleNamespace(id=1, username=username, is_admin=False)

            def is_workspace_admin(self, user_id, workspace):
                return False

            def list_mutation_conditions_for_user(self, user_id, workspace, types, parents=None):
                return [
                    SimpleNamespace(
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

        def _must_not_load(_type, _ids):
            raise AssertionError(
                "the gate loaded resources although the store filtered the ids; "
                "pushdown exists precisely to avoid this"
            )

        monkeypatch.setattr(auth_resources, "attrs_for_bulk", _must_not_load)
        context = ConditionContext(
            resource_type=resource_type,
            scope=ConditionScope.MUTATE,
            request=None,
            resource_ids=tuple(ids),
            parent_resource_id="e-1",
        )
        return auth_module.authorize_on_conditions("alice", "w", [context]), seen

    def test_a_fully_satisfied_set_permits_without_loading(self, monkeypatch):
        allowed, seen = self._gate(monkeypatch, fails=None)
        assert allowed is True
        assert seen["ids"] == ["r-1", "r-2"]

    def test_an_unsatisfied_id_denies_without_loading(self, monkeypatch):
        allowed, _ = self._gate(monkeypatch, fails="r-2")
        assert allowed is False, "an id the store reported as failing must deny"

    def test_an_absent_id_denies_like_a_failed_clause(self, monkeypatch):
        """Indistinguishable by design: a 404 would reveal which ids exist."""
        allowed, _ = self._gate(monkeypatch, fails="r-1")
        assert allowed is False

    def test_a_decline_falls_back_to_loading(self, monkeypatch):
        """A declining store must still be answered, not refused."""
        from mlflow.server import auth as auth_module
        from mlflow.server.auth.conditions import ConditionContext, ConditionScope

        class Store:
            def get_user(self, username):
                return SimpleNamespace(id=1, username=username, is_admin=False)

            def is_workspace_admin(self, user_id, workspace):
                return False

            def list_mutation_conditions_for_user(self, user_id, workspace, types, parents=None):
                return [
                    SimpleNamespace(
                        resource_type="run",
                        value_condition=None,
                        target_condition=self.__class__
                        and TestTheGateConsultsPushdownForExplicitIds.CONDITION,
                        resource_pattern="*",
                    )
                ]

        monkeypatch.setattr(auth_module, "store", Store())
        monkeypatch.setattr(
            auth_module,
            "_get_tracking_store",
            lambda: SimpleNamespace(
                find_failing_resource=lambda *a, **k: DECLINED,
            ),
        )
        loaded = {}

        def _bulk(resource_type, ids):
            loaded["ids"] = list(ids)
            return {i: SimpleNamespace(tags={TAG_KEY: "dev"}, aliases={}) for i in ids}

        monkeypatch.setattr(auth_resources, "attrs_for_bulk", _bulk)
        context = ConditionContext(
            resource_type="run",
            scope=ConditionScope.MUTATE,
            request=None,
            resource_ids=("r-1",),
            parent_resource_id="e-1",
        )
        assert auth_module.authorize_on_conditions("alice", "w", [context]) is True
        assert loaded["ids"] == ["r-1"], "a declining store must fall back to loading"

    def test_a_composite_id_is_pushed_as_parts(self, monkeypatch):
        """The store must never receive the auth layer's joined ``name/version``.

        The composite format is this layer's invention -- it percent-encodes the
        name because a name may itself contain ``/`` -- so the store is handed the
        decomposed parts and the gate recomposes to compare.
        """
        joined = version_resource_id("com.example/svc", "1.0.0")
        allowed, seen = self._gate(
            monkeypatch,
            fails=None,
            resource_type="registered_model_version",
            ids=(joined,),
        )
        assert allowed is True
        assert seen["ids"] == [("com.example/svc", "1.0.0")], (
            "a composite id must reach the store as parts, not as a joined string"
        )

    def test_one_unpushable_row_falls_back_for_the_whole_context(self, monkeypatch):
        """A conjunction must not be answered from the half the store understood."""
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
            _pushable_clauses([Clause(identifier="bogus", key="k", comparator="=", value="v")])
            is None
        )


class TestTheRightStoreAnswers:
    """A registry type must be asked of the registry store, not the tracking one.

    Found by mutation: collapsing ``_condition_store`` to always return the
    tracking store changed nothing, because the harness above points both getters
    at one fake. Asking the wrong store is not a correctness hole -- an unmapped
    entity declines rather than answering from the wrong table -- but it silently
    loses the pushdown, which is the kind of regression that shows up as a
    performance report months later rather than as a failure.
    """

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
                    SimpleNamespace(
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
        monkeypatch.setattr(
            auth_resources,
            "attrs_for_bulk",
            lambda *a: (_ for _ in ()).throw(AssertionError("fell back to loading")),
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

    def test_a_registry_entry_asks_the_registry_store(self, monkeypatch):
        assert self._which_store_was_asked(monkeypatch, "registered_model", "m-1") == ["registry"]

    def test_a_prompt_asks_the_registry_store(self, monkeypatch):
        """A prompt is stored as a registered model, so it is the registry's to answer."""
        assert self._which_store_was_asked(monkeypatch, "prompt", "p-1") == ["registry"]

    def test_a_run_asks_the_tracking_store(self, monkeypatch):
        assert self._which_store_was_asked(monkeypatch, "run", "r-1") == ["tracking"]

    def test_an_mcp_server_asks_the_tracking_store(self, monkeypatch):
        """MCP lives in the tracking store despite being a registry in name."""
        assert self._which_store_was_asked(monkeypatch, "mcp_server", "com.example/s") == [
            "tracking"
        ]


class TestAnUnpushableRowFallsBackWholesale:
    """One row the converter declines must take the WHOLE context in memory.

    Found by mutation: turning the row loop's ``return None`` into ``continue``
    changed nothing, because every authored condition converts -- the parser
    rejects an unknown identifier at authoring time, so the decline is
    unreachable from a stored condition. The converter is therefore stubbed to
    decline one row, which is the only way to exercise the loop's own contract.

    Pushing the rows it understood and loading for the rest would in fact be
    correct, but the rows are a conjunction and a partially-pushed conjunction is
    one refactor away from being evaluated as the whole thing.
    """

    def test_a_declining_row_makes_the_context_load_instead(self, monkeypatch):
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
                    SimpleNamespace(
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
        loaded = []
        monkeypatch.setattr(
            auth_resources,
            "attrs_for_bulk",
            lambda rt, ids: (
                loaded.append(list(ids))
                or {
                    i: SimpleNamespace(tags={TAG_KEY: "dev", "team": "ml"}, aliases={}) for i in ids
                }
            ),
        )
        context = ConditionContext(
            resource_type="run",
            scope=ConditionScope.MUTATE,
            request=None,
            resource_ids=("r-1",),
            parent_resource_id="e-1",
        )
        assert auth_module.authorize_on_conditions("alice", "w", [context]) is True
        assert loaded == [["r-1"]], "an undeclinable row must send the whole context to the loader"
        assert filtered == [], "no row may be pushed once one of them cannot be"


class TestTheCascadeCapIsFallbackOnly:
    """``MAX_CASCADE_CHILDREN`` no longer bounds a delete on a store that can filter.

    It was a global ceiling: an experiment with more children than the cap could not
    be deleted at all while any condition existed on those types -- including one
    scoped to a *different* experiment -- because enumeration returned ``None`` and
    an unevaluable condition must refuse. The UI rendered that as a bare
    "Permission denied" with nothing the caller could do (F-UI-2a).

    Pushdown removes it where it matters: the store answers "does this parent hold a
    failing child?" in one query and never enumerates. The cap still guards the
    fallback, so the refusal is now **backend-dependent** -- behaviour that is
    uniform everywhere else in this gate, which is exactly why it is pinned rather
    than left implicit.
    """

    @staticmethod
    def _delete_a_huge_experiment(monkeypatch, *, cascade_answer):
        from mlflow.server import auth as auth_module
        from mlflow.server.auth.conditions import ConditionContext, ConditionScope

        class Store:
            def get_user(self, username):
                return SimpleNamespace(id=1, username=username, is_admin=False)

            def is_workspace_admin(self, user_id, workspace):
                return False

            def list_mutation_conditions_for_user(self, user_id, workspace, types, parents=None):
                return [
                    SimpleNamespace(
                        resource_type="run",
                        value_condition=None,
                        target_condition=f"tags.{TAG_KEY} != 'prod'",
                        resource_pattern="*",
                    )
                ]

        monkeypatch.setattr(auth_module, "store", Store())
        monkeypatch.setattr(
            auth_module,
            "_get_tracking_store",
            lambda: SimpleNamespace(
                find_failing_resource=(
                    lambda e, c, *, ids=None, parent_id=None: (
                        cascade_answer if parent_id is not None else DECLINED
                    )
                ),
            ),
        )
        # The enumerator overflows its bound, which is what `None` means here.
        context = ConditionContext(
            resource_type="run",
            scope=ConditionScope.MUTATE,
            request=None,
            resource_ids=(),
            resource_id_resolver=lambda: None,
            parent_resource_id="exp-huge",
        )
        return auth_module.authorize_on_conditions("alice", "w", [context])

    def test_a_filtering_store_permits_a_delete_the_cap_would_have_refused(self, monkeypatch):
        """The fix: enumeration overflowing is irrelevant once the store answered."""
        assert self._delete_a_huge_experiment(monkeypatch, cascade_answer=None) is True

    def test_a_filtering_store_still_denies_a_genuinely_failing_child(self, monkeypatch):
        """Permissiveness must come from the cap going away, not from the gate relaxing."""
        assert self._delete_a_huge_experiment(monkeypatch, cascade_answer="child-3") is False

    def test_a_declining_store_still_refuses_an_overflowing_enumeration(self, monkeypatch):
        """The cap still protects the fallback, so the refusal is backend-dependent.

        An unevaluable condition must never pass vacuously, so this half must not be
        "fixed" by loosening it.
        """
        assert self._delete_a_huge_experiment(monkeypatch, cascade_answer=DECLINED) is False

    def test_the_cap_is_documented_as_fallback_only(self):
        """A behaviour that varies by backend has to say so where it is defined.

        Asserting on a comment is unusual, but this is the one property of the gate
        that is not uniform across backends, and the reason it is acceptable lives in
        prose rather than in code.
        """
        import inspect

        source = inspect.getsource(auth_resources)
        assert "FALLBACK-ONLY" in source, (
            "the cascade cap's backend-dependence must stay documented at its definition"
        )


class TestResourceScopeMatchingAndPushdown:
    """The two scope guarantees that no behavioural test reaches end to end.

    Both are asserted directly on the helpers. A behavioural test cannot see either: the
    fallback path answers identically to the pushdown, and an over-broad scope match only
    shows up as a *denial* on a resource the admin never named -- which no wired route
    exercises, because every route in the suite names a single resource whose pattern
    matches. Mutation testing found both escaping, which is what put them here.
    """

    @staticmethod
    def _row(resource_pattern):
        return SimpleNamespace(
            resource_type="run",
            value_condition=None,
            target_condition=f"tags.{TAG_KEY} != 'prod'",
            resource_pattern=resource_pattern,
        )

    def test_an_id_pattern_governs_only_the_resource_it_names(self):
        """Charging it against a sibling would deny a mutation on a resource the admin
        never pointed the condition at -- over-restriction rather than a hole, but still
        not what was written.
        """
        row = self._row("abc")
        assert auth_module._row_governs(row, "abc") is True
        assert auth_module._row_governs(row, "xyz") is False
        # Nothing to judge against: a create, or a cascade before resolution.
        assert auth_module._row_governs(row, None) is False

    def test_a_wildcard_pattern_governs_every_resource_including_unnamed(self):
        """The pre-scope behaviour, and what makes restricting a create possible at all."""
        row = self._row("*")
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
        return auth_module._target_pushdown(context, rows, parent_id="e-1"), asked

    def test_an_id_scoped_row_declines_the_cascade_pushdown(self, monkeypatch):
        """The cascade sends ONE clause set covering ALL of a parent\'s children, which a row
        naming a single resource breaks: its clauses apply to one child and must not be
        charged against its siblings. Declining is correct and costs only speed -- the
        fallback judges each child against the rows that govern it.

        Unreachable in practice: every cascade-reachable type (run, trace, logged model, and
        the three version types) is wildcard-only, so ``normalize_condition_scope`` refuses an
        id pattern for all six and such a row cannot be stored. Kept because a future tier
        with id-grain patterns would otherwise fail silently, in the fail-open direction.
        """
        verdict, asked = self._cascade_pushdown(monkeypatch, [self._row("abc")])
        assert verdict is DECLINED
        assert asked == [], "a scoped row must not be pushed at all, not pushed and ignored"

    def test_an_unscoped_cascade_row_is_still_pushed(self, monkeypatch):
        """The positive control: declining must be keyed on the pattern, not on having rows.

        Without this, returning ``DECLINED`` unconditionally would pass the test above and
        silently cost every cascade its pushdown.
        """
        verdict, asked = self._cascade_pushdown(monkeypatch, [self._row("*")])
        assert verdict is None
        assert len(asked) == 1, "an unscoped row must reach the store"

    def test_one_scoped_row_among_unscoped_ones_still_declines(self, monkeypatch):
        """Rows are conjunctive, so the context falls back as a whole."""
        verdict, _ = self._cascade_pushdown(monkeypatch, [self._row("*"), self._row("abc")])
        assert verdict is DECLINED


class TestPushdownColumnNamesResolve:
    """Every pushdown mapping must name columns that exist on its model.

    Found by conformance testing, not by these unit tests: ``logged_model`` declared the tag
    key/value columns as ``key``/``value``, but ``SqlLoggedModelTag`` calls them ``tag_key``
    and ``tag_value``. The result was an ``AttributeError`` inside the gate, surfacing as a
    **500 on every logged-model tag write** as soon as any logged-model condition existed --
    a fail-closed-by-crash, not a denial.

    The existing pushdown tests all build their own clauses against ``SqlTag``, whose columns
    really are ``key``/``value``, so the table itself was never exercised per entity. These two
    tests check the table rather than one path through it, so a future entry with a mistyped
    column fails here instead of in production.
    """

    @pytest.mark.parametrize(
        "store",
        [SqlAlchemyStore, RegistrySqlAlchemyStore],
        ids=["tracking", "registry"],
    )
    def test_every_namespace_mapping_names_real_columns(self, store):
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
    def test_every_cascade_entity_names_real_columns(self, store):
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
    def test_every_cascade_discriminator_is_well_formed(self, store):
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
                problems.append(
                    f"{entity}: discriminator widths differ -- child {child}, tag {tag}"
                )
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize(
        "store",
        [SqlAlchemyStore, RegistrySqlAlchemyStore],
        ids=["tracking", "registry"],
    )
    def test_a_composite_keyed_child_scopes_its_tag_subquery(self, store):
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


class TestDeclinedIsNotAVerdict:
    """``DECLINED`` must never be mistaken for "nothing failed".

    The hazard is one-directional. ``None`` means every resource satisfied every
    clause, and it is falsy; if a decline were also falsy, ``if failing:`` -- the
    obvious way to write the check -- would read "I cannot evaluate this predicate" as
    "this predicate passed" and let every mutation through unjudged. So the sentinel
    refuses to answer the question at all.
    """

    def test_truthiness_raises_rather_than_guessing(self):
        with pytest.raises(TypeError, match="not a verdict"):
            bool(DECLINED)

    def test_it_is_identity_comparable_and_distinct_from_none(self):
        assert DECLINED is DECLINED
        assert DECLINED is not None

    def test_the_abstract_default_declines(self):
        """A store that implements nothing must decline, not report a pass."""
        from mlflow.store.model_registry.abstract_store import (
            AbstractStore as RegistryAbstractStore,
        )
        from mlflow.store.tracking.abstract_store import AbstractStore as TrackingAbstractStore

        for cls in (TrackingAbstractStore, RegistryAbstractStore):
            answer = cls.find_failing_resource(
                object(), "run", [("tags", TAG_KEY, "=", "x")], ids=["r1"]
            )
            assert answer is DECLINED, cls.__name__


class TestTheSelectorContract:
    """Exactly one selector, enforced loudly.

    Neither selector is a safe default. Defaulting to ``ids=[]`` would answer "nothing
    failed" for a cascade whose parent was never passed, permitting the whole cascade;
    defaulting to a parent would judge the wrong population. A programming error here
    is not a runtime condition, so it raises rather than declining.
    """

    @pytest.mark.parametrize(
        "kwargs", [{}, {"ids": ["a"], "parent_id": "1"}], ids=["neither", "both"]
    )
    def test_exactly_one_selector_is_required(self, store_with_runs, kwargs):
        store, _, _ = store_with_runs
        with pytest.raises(ValueError, match="exactly one"):
            store.find_failing_resource("run", [], **kwargs)

    def test_the_registry_enforces_it_too(self, monkeypatch):
        import tempfile

        d = tempfile.mkdtemp()
        registry = RegistrySqlAlchemyStore(f"sqlite:///{d}/registry.db")
        with pytest.raises(ValueError, match="exactly one"):
            registry.find_failing_resource("registered_model", [], ids=["m"], parent_id="p")


class TestVersionCascades:
    """A version tier's cascade is answered in SQL, like every other cascade.

    These three tiers were the last to enumerate instead of pushing down, which is
    why they alone could hit ``MAX_CASCADE_CHILDREN`` and refuse a delete outright
    past 2000 versions. What kept them out was not the join but *identity*: the
    satisfying subquery selects the child id, and one column names a child only
    when it is globally unique. ``run_uuid``, ``request_id`` and ``model_id`` are;
    a version is not -- ``name`` matches every version of the model and ``version``
    matches version 3 of any model.

    Scoping the subquery to the parent restores it, because ``version`` is unique
    within one ``name``. That makes the parent filter on the *inner* half
    load-bearing rather than redundant, which is what
    ``test_the_cascade_judges_only_its_own_parents_versions`` exists to prove.
    """

    @pytest.fixture
    def registry(self):
        from mlflow.entities.model_registry import ModelVersionTag

        d = tempfile.mkdtemp()
        store = RegistrySqlAlchemyStore(f"sqlite:///{d}/registry.db")
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

    # Tag state per parent, for the parity check. ``None`` is an untagged version.
    PARENTS = {
        "m-prod": ["prod"],
        "m-dev": ["dev"],
        "m-mixed": ["dev", "prod"],
        "m-bare": [None],
    }

    SATISFY_DEV = [("tags", TAG_KEY, "=", "dev")]

    @pytest.mark.parametrize("entity", ["registered_model_version", "prompt_version"])
    def test_a_failing_version_is_named_rather_than_enumerated(self, registry, entity):
        """The cascade answers, and the answer identifies the version that failed.

        Asked of the two-version model so the failing version is the *second* one --
        a query that simply returned the parent's first version would pass a
        single-version fixture by accident.

        A prompt *is* a registered model (T12.9) -- no prompt tables exist -- so both
        entity names resolve to the same rows and both must be mapped.
        """
        failing = registry.find_failing_resource(entity, self.SATISFY_DEV, parent_id="m-mixed")
        assert failing is not DECLINED, "the version cascade must be pushed down, not declined"
        assert failing == ("m-mixed", "2"), "version 2 is the prod-tagged one"

    def test_a_wholly_satisfying_parent_permits(self, registry):
        assert (
            registry.find_failing_resource(
                "registered_model_version", self.SATISFY_DEV, parent_id="m-dev"
            )
            is None
        )

    def test_the_cascade_judges_only_its_own_parents_versions(self, registry):
        """The fail-OPEN test: an unscoped subquery acquits via a sibling model.

        ``m-prod`` holds exactly one version, numbered 1 and tagged ``prod``, so it
        fails ``= 'dev'`` and the delete must be refused. But ``m-dev`` and
        ``m-mixed`` each have a *dev-tagged version 1* too. Unscoped, the satisfying
        set is the bare number ``{1}``, ``m-prod``'s version 1 tests as a member of
        it, and the cascade permits -- granting a delete the condition forbids on the
        strength of another model's tag.

        Scoped to the parent the satisfying set is empty, and the refusal is correct.
        """
        failing = registry.find_failing_resource(
            "registered_model_version", self.SATISFY_DEV, parent_id="m-prod"
        )
        assert failing is not DECLINED
        assert failing == ("m-prod", "1"), (
            "permitting here means the satisfying set leaked other models' version "
            "numbers across the parent boundary"
        )

    def test_an_untagged_version_still_fails(self, registry):
        """D20 on the cascade path: absence satisfies nothing."""
        failing = registry.find_failing_resource(
            "registered_model_version", self.SATISFY_DEV, parent_id="m-bare"
        )
        assert failing == ("m-bare", "1")

    @pytest.mark.parametrize(("comparator", "value"), REGISTRY_COMPARATORS)
    def test_every_comparator_agrees_with_in_memory(self, registry, comparator, value):
        """Parity on the cascade selector, asked per parent.

        Per parent rather than in bulk because a cascade answers one parent's question;
        and every parent is asked because the comparators differ in how they treat the
        untagged version.
        """
        clauses = [("tags", TAG_KEY, comparator, value)]
        for parent, versions in self.PARENTS.items():
            pushed = registry.find_failing_resource(
                "registered_model_version", clauses, parent_id=parent
            )
            assert pushed is not DECLINED, "the version cascade must be pushed down"
            any_fails = any(not _matches_in_memory(tag, comparator, value) for tag in versions)
            assert (pushed is not None) is any_fails, (
                f"{parent}: SQL and the in-memory matcher disagree on {comparator} {value!r}"
            )

    def test_the_registry_cascade_scopes_the_satisfying_set_too(self, registry, monkeypatch):
        """The registry twin of the tracking test of the same name.

        Needed separately because the registry's ``_get_query`` scopes by
        ``workspace`` over tables keyed by *name*, which is not unique across
        workspaces -- so scoping only the outer half reads children in scope and
        judges them against another tenant's tag rows.
        """
        from mlflow.store.model_registry.dbmodels.models import SqlModelVersionTag

        original = registry._get_query

        def scoped(session, model):
            query = original(session, model)
            return query.filter(sql.false()) if model is SqlModelVersionTag else query

        monkeypatch.setattr(registry, "_get_query", scoped)
        # Honoured: no tag row is visible, so no version satisfies and every version
        # fails -> deny. Bypassed: the dev tags are read out of scope, every version
        # satisfies -> permit.
        failing = registry.find_failing_resource(
            "registered_model_version", self.SATISFY_DEV, parent_id="m-dev"
        )
        assert failing is not None
        assert failing is not DECLINED

    def test_an_mcp_server_version_cascade_is_answered(self, monkeypatch, tmp_path):
        """The third tier, on the tracking store, whose version column is a VARCHAR."""
        store = SqlAlchemyStore(f"sqlite:///{tmp_path}/mcp.db", str(tmp_path))
        store.create_mcp_server("demo/gateway")
        for version, tag in (("1.0.0", "prod"), ("2.0.0", "dev")):
            store.create_mcp_server_version({"name": "demo/gateway", "version": version})
            store.set_mcp_server_version_tag("demo/gateway", version, TAG_KEY, tag)
        failing = store.find_failing_resource(
            "mcp_server_version", self.SATISFY_DEV, parent_id="demo/gateway"
        )
        assert failing is not DECLINED, "the MCP version cascade must be pushed down"
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
    assert answer is not DECLINED, "a wide IN list must still push down"
    assert answer is not None, "absent ids satisfy nothing, so one must be reported"
