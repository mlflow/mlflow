"""Parity between pushed-down and in-memory evaluation of a target condition.

``AbstractStore.filter_ids_by_tag_clauses`` lets a store answer "which of these
resources satisfy this tag predicate" without loading the resources. That makes
**two** implementations of one semantic: the store's SQL predicate and the
auth layer's :func:`evaluate_resource`. Nothing in the type system forces them
to agree, and a disagreement is not symmetric -- a pushdown that matches a
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

from mlflow.entities import RunTag, ViewType
from mlflow.server.auth import resources as auth_resources
from mlflow.server.auth.conditions import (
    NAMESPACE_RESOURCE,
    evaluate_resource,
    parse_condition,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

TAG_KEY = "lifecycle"

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


@pytest.mark.parametrize(("comparator", "value", "filter_text"), COMPARATORS)
def test_pushdown_agrees_with_in_memory_evaluation(store_with_runs, comparator, value, filter_text):
    store, _, ids = store_with_runs
    by_id = {run_id: label for label, run_id in ids.items()}

    pushed = store.filter_ids_by_tag_clauses("run", list(by_id), [(TAG_KEY, comparator, value)])
    assert pushed is not None, (
        f"SqlAlchemyStore must push {comparator} down; returning None falls back to "
        "loading every resource, which is the cost this exists to avoid"
    )

    assert {by_id[i] for i in pushed} == _in_memory(filter_text, ids)


@pytest.mark.parametrize(("comparator", "value", "_filter_text"), COMPARATORS)
def test_an_untagged_resource_satisfies_no_comparator(
    store_with_runs, comparator, value, _filter_text
):
    """D20, stated directly rather than inferred from the parity above.

    Parity alone would still pass if *both* implementations wrongly included the
    untagged resource, so the absence rule is asserted on its own.
    """
    store, _, ids = store_with_runs
    pushed = store.filter_ids_by_tag_clauses(
        "run", list(ids.values()), [(TAG_KEY, comparator, value)]
    )
    assert pushed is not None
    assert ids["untagged"] not in pushed, (
        f"{comparator} matched a resource with no {TAG_KEY!r} tag; on the target side "
        "an absent tag must fail every comparator"
    )


def test_every_clause_must_hold(store_with_runs):
    """Clauses are conjunctive, matching ``combine``'s AND-only semantics."""
    store, _, ids = store_with_runs
    pushed = store.filter_ids_by_tag_clauses(
        "run",
        list(ids.values()),
        [(TAG_KEY, "=", "prod"), (TAG_KEY, "=", "dev")],
    )
    assert pushed is not None
    assert pushed == set(), "no run can hold two different values for one tag key"


def test_no_clauses_matches_everything_and_no_ids_matches_nothing(store_with_runs):
    """Neither empty input may be confused with ``None``'s "cannot push down"."""
    store, _, ids = store_with_runs
    assert store.filter_ids_by_tag_clauses("run", list(ids.values()), []) == set(ids.values())
    assert store.filter_ids_by_tag_clauses("run", [], [(TAG_KEY, "=", "prod")]) == set()


def test_an_id_list_larger_than_the_sql_parameter_cap_still_works(store_with_runs):
    """Ids bind one SQL parameter each, and backends cap how many a statement carries.

    Unchunked, SQLite raises "too many SQL variables" above ~32k -- and a caller has no
    way to know the cap, so the failure would surface as an opaque 500 on a large bulk
    delete rather than a refusal. Verified to fail before chunking was added.
    """
    store, _, ids = store_with_runs
    padded = list(ids.values()) + [f"absent-{i}" for i in range(60_000)]
    pushed = store.filter_ids_by_tag_clauses("run", padded, [(TAG_KEY, "!=", "prod")])
    assert pushed == {ids["other"]}, (
        "only the dev-tagged run satisfies != 'prod'; absent ids must not match, and "
        "the untagged run must not either"
    )


def test_an_unknown_entity_declines_rather_than_matching(store_with_runs):
    """An unmapped entity must return ``None``, never a wrong or empty answer.

    Returning ``set()`` would read as "nothing matched" and deny every mutation;
    returning the input would permit every one. Only ``None`` routes the caller
    to in-memory evaluation.
    """
    store, _, ids = store_with_runs
    assert (
        store.filter_ids_by_tag_clauses(
            "not_an_entity", list(ids.values()), [(TAG_KEY, "=", "prod")]
        )
        is None
    )


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
        return store.any_child_failing_tag_clauses(
            "run", experiment_id, [(TAG_KEY, comparator, value)]
        )

    def test_a_failing_child_is_found(self, store_with_runs):
        """The fixture holds a prod run and an untagged one, both failing."""
        store, experiment_id, _ = store_with_runs
        assert self._answer(store, experiment_id) is True

    def test_all_satisfying_children_pass(self, monkeypatch):
        store, experiment_id = _store_with(monkeypatch, [("a", "dev"), ("b", "dev")])
        assert self._answer(store, experiment_id) is False

    def test_an_untagged_child_fails(self, monkeypatch):
        """D20 again, and the case a complement-based query gets wrong.

        Searching for children that *violate* the filter cannot work, because
        the complement of ``!= 'prod'`` is not ``= 'prod'`` when absence fails
        on both. An untagged child satisfies neither and must still deny.
        """
        store, experiment_id = _store_with(monkeypatch, [("a", "dev"), ("b", None)])
        assert self._answer(store, experiment_id) is True

    def test_a_parent_with_no_children_passes(self, monkeypatch):
        """Vacuous, and distinct from "could not enumerate", which denied."""
        store, experiment_id = _store_with(monkeypatch, [])
        assert self._answer(store, experiment_id) is False

    def test_no_clauses_cannot_fail(self, store_with_runs):
        store, experiment_id, _ = store_with_runs
        assert store.any_child_failing_tag_clauses("run", experiment_id, []) is False

    def test_a_child_failing_only_the_second_clause_is_found(self, monkeypatch):
        """Clauses are conjunctive, so failing any one of them fails the child."""
        store, experiment_id = _store_with(monkeypatch, [("a", "dev")])
        assert (
            store.any_child_failing_tag_clauses(
                "run",
                experiment_id,
                [(TAG_KEY, "!=", "prod"), (TAG_KEY, "=", "prod")],
            )
            is True
        )

    def test_a_sibling_parents_children_are_not_considered(self, monkeypatch):
        """The query must be scoped to the parent, or one experiment's
        protected run would block deleting an unrelated experiment.
        """
        store, experiment_id = _store_with(monkeypatch, [("a", "dev")])
        other = store.create_experiment("other")
        run_id = store.create_run(other, "u", 0, [], "p").info.run_id
        store.set_tag(run_id, RunTag(TAG_KEY, "prod"))
        assert self._answer(store, experiment_id) is False
        assert self._answer(store, other) is True

    def test_an_unknown_entity_declines(self, store_with_runs):
        store, experiment_id, _ = store_with_runs
        assert (
            store.any_child_failing_tag_clauses(
                "not_an_entity", experiment_id, [(TAG_KEY, "=", "prod")]
            )
            is None
        )

    def test_the_answer_matches_judging_each_child_individually(self, monkeypatch):
        """Parity with the enumerate-and-evaluate path this replaces."""
        filter_text = f"tags.{TAG_KEY} != 'prod'"
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
            assert self._answer(store, experiment_id) is individually, runs


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
                    )
                ]

        monkeypatch.setattr(auth_module, "store", Store())
        monkeypatch.setattr(
            auth_module,
            "_get_tracking_store",
            lambda: SimpleNamespace(any_child_failing_tag_clauses=lambda *a, **k: pushdown_answer),
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
        assert self._gate(monkeypatch, True) is False

    def test_a_reported_pass_permits_without_enumerating(self, monkeypatch):
        assert self._gate(monkeypatch, False) is True

    def test_a_decline_falls_back_to_enumeration(self, monkeypatch):
        """And the fallback is reached, proving the decline is honoured."""
        with pytest.raises(AssertionError, match="the gate enumerated children"):
            self._gate(monkeypatch, None)


class TestOnlyTagClausesArePushed:
    """A clause the store cannot express must take the whole row in memory.

    The pushdown hooks speak only about tags, but the resource namespace also
    has ``aliases.<name>``. Pushing a row's tag clauses and silently ignoring
    its alias clause would judge a conjunction against a subset of itself --
    the fail-open direction, and one that no end-to-end case detects, because
    aliases live on version types while the cascade entities are runs, traces
    and logged models. So the guard is asserted directly on the helper.
    """

    @staticmethod
    def _triples(filter_text):
        from mlflow.server.auth import _tag_clause_triples

        return _tag_clause_triples(parse_condition(filter_text, NAMESPACE_RESOURCE))

    def test_tag_clauses_convert(self):
        assert self._triples(f"tags.{TAG_KEY} != 'prod'") == [(TAG_KEY, "!=", "prod")]

    def test_several_tag_clauses_convert_in_order(self):
        triples = self._triples(f"tags.{TAG_KEY} != 'prod' AND tags.team = 'ml'")
        assert triples == [(TAG_KEY, "!=", "prod"), ("team", "=", "ml")]

    def test_an_alias_clause_declines(self):
        assert self._triples("aliases.production = 'yes'") is None

    def test_a_row_mixing_tags_and_aliases_declines_whole(self):
        """Not "push the tag half" -- the row is indivisible."""
        assert self._triples(f"tags.{TAG_KEY} != 'prod' AND aliases.production = 'yes'") is None
