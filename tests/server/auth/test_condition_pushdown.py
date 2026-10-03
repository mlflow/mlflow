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

import pytest

from mlflow.entities import RunTag
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
