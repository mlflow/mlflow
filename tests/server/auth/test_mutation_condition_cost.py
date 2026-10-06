# Cost assertions for the mutation-condition gate.
#
# The laziness IS the design: conditions are meant to cost nothing on the paths that have
# none, one indexed query on the paths that do, and a resource read only when a resource
# condition actually needs one. Every one of those is a silent property -- if the gate
# started loading conditions for an admin, or reading a resource to evaluate a
# request-only condition, nothing would fail. The behaviour would stay correct and simply
# get slower on every mutating request.
#
# So these tests count. They drive `authorize_on_conditions` directly rather than over
# HTTP, because the numbers are the assertion and a real server adds reads that have
# nothing to do with the gate.

from types import SimpleNamespace

import pytest

from mlflow.server import auth as auth_module
from mlflow.server.auth import resources as auth_resources
from mlflow.server.auth.conditions import (
    ConditionContext,
    ConditionScope,
    ExperimentRequestValues,
    ExperimentResourceValues,
    MutationConditionSpec,
    RegisteredModelRequestValues,
    RegisteredModelVersionRequestValues,
    RegisteredModelVersionResourceValues,
    RunRequestValues,
    RunResourceValues,
    TraceRequestValues,
    TraceResourceValues,
)

from tests.server.auth.condition_store_fakes import answering_store, permissive_store

_WORKSPACE = "team-a"


@pytest.fixture(autouse=True)
def _pushdown_declines(monkeypatch):
    """Make the stores decline pushdown, so these tests pin the fallback path.

    The cases in this module stub the *resource layer* -- enumerators, bulk
    loaders, counting shims -- rather than the store, so once the gate started
    asking the store first they reached the real default store and failed on a
    missing database. Declining here keeps them exercising the enumerate-and-judge
    path, which is still what every non-SQL backend uses, and is therefore a path
    that needs its own coverage rather than being an accident of the stubs.

    Autouse but not binding: a test that wants the pushdown consulted can
    monkeypatch the store again, and its own patch wins.
    """

    from mlflow.server import auth as auth_module

    permissive = permissive_store()
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: permissive)
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: permissive, raising=False)


class CountingStore:
    """Counts the auth-store calls the gate makes."""

    def __init__(self, rows=(), is_admin=False, is_ws_admin=False):
        self.rows = list(rows)
        self.is_admin = is_admin
        self.is_ws_admin = is_ws_admin
        self.workspace_admin_checks = 0
        self.user_loads = 0
        self.condition_loads = []

    def get_user(self, username):
        self.user_loads += 1
        return SimpleNamespace(id=1, username=username, is_admin=self.is_admin)

    def is_workspace_admin(self, user_id, workspace):
        self.workspace_admin_checks += 1
        return self.is_ws_admin

    def list_mutation_conditions_for_user(self, user_id, workspace, resource_types, parents=None):
        self.condition_loads.append(set(resource_types))
        return list(self.rows)


class CountingResources:
    """Counts resource attribute reads, and serves them from a supplied map."""

    def __init__(self, values=None):
        self.values = values or {}
        self.single_reads = []

    def attrs_for(self, resource_type, resource_id):
        self.single_reads.append((resource_type, resource_id))
        return self.values.get((resource_type, resource_id))


@pytest.fixture
def gate(monkeypatch):
    """Installs the counters and returns a helper that runs the gate.

    Yields `(run, store_holder, resources_holder)` where `run(contexts, rows=..., ...)`
    calls the gate and the holders carry the counts.
    """
    state = {}

    def run(
        contexts,
        rows=(),
        is_admin=False,
        values=None,
        workspace=_WORKSPACE,
        is_ws_admin=False,
    ):
        store = CountingStore(rows, is_admin, is_ws_admin)
        resources = CountingResources(values)
        # The store decides every target verdict, so `values` feeds IT and the call log is
        # where the cost of judging resources now shows up. `attrs_for` survives for
        # attribution only -- a read on the denial path, never on the happy path.
        asked = []
        monkeypatch.setattr(auth_module, "store", store)
        # A test that installed its own pushdown store (the `pushdown` fixture) keeps it:
        # it is asserting on that store's own counters, and silently replacing it here
        # would make those assertions vacuous rather than failing.
        if not getattr(auth_module._get_tracking_store, "_test_pushdown", False):
            answering = answering_store(values or {}, calls=asked)
            monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: answering)
            monkeypatch.setattr(
                auth_module, "_get_model_registry_store", lambda: answering, raising=False
            )
        monkeypatch.setattr(auth_resources, "attrs_for", resources.attrs_for)
        state["store"] = store
        state["resources"] = resources
        state["asked"] = asked
        return auth_module.authorize_on_conditions("alice", workspace, contexts)

    return run, state


def _mutate(resource_type, resource_id, request):
    return ConditionContext(
        resource_type=resource_type,
        scope=ConditionScope.MUTATE,
        request=request,
        resource_ids=(resource_id,),
    )


# ---- The paths that must cost nothing ---------------------------------------


def test_no_contexts_touches_no_store(gate):
    """A route that declares nothing does not even resolve the user."""
    run, state = gate
    assert run([]) is True
    assert state["store"].user_loads == 0
    assert state["store"].condition_loads == []
    assert state["resources"].single_reads == []


def test_an_admin_loads_no_conditions(gate):
    """Admin bypass precedes the condition load, so an admin's mutations pay one user
    lookup -- which the grant path already did -- and nothing else.
    """
    run, state = gate
    allowed = run(
        [_mutate("run", "r1", RunRequestValues(tags=(("k", "v"),)))],
        rows=[MutationConditionSpec("run", value_condition="tag_key != 'k'")],
        is_admin=True,
    )
    assert allowed is True
    assert state["store"].condition_loads == []
    assert state["resources"].single_reads == []


def test_nothing_configured_is_one_query_and_no_reads(gate):
    """The overwhelmingly common case, and the one that makes an empty table behave
    exactly like today's server: a single indexed query, then out.
    """
    run, state = gate
    assert run([_mutate("run", "r1", RunRequestValues())]) is True
    assert len(state["store"].condition_loads) == 1
    assert state["resources"].single_reads == []


def test_only_the_types_in_play_are_loaded(gate):
    """The query is narrowed to the declared types, so a run mutation does not load the
    conditions for every resource type in the workspace.
    """
    run, state = gate
    run([_mutate("run", "r1", RunRequestValues())])
    assert state["store"].condition_loads == [{"run"}]


def test_a_request_only_condition_reads_no_resource(gate):
    """Configured, but with no target condition: the resource is never read. This is the
    case a naive implementation gets wrong, because it has conditions to evaluate and so
    looks like it needs state.
    """
    run, state = gate
    allowed = run(
        [_mutate("run", "r1", RunRequestValues(tags=(("notes", "x"),)))],
        rows=[MutationConditionSpec("run", value_condition="tag_key != 'locked'")],
    )
    assert allowed is True
    assert state["resources"].single_reads == []


def test_create_scope_reads_no_resource(gate):
    """A create has no prior state, so a target condition cannot apply and nothing is
    read -- even though one is configured.
    """
    run, state = gate
    allowed = run(
        [
            ConditionContext(
                resource_type="run",
                scope=ConditionScope.CREATE,
                request=RunRequestValues(tags=(("notes", "x"),)),
            )
        ],
        rows=[MutationConditionSpec("run", target_condition="tags.stage = 'prod'")],
    )
    assert allowed is True
    assert state["resources"].single_reads == []


def test_a_request_denial_short_circuits_before_reading(gate):
    """Request conditions are pure, so evaluating them first means a denial costs no
    resource read at all. Ordering, not an optimization to be reshuffled.
    """
    run, state = gate
    allowed = run(
        [_mutate("run", "r1", RunRequestValues(tags=(("locked", "yes"),)))],
        rows=[
            MutationConditionSpec(
                "run",
                value_condition="tag_key != 'locked'",
                target_condition="tags.stage = 'prod'",
            )
        ],
    )
    assert allowed is False
    assert state["resources"].single_reads == []


# ---- The paths that must cost exactly one read ------------------------------


def test_a_target_condition_reads_the_resource_once(gate):
    """One read for one resource -- not one per clause and not one per role."""
    run, state = gate
    allowed = run(
        [_mutate("run", "r1", RunRequestValues())],
        rows=[
            MutationConditionSpec("run", target_condition="tags.stage = 'prod'"),
            MutationConditionSpec("run", target_condition="tags.team = 'ml'"),
        ],
        values={("run", "r1"): RunResourceValues("r1", tags={"stage": "prod", "team": "ml"})},
    )
    assert allowed is True
    assert state["asked"] == [("ids", "run", ("r1",))] * 2, (
        "one store call per ROW -- not one per clause -- and the resource itself is never "
        f"read on the happy path: {state['asked']}"
    )
    assert state["resources"].single_reads == []


@pytest.mark.parametrize(
    ("resource_type", "request_values", "resource_values"),
    [
        ("experiment", ExperimentRequestValues(), ExperimentResourceValues("1", tags={})),
        (
            "registered_model_version",
            RegisteredModelVersionRequestValues(),
            RegisteredModelVersionResourceValues("m/1", tags={}),
        ),
        ("trace", TraceRequestValues(), TraceResourceValues("t1", tags={})),
    ],
)
def test_types_not_already_loaded_read_exactly_once(
    gate, resource_type, request_values, resource_values
):
    """The types whose base check does not already hold the entity pay one read, never
    more. Parametrized so a newly wired type of this kind enrols here.
    """
    run, state = gate
    resource_id = resource_values.resource_id
    run(
        [_mutate(resource_type, resource_id, request_values)],
        rows=[MutationConditionSpec(resource_type, target_condition="tags.stage = 'prod'")],
        values={(resource_type, resource_id): resource_values},
    )
    # The store is asked in ITS key shape: a version goes down decomposed as
    # ``(name, version)``, never as the auth layer's opaque ``name/version``.
    expected_key = auth_module._condition_pushdown_key(resource_type, resource_id)
    assert state["asked"] == [("ids", resource_type, (expected_key,))], (
        f"one store call for one resource, never more: {state['asked']}"
    )


def test_an_absent_resource_denies_without_a_second_look(gate):
    """A missing resource denies rather than 404ing (so the response is not an oracle),
    and it does so on the first read rather than retrying.
    """
    run, state = gate
    allowed = run(
        [_mutate("run", "r1", RunRequestValues())],
        rows=[MutationConditionSpec("run", target_condition="tags.stage = 'prod'")],
        values={},
    )
    assert allowed is False
    assert state["asked"] == [("ids", "run", ("r1",))], (
        f"one question asked, not a retry: {state['asked']}"
    )


# ---- The unenumerable case, whose cost changed when D21 closed the hole ------


def test_an_unenumerable_target_denies_without_reading(gate):
    """D21. A MUTATE context naming no resource is refused, and the refusal happens
    before any fetch.

    Worth asserting the cost as well as the outcome: this path no longer exits at
    `needs_resource_values` -- it deliberately proceeds into the target loop so it can
    deny -- so the natural worry is that it now reads something. It does not.
    """
    run, state = gate
    allowed = run(
        [
            ConditionContext(
                resource_type="trace",
                scope=ConditionScope.MUTATE,
                request=TraceRequestValues(),
                resource_ids=(),
            )
        ],
        rows=[MutationConditionSpec("trace", target_condition="tags.reviewed = 'yes'")],
    )
    assert allowed is False
    assert state["resources"].single_reads == []


def test_an_unenumerable_target_costs_nothing_without_a_target_condition(gate):
    """The other half: with no target condition the gate returns before the loop, so
    predicate-mode routes keep their current cost exactly.
    """
    run, state = gate
    allowed = run(
        [
            ConditionContext(
                resource_type="trace",
                scope=ConditionScope.MUTATE,
                request=TraceRequestValues(),
                resource_ids=(),
            )
        ],
        rows=[MutationConditionSpec("trace", value_condition="tag_key != 'reviewed'")],
    )
    assert allowed is True
    assert state["resources"].single_reads == []


# ---- One query regardless of how much is configured ------------------------


def test_many_roles_and_types_still_load_in_one_query(gate):
    """Conditions across several roles and types resolve from a single query, the same
    shape `list_grants` uses. Otherwise a user in many roles would pay per role.
    """
    run, state = gate
    contexts = [
        _mutate("run", "r1", RunRequestValues()),
        _mutate("registered_model", "m", RegisteredModelRequestValues()),
    ]
    rows = [
        MutationConditionSpec("run", value_condition="tag_key != 'a'"),
        MutationConditionSpec("run", value_condition="tag_key != 'b'"),
        MutationConditionSpec("registered_model", value_condition="tag_key != 'c'"),
    ]
    assert run(contexts, rows=rows) is True
    assert len(state["store"].condition_loads) == 1
    assert state["store"].condition_loads == [{"run", "registered_model"}]
    assert state["resources"].single_reads == []


def test_the_user_is_resolved_once(gate):
    """Several contexts share one user lookup."""
    run, state = gate
    run([
        _mutate("run", "r1", RunRequestValues()),
        _mutate("registered_model", "m", RegisteredModelRequestValues()),
    ])
    assert state["store"].user_loads == 1


def test_many_ids_cost_one_bulk_call(gate):
    """D11. A bulk delete naming N resources evaluates its condition from ONE fetch.

    Read per id instead, a 100-trace delete would issue 100 round trips to answer a single
    condition -- correct, and slow in proportion to the request. The resource layer had a
    bulk path from the start; this asserts the gate actually takes it, which it did not
    originally do.
    """
    run, state = gate
    trace_ids = tuple(f"t{i}" for i in range(50))
    allowed = run(
        [
            ConditionContext(
                resource_type="trace",
                scope=ConditionScope.MUTATE,
                request=TraceRequestValues(),
                resource_ids=trace_ids,
            )
        ],
        rows=[MutationConditionSpec("trace", target_condition="tags.reviewed = 'yes'")],
        values={
            ("trace", trace_id): TraceResourceValues(trace_id, tags={"reviewed": "yes"})
            for trace_id in trace_ids
        },
    )
    assert allowed is True
    assert state["asked"] == [("ids", "trace", tuple(trace_ids))], (
        f"N traces cost ONE store call naming all N, never one call each: {state['asked']}"
    )
    assert state["resources"].single_reads == []


def test_one_failing_resource_in_a_bulk_set_denies(gate):
    """The bulk path must keep any-fails-denies: one trace not matching denies the whole
    request, rather than deleting the ones that happen to match.
    """
    run, state = gate
    trace_ids = ("t1", "t2", "t3")
    allowed = run(
        [
            ConditionContext(
                resource_type="trace",
                scope=ConditionScope.MUTATE,
                request=TraceRequestValues(),
                resource_ids=trace_ids,
            )
        ],
        rows=[MutationConditionSpec("trace", target_condition="tags.reviewed = 'yes'")],
        values={
            ("trace", "t1"): TraceResourceValues("t1", tags={"reviewed": "yes"}),
            ("trace", "t2"): TraceResourceValues("t2", tags={"reviewed": "no"}),
            ("trace", "t3"): TraceResourceValues("t3", tags={"reviewed": "yes"}),
        },
    )
    assert allowed is False


# ---- The composed experiment validators must query once ----------------------


@pytest.mark.parametrize(
    ("validator_name", "path", "body"),
    [
        (
            "validate_can_set_experiment_tag",
            "/api/2.0/mlflow/experiments/set-experiment-tag",
            {"experiment_id": "1", "key": "team", "value": "ml"},
        ),
        (
            "validate_can_delete_experiment_tag",
            "/api/2.0/mlflow/experiments/delete-experiment-tag",
            {"experiment_id": "1", "key": "team"},
        ),
    ],
)
def test_an_experiment_tag_validator_loads_conditions_once(monkeypatch, validator_name, path, body):
    """The legacy experiment surface composes its grant check and its condition check by hand,
    so a shared grant helper that also evaluates conditions doubles the query. The rest of this
    suite drives the gate directly and cannot see that, so this asserts on the composed
    validator.
    """
    store = CountingStore(
        rows=[MutationConditionSpec("experiment", value_condition="tag_key != 'x'")]
    )
    monkeypatch.setattr(auth_module, "store", store)
    monkeypatch.setattr(
        auth_module, "authenticate_request", lambda: SimpleNamespace(username="alice")
    )
    monkeypatch.setattr(
        auth_module,
        "_get_permission_from_experiment_id",
        lambda: SimpleNamespace(can_update=True),
    )
    monkeypatch.setattr(auth_module, "get_anchor_workspace", lambda *a, **k: _WORKSPACE)
    validator = getattr(auth_module, validator_name)
    with auth_module.app.test_request_context(path, method="POST", json=body):
        assert validator() is True
    assert len(store.condition_loads) == 1, (
        f"{validator_name} loaded conditions {len(store.condition_loads)} times; the grant "
        f"helper it delegates to must not evaluate conditions as well"
    )


# ---- A workspace admin is not restrictable either ---------------------------


def test_a_workspace_admin_bypasses_a_failing_condition(gate):
    """The grant half returns MANAGE for a workspace admin ahead of every other rule including
    DENY, so conditions must not override that. Otherwise a condition becomes a way to
    constrain an admin, which is exactly what conditions are not for.

    A workspace admin normally has `is_admin` false, so the system-admin bypass does not cover
    this case.
    """
    run, state = gate
    allowed = run(
        [
            ConditionContext(
                "run", ConditionScope.MUTATE, RunRequestValues(tags=(("pii", "yes"),)), ("r1",)
            )
        ],
        rows=[MutationConditionSpec("run", value_condition="tag_key != 'pii'")],
        is_ws_admin=True,
    )
    assert allowed is True, "a workspace admin was restricted by a condition"
    resources = state["resources"]
    assert resources.single_reads == [], "an admin bypass must precede every resource read"


def test_the_workspace_admin_check_is_skipped_when_nothing_is_configured(gate):
    """The bypass must not cost a query on the common path: with no conditions configured the
    gate still returns after exactly one lookup and never asks whether the user is an admin.
    """
    run, state = gate
    allowed = run(
        [ConditionContext("run", ConditionScope.MUTATE, RunRequestValues(), ("r1",))],
        rows=[],
    )
    assert allowed is True
    store = state["store"]
    assert len(store.condition_loads) == 1
    assert store.workspace_admin_checks == 0, (
        "the workspace-admin lookup ran even though no condition was configured"
    )


# ---- What the pushdown is for -----------------------------------------------
#
# The whole point of pushing a predicate into SQL is cost, and cost is invisible to
# a correctness test: delete the pushdown call and every behavioural assertion still
# passes, because the enumerate-and-judge fallback answers identically. That is not
# hypothetical -- it is exactly what happened while wiring the cascade, where the
# only thing that caught a missing pushdown call was making the enumerator raise.
#
# So these count instead. They re-patch the stores, which beats the module's autouse
# `_pushdown_declines` fixture, because here the pushdown is the subject rather than
# the thing being held out of the way.


class CountingPushdown:
    """A store that answers predicates and counts how often it was asked.

    ``named_fails`` and ``child_fails`` are what ``find_failing_resource`` returns for
    each selector, stated in that method's own vocabulary: ``None`` for "every resource
    satisfies every clause" and a resource id for "this one failed". There is no third
    answer -- a store that cannot answer raises, because the caller has no fallback.
    """

    def __init__(self, *, named_fails=None, child_fails=None):
        self._named_fails = named_fails
        self._child_fails = child_fails
        self.filter_calls = []
        self.cascade_calls = []

    def find_failing_resource(self, entity, clauses, *, ids=None, parent_id=None):
        # One method, two selectors -- so the counters split on which selector was used
        # rather than on which method was called.
        if parent_id is not None:
            self.cascade_calls.append((entity, parent_id, list(clauses)))
            return self._child_fails
        self.filter_calls.append((entity, list(ids), list(clauses)))
        return self._named_fails


@pytest.fixture
def pushdown(monkeypatch):
    """Install a counting pushdown store, overriding the module's declining one."""

    def install(store):
        def get():
            return store

        # Marked so the `gate` fixture leaves it alone -- the test is asserting on this
        # store's counters.
        get._test_pushdown = True
        monkeypatch.setattr(auth_module, "_get_tracking_store", get)
        monkeypatch.setattr(auth_module, "_get_model_registry_store", get, raising=False)
        return store

    return install


def _cascade(resource_type, parent_id, enumerated=None):
    """A cascade context: no ids, just a parent.

    `enumerated` is accepted and ignored. There is no resolver to run any more -- the
    children are never listed, which is the cost the pushdown exists to avoid -- so a
    caller that still passes a collector gets an empty one, and an assertion that nothing
    was enumerated holds by construction.
    """
    return ConditionContext(
        resource_type=resource_type,
        scope=ConditionScope.MUTATE,
        request=RunRequestValues(),
        resource_ids=(),
        parent_resource_id=parent_id,
    )


def test_a_cascade_costs_one_pushdown_call_and_no_enumeration(gate, pushdown):
    """The headline saving: a 2000-run experiment answers in one query.

    Previously this cost one search page per 500 children plus a fetch each. The
    enumeration is what must be zero -- not merely small.
    """
    run, state = gate
    store = pushdown(CountingPushdown(child_fails=None))
    enumerated = []
    allowed = run(
        [_cascade("run", "exp-1", enumerated)],
        rows=[MutationConditionSpec("run", target_condition="tags.lifecycle != 'prod'")],
    )
    assert allowed is True
    assert len(store.cascade_calls) == 1
    assert enumerated == [], "the children must never be enumerated when the store answered"
    assert state["resources"].single_reads == []


def test_a_denying_cascade_reads_only_the_child_it_names(gate, pushdown):
    """A refusal pays ONE read to say which child caused it, and never enumerates.

    The store answers "this child failed" without enumerating anything, and naming the
    *clause* it broke then costs a single fetch of that one child. That is the whole
    attribution budget: no enumeration, no bulk read, and one single read regardless of
    how many children the parent holds.
    """
    run, state = gate
    store = pushdown(CountingPushdown(child_fails="child-7"))
    enumerated = []
    allowed = run(
        [_cascade("run", "exp-1", enumerated)],
        rows=[MutationConditionSpec("run", target_condition="tags.lifecycle != 'prod'")],
    )
    assert allowed is False
    assert len(store.cascade_calls) == 1
    assert enumerated == [], "the children must never be enumerated when the store answered"
    assert state["resources"].single_reads == [("run", "child-7")], (
        "exactly the child the store named, and only that one"
    )


def test_cost_scales_with_conditions_not_with_children(gate, pushdown):
    """Two conditions is two queries, independent of how many children exist."""
    run, _ = gate
    store = pushdown(CountingPushdown(child_fails=None))
    enumerated = []
    allowed = run(
        [_cascade("run", "exp-1", enumerated)],
        rows=[
            MutationConditionSpec("run", target_condition="tags.lifecycle != 'prod'"),
            MutationConditionSpec("run", target_condition="tags.team = 'ml'"),
        ],
    )
    assert allowed is True
    assert len(store.cascade_calls) == 2
    assert enumerated == []


def test_an_explicit_batch_costs_one_pushdown_call_and_no_read(gate, pushdown):
    """The common case: the request names its ids, so nothing is loaded."""
    run, state = gate
    store = pushdown(CountingPushdown(named_fails=None))
    allowed = run(
        [
            ConditionContext(
                resource_type="run",
                scope=ConditionScope.MUTATE,
                request=RunRequestValues(),
                resource_ids=("r-1", "r-2", "r-3"),
            )
        ],
        rows=[MutationConditionSpec("run", target_condition="tags.lifecycle != 'prod'")],
    )
    assert allowed is True
    assert len(store.filter_calls) == 1, "N ids must cost one query, not N"
    assert store.filter_calls[0][1] == ["r-1", "r-2", "r-3"]
    assert state["resources"].single_reads == []


def test_many_ids_are_still_one_call(gate, pushdown):
    """The bulk-delete shape (D11), where per-id cost was the original complaint."""
    run, state = gate
    store = pushdown(CountingPushdown(named_fails=None))
    ids = tuple(f"r-{i}" for i in range(500))
    allowed = run(
        [
            ConditionContext(
                resource_type="run",
                scope=ConditionScope.MUTATE,
                request=RunRequestValues(),
                resource_ids=ids,
            )
        ],
        rows=[MutationConditionSpec("run", target_condition="tags.lifecycle != 'prod'")],
    )
    assert allowed is True
    assert len(store.filter_calls) == 1


def test_an_admin_asks_the_store_no_predicate(gate, pushdown):
    """Admin bypass precedes everything, including the pushdown."""
    run, _ = gate
    store = pushdown(CountingPushdown(named_fails=None, child_fails=None))
    enumerated = []
    allowed = run(
        [_cascade("run", "exp-1", enumerated)],
        rows=[MutationConditionSpec("run", target_condition="tags.lifecycle != 'prod'")],
        is_admin=True,
    )
    assert allowed is True
    assert store.cascade_calls == []
    assert store.filter_calls == []


def test_a_request_only_condition_asks_the_store_no_predicate(gate, pushdown):
    """No target condition means no resource question, so nothing is pushed."""
    run, _ = gate
    store = pushdown(CountingPushdown(named_fails=None, child_fails=None))
    allowed = run(
        [_mutate("run", "r-1", RunRequestValues(tags=(("lifecycle", "dev"),)))],
        rows=[MutationConditionSpec("run", value_condition="tag_value != 'prod'")],
    )
    assert allowed is True
    assert store.filter_calls == []
    assert store.cascade_calls == []


def test_a_request_denial_short_circuits_before_any_predicate(gate, pushdown):
    """Request conditions are pure, so a denial there must not reach the database."""
    run, _ = gate
    store = pushdown(CountingPushdown(named_fails=None))
    allowed = run(
        [_mutate("run", "r-1", RunRequestValues(tags=(("lifecycle", "prod"),)))],
        rows=[
            MutationConditionSpec(
                "run",
                value_condition="tag_value != 'prod'",
                target_condition="tags.lifecycle != 'prod'",
            )
        ],
    )
    assert allowed is False
    assert store.filter_calls == [], "a pure denial must not pay for a query"


# ---- A resource-scoped row is still pushable -------------------------------


def _recording_pushdown_store(monkeypatch, answer=None):
    """A store that records every pushdown call and reports total satisfaction.

    Overrides the autouse permissive store, whose own docstring invites exactly this.
    """
    from types import SimpleNamespace

    from mlflow.server import auth as auth_module

    calls = []

    def find_failing_resource(entity, clauses, *, ids=None, parent_id=None):
        calls.append((entity, list(ids), list(clauses)))
        # ``None`` is the new "nothing failed", so a caller asking for the default
        # answer gets a pass rather than an echo of the ids.
        return answer

    pushing = SimpleNamespace(
        find_failing_resource=find_failing_resource,
    )

    def get():
        return pushing

    # Marked so `gate` keeps it: this store's call log is the assertion.
    get._test_pushdown = True
    monkeypatch.setattr(auth_module, "_get_tracking_store", get)
    monkeypatch.setattr(auth_module, "_get_model_registry_store", get, raising=False)
    return calls


def test_a_resource_scoped_row_is_pushed_with_only_the_ids_it_governs(gate, monkeypatch):
    """A scoped row must not force the whole context in-memory.

    The ids ARE the query input, so a row naming one resource is pushable -- it just
    narrows which ids go down with it. Charging a scoped row against ids it does not
    govern would be wrong, which is why the ids are narrowed PER ROW rather than the
    pushdown being abandoned.

    Asserted by cost, not by outcome: a fallback would answer identically and only a
    read count distinguishes them.
    """
    run, state = gate
    calls = _recording_pushdown_store(monkeypatch)
    allowed = run(
        [
            ConditionContext(
                resource_type="run",
                scope=ConditionScope.MUTATE,
                request=RunRequestValues(),
                resource_ids=("r1", "r2"),
            )
        ],
        rows=[
            MutationConditionSpec(
                "run", target_condition="tags.stage = 'prod'", resource_pattern="r1"
            ),
            MutationConditionSpec("run", target_condition="tags.team = 'ml'"),
        ],
        values={
            ("run", "r1"): RunResourceValues("r1", tags={"stage": "prod", "team": "ml"}),
            ("run", "r2"): RunResourceValues("r2", tags={"stage": "dev", "team": "ml"}),
        },
    )
    assert allowed is True
    pushed_ids = [ids for _entity, ids, _clauses in calls]
    assert pushed_ids == [["r1"], ["r1", "r2"]], (
        f"the scoped row should carry only the id it governs and the wildcard row both; "
        f"got {pushed_ids}"
    )


def test_a_row_governing_none_of_the_ids_costs_no_query(gate, monkeypatch):
    """A row scoped to a resource not in play is skipped before the store is asked.

    This is a saved query, not a correctness guard, and the distinction is worth pinning:
    pushing an empty id list would be harmless -- the store contract says empty ids return
    an empty set, which compares equal to the empty expectation and denies nothing -- so
    removing the skip changes cost and not outcome. Hence a cost assertion.
    """
    run, state = gate
    calls = _recording_pushdown_store(monkeypatch)
    allowed = run(
        [
            ConditionContext(
                resource_type="run",
                scope=ConditionScope.MUTATE,
                request=RunRequestValues(),
                resource_ids=("r1",),
            )
        ],
        rows=[
            MutationConditionSpec(
                "run", target_condition="tags.stage = 'prod'", resource_pattern="other-run"
            ),
        ],
        values={("run", "r1"): RunResourceValues("r1", tags={})},
    )
    assert allowed is True
    assert calls == [], f"a row governing no id in play must not reach the store: {calls}"
