# Tests for ``mlflow.server.auth.resources`` -- the auth layer's single resource
# read path, and its request-scoped caches.
#
# The laziness and the memoization are the design here, so a regression would be silent
# rather than loud: these tests assert *how many* store calls happen, and that nothing
# survives the request.

import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from mlflow.entities import Experiment, LifecycleStage, Run, RunData, RunInfo
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.server.auth import resources as auth_resources
from mlflow.server.auth.conditions import (
    RegisteredModelResourceValues,
    RegisteredModelVersionResourceValues,
)


@pytest.fixture(autouse=True)
def clean_cache():
    auth_resources.clear_cache()
    yield
    auth_resources.clear_cache()


def _missing(*_args, **_kwargs):
    raise MlflowException("does not exist", RESOURCE_DOES_NOT_EXIST)


class FakeRegistryStore:
    def __init__(self, entries=None, versions=None):
        self.entries = entries or {}
        self.versions = versions or {}
        self.entry_calls = []
        self.version_calls = []

    def get_registered_model(self, name):
        self.entry_calls.append(name)
        if name not in self.entries:
            _missing()
        return self.entries[name]

    def get_model_version(self, name, version):
        self.version_calls.append((name, version))
        if (name, version) not in self.versions:
            _missing()
        return self.versions[(name, version)]


class FakeTrackingStore:
    def __init__(self, runs=None, experiments=None, traces=None):
        self.runs = runs or {}
        self.experiments = experiments or {}
        self.traces = traces or {}
        self.run_calls = []
        self.experiment_calls = []
        self.trace_calls = []
        self.batch_calls = []

    def get_run(self, run_id):
        self.run_calls.append(run_id)
        if run_id not in self.runs:
            _missing()
        return self.runs[run_id]

    def get_experiment(self, experiment_id):
        self.experiment_calls.append(experiment_id)
        if experiment_id not in self.experiments:
            _missing()
        return self.experiments[experiment_id]

    def get_trace_info(self, trace_id):
        self.trace_calls.append(trace_id)
        if trace_id not in self.traces:
            _missing()
        return self.traces[trace_id]

    def batch_get_trace_infos(self, trace_ids, location=None, experiment_ids=None):
        self.batch_calls.append(list(trace_ids))
        return [self.traces[t] for t in trace_ids if t in self.traces]


class FakeTraceInfo:
    def __init__(self, trace_id, tags=None):
        self.trace_id = trace_id
        self.request_id = trace_id
        self._tags = tags or {}


def _registered_model(name, tags=None, aliases=None):
    rm = RegisteredModel(name, 0)
    for key, value in (tags or {}).items():
        rm._tags[key] = value
    if aliases:
        rm._aliases = dict(aliases)
    return rm


def _run(run_id, experiment_id="0", tags=None):
    info = RunInfo(
        run_id=run_id,
        experiment_id=experiment_id,
        user_id="u",
        status="FINISHED",
        start_time=0,
        end_time=1,
        lifecycle_stage=LifecycleStage.ACTIVE,
    )
    return Run(info, RunData(metrics=[], params=[], tags=[]))


@pytest.fixture
def registry(monkeypatch):
    store = FakeRegistryStore()
    monkeypatch.setattr(auth_resources, "_registry_store", lambda: store)
    return store


@pytest.fixture
def tracking(monkeypatch):
    store = FakeTrackingStore()
    monkeypatch.setattr(auth_resources, "_tracking_store", lambda: store)
    return store


# ---- storage_kind collapsing ----------------------------------------------


def test_prompt_and_registered_model_share_one_cache_entry(registry):
    """Required, not an optimisation: the fetch *precedes* the classification, because
    the routes that tell the families apart do so by reading ``rm._is_prompt()`` off
    the fetched row. At fetch time the auth type is not yet known.
    """
    registry.entries["m"] = _registered_model("m", tags={"lifecycle": "dev"})

    first = auth_resources.attrs_for("registered_model", "m")
    second = auth_resources.attrs_for("prompt", "m")

    assert registry.entry_calls == ["m"], "expected one fetch for both families"
    assert first == second


def test_version_types_share_one_cache_entry(registry):
    registry.versions[("m", "3")] = ModelVersion("m", "3", 0, tags=[])

    auth_resources.attrs_for(
        "registered_model_version", auth_resources.version_resource_id("m", "3")
    )
    auth_resources.attrs_for("prompt_version", auth_resources.version_resource_id("m", "3"))

    assert registry.version_calls == [("m", "3")]


@pytest.mark.parametrize(
    ("resource_type", "expected"),
    [
        ("registered_model", "registry_entry"),
        ("prompt", "registry_entry"),
        ("registered_model_version", "registry_version"),
        ("prompt_version", "registry_version"),
        ("run", "run"),
        ("trace", "trace"),
        ("experiment", "experiment"),
        ("logged_model", "logged_model"),
    ],
)
def test_storage_kind_mapping(resource_type, expected):
    assert auth_resources.storage_kind(resource_type) == expected


# ---- Memoization -----------------------------------------------------------


def test_repeat_fetches_hit_the_cache(tracking):
    tracking.runs["r1"] = _run("r1")
    auth_resources.fetch_run("r1")
    auth_resources.fetch_run("r1")
    auth_resources.attrs_for("run", "r1")
    assert tracking.run_calls == ["r1"]


def test_validator_fetch_makes_the_gate_free(registry):
    """The point of the layer: a base check that already loaded the entity leaves the
    condition gate with zero extra queries.
    """
    registry.entries["m"] = _registered_model("m", tags={"lifecycle": "dev"})

    auth_resources.fetch_registered_model("m")  # what the validator does
    before = len(registry.entry_calls)
    auth_resources.attrs_for("registered_model", "m")  # what the gate does
    assert len(registry.entry_calls) == before == 1


def test_missing_resource_is_cached_negatively(tracking):
    """A second lookup of a missing resource must not re-query -- the answer is the
    same and the query is not free.
    """
    assert auth_resources.attrs_for("run", "nope") is None
    assert auth_resources.attrs_for("run", "nope") is None
    assert tracking.run_calls == ["nope"]


def test_experiment_is_fetched_here_not_via_workspace_resolution(tracking):
    """``_get_resource_workspace`` returns from a TTL cache before calling its fetcher
    and discards the entity anyway, so experiment attributes cannot come from there --
    this layer reads it at evaluation time like the version types.
    """
    tracking.experiments["0"] = Experiment("0", "e", "loc", LifecycleStage.ACTIVE, tags=[])
    values = auth_resources.attrs_for("experiment", "0")
    assert values is not None
    assert tracking.experiment_calls == ["0"]


# ---- Projection ------------------------------------------------------------


def test_tags_are_projected_from_private_attribute(registry):
    """``RegisteredModel.tags`` strips the prompt marker ("should not be user-facing"),
    so reading the public property would make a resource condition on
    ``tags.mlflow.prompt.is_prompt`` silently see nothing -- and D4 permits exactly
    that key on the resource side.
    """
    registry.entries["m"] = _registered_model(
        "m", tags={"mlflow.prompt.is_prompt": "true", "lifecycle": "dev"}
    )
    values = auth_resources.attrs_for("prompt", "m")

    assert values.tags["mlflow.prompt.is_prompt"] == "true"
    assert "mlflow.prompt.is_prompt" not in registry.entries["m"].tags, (
        "guard: the public property is expected to strip it"
    )


def test_aliases_populated_only_for_owning_types(registry):
    """D18: aliases are stored on the registry entry, so a version's alias list names
    its *parent's* aliases. Treating them as the version's own state would let one
    alias be governed under two resource types.

    The version's values carry no ``aliases`` field at all, which is stronger than an
    empty one: an empty mapping says "no aliases right now", while a missing field says
    the type can never have them. Both deny an alias clause, but only the second makes
    projecting one a programming error rather than a silent no-op.
    """
    registry.entries["m"] = _registered_model("m", aliases={"champion": "3"})
    registry.versions[("m", "3")] = ModelVersion("m", "3", 0, tags=[], aliases=["champion"])

    entry = auth_resources.attrs_for("registered_model", "m")
    version = auth_resources.attrs_for(
        "registered_model_version", auth_resources.version_resource_id("m", "3")
    )

    assert entry.aliases == {"champion": "3"}
    assert not hasattr(version, "aliases")
    assert type(version) is RegisteredModelVersionResourceValues
    assert type(entry) is RegisteredModelResourceValues


def test_trace_tags_are_projected(tracking):
    tracking.traces["t1"] = FakeTraceInfo("t1", tags={"lifecycle": "dev"})
    values = auth_resources.attrs_for("trace", "t1")
    assert values.tags == {"lifecycle": "dev"}


# ---- Composite ids ---------------------------------------------------------


def test_version_resource_id_survives_a_slash_in_the_name():
    """The name is percent-encoded so the separator stays unambiguous, following the
    existing ``_scorer_pattern`` precedent.
    """
    composite = auth_resources.version_resource_id("team/model", "3")
    assert composite == "team%2Fmodel/3"
    assert auth_resources._split_version_resource_id(composite) == ("team/model", "3")


def test_version_fetch_uses_the_decoded_name(registry):
    registry.versions[("team/model", "3")] = ModelVersion("team/model", "3", 0, tags=[])
    values = auth_resources.attrs_for(
        "registered_model_version", auth_resources.version_resource_id("team/model", "3")
    )
    assert values is not None
    assert registry.version_calls == [("team/model", "3")]


# ---- Bulk ------------------------------------------------------------------


def test_bulk_traces_issue_one_call_not_n(tracking):
    """D11: ``batch_get_trace_infos`` is already on the abstract store, batches
    internally and skips spans, so N ids cost one call -- no new store method, no cap.
    """
    for i in range(50):
        tracking.traces[f"t{i}"] = FakeTraceInfo(f"t{i}", tags={"i": str(i)})

    resolved = auth_resources.attrs_for_bulk("trace", [f"t{i}" for i in range(50)])

    assert len(resolved) == 50
    assert len(tracking.batch_calls) == 1
    assert tracking.trace_calls == [], "per-id fetches should not run after a bulk fill"


def test_bulk_marks_absent_traces_without_refetching(tracking):
    tracking.traces["t1"] = FakeTraceInfo("t1")
    resolved = auth_resources.attrs_for_bulk("trace", ["t1", "gone"])
    assert resolved["t1"] is not None
    assert resolved["gone"] is None
    assert tracking.trace_calls == [], "a trace the batch omitted does not exist"


def test_bulk_reuses_already_cached_entries(tracking):
    tracking.traces["t1"] = FakeTraceInfo("t1")
    tracking.traces["t2"] = FakeTraceInfo("t2")
    auth_resources.attrs_for("trace", "t1")

    auth_resources.attrs_for_bulk("trace", ["t1", "t2"])
    assert tracking.batch_calls == [["t2"]], "only the miss should be batched"


def test_bulk_dedups_repeated_ids(tracking):
    tracking.traces["t1"] = FakeTraceInfo("t1")
    auth_resources.attrs_for_bulk("trace", ["t1", "t1", "t1"])
    assert tracking.batch_calls == [["t1"]]


def test_bulk_falls_back_to_per_id_for_other_types(registry):
    registry.entries["a"] = _registered_model("a")
    registry.entries["b"] = _registered_model("b")
    resolved = auth_resources.attrs_for_bulk("registered_model", ["a", "b"])
    assert set(resolved) == {"a", "b"}
    assert registry.entry_calls == ["a", "b"]


def test_bulk_failure_falls_back_rather_than_denying(tracking, monkeypatch):
    """A bulk failure must not deny on its own -- the per-id path reaches the same
    decision, so failing the whole request would turn a transient store error into a
    permission error.
    """
    tracking.traces["t1"] = FakeTraceInfo("t1")

    def boom(*_a, **_k):
        raise MlflowException("bulk unavailable")

    monkeypatch.setattr(tracking, "batch_get_trace_infos", boom)
    resolved = auth_resources.attrs_for_bulk("trace", ["t1"])
    assert resolved["t1"] is not None
    assert tracking.trace_calls == ["t1"]


# ---- Strict accessors ------------------------------------------------------


def test_strict_accessors_raise_like_the_store(registry, tracking):
    """A few call sites deliberately let the store's exception propagate and their
    callers catch it. Returning ``None`` there would surface later as an
    ``AttributeError`` instead.
    """
    with pytest.raises(MlflowException, match="does not exist"):
        auth_resources.fetch_registered_model_strict("gone")
    with pytest.raises(MlflowException, match="does not exist"):
        auth_resources.fetch_run_strict("gone")


def test_strict_accessor_returns_the_entity_when_present(registry):
    registry.entries["m"] = _registered_model("m")
    assert auth_resources.fetch_registered_model_strict("m").name == "m"


def test_strict_and_lenient_share_the_cache(registry):
    registry.entries["m"] = _registered_model("m")
    auth_resources.fetch_registered_model("m")
    auth_resources.fetch_registered_model_strict("m")
    assert registry.entry_calls == ["m"]


# ---- Cache lifetime --------------------------------------------------------


def test_clear_cache_empties_both_caches(registry):
    registry.entries["m"] = _registered_model("m")
    auth_resources.attrs_for("registered_model", "m")
    assert auth_resources.cache_sizes() == (1, 1)

    auth_resources.clear_cache()
    assert auth_resources.cache_sizes() == (0, 0)


def test_cache_does_not_leak_across_sequential_uses_on_one_thread(registry):
    """The failure the ``finally`` in both funnels exists to prevent. Unlike Flask's
    ``g``, a ContextVar is not torn down for us, so without an explicit clear a pooled
    worker thread would serve the next request another request's tags.
    """
    registry.entries["m"] = _registered_model("m", tags={"lifecycle": "dev"})
    first = auth_resources.attrs_for("registered_model", "m")
    assert first.tags == {"lifecycle": "dev"}

    auth_resources.clear_cache()  # what before-request completion does

    registry.entries["m"] = _registered_model("m", tags={"lifecycle": "prod"})
    second = auth_resources.attrs_for("registered_model", "m")
    assert second.tags == {"lifecycle": "prod"}, "stale tags leaked past the clear"


def test_caches_are_per_thread(registry):
    """ContextVars are per-execution-context, so two worker threads cannot see each
    other's entries even before the explicit clear.
    """
    registry.entries["m"] = _registered_model("m", tags={"lifecycle": "dev"})
    auth_resources.attrs_for("registered_model", "m")

    seen = {}

    def other_thread():
        seen["sizes"] = auth_resources.cache_sizes()

    thread = threading.Thread(target=other_thread, name="auth-resources-cache-probe")
    thread.start()
    thread.join()

    assert seen["sizes"] == (0, 0)


def test_before_request_clears_the_cache_on_every_exit(monkeypatch):
    """``_before_request`` has six-plus exit points, and a denied request is exactly
    the one whose leftover state would matter -- so the clear is a ``finally`` around
    the whole body rather than a call at the end of the happy path.
    """
    from mlflow.server import auth as auth_module

    cleared = []
    monkeypatch.setattr(auth_resources, "clear_cache", lambda: cleared.append(True))

    # Unprotected-route early return.
    monkeypatch.setattr(auth_module, "_authorize_before_request", lambda: None)
    auth_module._before_request()
    assert len(cleared) == 1

    # A denial.
    monkeypatch.setattr(auth_module, "_authorize_before_request", lambda: "forbidden")
    auth_module._before_request()
    assert len(cleared) == 2

    # An exception on the way through -- call the undecorated function so the
    # exception is not converted to a response before we see it.
    def raises():
        raise MlflowException("boom")

    monkeypatch.setattr(auth_module, "_authorize_before_request", raises)
    with pytest.raises(MlflowException, match="boom"):
        auth_module._before_request.__wrapped__()
    assert len(cleared) == 3


def test_cache_is_scoped_to_the_flask_request_context(registry):
    """Regression guard, from a real failure this caused.

    Validators are also invoked directly -- by tests, and by the GraphQL path -- without
    ``_before_request`` running to clear anything. With the caches held only in
    ContextVars, two such invocations on one thread shared a cache, so the second saw
    the first's entity and never consulted the (possibly re-patched) store. Anchoring to
    ``g`` when a Flask request context exists makes each context start clean, because
    Flask tears ``g`` down for us.
    """
    from mlflow.server import auth as auth_module

    registry.entries["m"] = _registered_model("m", tags={"lifecycle": "dev"})

    with auth_module.app.test_request_context("/api/2.0/mlflow/registered-models/get"):
        auth_resources.attrs_for("registered_model", "m")
        assert auth_resources.cache_sizes() == (1, 1)

    with auth_module.app.test_request_context("/api/2.0/mlflow/registered-models/get"):
        assert auth_resources.cache_sizes() == (0, 0), (
            "a new request context must start with an empty cache"
        )


def test_an_error_still_propagates_in_a_fresh_request_context(registry):
    """The other half of the same regression: a cached entity from an earlier
    invocation would stop a later one from ever calling the store, so a store that
    had started failing would look healthy.
    """
    from mlflow.server import auth as auth_module

    registry.entries["m"] = _registered_model("m")

    with auth_module.app.test_request_context("/"):
        assert auth_resources.fetch_registered_model("m") is not None

    class Broken:
        def get_registered_model(self, name):
            raise RuntimeError("registry store backend is down")

    with auth_module.app.test_request_context("/"):
        with mock.patch.object(auth_resources, "_registry_store", lambda: Broken()):
            with pytest.raises(RuntimeError, match="backend is down"):
                auth_resources.fetch_registered_model("m")


def test_stores_are_resolved_through_the_auth_module_seam(monkeypatch):
    """Regression guard: importing ``_get_tracking_store`` /
    ``_get_model_registry_store`` from ``mlflow.server.handlers`` directly would bypass
    every existing test that substitutes a fake store by patching the auth module's
    attribute -- and authorization would silently run against the real store.
    """
    from mlflow.server import auth as auth_module

    sentinel = object()
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: sentinel)
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: sentinel)

    assert auth_resources._tracking_store() is sentinel
    assert auth_resources._registry_store() is sentinel


def test_cache_key_includes_workspace(registry, monkeypatch):
    """A resource id is not globally unique -- for a registered model the id *is* the
    name, and names are unique per workspace.
    """
    registry.entries["m"] = _registered_model("m", tags={"lifecycle": "dev"})

    with mock.patch("mlflow.utils.workspace_context.get_request_workspace", return_value="ws1"):
        auth_resources.attrs_for("registered_model", "m")
    with mock.patch("mlflow.utils.workspace_context.get_request_workspace", return_value="ws2"):
        auth_resources.attrs_for("registered_model", "m")

    assert registry.entry_calls == ["m", "m"], "workspaces must not share a cache entry"


def test_projected_values_are_always_strings():
    """Every comparison in a condition is a string comparison, so a projected value of any
    other type can satisfy no clause an admin could write. On the resource side that denies
    every mutation of the type rather than allowing it, so the failure is quiet: the
    condition looks configured and simply never matches.

    The registry store returns an alias's version as an ``int``, which is how this was
    found -- ``aliases.champion = '1'`` compared ``1`` against ``'1'``.
    """
    entity = SimpleNamespace(
        _tags={"count": 3, 7: "seven"},
        aliases={"champion": 1, "candidate": 2},
    )
    values = auth_resources.values_for_entity("registered_model", "m", entity)

    assert values.tags == {"count": "3", "7": "seven"}
    assert values.aliases == {"champion": "1", "candidate": "2"}
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in values.tags.items())
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in values.aliases.items())
