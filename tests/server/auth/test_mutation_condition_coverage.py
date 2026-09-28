# Coverage guards for mutation-condition wiring.
#
# An in-scope mutating route that forgets to declare a ConditionContext is **fail-open**:
# it keeps working, every test keeps passing, and the conditions an admin configured
# silently do not apply to it. Nothing else in the suite notices, because the route's own
# tests only assert its permission behaviour.
#
# These guards are therefore what makes the decision not to refactor every validator safe.
# They are deliberately BEHAVIOURAL: each drives the real validator through a request
# context and observes whether a context was declared. A guard that instead checked a
# lookup table would prove only that an entry exists, not that the validator consults it --
# which is exactly the assurance that turned out to be false when the extractor registry
# was removed.

from types import SimpleNamespace

import flask
import pytest

from mlflow.server import auth as auth_module
from mlflow.server.auth import resources as auth_resources
from mlflow.server.auth.conditions import (
    ConditionContext,
    ConditionScope,
    MutationConditionSpec,
    TraceRequestValues,
    TraceResourceValues,
)

_WORKSPACE = "team-a"


class _Recorder:
    """Captures the ConditionContexts a validator declares."""

    def __init__(self):
        self.contexts = []

    def record(self, *contexts_lists):
        for contexts in contexts_lists:
            self.contexts.extend(contexts or ())

    def types_at(self, scope):
        return {c.resource_type for c in self.contexts if c.scope is scope}


@pytest.fixture
def recorder(monkeypatch):
    """Intercepts both gates, so a route is covered whether it routes conditions through
    `authorize` or calls the conditions half directly (the legacy experiment surface).
    """
    rec = _Recorder()

    def fake_authorize(username, anchor, requirements, *, conditions=(), workspace=None):
        rec.record(conditions)
        return True

    def fake_conditions(username, workspace, contexts):
        rec.record(contexts)
        return True

    monkeypatch.setattr(auth_module, "authorize", fake_authorize)
    monkeypatch.setattr(auth_module, "authorize_on_conditions", fake_conditions)
    monkeypatch.setattr(
        auth_module, "authenticate_request", lambda: SimpleNamespace(username="alice")
    )
    monkeypatch.setattr(auth_module, "_user_can_create_in_workspace", lambda: True)

    def fake_create_not_denied(username, created_type, resource_id="*", *, conditions=()):
        # The create path passes its conditions here rather than to `authorize`, so the
        # capture has to happen at this seam too or a create would look unwired.
        rec.record(conditions)
        return True

    monkeypatch.setattr(auth_module, "_create_not_denied", fake_create_not_denied)
    monkeypatch.setattr(auth_module, "get_anchor_workspace", lambda *a, **k: _WORKSPACE)

    # The fetches each wired helper performs before it declares its context. Stubbed so the
    # guard exercises the wiring rather than a store.
    def a_registered_model(name):
        # `_is_prompt` is a method on the real entity, and the strict variant is what the
        # registry classification calls.
        return SimpleNamespace(
            name=name, _tags={}, aliases={}, _is_prompt=lambda: False, workspace=_WORKSPACE
        )

    monkeypatch.setattr(auth_resources, "fetch_registered_model", a_registered_model)
    monkeypatch.setattr(auth_resources, "fetch_registered_model_strict", a_registered_model)
    monkeypatch.setattr(
        auth_resources,
        "fetch_run",
        lambda run_id: SimpleNamespace(info=SimpleNamespace(experiment_id="1", run_id=run_id)),
    )
    monkeypatch.setattr(
        auth_resources,
        "fetch_trace_info",
        lambda trace_id: SimpleNamespace(experiment_id="1", trace_id=trace_id),
    )
    monkeypatch.setattr(
        auth_resources,
        "fetch_logged_model",
        lambda model_id: SimpleNamespace(experiment_id="1", model_id=model_id),
    )
    monkeypatch.setattr(auth_module, "_entity_is_prompt", lambda msg: False)
    monkeypatch.setattr(auth_module, "_request_targets_prompt", lambda *a, **k: False)
    return rec


# Each row: the validator, the REAL route (path and method) it is registered on, and the
# resource type + scope its condition must name.
#
# The path and method are part of the fixture because the request's shape decides where a
# value is read from: a POST carries it in the body, a DELETE in the query string or the
# path, and `_get_request_param` merges `view_args` last. An earlier version of this guard
# posted a synthetic body to every route, which let `DeleteLoggedModelTag` pass while
# asking for `key` when its path parameter is named `tag_key` -- the route 400'd in
# production and the guard was green. Driving the registered shape is what closes that.
#
# The scope is asserted because a mutating route declaring CREATE scope would skip every
# resource condition -- a subtler fail-open than declaring none at all.
_WIRED_MUTATIONS = [
    # Registered models and prompts.
    (
        "_validate_can_set_registered_model_or_prompt_tag",
        "/api/2.0/mlflow/registered-models/set-tag",
        "POST",
        {"name": "m", "key": "k", "value": "v"},
        "registered_model",
        ConditionScope.MUTATE,
    ),
    (
        "_validate_can_delete_registered_model_or_prompt_tag",
        "/api/2.0/mlflow/registered-models/delete-tag",
        "DELETE",
        {"name": "m", "key": "k"},
        "registered_model",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_set_model_or_prompt_version_alias",
        "/api/2.0/mlflow/registered-models/alias",
        "POST",
        {"name": "m", "alias": "champion", "version": "1"},
        "registered_model",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_delete_model_or_prompt_version_alias",
        "/api/2.0/mlflow/registered-models/alias",
        "DELETE",
        {"name": "m", "alias": "champion"},
        "registered_model",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_create_registered_model",
        "/api/2.0/mlflow/registered-models/create",
        "POST",
        {"name": "m"},
        "registered_model",
        ConditionScope.CREATE,
    ),
    # Model versions.
    (
        "validate_can_set_model_or_prompt_version_tag",
        "/api/2.0/mlflow/model-versions/set-tag",
        "POST",
        {"name": "m", "version": "1", "key": "k", "value": "v"},
        "registered_model_version",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_delete_model_or_prompt_version_tag",
        "/api/2.0/mlflow/model-versions/delete-tag",
        "DELETE",
        {"name": "m", "version": "1", "key": "k"},
        "registered_model_version",
        ConditionScope.MUTATE,
    ),
    # Runs.
    (
        "validate_can_set_run_tag",
        "/api/2.0/mlflow/runs/set-tag",
        "POST",
        {"run_id": "r1", "key": "k", "value": "v"},
        "run",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_delete_run_tag",
        "/api/2.0/mlflow/runs/delete-tag",
        "POST",
        {"run_id": "r1", "key": "k"},
        "run",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_log_batch",
        "/api/2.0/mlflow/runs/log-batch",
        "POST",
        {"run_id": "r1", "tags": [{"key": "k", "value": "v"}]},
        "run",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_create_run",
        "/api/2.0/mlflow/runs/create",
        "POST",
        {"experiment_id": "1"},
        "run",
        ConditionScope.CREATE,
    ),
    # Traces. The id is a PATH parameter on the tag routes.
    (
        "validate_can_set_trace_tag_by_request_id",
        "/api/2.0/mlflow/traces/t1/tags",
        "PATCH",
        {"key": "k", "value": "v"},
        "trace",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_delete_trace_tag_by_request_id",
        "/api/2.0/mlflow/traces/t1/tags?key=k",
        "DELETE",
        None,
        "trace",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_set_trace_tag_by_trace_id",
        "/api/3.0/mlflow/traces/t1/tags",
        "PATCH",
        {"key": "k", "value": "v"},
        "trace",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_delete_trace_tag_by_trace_id",
        "/api/3.0/mlflow/traces/t1/tags?key=k",
        "DELETE",
        None,
        "trace",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_delete_traces",
        "/api/2.0/mlflow/traces/delete-traces",
        "POST",
        {"experiment_id": "1", "request_ids": ["t1"]},
        "trace",
        ConditionScope.MUTATE,
    ),
    # Logged models. The tag key is a PATH parameter on the delete, named `tag_key`.
    (
        "validate_can_set_logged_model_tags",
        "/api/2.0/mlflow/logged-models/m1/tags",
        "PATCH",
        {"tags": [{"key": "k", "value": "v"}]},
        "logged_model",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_delete_logged_model_tag",
        "/api/2.0/mlflow/logged-models/m1/tags/k",
        "DELETE",
        None,
        "logged_model",
        ConditionScope.MUTATE,
    ),
    # Experiments -- the legacy surface, which calls the conditions half directly.
    (
        "validate_can_set_experiment_tag",
        "/api/2.0/mlflow/experiments/set-experiment-tag",
        "POST",
        {"experiment_id": "1", "key": "k", "value": "v"},
        "experiment",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_delete_experiment_tag",
        "/api/2.0/mlflow/experiments/delete-experiment-tag",
        "POST",
        {"experiment_id": "1", "key": "k"},
        "experiment",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_create_experiment",
        "/api/2.0/mlflow/experiments/create",
        "POST",
        {"name": "e"},
        "experiment",
        ConditionScope.CREATE,
    ),
]


@pytest.mark.parametrize(
    ("validator", "path", "method", "body", "resource_type", "scope"), _WIRED_MUTATIONS
)
def test_every_wired_mutation_declares_a_condition(
    recorder, monkeypatch, validator, path, method, body, resource_type, scope
):
    """The guard that makes leaving other validators untouched safe.

    Fails if a route stops declaring a context -- which would not otherwise fail anything,
    because the route keeps authorizing exactly as before and only the conditions go quiet.
    """
    if validator in ("validate_can_set_experiment_tag", "validate_can_delete_experiment_tag"):
        # The grant half of the legacy experiment surface. Stubbed at the permission lookup
        # rather than at `validate_can_update_experiment`, because the tag validators no longer
        # delegate to it -- doing so ran the condition query twice.
        monkeypatch.setattr(
            auth_module,
            "_get_permission_from_experiment_id",
            lambda: SimpleNamespace(can_update=True),
        )
    if validator.endswith("_alias"):
        monkeypatch.setattr(auth_module, "_alias_version_requirement_met", lambda: True)

    with auth_module.app.test_request_context(path, method=method, json=body):
        getattr(auth_module, validator)()

    assert recorder.contexts, f"{validator} declared no ConditionContext -- it is fail-open"
    assert resource_type in recorder.types_at(scope), (
        f"{validator} declared no {resource_type} condition at {scope.name} scope; "
        f"got {[(c.resource_type, c.scope.name) for c in recorder.contexts]}"
    )


@pytest.mark.parametrize(
    ("validator", "path", "method", "body", "resource_type", "scope"), _WIRED_MUTATIONS
)
def test_every_wired_mutation_extracts_its_values(
    recorder, monkeypatch, validator, path, method, body, resource_type, scope
):
    """Declaring a context is necessary but not sufficient: it must carry the values the
    request actually names.

    A route that declares a correctly-typed context and passes nothing looks wired, and
    then permits everything -- empty values read as vacuous (D20). So every route that
    names a tag or an alias must be seen to have extracted one. The create rows whose body
    carries no tag are the exception, and are listed as such.
    """
    if validator in ("validate_can_set_experiment_tag", "validate_can_delete_experiment_tag"):
        # The grant half of the legacy experiment surface. Stubbed at the permission lookup
        # rather than at `validate_can_update_experiment`, because the tag validators no longer
        # delegate to it -- doing so ran the condition query twice.
        monkeypatch.setattr(
            auth_module,
            "_get_permission_from_experiment_id",
            lambda: SimpleNamespace(can_update=True),
        )
    if validator.endswith("_alias"):
        monkeypatch.setattr(auth_module, "_alias_version_requirement_met", lambda: True)

    # These bodies deliberately name neither a tag nor an alias.
    carries_nothing = {
        "validate_can_create_registered_model",
        "validate_can_create_run",
        "validate_can_create_experiment",
        "validate_can_delete_traces",
    }

    with auth_module.app.test_request_context(path, method=method, json=body):
        getattr(auth_module, validator)()

    extracted = [(c.request.tags, getattr(c.request, "aliases", ())) for c in recorder.contexts]
    if validator in carries_nothing:
        return
    assert any(tags or aliases for tags, aliases in extracted), (
        f"{validator} declared a context but extracted no tag or alias from the request, "
        f"so its condition is vacuous and permits everything; got {extracted}"
    )


def test_a_tag_bearing_body_reaches_the_condition(recorder, monkeypatch):
    """Declaring a context is necessary but not sufficient: it must carry the body's tag.

    A route could declare a correctly-typed context and pass empty values, which a request
    condition then treats as vacuous (D20) and permits. That is the same fail-open, one
    level down, so the tag itself is asserted.
    """
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/x",
        method="POST",
        json={"run_id": "r1", "key": "lifecycle", "value": "prod"},
    ):
        auth_module.validate_can_set_run_tag()

    (context,) = recorder.contexts
    assert context.request.tags == (("lifecycle", "prod"),)


def test_a_delete_reaches_the_condition_with_a_none_value(recorder, monkeypatch):
    """The delete signal is the pairing `(key, None)`: it gates on the key (D12) while
    leaving a `tag_value` clause vacuous (D13). An empty string would satisfy a value
    clause instead, so the None is asserted rather than just the key.
    """
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/x", method="POST", json={"run_id": "r1", "key": "lifecycle"}
    ):
        auth_module.validate_can_delete_run_tag()

    (context,) = recorder.contexts
    assert context.request.tags == (("lifecycle", None),)


# ---- The FastAPI funnel ------------------------------------------------------
#
# Native FastAPI routes (`/gateway/*`, `/v1/traces`, `/ajax-api/3.0/jobs`, the assistant,
# the artifact proxy, MCP) bypass Flask's `_before_request` entirely, so they reach the gate
# with NO Flask request context. `/v1/traces` is OTel trace ingest, and it authorizes as a
# trace create -- so the CREATE-scope context really does get declared on that path.
#
# Nothing in the gate may therefore touch a Flask global. A regression here would be
# invisible under the test suite's Flask client and break uvicorn deployments only, which is
# the worst possible place to find out.


def _conditioned_store(**kwargs):
    class Store:
        def get_user(self, username):
            return SimpleNamespace(id=1, username=username, is_admin=False)

        def is_workspace_admin(self, user_id, workspace):
            return False

        def list_mutation_conditions_for_user(self, user_id, workspace, resource_types):
            return [MutationConditionSpec("trace", **kwargs)]

    return Store()


def test_the_gate_evaluates_a_request_condition_with_no_flask_context(monkeypatch):
    """The OTel-ingest shape: a CREATE-scope decision taken outside any request context."""
    from flask import has_request_context

    assert not has_request_context(), "this test must run outside a request context"
    monkeypatch.setattr(
        auth_module, "store", _conditioned_store(value_condition="tag_key != 'pii'")
    )
    context = ConditionContext(
        resource_type="trace",
        scope=ConditionScope.CREATE,
        request=TraceRequestValues(tags=(("pii", "x"),)),
    )
    assert auth_module.authorize_on_conditions("alice", "ws", [context]) is False


def test_the_gate_evaluates_a_target_condition_with_no_flask_context(monkeypatch):
    """The resource-reading half, which is where a Flask `g` access would hide: the caches
    are keyed on `g` when a request context exists and on ContextVars when it does not.
    """
    from flask import has_request_context

    assert not has_request_context()
    monkeypatch.setattr(
        auth_module, "store", _conditioned_store(target_condition="tags.reviewed = 'yes'")
    )
    monkeypatch.setattr(
        auth_resources,
        "attrs_for_bulk",
        lambda rt, ids: {i: TraceResourceValues(i, tags={"reviewed": "no"}) for i in ids},
    )
    context = ConditionContext(
        resource_type="trace",
        scope=ConditionScope.MUTATE,
        request=TraceRequestValues(),
        resource_ids=("t1",),
    )
    assert auth_module.authorize_on_conditions("alice", "ws", [context]) is False


def test_clearing_the_cache_outside_a_request_context_is_safe():
    """The FastAPI middleware calls this on every request, including ones that never
    touched a resource. It must not require a Flask context to do so.
    """
    from flask import has_request_context

    assert not has_request_context()
    auth_resources.clear_cache()


# ---- Reads must declare no condition at all ---------------------------------
#
# The invariant is that conditions gate mutations and never reads. Asserting it on the
# *declaration* rather than the outcome is what makes it checkable: a read that declares a
# MUTATE context is already wrong, even if the currently-configured conditions happen to
# permit it. A request-only condition hides this, because empty request tags are vacuous and
# pass -- so the bug only shows with a target condition, on the read paths that share a helper
# with their mutating siblings.

_READ_VALIDATORS = [
    ("validate_can_read_run", "/api/2.0/mlflow/runs/get", "GET", {"run_id": "r1"}),
    (
        "validate_can_read_trace_by_request_id",
        "/api/2.0/mlflow/traces/get",
        "GET",
        {"request_id": "t1"},
    ),
    ("validate_can_read_trace_by_trace_id", "/api/3.0/mlflow/traces/t1", "GET", {"trace_id": "t1"}),
    (
        "validate_can_read_logged_model",
        "/api/2.0/mlflow/logged-models/m1",
        "GET",
        {"model_id": "m1"},
    ),
]


@pytest.mark.parametrize(("validator_name", "path", "method", "params"), _READ_VALIDATORS)
def test_a_read_declares_no_condition(recorder, monkeypatch, validator_name, path, method, params):
    validator = getattr(auth_module, validator_name, None)
    if validator is None:
        pytest.skip(f"{validator_name} is not defined on this branch")
    with auth_module.app.test_request_context(path, method=method, query_string=params):
        for key, value in params.items():
            monkeypatch.setitem(flask.request.view_args or {}, key, value)
        validator()
    assert recorder.types_at(ConditionScope.MUTATE) == set(), (
        f"{validator_name} is a READ but declared a MUTATE condition; a target condition would "
        f"then deny reading the resource, which conditions must never do"
    )
    assert recorder.types_at(ConditionScope.CREATE) == set(), (
        f"{validator_name} is a READ but declared a CREATE condition"
    )


# ---- Creating a version must not smuggle a tag past the condition -----------


def test_creating_a_model_version_declares_its_tags(recorder, monkeypatch):
    """`CreateModelVersion` carries `tags` and the handler passes them straight to the store,
    so a tag condition that gates `SetModelVersionTag` is worthless unless the create declares
    the same values: a restricted tag could simply be set at creation time instead.
    """
    from mlflow.protos.model_registry_pb2 import CreateModelVersion

    msg = CreateModelVersion(name="m", source="s")
    msg.tags.add(key="approved", value="yes")
    monkeypatch.setattr(auth_module, "_get_request_message", lambda _proto: msg)
    monkeypatch.setattr(
        auth_module, "_registered_model_or_prompt_target", lambda: ("registered_model", "m")
    )
    monkeypatch.setattr(auth_module, "is_models_uri", lambda _s: False)
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/create", method="POST", json={"name": "m", "source": "s"}
    ):
        auth_module.validate_can_create_model_version()
    declared = [
        (c.resource_type, tuple(c.request.tags))
        for c in recorder.contexts
        if c.scope is ConditionScope.CREATE
    ]
    assert declared, "create-version declared no CREATE condition, so its tags are unconditioned"
    assert any("approved" in [k for k, _ in tags] for _t, tags in declared), (
        f"the request's tag never reached a condition; declared={declared}"
    )


# ---- Trace producers must declare their own tags ----------------------------
#
# All three carry tags the handler persists, so a condition gating SetTraceTag is avoidable
# unless the producers declare the same values.


def _declared_tag_keys(recorder, scope):
    keys = []
    for c in recorder.contexts:
        if c.scope is scope:
            keys.extend(k for k, _ in (c.request.tags or ()))
    return keys


def test_starting_a_trace_declares_its_tags(recorder, monkeypatch):
    from mlflow.protos.service_pb2 import StartTrace

    msg = StartTrace(experiment_id="1")
    msg.tags.add(key="approved", value="yes")
    monkeypatch.setattr(auth_module, "_get_request_message", lambda _proto: msg)
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/traces", method="POST", json={"experiment_id": "1"}
    ):
        auth_module.validate_can_start_trace()
    assert "approved" in _declared_tag_keys(recorder, ConditionScope.CREATE), (
        "StartTrace persists its tags but declared none to the condition"
    )


def test_starting_a_trace_v3_declares_its_tags(recorder, monkeypatch):
    from mlflow.protos.service_pb2 import StartTraceV3

    msg = StartTraceV3()
    msg.trace.trace_info.trace_location.mlflow_experiment.experiment_id = "1"
    msg.trace.trace_info.tags["approved"] = "yes"
    monkeypatch.setattr(auth_module, "_get_request_message", lambda _proto: msg)
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/traces", method="POST", json={"trace": {}}
    ):
        auth_module.validate_can_start_trace_v3()
    assert "approved" in _declared_tag_keys(recorder, ConditionScope.CREATE), (
        "StartTraceV3 persists trace_info.tags but declared none to the condition"
    )


def test_ending_a_trace_declares_its_tags(recorder, monkeypatch):
    from mlflow.protos.service_pb2 import EndTrace

    msg = EndTrace(request_id="t1")
    msg.tags.add(key="approved", value="yes")
    monkeypatch.setattr(auth_module, "_get_request_message", lambda _proto: msg)
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/traces/t1", method="PATCH", json={"request_id": "t1"}
    ):
        auth_module.validate_can_update_trace_by_request_id()
    assert "approved" in _declared_tag_keys(recorder, ConditionScope.MUTATE), (
        "EndTrace persists its tags but declared none to the condition"
    )


# ---- The other two run producers --------------------------------------------


def test_a_promptlab_run_create_declares_its_tags(recorder, monkeypatch):
    """PromptLab creates a run with caller-supplied tags, so it is a run create like any other
    and its tags must be judged. Only the primary CreateRun producer was covered.
    """
    body = {
        "experiment_id": "1",
        "prompt_template": "t",
        "prompt_parameters": [],
        "model_route": "r",
        "model_input": "i",
        "tags": [{"key": "pii", "value": "yes"}],
    }
    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow/runs/create-promptlab-run", method="POST", json=body
    ):
        auth_module.validate_can_create_promptlab_run()
    keys = [k for c in recorder.contexts for k, _ in (c.request.tags or ())]
    assert "pii" in keys, f"the caller's tag never reached a condition; got {keys}"


def test_an_issue_detection_run_create_declares_its_derived_tags(recorder, monkeypatch):
    """Issue detection DERIVES its run tags in the handler rather than passing them through, so
    the auth layer reconstructs them. Pinning the reconstruction is what makes the coupling
    survivable: if the handler's tag set changes, this fails rather than a condition going
    quiet.
    """
    body = {
        "experiment_id": "1",
        "categories": ["toxicity", "pii"],
        "endpoint_name": "my-endpoint",
        "trace_ids": ["t1", "t2", "t3"],
    }
    with auth_module.app.test_request_context(
        "/ajax-api/3.0/mlflow/issues/detect", method="POST", json=body
    ):
        auth_module.validate_can_invoke_issue_detection()
    declared = dict(pair for c in recorder.contexts for pair in (c.request.tags or ()) if pair[0])
    assert declared.get("categories") == "toxicity,pii"
    assert declared.get("model") == "gateway:/my-endpoint"
    assert declared.get("total_traces") == "3", "an int must be projected as a string"
    assert declared.get("endpoint_name") == "my-endpoint"


def test_an_issue_detection_projection_matches_the_handlers_normalization(recorder, monkeypatch):
    """The handler lowercases the provider and deduplicates the trace ids BEFORE building its
    tags, so projecting the raw request values would judge a different string than the one
    stored -- and a condition would permit exactly the value it was written to reject.

    The earlier test used an endpoint name and unique ids, so it could not see either drift.
    """
    body = {
        "experiment_id": "1",
        "categories": ["toxicity"],
        "provider": "OpenAI",
        "model": "gpt",
        "trace_ids": ["t1", "t1", "t2"],
    }
    with auth_module.app.test_request_context(
        "/ajax-api/3.0/mlflow/issues/detect", method="POST", json=body
    ):
        auth_module.validate_can_invoke_issue_detection()
    declared = dict(pair for c in recorder.contexts for pair in (c.request.tags or ()) if pair[0])
    assert declared.get("model") == "openai:/gpt", (
        "the provider must be lowercased as the handler lowercases it"
    )
    assert declared.get("total_traces") == "2", (
        "duplicate trace ids must be collapsed as the handler collapses them"
    )


def test_an_issue_detection_projection_survives_a_malformed_trace_id(recorder, monkeypatch):
    """A JSON body can carry a list or object where a trace id belongs, and those are
    unhashable. Deduplicating them raised TypeError from authorization, before the request was
    ever judged -- so a malformed body failed as a crash in the auth layer instead of in the
    handler's own validation.
    """
    body = {
        "experiment_id": "1",
        "categories": ["toxicity"],
        "provider": "openai",
        "model": "gpt",
        "trace_ids": ["t1", ["nested"], {"k": "v"}, "t1"],
    }
    with auth_module.app.test_request_context(
        "/ajax-api/3.0/mlflow/issues/detect", method="POST", json=body
    ):
        auth_module.validate_can_invoke_issue_detection()
    declared = dict(pair for c in recorder.contexts for pair in (c.request.tags or ()) if pair[0])
    assert declared.get("total_traces") == "1", (
        f"the hashable id should still be counted once; got {declared.get('total_traces')}"
    )
