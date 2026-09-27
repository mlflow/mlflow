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

import pytest

from mlflow.server import auth as auth_module
from mlflow.server.auth import resources as auth_resources
from mlflow.server.auth.conditions import ConditionScope

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


# Each row: the validator, a request body, and the resource type + scope its condition must
# name. The scope is part of the assertion because a mutating route declaring CREATE scope
# would skip every resource condition -- a subtler fail-open than declaring none at all.
_WIRED_MUTATIONS = [
    # Registered models and prompts.
    (
        "_validate_can_set_registered_model_or_prompt_tag",
        {"name": "m", "key": "k", "value": "v"},
        "registered_model",
        ConditionScope.MUTATE,
    ),
    (
        "_validate_can_delete_registered_model_or_prompt_tag",
        {"name": "m", "key": "k"},
        "registered_model",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_set_model_or_prompt_version_alias",
        {"name": "m", "alias": "champion", "version": "1"},
        "registered_model",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_delete_model_or_prompt_version_alias",
        {"name": "m", "alias": "champion"},
        "registered_model",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_create_registered_model",
        {"name": "m"},
        "registered_model",
        ConditionScope.CREATE,
    ),
    # Model versions.
    (
        "validate_can_set_model_or_prompt_version_tag",
        {"name": "m", "version": "1", "key": "k", "value": "v"},
        "registered_model_version",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_delete_model_or_prompt_version_tag",
        {"name": "m", "version": "1", "key": "k"},
        "registered_model_version",
        ConditionScope.MUTATE,
    ),
    # Runs.
    (
        "validate_can_set_run_tag",
        {"run_id": "r1", "key": "k", "value": "v"},
        "run",
        ConditionScope.MUTATE,
    ),
    ("validate_can_delete_run_tag", {"run_id": "r1", "key": "k"}, "run", ConditionScope.MUTATE),
    (
        "validate_can_log_batch",
        {"run_id": "r1", "tags": [{"key": "k", "value": "v"}]},
        "run",
        ConditionScope.MUTATE,
    ),
    ("validate_can_create_run", {"experiment_id": "1"}, "run", ConditionScope.CREATE),
    # Traces.
    (
        "validate_can_set_trace_tag_by_request_id",
        {"request_id": "t1", "key": "k", "value": "v"},
        "trace",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_delete_trace_tag_by_request_id",
        {"request_id": "t1", "key": "k"},
        "trace",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_delete_traces",
        {"experiment_id": "1", "request_ids": ["t1"]},
        "trace",
        ConditionScope.MUTATE,
    ),
    # Logged models.
    (
        "validate_can_set_logged_model_tags",
        {"model_id": "m1", "tags": [{"key": "k", "value": "v"}]},
        "logged_model",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_delete_logged_model_tag",
        {"model_id": "m1", "key": "k"},
        "logged_model",
        ConditionScope.MUTATE,
    ),
    # Experiments -- the legacy surface, which calls the conditions half directly.
    (
        "validate_can_set_experiment_tag",
        {"experiment_id": "1", "key": "k", "value": "v"},
        "experiment",
        ConditionScope.MUTATE,
    ),
    (
        "validate_can_delete_experiment_tag",
        {"experiment_id": "1", "key": "k"},
        "experiment",
        ConditionScope.MUTATE,
    ),
    ("validate_can_create_experiment", {"name": "e"}, "experiment", ConditionScope.CREATE),
]


@pytest.mark.parametrize(("validator", "body", "resource_type", "scope"), _WIRED_MUTATIONS)
def test_every_wired_mutation_declares_a_condition(
    recorder, monkeypatch, validator, body, resource_type, scope
):
    """The guard that makes leaving other validators untouched safe.

    Fails if a route stops declaring a context -- which would not otherwise fail anything,
    because the route keeps authorizing exactly as before and only the conditions go quiet.
    """
    if validator == "validate_can_set_experiment_tag":
        monkeypatch.setattr(auth_module, "validate_can_update_experiment", lambda: True)
    if validator == "validate_can_delete_experiment_tag":
        monkeypatch.setattr(auth_module, "validate_can_update_experiment", lambda: True)
    if validator.endswith("_alias"):
        monkeypatch.setattr(auth_module, "_alias_version_requirement_met", lambda: True)

    with auth_module.app.test_request_context("/api/2.0/mlflow/x", method="POST", json=body):
        getattr(auth_module, validator)()

    assert recorder.contexts, f"{validator} declared no ConditionContext -- it is fail-open"
    assert resource_type in recorder.types_at(scope), (
        f"{validator} declared no {resource_type} condition at {scope.name} scope; "
        f"got {[(c.resource_type, c.scope.name) for c in recorder.contexts]}"
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
