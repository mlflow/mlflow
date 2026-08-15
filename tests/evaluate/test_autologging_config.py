import contextlib
from unittest import mock

import pytest

from mlflow.models.evaluation.utils.trace import (
    _should_enable_tracing,
    configure_autologging_for_evaluation,
)
from mlflow.utils.import_hooks import _post_import_hooks

# Module names that are guaranteed to be absent from `sys.modules`, so that
# `register_post_import_hook` stores the hook instead of firing it immediately.
# The deferred path is the one the lazy-configuration design targets, and the
# only one affected by the late-binding bug: an immediate fire happens inside
# the loop iteration, where the loop variables still hold the right values.
_FLAVOR_TO_MODULE = {
    "openai": "_fake_module_openai",
    "langchain": "_fake_module_langchain",
}


@pytest.fixture
def cleanup_import_hooks():
    yield
    for module in _FLAVOR_TO_MODULE.values():
        _post_import_hooks.pop(module, None)


def _fire_deferred_hooks(modules=None):
    """Invoke every hook that is still pending for the given modules."""
    for module in modules or _FLAVOR_TO_MODULE.values():
        for hook in _post_import_hooks.get(module) or []:
            hook(mock.Mock(__name__=module))


@contextlib.contextmanager
def _configured(get_autolog_function, integrations=None):
    """Enter `configure_autologging_for_evaluation` with the flavor registry stubbed.

    `_is_trace_autologging_supported` is deliberately NOT patched, so the real
    `log_traces`-in-signature gate is exercised.
    """
    with (
        mock.patch("mlflow.models.evaluation.utils.trace.FLAVOR_TO_MODULE_NAME", _FLAVOR_TO_MODULE),
        mock.patch(
            "mlflow.models.evaluation.utils.trace.AUTOLOGGING_INTEGRATIONS",
            integrations if integrations is not None else {},
        ),
        mock.patch("mlflow.models.evaluation.utils.trace.is_autolog_supported", return_value=True),
        mock.patch(
            "mlflow.models.evaluation.utils.trace.get_autolog_function",
            side_effect=get_autolog_function,
        ),
        configure_autologging_for_evaluation(),
    ):
        yield


def test_deferred_hook_configures_its_own_flavor(cleanup_import_hooks):
    configured_flavors = []

    def get_autolog_function(flavor):
        def autolog(log_traces=True, log_models=True, silent=False, disable=False):
            configured_flavors.append(flavor)

        return autolog

    with _configured(get_autolog_function):
        # Assert inside the context manager: teardown also calls
        # `get_autolog_function`, which would pollute `configured_flavors`.
        _fire_deferred_hooks()
        assert configured_flavors == ["openai", "langchain"]


def test_deferred_hook_uses_its_own_original_config(cleanup_import_hooks):
    integrations = {
        "openai": {"extra_flag": "from_openai"},
        "langchain": {"extra_flag": "from_langchain"},
    }
    received_flags = []

    def get_autolog_function(flavor):
        def autolog(log_traces=True, log_models=True, silent=False, extra_flag=None):
            received_flags.append(extra_flag)

        return autolog

    with _configured(get_autolog_function, integrations=integrations):
        _fire_deferred_hooks()
        assert received_flags == ["from_openai", "from_langchain"]


def _mixed_traceability_autolog(record):
    """`openai` mimics a non-traceable flavor (no `log_traces`, e.g. transformers);
    `langchain` mimics a traceable one. Giving the two flavors different shapes means
    these tests also fail if the hooks are bound to the wrong flavor.
    """

    def get_autolog_function(flavor):
        if flavor == "openai":

            def autolog(log_models=True, silent=False, disable=False):
                record.append((flavor, {"disable": disable}))

        else:

            def autolog(log_traces=True, log_models=True, silent=False, disable=False):
                record.append((
                    flavor,
                    {"log_traces": log_traces, "log_models": log_models, "silent": silent},
                ))

        return autolog

    return get_autolog_function


def test_deferred_hook_enables_tracing_for_traceable_flavor(cleanup_import_hooks):
    record = []

    with _configured(_mixed_traceability_autolog(record)):
        _fire_deferred_hooks([_FLAVOR_TO_MODULE["langchain"]])
        # Tracing on; model logging off so eval does not create LoggedModels.
        assert record == [("langchain", {"log_traces": True, "log_models": False, "silent": True})]


def test_deferred_hook_disables_autolog_for_non_traceable_flavor(cleanup_import_hooks):
    record = []

    with _configured(_mixed_traceability_autolog(record)):
        _fire_deferred_hooks([_FLAVOR_TO_MODULE["openai"]])
        # `_should_enable_tracing` must gate this flavor out and disable it instead.
        assert record == [("openai", {"disable": True})]


@pytest.mark.parametrize(
    ("flavor", "expected"),
    [
        ("langchain", True),
        ("openai", True),
        ("dspy", True),
        ("sklearn", False),
        ("xgboost", False),
        ("transformers", False),
    ],
)
def test_should_enable_tracing_gate(flavor: str, expected: bool):
    # No mocking: exercises the real autolog signatures.
    assert _should_enable_tracing(flavor, {}) is expected


def test_should_enable_tracing_respects_explicit_opt_out():
    assert _should_enable_tracing("langchain", {"langchain": {"log_traces": False}}) is False
    assert _should_enable_tracing("langchain", {"mlflow": {"disable": True}}) is False


def test_teardown_removes_hooks_that_had_no_prior_registration(cleanup_import_hooks):
    def get_autolog_function(flavor):
        def autolog(log_traces=True, log_models=True, silent=False, disable=False):
            pass

        return autolog

    with _configured(get_autolog_function):
        assert all(module in _post_import_hooks for module in _FLAVOR_TO_MODULE.values())

    # `ImportHookFinder` gates on key presence, so leaving a `None` value behind
    # keeps MLflow intercepting imports of these modules for the process lifetime.
    for module in _FLAVOR_TO_MODULE.values():
        assert module not in _post_import_hooks


def test_teardown_restores_pre_existing_hooks(cleanup_import_hooks):
    module = _FLAVOR_TO_MODULE["langchain"]

    def pre_existing_hook(mod):
        pass

    _post_import_hooks[module] = [pre_existing_hook]

    def get_autolog_function(flavor):
        def autolog(log_traces=True, log_models=True, silent=False, disable=False):
            pass

        return autolog

    with _configured(get_autolog_function):
        pass

    assert _post_import_hooks[module] == [pre_existing_hook]
