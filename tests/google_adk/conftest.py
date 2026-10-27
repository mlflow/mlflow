import sys

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import Tracer as SDKTracer

import mlflow


def _rebind_adk_tracers():
    """
    ADK keeps module-level tracers bound to the tracer provider that existed when they were
    created. ``reset_tracing`` replaces the global provider after each test, so point them back
    at whichever provider the next test installs.
    """
    for name, module in list(sys.modules.items()):
        if not name.startswith("google.adk"):
            continue
        for attr, value in list(vars(module).items()):
            if isinstance(value, otel_trace.ProxyTracer):
                value._real_tracer = None
            elif isinstance(value, SDKTracer):
                setattr(module, attr, otel_trace.ProxyTracer(value.instrumentation_info.name))


@pytest.fixture(autouse=True)
def clear_autolog_state(reset_tracing):
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    for key in list(AUTOLOGGING_INTEGRATIONS.keys()):
        AUTOLOGGING_INTEGRATIONS[key].clear()
    mlflow.utils.import_hooks._post_import_hooks = {}

    yield

    mlflow.google_adk.autolog(disable=True)
    _rebind_adk_tracers()
