import pytest
from opentelemetry import trace as otel_trace

import mlflow
from mlflow.tracing.provider import provider


@pytest.fixture(autouse=True)
def clear_autolog_state(reset_tracing):
    # Reset strands tracer singleton to clear cached tracer provider
    try:
        import strands.telemetry.tracer as strands_tracer

        strands_tracer._tracer_instance = None
    except Exception:
        pass

    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    for key in list(AUTOLOGGING_INTEGRATIONS.keys()):
        AUTOLOGGING_INTEGRATIONS[key].clear()
    mlflow.utils.import_hooks._post_import_hooks = {}

    yield

    # Reset OTel provider state for test isolation when switching between
    # MLFLOW_USE_DEFAULT_TRACER_PROVIDER modes (pattern from test_integration.py)
    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE._done = False
    provider._global_provider_init_once._done = False
    provider._isolated_tracer_provider_once._done = False
