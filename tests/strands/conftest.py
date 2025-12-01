import pytest

import mlflow


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
