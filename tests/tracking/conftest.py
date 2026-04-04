import pytest

import mlflow
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_TRACE_LOGGING


@pytest.fixture(autouse=True)
def enable_async_trace_logging(monkeypatch):
    """Enable async trace logging for all tests in tests/tracking/ to exercise the async path.

    Overrides the global disable_async_trace_logging fixture from tests/conftest.py.
    Terminates async queues on teardown to prevent thread leaks between tests.
    """
    monkeypatch.setenv(MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.name, "true")

    yield

    from mlflow.tracing.processor.base_mlflow import flush_all_batch_processors
    from mlflow.tracing.provider import _get_trace_exporter

    try:
        flush_all_batch_processors(terminate=True)
    except Exception:
        pass
    try:
        if exporter := _get_trace_exporter():
            if hasattr(exporter, "_async_queue"):
                exporter._async_queue.flush(terminate=True)
    except Exception:
        pass


@pytest.fixture
def reset_active_experiment():
    yield
    mlflow.tracking.fluent._active_experiment_id = None
