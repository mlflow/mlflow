import pytest

import mlflow
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_TRACE_LOGGING
from mlflow.tracing.fluent import _flush_pending_async_trace_writes


@pytest.fixture(autouse=True)
def enable_async_trace_logging(monkeypatch):
    """Enable async trace logging for all tests in tests/tracking/ to exercise the async path.

    Overrides the global disable_async_trace_logging fixture from tests/conftest.py.
    Terminates async queues on teardown to prevent thread leaks between tests.
    """
    monkeypatch.setenv(MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.name, "true")

    yield

    _flush_pending_async_trace_writes(terminate=True)


@pytest.fixture
def reset_active_experiment():
    yield
    mlflow.tracking.fluent._active_experiment_id = None
