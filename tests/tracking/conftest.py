import pytest

import mlflow
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_TRACE_LOGGING


@pytest.fixture(autouse=True)
def enable_async_trace_logging(monkeypatch):
    """Enable async trace logging for all tests in tests/tracking/ to exercise the async path.

    Overrides the global disable_async_trace_logging fixture from tests/conftest.py.
    """
    monkeypatch.setenv(MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.name, "true")


@pytest.fixture
def reset_active_experiment():
    yield
    mlflow.tracking.fluent._active_experiment_id = None
