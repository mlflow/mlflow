import pytest

import mlflow
import mlflow.telemetry.utils
from mlflow.telemetry.client import set_telemetry_client


@pytest.fixture(autouse=True)
def mock_is_ci_env(monkeypatch):
    # patch this so we can run telemetry tests, but avoid
    # tracking other tests in CI
    monkeypatch.setattr(mlflow.telemetry.utils, "_IS_IN_CI_ENV_OR_TESTING", False)
    monkeypatch.setattr(mlflow.telemetry.utils, "_IS_MLFLOW_DEV_VERSION", False)
    set_telemetry_client()
