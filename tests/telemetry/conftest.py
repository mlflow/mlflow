from unittest.mock import patch

import pytest

from mlflow.telemetry.client import set_telemetry_client


@pytest.fixture(autouse=True)
def mock_is_ci_env():
    # patch this so we can run telemetry tests, but avoid
    # tracking other tests in CI
    with patch("mlflow.telemetry.utils._is_ci_env", return_value=False):
        set_telemetry_client()
        yield
