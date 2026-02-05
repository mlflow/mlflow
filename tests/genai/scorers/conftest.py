"""Conftest for scorer tests - imports telemetry fixtures for telemetry testing."""

from unittest import mock

import pytest

import mlflow.telemetry.utils

# Import telemetry fixtures from the telemetry test module
# Autouse fixtures (terminate_telemetry_client, mock_requests_get)
# are imported for their side effects and run automatically
from tests.telemetry.conftest import (  # noqa: F401
    mock_requests,
    mock_requests_get,
    mock_telemetry_client,
    terminate_telemetry_client,
)


@pytest.fixture(autouse=True)
def mock_get_telemetry_client(mock_telemetry_client, monkeypatch):
    # Enable telemetry for tests using this fixture
    monkeypatch.setattr(mlflow.telemetry.utils, "_IS_MLFLOW_TESTING_TELEMETRY", True)
    with mock.patch(
        "mlflow.telemetry.track.get_telemetry_client", return_value=mock_telemetry_client
    ):
        yield
