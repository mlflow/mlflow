"""Conftest for scorer tests - imports telemetry fixtures for telemetry testing."""

from unittest import mock

import pytest

# Import telemetry fixtures from the telemetry test module
# These fixtures are used by pytest for test setup and are not directly referenced
from tests.telemetry.conftest import (  # noqa: F401
    bypass_env_check,
    is_mlflow_testing,
    mock_requests,
    mock_requests_get,
    mock_telemetry_client,
    terminate_telemetry_client,
)


@pytest.fixture(autouse=True)
def mock_get_telemetry_client(mock_telemetry_client):
    """Auto-patch get_telemetry_client to return the mock client."""
    with mock.patch(
        "mlflow.telemetry.track.get_telemetry_client", return_value=mock_telemetry_client
    ):
        yield
