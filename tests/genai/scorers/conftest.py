"""Conftest for scorer tests - imports telemetry fixtures for telemetry testing."""

import sys
from pathlib import Path
from unittest import mock

import pytest

# Add telemetry tests to path so we can import fixtures
telemetry_tests_path = Path(__file__).parent.parent.parent / "telemetry"
if str(telemetry_tests_path) not in sys.path:
    sys.path.insert(0, str(telemetry_tests_path))

# Import telemetry fixtures
from conftest import (  # noqa: E402, F401
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
