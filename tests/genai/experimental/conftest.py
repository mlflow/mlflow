"""
Configuration and fixtures for experimental genai tests.

This conftest.py contains fixtures specific to experimental features
that should be isolated from the main genai test suite.

Testing approach:
- If the databricks_ingest package is installed, tests will use the real package
- Without the real package: mocking is applied automatically
- Key tests should be designed to run against both implementations when possible

Examples:
    # Run tests (automatically detects real vs mock implementations)
    pytest tests/genai/experimental/
"""

from enum import Enum
from unittest import mock

import pytest


def _is_zerobus_sdk_available():
    """Check if the real zerobus_sdk package is available."""
    try:
        import zerobus_sdk  # noqa: F401

        return True
    except ImportError:
        return False


class MockStreamState(Enum):
    """Mock StreamState enum that behaves like the real one."""

    UNINITIALIZED = 0
    OPENED = 1
    FLUSHING = 2
    CLOSED = 3
    RECOVERING = 4
    FAILED = 5

    def __str__(self):
        return f"StreamState.{self.name}"


def _create_mock_zerobus_context():
    """Create a mock context for zerobus_sdk when it's not available."""
    # Create a mock stream with the methods we need
    mock_stream = mock.MagicMock()
    mock_stream.get_state.return_value = MockStreamState.OPENED
    mock_stream.ingest_record.return_value = None
    mock_stream.flush.return_value = None

    # Create a mock SDK that returns our mock stream
    mock_sdk = mock.MagicMock()
    mock_sdk.create_stream.return_value = mock_stream

    # Create the mock modules properly
    mock_zerobus_sdk = mock.MagicMock(
        ZerobusSdk=mock.MagicMock(return_value=mock_sdk),
        TableProperties=mock.MagicMock(side_effect=lambda *args: mock.MagicMock()),
    )

    # Create the definitions module with StreamState
    mock_definitions = mock.MagicMock()
    mock_definitions.StreamState = MockStreamState

    # Create the shared module
    mock_shared = mock.MagicMock()
    mock_shared.definitions = mock_definitions

    # Mock the entire module hierarchy
    return mock.patch.dict(
        "sys.modules",
        {
            "zerobus_sdk": mock_zerobus_sdk,
            "zerobus_sdk.shared": mock_shared,
            "zerobus_sdk.shared.definitions": mock_definitions,
        },
    )


@pytest.fixture(autouse=True)
def mock_zerobus_sdk():
    """
    Conditionally mock the zerobus_sdk module when it's not available.

    This fixture checks if the real zerobus_sdk package is available:
    - If available, use the real package
    - If not available, use enum-compatible mocks

    This allows tests to run against the real package when available while
    maintaining compatibility when it's not installed.
    """
    # If real package is available, let tests use the real package
    if _is_zerobus_sdk_available():
        # Real package is available - no mocking needed
        yield
        return

    # Real package not available - apply enum-compatible mocks
    with _create_mock_zerobus_context():
        # Inject StreamState and TableProperties into the module namespace
        # when they're not available from the import
        import mlflow.tracing.export.databricks_delta as delta_module

        if not hasattr(delta_module, "StreamState"):
            delta_module.StreamState = MockStreamState
        if not hasattr(delta_module, "TableProperties"):
            delta_module.TableProperties = mock.MagicMock(
                side_effect=lambda *args: mock.MagicMock()
            )
        yield
