import threading

import pytest

from tests.telemetry.helper_functions import MockTelemetryServer


@pytest.fixture(autouse=True, scope="module")
def cleanup_zombie_threads():
    for thread in threading.enumerate():
        if thread != threading.main_thread():
            thread.join(timeout=1)


@pytest.fixture
def mock_server():
    """Fixture to provide a mock telemetry server."""
    server = MockTelemetryServer(port=8001)
    server.start()
    yield server
    server.stop()
