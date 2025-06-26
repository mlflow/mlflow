import logging
import time
from threading import Thread

import pytest
import requests
import uvicorn
from fastapi import FastAPI

from mlflow.telemetry.client import TelemetryClient, get_telemetry_client

_logger = logging.getLogger(__name__)


class MockTelemetryServer:
    """Mock FastAPI server for testing telemetry client."""

    def __init__(self, port=8000):
        self.port = port
        self.app = FastAPI()
        self.received_records = []
        self.server_thread = None
        self.server = None

        @self.app.post("/telemetry")
        async def receive_telemetry(request_data: dict):
            _logger.warning(f"Received telemetry records {request_data} from {self.port}")
            self.received_records.extend(request_data.get("records", []))
            return {"status": "success", "count": len(request_data.get("records", []))}

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}

    def start(self):
        """Start the server in a background thread."""
        self.server = uvicorn.Server(
            config=uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error")
        )
        self.server_thread = Thread(target=self.server.run, daemon=True)
        self.server_thread.start()

        # Wait for server to start
        time.sleep(1)

        # Verify server is running
        try:
            response = requests.get(f"http://127.0.0.1:{self.port}/health")
            assert response.status_code == 200
        except Exception as e:
            pytest.fail(f"Failed to start mock server: {e}")

    def stop(self):
        """Stop the server."""
        if self.server:
            self.server.should_exit = True
        if self.server_thread:
            self.server_thread.join(timeout=5)

    def get_record_count(self):
        """Get the number of received records."""
        return len(self.received_records)


def wait_for_telemetry_threads(client: TelemetryClient = None, timeout: float = 5.0):
    """Wait for telemetry threads to finish to avoid race conditions in tests."""
    telemetry_client = client or get_telemetry_client()
    if telemetry_client is None:
        return

    # Flush the telemetry client to ensure all pending records are processed
    telemetry_client.flush()

    # Wait for the queue to be empty
    start_time = time.time()
    while not telemetry_client._queue.empty() and (time.time() - start_time) < timeout:
        time.sleep(0.1)
