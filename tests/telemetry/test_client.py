import json
import threading
import time

import pytest

from mlflow.environment_variables import (
    _MLFLOW_TELEMETRY_MAX_QUEUE_SIZE,
    _MLFLOW_TELEMETRY_MAX_WORKERS,
)
from mlflow.telemetry.client import TelemetryClient
from mlflow.telemetry.schemas import APIRecord, APIStatus, LogModelParams, ModelType

from tests.telemetry.helper_functions import wait_for_telemetry_threads


@pytest.fixture
def telemetry_client():
    """Fixture to provide a telemetry client."""
    return TelemetryClient()


def test_telemetry_client_initialization():
    """Test that TelemetryClient initializes correctly."""
    client = TelemetryClient()
    assert client.info is not None
    assert client._queue.maxsize == _MLFLOW_TELEMETRY_MAX_QUEUE_SIZE.get()
    assert client._max_workers == _MLFLOW_TELEMETRY_MAX_WORKERS.get()
    assert not client.is_active


def test_add_record_and_send(telemetry_client: TelemetryClient, mock_requests):
    """Test adding a record and sending it to the mock server."""
    # Create a test record
    record = APIRecord(
        api_name="test_api",
        status=APIStatus.SUCCESS,
        params=LogModelParams(flavor="test_flavor", model=ModelType.PYTHON_FUNCTION),
    )

    # Add record and wait for processing
    telemetry_client.add_record(record)
    wait_for_telemetry_threads(client=telemetry_client)

    # Verify record was sent
    assert len(mock_requests) == 1

    # Check the received data structure
    received_record = mock_requests[0]
    assert "data" in received_record
    assert "partition-key" in received_record

    # Parse the data to verify content
    data = json.loads(received_record["data"])
    assert data["api_name"] == "test_api"
    assert data["status"] == "success"


def test_batch_processing(telemetry_client: TelemetryClient, mock_requests):
    """Test that multiple records are batched correctly."""
    telemetry_client._batch_size = 3  # Set small batch size for testing

    # Add multiple records
    for i in range(5):
        record = APIRecord(
            api_name=f"test_api_{i}",
            status=APIStatus.SUCCESS,
            params=LogModelParams(flavor="test_flavor", model=ModelType.PYTHON_FUNCTION),
        )
        telemetry_client.add_record(record)

    wait_for_telemetry_threads(client=telemetry_client)

    assert len(mock_requests) == 5


def test_flush_functionality(telemetry_client: TelemetryClient, mock_requests):
    """Test that flush properly sends pending records."""
    record = APIRecord(
        api_name="test_api",
        status=APIStatus.SUCCESS,
        params=LogModelParams(flavor="test_flavor", model=ModelType.PYTHON_FUNCTION),
    )
    telemetry_client.add_record(record)

    telemetry_client.flush()

    assert len(mock_requests) == 1


def test_client_shutdown(telemetry_client: TelemetryClient, mock_requests):
    for _ in range(100):
        record = APIRecord(
            api_name="test_api",
            status=APIStatus.SUCCESS,
            params=LogModelParams(flavor="test_flavor", model=ModelType.PYTHON_FUNCTION),
        )
        telemetry_client.add_record(record)
    assert len(mock_requests) == 0

    start_time = time.time()
    telemetry_client.flush(terminate=True)
    end_time = time.time()

    assert end_time - start_time < 5
    assert len(mock_requests) == 100

    assert not telemetry_client.is_active


def test_error_handling():
    """Test that client handles server errors gracefully."""
    client = TelemetryClient()
    client.telemetry_url = "http://127.0.0.1:9999/nonexistent"  # Invalid URL

    # Add a record - should not crash
    record = APIRecord(
        api_name="test_api",
        status=APIStatus.SUCCESS,
        params=LogModelParams(flavor="test_flavor", model=ModelType.PYTHON_FUNCTION),
    )
    client.add_record(record)

    wait_for_telemetry_threads(client=client)

    # Client should still be active despite errors
    assert client.is_active


def test_stop_event_handling(telemetry_client: TelemetryClient, mock_requests):
    """Test that records are not added when stop event is set."""
    # Set stop event
    telemetry_client._stop_event.set()

    record = APIRecord(
        api_name="test_api",
        status=APIStatus.SUCCESS,
        params=LogModelParams(flavor="test_flavor", model=ModelType.PYTHON_FUNCTION),
    )
    telemetry_client.add_record(record)

    wait_for_telemetry_threads(client=telemetry_client)

    # No records should be sent
    assert len(mock_requests) == 0


def test_concurrent_record_addition(telemetry_client: TelemetryClient, mock_requests):
    """Test adding records from multiple threads."""

    def add_records(thread_id):
        for i in range(5):
            record = APIRecord(
                api_name=f"thread_{thread_id}_api_{i}",
                status=APIStatus.SUCCESS,
                params=LogModelParams(flavor="test_flavor", model=ModelType.PYTHON_FUNCTION),
            )
            telemetry_client.add_record(record)
            time.sleep(0.1)

    # Start multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=add_records, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    wait_for_telemetry_threads(client=telemetry_client)

    # Should have received records from all threads
    assert len(mock_requests) == 15


def test_telemetry_info_inclusion(telemetry_client: TelemetryClient, mock_requests):
    """Test that telemetry info is included in records."""
    record = APIRecord(
        api_name="test_api",
        status=APIStatus.SUCCESS,
        params=LogModelParams(flavor="test_flavor", model=ModelType.PYTHON_FUNCTION),
    )
    telemetry_client.add_record(record)

    wait_for_telemetry_threads(client=telemetry_client)

    # Verify telemetry info is included
    received_record = mock_requests[0]
    data = json.loads(received_record["data"])

    # Check that telemetry info fields are present
    for field in [
        "session_id",
        "mlflow_version",
        "python_version",
        "operating_system",
        "backend_store",
    ]:
        assert data[field] == getattr(telemetry_client.info, field)

    # Check that record fields are present
    assert data["api_name"] == "test_api"
    assert data["status"] == "success"


def test_partition_key(telemetry_client: TelemetryClient, mock_requests):
    """Test that partition key is set correctly."""
    record = APIRecord(
        api_name="test_api",
        status=APIStatus.SUCCESS,
        params=LogModelParams(flavor="test_flavor", model=ModelType.PYTHON_FUNCTION),
    )
    telemetry_client.add_record(record)

    wait_for_telemetry_threads(client=telemetry_client)

    # Verify partition key
    received_record = mock_requests[0]
    assert received_record["partition-key"] == "test"


def test_max_workers_setup(monkeypatch):
    """Test max_workers configuration and validation."""
    # Test default value
    client = TelemetryClient()
    assert client._max_workers == _MLFLOW_TELEMETRY_MAX_WORKERS.get()

    # Test invalid value (0 or negative)
    monkeypatch.setenv("_MLFLOW_TELEMETRY_MAX_WORKERS", "0")
    client = TelemetryClient()
    assert client._max_workers == 1  # Should default to 1

    monkeypatch.setenv("_MLFLOW_TELEMETRY_MAX_WORKERS", "-5")
    client = TelemetryClient()
    assert client._max_workers == 1  # Should default to 1

    # Test valid value
    monkeypatch.setenv("_MLFLOW_TELEMETRY_MAX_WORKERS", "8")
    client = TelemetryClient()
    assert client._max_workers == 8

    # Test that correct number of threads are created
    client.activate()
    assert len(client._consumer_threads) == 8

    # Verify thread names
    for i, thread in enumerate(client._consumer_threads):
        assert thread.name == f"MLflowTelemetryConsumer-{i}"
        assert thread.daemon is True
