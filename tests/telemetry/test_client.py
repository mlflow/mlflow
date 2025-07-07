import json
import threading
import time

import pytest

from mlflow.telemetry.client import MAX_QUEUE_SIZE, MAX_WORKERS, TelemetryClient
from mlflow.telemetry.schemas import APIRecord, APIStatus, LogModelParams, ModelType


@pytest.fixture
def telemetry_client():
    """Fixture to provide a telemetry client."""
    client = TelemetryClient()
    yield client
    # Cleanup
    client._wait_for_consumer_threads(terminate=True)


def test_telemetry_client_initialization(telemetry_client: TelemetryClient):
    """Test that TelemetryClient initializes correctly."""
    assert telemetry_client.info is not None
    assert telemetry_client._queue.maxsize == MAX_QUEUE_SIZE
    assert telemetry_client._max_workers == MAX_WORKERS
    assert not telemetry_client.is_active


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
    telemetry_client._wait_for_consumer_threads()

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

    telemetry_client._wait_for_consumer_threads()

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

    assert end_time - start_time < 0.1
    # remaining records are dropped
    assert len(mock_requests) == 0

    assert not telemetry_client.is_active


def test_error_handling(mock_requests, telemetry_client):
    """Test that client handles server errors gracefully."""
    telemetry_client.telemetry_url = "http://127.0.0.1:9999/nonexistent"  # Invalid URL

    # Add a record - should not crash
    record = APIRecord(
        api_name="test_api",
        status=APIStatus.SUCCESS,
        params=LogModelParams(flavor="test_flavor", model=ModelType.PYTHON_FUNCTION),
    )
    telemetry_client.add_record(record)

    telemetry_client._wait_for_consumer_threads()

    # Client should still be active despite errors
    assert telemetry_client.is_active
    assert len(mock_requests) == 0


def test_stop_event(telemetry_client: TelemetryClient, mock_requests):
    """Test that records are not added when telemetry client is stopped."""
    telemetry_client._is_stopped = True

    record = APIRecord(
        api_name="test_api",
        status=APIStatus.SUCCESS,
        params=LogModelParams(flavor="test_flavor", model=ModelType.PYTHON_FUNCTION),
    )
    telemetry_client.add_record(record)

    telemetry_client._wait_for_consumer_threads()

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

    telemetry_client._wait_for_consumer_threads()

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

    telemetry_client._wait_for_consumer_threads()

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

    telemetry_client._wait_for_consumer_threads()

    # Verify partition key
    received_record = mock_requests[0]
    assert received_record["partition-key"] == "test"


def test_max_workers_setup(telemetry_client: TelemetryClient):
    """Test max_workers configuration and validation."""
    # Test default value
    assert telemetry_client._max_workers == MAX_WORKERS

    telemetry_client._max_workers = 8
    # Test that correct number of threads are created
    telemetry_client.activate()
    assert len(telemetry_client._consumer_threads) == 8

    # Verify thread names
    for i, thread in enumerate(telemetry_client._consumer_threads):
        assert thread.name == f"MLflowTelemetryConsumer-{i}"
        assert thread.daemon is True


def test_batch_time_interval(mock_requests, telemetry_client):
    """Test that batching respects time interval configuration."""

    # Set batch time interval to 1 second for testing
    telemetry_client._batch_time_interval = 1

    # Add first record
    record1 = APIRecord(
        api_name="test_api_1",
        status=APIStatus.SUCCESS,
        params=LogModelParams(flavor="test_flavor", model=ModelType.PYTHON_FUNCTION),
    )
    telemetry_client.add_record(record1)

    # Should not send immediately since batch size is not reached
    assert len(mock_requests) == 0

    # Add second record before time interval
    record2 = APIRecord(
        api_name="test_api_2",
        status=APIStatus.SUCCESS,
        params=LogModelParams(flavor="test_flavor", model=ModelType.PYTHON_FUNCTION),
    )
    telemetry_client.add_record(record2)

    # Should still not send
    assert len(mock_requests) == 0

    # Wait for time interval to pass
    time.sleep(1.1)

    # Add third record which should trigger sending due to time interval
    record3 = APIRecord(
        api_name="test_api_3",
        status=APIStatus.SUCCESS,
        params=LogModelParams(flavor="test_flavor", model=ModelType.PYTHON_FUNCTION),
    )
    telemetry_client.add_record(record3)

    # Wait for processing
    telemetry_client._wait_for_consumer_threads()

    # Should have sent all 3 records in one batch
    assert len(mock_requests) == 3

    # Verify all records were sent
    api_names = {json.loads(req["data"])["api_name"] for req in mock_requests}
    assert api_names == {"test_api_1", "test_api_2", "test_api_3"}
