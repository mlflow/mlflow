import logging
import threading
import time
from unittest import mock

import pytest

from mlflow.telemetry.client import (
    MAX_QUEUE_SIZE,
    MAX_WORKERS,
    TelemetryClient,
    get_telemetry_client,
    set_telemetry_client,
)
from mlflow.telemetry.schemas import APIRecord, APIStatus, SourceSDK, TelemetryConfig
from mlflow.utils.os import is_windows
from mlflow.version import VERSION


@pytest.fixture
def telemetry_client():
    """Fixture to provide a telemetry client."""
    client = get_telemetry_client()
    yield client
    # Cleanup
    client.flush(terminate=True)


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
        api_module="test_module",
        api_name="test_api",
        timestamp_ns=time.time_ns(),
        status=APIStatus.SUCCESS,
    )

    # Add record and wait for processing
    telemetry_client.add_record(record)
    telemetry_client.flush()

    received_record = mock_requests[0]
    assert "data" in received_record
    assert "partition-key" in received_record

    data = received_record["data"]
    assert data["api_module"] == "test_module"
    assert data["api_name"] == "test_api"
    assert data["status"] == "success"


def test_batch_processing(telemetry_client: TelemetryClient, mock_requests):
    """Test that multiple records are batched correctly."""
    telemetry_client._batch_size = 3  # Set small batch size for testing

    # Add multiple records
    for i in range(5):
        record = APIRecord(
            api_module="test_module",
            api_name=f"test_api_{i}",
            timestamp_ns=time.time_ns(),
            status=APIStatus.SUCCESS,
        )
        telemetry_client.add_record(record)

    telemetry_client.flush()

    assert len(mock_requests) == 5


def test_flush_functionality(telemetry_client: TelemetryClient, mock_requests):
    """Test that flush properly sends pending records."""
    record = APIRecord(
        api_module="test_module",
        api_name="test_api",
        timestamp_ns=time.time_ns(),
        status=APIStatus.SUCCESS,
    )
    telemetry_client.add_record(record)

    telemetry_client.flush()

    assert len(mock_requests) == 1


def test_client_shutdown(telemetry_client: TelemetryClient, mock_requests):
    for _ in range(100):
        record = APIRecord(
            api_module="test_module",
            api_name="test_api",
            timestamp_ns=time.time_ns(),
            status=APIStatus.SUCCESS,
        )
        telemetry_client.add_record(record)
    assert len(mock_requests) == 0

    start_time = time.time()
    telemetry_client.flush(terminate=True)
    end_time = time.time()

    # 1 second for consumer threads
    # add some buffer for processing time
    assert end_time - start_time < 2
    # remaining records are processed directly
    assert len(mock_requests) == 100

    assert not telemetry_client.is_active


def test_error_handling(mock_requests, telemetry_client):
    """Test that client handles server errors gracefully."""
    telemetry_client.config.ingestion_url = "http://127.0.0.1:9999/nonexistent"  # Invalid URL

    # Add a record - should not crash
    record = APIRecord(
        api_module="test_module",
        api_name="test_api",
        timestamp_ns=time.time_ns(),
        status=APIStatus.SUCCESS,
    )
    telemetry_client.add_record(record)

    telemetry_client.flush()

    # Client should still be active despite errors
    assert telemetry_client.is_active
    assert len(mock_requests) == 0


def test_stop_event(telemetry_client: TelemetryClient, mock_requests):
    """Test that records are not added when telemetry client is stopped."""
    telemetry_client._is_stopped = True

    record = APIRecord(
        api_module="test_module",
        api_name="test_api",
        timestamp_ns=time.time_ns(),
        status=APIStatus.SUCCESS,
    )
    telemetry_client.add_record(record)

    # we need to terminate since the threads are stopped
    telemetry_client.flush(terminate=True)

    # No records should be sent
    assert len(mock_requests) == 0


def test_concurrent_record_addition(telemetry_client: TelemetryClient, mock_requests):
    """Test adding records from multiple threads."""

    def add_records(thread_id):
        for i in range(5):
            record = APIRecord(
                api_module="test_module",
                api_name=f"thread_{thread_id}_api_{i}",
                timestamp_ns=time.time_ns(),
                status=APIStatus.SUCCESS,
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

    telemetry_client.flush()

    # Should have received records from all threads
    assert len(mock_requests) == 15


def test_telemetry_info_inclusion(telemetry_client: TelemetryClient, mock_requests):
    """Test that telemetry info is included in records."""
    record = APIRecord(
        api_module="test_module",
        api_name="test_api",
        timestamp_ns=time.time_ns(),
        status=APIStatus.SUCCESS,
    )
    telemetry_client.add_record(record)

    telemetry_client.flush()

    # Verify telemetry info is included
    received_record = mock_requests[0]
    data = received_record["data"]

    # Check that telemetry info fields are present
    assert telemetry_client.info.items() <= data.items()

    # Check that record fields are present
    assert data["api_name"] == "test_api"
    assert data["status"] == "success"


def test_partition_key(telemetry_client: TelemetryClient, mock_requests):
    """Test that partition key is set correctly."""
    record = APIRecord(
        api_module="test_module",
        api_name="test_api",
        timestamp_ns=time.time_ns(),
        status=APIStatus.SUCCESS,
    )
    telemetry_client.add_record(record)
    telemetry_client.add_record(record)

    telemetry_client.flush()

    # Verify partition key is random
    assert mock_requests[0]["partition-key"] != mock_requests[1]["partition-key"]


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


def test_log_suppression_in_consumer_thread(mock_requests, capsys, telemetry_client):
    """Test that logs are suppressed in the consumer thread but not in main thread."""
    # Clear any existing captured output
    capsys.readouterr()

    # Log from main thread - this should be captured
    logger = logging.getLogger("mlflow.telemetry.client")
    logger.info("TEST LOG FROM MAIN THREAD")

    original_process = telemetry_client._process_records

    def process_with_log(records):
        logger.info("TEST LOG FROM CONSUMER THREAD")
        original_process(records)

    telemetry_client._process_records = process_with_log

    record = APIRecord(
        api_module="test_module",
        api_name="test_api",
        timestamp_ns=time.time_ns(),
        status=APIStatus.SUCCESS,
    )
    telemetry_client.add_record(record)

    telemetry_client.flush()
    assert len(mock_requests) == 1

    captured = capsys.readouterr()

    assert "TEST LOG FROM MAIN THREAD" in captured.err
    # Verify that the consumer thread log was suppressed
    assert "TEST LOG FROM CONSUMER THREAD" not in captured.err


def test_consumer_thread_no_stderr_output(mock_requests, capsys, telemetry_client):
    """Test that consumer thread produces no stderr output at all."""
    # Clear any existing captured output
    capsys.readouterr()

    # Log from main thread - this should be captured
    logger = logging.getLogger("mlflow.telemetry.client")
    logger.info("MAIN THREAD LOG BEFORE CLIENT")

    # Clear output after client initialization to focus on consumer thread output
    capsys.readouterr()

    # Add multiple records to ensure consumer thread processes them
    for i in range(5):
        record = APIRecord(
            api_module="test_module",
            api_name=f"test_api_{i}",
            timestamp_ns=time.time_ns(),
            status=APIStatus.SUCCESS,
        )
        telemetry_client.add_record(record)

    telemetry_client.flush()
    # Wait for all records to be processed
    assert len(mock_requests) == 5

    # Capture output after consumer thread has processed all records
    captured = capsys.readouterr()

    # Verify consumer thread produced no stderr output
    assert captured.err == ""

    # Log from main thread after processing - this should be captured
    logger.info("MAIN THREAD LOG AFTER PROCESSING")
    captured_after = capsys.readouterr()
    assert "MAIN THREAD LOG AFTER PROCESSING" in captured_after.err


def test_batch_time_interval(mock_requests, telemetry_client: TelemetryClient):
    """Test that batching respects time interval configuration."""

    # Set batch time interval to 1 second for testing
    telemetry_client._batch_time_interval = 1

    # Add first record
    record1 = APIRecord(
        api_module="test_module",
        api_name="test_api_1",
        timestamp_ns=time.time_ns(),
        status=APIStatus.SUCCESS,
    )
    telemetry_client.add_record(record1)

    # Should not send immediately since batch size is not reached
    assert len(mock_requests) == 0

    # Add second record before time interval
    record2 = APIRecord(
        api_module="test_module",
        api_name="test_api_2",
        timestamp_ns=time.time_ns(),
        status=APIStatus.SUCCESS,
    )
    telemetry_client.add_record(record2)

    # Should still not send
    assert len(mock_requests) == 0

    # Wait for time interval to pass
    time.sleep(1.1)
    # records are sent due to time interval
    assert len(mock_requests) == 2

    record3 = APIRecord(
        api_module="test_module",
        api_name="test_api_3",
        timestamp_ns=time.time_ns(),
        status=APIStatus.SUCCESS,
    )
    telemetry_client.add_record(record3)
    assert len(mock_requests) == 2
    telemetry_client.flush()

    # Verify all records were sent
    api_names = {req["data"]["api_name"] for req in mock_requests}
    assert api_names == {"test_api_1", "test_api_2", "test_api_3"}


def test_set_telemetry_client_non_blocking():
    start_time = time.time()
    set_telemetry_client()
    assert time.time() - start_time < 1
    assert get_telemetry_client() is not None
    time.sleep(1.1)
    assert not any(thread.name.startswith("GetTelemetryConfig") for thread in threading.enumerate())


@pytest.mark.parametrize(
    "mock_requests_return_value",
    [
        mock.Mock(status_code=403),
        mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": True,
                }
            ),
        ),
        mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": "1.0.0",
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                }
            ),
        ),
        mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 0,
                }
            ),
        ),
        mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 70,
                }
            ),
        ),
    ],
)
@pytest.mark.no_mock_requests_get
def test_client_get_config_none(mock_requests_return_value):
    with (
        mock.patch("mlflow.telemetry.client.requests.get") as mock_requests,
        mock.patch("random.randint", return_value=80),
    ):
        mock_requests.return_value = mock_requests_return_value
        set_telemetry_client()
        client = get_telemetry_client()
        client._get_config()
        assert client.config is None


@pytest.mark.no_mock_requests_get
def test_client_get_config_not_none():
    with (
        mock.patch("mlflow.telemetry.client.requests.get") as mock_requests,
        mock.patch("random.randint", return_value=50),
    ):
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 70,
                }
            ),
        )
        set_telemetry_client()
        client = get_telemetry_client()
        client._get_config()
        assert client.config == TelemetryConfig(
            ingestion_url="http://localhost:9999",
            disable_api_map={},
        )

    with mock.patch("mlflow.telemetry.client.requests.get") as mock_requests:
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                }
            ),
        )
        set_telemetry_client()
        client = get_telemetry_client()
        client._get_config()
        assert client.config == TelemetryConfig(
            ingestion_url="http://localhost:9999",
            disable_api_map={},
        )

    with mock.patch("mlflow.telemetry.client.requests.get") as mock_requests:
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                    "disable_api_map": {},
                    "disable_sdks": ["mlflow-tracing"],
                }
            ),
        )
        with mock.patch(
            "mlflow.telemetry.client.get_source_sdk", return_value=SourceSDK.MLFLOW_TRACING
        ):
            set_telemetry_client()
            client = get_telemetry_client()
            client._get_config()
            assert client.config is None

        with mock.patch(
            "mlflow.telemetry.client.get_source_sdk", return_value=SourceSDK.MLFLOW_SKINNY
        ):
            set_telemetry_client()
            client = get_telemetry_client()
            client._get_config()
            assert client.config == TelemetryConfig(
                ingestion_url="http://localhost:9999",
                disable_api_map={},
            )

        with mock.patch("mlflow.telemetry.client.get_source_sdk", return_value=SourceSDK.MLFLOW):
            set_telemetry_client()
            client = get_telemetry_client()
            client._get_config()
            assert client.config == TelemetryConfig(
                ingestion_url="http://localhost:9999",
                disable_api_map={},
            )


@pytest.mark.no_mock_requests_get
@pytest.mark.skipif(is_windows(), reason="This test only passes on non-Windows")
def test_get_config_disable_non_windows():
    with mock.patch("mlflow.telemetry.client.requests.get") as mock_requests:
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                    "disable_os": ["linux", "darwin"],
                }
            ),
        )
        set_telemetry_client()
        client = get_telemetry_client()
        client._get_config()
        assert client.config is None

    with mock.patch("mlflow.telemetry.client.requests.get") as mock_requests:
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                    "disable_os": ["win32"],
                }
            ),
        )
        set_telemetry_client()
        client = get_telemetry_client()
        client._get_config()
        assert client.config == TelemetryConfig(
            ingestion_url="http://localhost:9999",
            disable_api_map={},
        )


@pytest.mark.no_mock_requests_get
@pytest.mark.skipif(not is_windows(), reason="This test only passes on Windows")
def test_get_config_windows():
    with mock.patch("mlflow.telemetry.client.requests.get") as mock_requests:
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                    "disable_os": ["win32"],
                }
            ),
        )
        set_telemetry_client()
        client = get_telemetry_client()
        client._get_config()
        assert client.config is None

    with mock.patch("mlflow.telemetry.client.requests.get") as mock_requests:
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                    "disable_os": ["linux", "darwin"],
                }
            ),
        )
        set_telemetry_client()
        client = get_telemetry_client()
        client._get_config()
        assert client.config == TelemetryConfig(
            ingestion_url="http://localhost:9999",
            disable_api_map={},
        )


@pytest.mark.no_mock_requests_get
def test_client_set_to_none_if_config_none():
    with mock.patch("mlflow.telemetry.client.requests.get") as mock_requests:
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": True,
                }
            ),
        )
        set_telemetry_client()
        client = get_telemetry_client()
        assert client is not None
        client._config_thread.join(timeout=3)
        assert not client._config_thread.is_alive()
        assert client.config is None
        assert client._is_config_fetched is True
        assert client._is_stopped


@pytest.mark.no_mock_requests_get
@pytest.mark.parametrize("terminate", [True, False])
def test_records_not_dropped_when_fetching_config(mock_requests, terminate):
    record = APIRecord(
        api_module="test_module",
        api_name="test_api",
        timestamp_ns=time.time_ns(),
        status=APIStatus.SUCCESS,
    )
    with mock.patch("mlflow.telemetry.client.requests.get") as mock_requests_get:
        mock_requests_get.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                }
            ),
        )
        set_telemetry_client()
        client = get_telemetry_client()
        client.add_record(record)
        assert len(client._pending_records) == 1
        client.flush(terminate=terminate)
        assert len(mock_requests) == 1


@pytest.mark.no_mock_requests_get
@pytest.mark.parametrize("terminate", [True, False])
def test_records_not_processed_when_fetching_config_failed(mock_requests, terminate):
    record = APIRecord(
        api_module="test_module",
        api_name="test_api",
        timestamp_ns=time.time_ns(),
        status=APIStatus.SUCCESS,
    )

    def mock_requests_get(*args, **kwargs):
        time.sleep(1)
        return mock.Mock(status_code=403)

    with mock.patch("mlflow.telemetry.client.requests.get", side_effect=mock_requests_get):
        set_telemetry_client()
        client = get_telemetry_client()
        client.add_record(record)
        assert len(client._pending_records) == 1
        client.flush(terminate=terminate)
        assert len(mock_requests) == 0

        # clean up
        if terminate is False:
            client.flush(terminate=True)
