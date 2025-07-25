import logging
import threading
import time
import warnings
from unittest import mock

import pytest

import mlflow
from mlflow.telemetry.client import (
    MAX_QUEUE_SIZE,
    MAX_WORKERS,
    TelemetryClient,
    get_telemetry_client,
)
from mlflow.telemetry.events import CreateLoggedModelEvent, CreateRunEvent
from mlflow.telemetry.schemas import Record, SourceSDK, Status, TelemetryConfig
from mlflow.utils.os import is_windows
from mlflow.version import IS_TRACING_SDK_ONLY, VERSION

from tests.telemetry.helper_functions import validate_telemetry_record

if not IS_TRACING_SDK_ONLY:
    from mlflow.tracking._tracking_service.utils import _use_tracking_uri


def test_telemetry_client_initialization(mock_telemetry_client: TelemetryClient):
    """Test that TelemetryClient initializes correctly."""
    assert mock_telemetry_client.info is not None
    assert mock_telemetry_client._queue.maxsize == MAX_QUEUE_SIZE
    assert mock_telemetry_client._max_workers == MAX_WORKERS
    assert not mock_telemetry_client.is_active


def test_add_record_and_send(mock_telemetry_client: TelemetryClient, mock_requests):
    """Test adding a record and sending it to the mock server."""
    # Create a test record
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )

    # Add record and wait for processing
    mock_telemetry_client.add_record(record)
    mock_telemetry_client.flush()

    received_record = mock_requests[0]
    assert "data" in received_record
    assert "partition-key" in received_record

    data = received_record["data"]
    assert data["event_name"] == "test_event"
    assert data["status"] == "success"


def test_batch_processing(mock_telemetry_client: TelemetryClient, mock_requests):
    """Test that multiple records are batched correctly."""
    mock_telemetry_client._batch_size = 3  # Set small batch size for testing

    # Add multiple records
    for i in range(5):
        record = Record(
            event_name=f"test_event_{i}",
            timestamp_ns=time.time_ns(),
            status=Status.SUCCESS,
        )
        mock_telemetry_client.add_record(record)

    mock_telemetry_client.flush()

    assert len(mock_requests) == 5


def test_flush_functionality(mock_telemetry_client: TelemetryClient, mock_requests):
    """Test that flush properly sends pending records."""
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record)

    mock_telemetry_client.flush()

    assert len(mock_requests) == 1


def test_first_record_sent(mock_telemetry_client: TelemetryClient, mock_requests):
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record)
    time.sleep(1)
    assert len(mock_requests) == 1


def test_client_shutdown(mock_telemetry_client: TelemetryClient, mock_requests):
    for _ in range(100):
        record = Record(
            event_name="test_event",
            timestamp_ns=time.time_ns(),
            status=Status.SUCCESS,
        )
        mock_telemetry_client.add_record(record)
    assert len(mock_requests) == 0

    start_time = time.time()
    mock_telemetry_client.flush(terminate=True)
    end_time = time.time()
    assert end_time - start_time < 0.1
    # remaining records are dropped
    assert len(mock_requests) == 0

    assert not mock_telemetry_client.is_active


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9999/nonexistent",
        "http://127.0.0.1:9999/unauthorized",
        "http://127.0.0.1:9999/forbidden",
        "http://127.0.0.1:9999/bad_request",
    ],
)
@pytest.mark.parametrize("terminate", [True, False])
def test_telemetry_collection_stopped_on_error(
    mock_requests, mock_telemetry_client, url, terminate
):
    mock_telemetry_client.config.ingestion_url = url

    # Add a record - should not crash
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record)

    mock_telemetry_client.flush(terminate=terminate)

    assert mock_telemetry_client._is_stopped is True
    assert mock_telemetry_client.is_active is False
    assert len(mock_requests) == 0
    assert mock_telemetry_client._pending_records == []

    # add record after stopping should be no-op
    mock_telemetry_client.add_record(record)
    mock_telemetry_client.flush(terminate=terminate)
    assert len(mock_requests) == 0


@pytest.mark.parametrize("error_code", [429, 500])
@pytest.mark.parametrize("terminate", [True, False])
def test_telemetry_retry_on_error(mock_telemetry_client, error_code, terminate):
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )

    class MockPostTracker:
        def __init__(self):
            self.count = 0
            self.responses = []

        def mock_post(self, *args, **kwargs):
            self.count += 1
            if self.count < 3:
                return mock.Mock(status_code=error_code)
            else:
                self.responses.append(record)
                return mock.Mock(status_code=200)

    tracker = MockPostTracker()

    with mock.patch("requests.post", side_effect=tracker.mock_post):
        mock_telemetry_client.add_record(record)
        start_time = time.time()
        mock_telemetry_client.flush(terminate=terminate)
        duration = time.time() - start_time
        if terminate:
            assert duration < 1.5
        else:
            assert duration < 2.5

    # no retry when terminating
    if terminate:
        assert tracker.responses == []
    else:
        assert tracker.responses == [record]


@pytest.mark.parametrize("error_type", [ConnectionError, TimeoutError])
@pytest.mark.parametrize("terminate", [True, False])
def test_telemetry_retry_on_request_error(mock_telemetry_client, error_type, terminate):
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )

    class MockPostTracker:
        def __init__(self):
            self.count = 0
            self.responses = []

        def mock_post(self, *args, **kwargs):
            self.count += 1
            if self.count < 3:
                raise error_type()
            else:
                self.responses.append(record)
                return mock.Mock(status_code=200)

    tracker = MockPostTracker()

    with mock.patch("requests.post", side_effect=tracker.mock_post):
        mock_telemetry_client.add_record(record)
        start_time = time.time()
        mock_telemetry_client.flush(terminate=terminate)
        duration = time.time() - start_time
        if terminate:
            assert duration < 1.5
        else:
            assert duration < 2.5

    # no retry when terminating
    if terminate:
        assert tracker.responses == []
    else:
        assert tracker.responses == [record]


def test_stop_event(mock_telemetry_client: TelemetryClient, mock_requests):
    """Test that records are not added when telemetry client is stopped."""
    mock_telemetry_client._is_stopped = True

    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record)

    # we need to terminate since the threads are stopped
    mock_telemetry_client.flush(terminate=True)

    # No records should be sent
    assert len(mock_requests) == 0


def test_concurrent_record_addition(mock_telemetry_client: TelemetryClient, mock_requests):
    """Test adding records from multiple threads."""

    def add_records(thread_id):
        for i in range(5):
            record = Record(
                event_name=f"test_event_{thread_id}_{i}",
                timestamp_ns=time.time_ns(),
                status=Status.SUCCESS,
            )
            mock_telemetry_client.add_record(record)
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

    mock_telemetry_client.flush()

    # Should have received records from all threads
    assert len(mock_requests) == 15


def test_telemetry_info_inclusion(mock_telemetry_client: TelemetryClient, mock_requests):
    """Test that telemetry info is included in records."""
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record)

    mock_telemetry_client.flush()

    # Verify telemetry info is included
    received_record = mock_requests[0]
    data = received_record["data"]

    # Check that telemetry info fields are present
    assert mock_telemetry_client.info.items() <= data.items()

    # Check that record fields are present
    assert data["event_name"] == "test_event"
    assert data["status"] == "success"


def test_partition_key(mock_telemetry_client: TelemetryClient, mock_requests):
    """Test that partition key is set correctly."""
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record)
    mock_telemetry_client.add_record(record)

    mock_telemetry_client.flush()

    # Verify partition key is random
    assert mock_requests[0]["partition-key"] != mock_requests[1]["partition-key"]


def test_max_workers_setup(mock_telemetry_client: TelemetryClient):
    """Test max_workers configuration and validation."""
    # Test default value
    assert mock_telemetry_client._max_workers == MAX_WORKERS

    mock_telemetry_client._max_workers = 8
    # Test that correct number of threads are created
    mock_telemetry_client.activate()
    assert len(mock_telemetry_client._consumer_threads) == 8

    # Verify thread names
    for i, thread in enumerate(mock_telemetry_client._consumer_threads):
        assert thread.name == f"MLflowTelemetryConsumer-{i}"
        assert thread.daemon is True


def test_log_suppression_in_consumer_thread(mock_requests, capsys, mock_telemetry_client):
    """Test that logs are suppressed in the consumer thread but not in main thread."""
    # Clear any existing captured output
    capsys.readouterr()

    # Log from main thread - this should be captured
    logger = logging.getLogger("mlflow.telemetry.client")
    logger.info("TEST LOG FROM MAIN THREAD")

    original_process = mock_telemetry_client._process_records

    def process_with_log(records):
        logger.info("TEST LOG FROM CONSUMER THREAD")
        original_process(records)

    mock_telemetry_client._process_records = process_with_log

    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record)

    mock_telemetry_client.flush()
    assert len(mock_requests) == 1

    captured = capsys.readouterr()

    assert "TEST LOG FROM MAIN THREAD" in captured.err
    # Verify that the consumer thread log was suppressed
    assert "TEST LOG FROM CONSUMER THREAD" not in captured.err


def test_consumer_thread_no_stderr_output(mock_requests, capsys, mock_telemetry_client):
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
        record = Record(
            event_name=f"test_event_{i}",
            timestamp_ns=time.time_ns(),
            status=Status.SUCCESS,
        )
        mock_telemetry_client.add_record(record)

    mock_telemetry_client.flush()
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


def test_batch_time_interval(mock_requests, mock_telemetry_client: TelemetryClient):
    """Test that batching respects time interval configuration."""

    # Set batch time interval to 1 second for testing
    mock_telemetry_client.config.batch_time_interval_seconds = 1

    # Add first record
    record1 = Record(
        event_name="test_event_1",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record1)

    # Should not send immediately since batch size is not reached
    assert len(mock_requests) == 0

    # Add second record before time interval
    record2 = Record(
        event_name="test_event_2",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record2)

    # Should still not send
    assert len(mock_requests) == 0

    # Wait for time interval to pass
    time.sleep(1.1)
    # records are sent due to time interval
    assert len(mock_requests) == 2

    record3 = Record(
        event_name="test_event_3",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record3)
    assert len(mock_requests) == 2
    mock_telemetry_client.flush()

    # Verify all records were sent
    event_names = {req["data"]["event_name"] for req in mock_requests}
    assert event_names == {"test_event_1", "test_event_2", "test_event_3"}


def test_set_telemetry_client_non_blocking():
    start_time = time.time()
    client = TelemetryClient()
    assert time.time() - start_time < 1
    assert client is not None
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
@pytest.mark.no_mock_telemetry_client
def test_client_get_config_none(mock_requests_return_value):
    with (
        mock.patch("mlflow.telemetry.client.requests.get") as mock_requests,
        mock.patch("random.randint", return_value=80),
    ):
        mock_requests.return_value = mock_requests_return_value
        client = TelemetryClient()
        client._get_config()
        assert client.config is None


@pytest.mark.no_mock_telemetry_client
def test_client_get_config_not_none():
    default_configs = {
        "batch_time_interval_seconds": 30,
        "retryable_error_codes": {429, 500},
        "stop_on_error_codes": {400, 401, 403, 404},
    }
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
                    **default_configs,
                }
            ),
        )
        client = TelemetryClient()
        client._get_config()
        assert client.config == TelemetryConfig(
            ingestion_url="http://localhost:9999",
            disable_events=set(),
            **default_configs,
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
                    **default_configs,
                }
            ),
        )
        client = TelemetryClient()
        client._get_config()
        assert client.config == TelemetryConfig(
            ingestion_url="http://localhost:9999",
            disable_events=set(),
            **default_configs,
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
                    "disable_events": [],
                    "disable_sdks": ["mlflow-tracing"],
                    **default_configs,
                }
            ),
        )
        with mock.patch(
            "mlflow.telemetry.client.get_source_sdk", return_value=SourceSDK.MLFLOW_TRACING
        ):
            client = TelemetryClient()
            client._get_config()
            assert client.config is None

        with mock.patch(
            "mlflow.telemetry.client.get_source_sdk", return_value=SourceSDK.MLFLOW_SKINNY
        ):
            client = TelemetryClient()
            client._get_config()
            assert client.config == TelemetryConfig(
                ingestion_url="http://localhost:9999",
                disable_events=set(),
                **default_configs,
            )

        with mock.patch("mlflow.telemetry.client.get_source_sdk", return_value=SourceSDK.MLFLOW):
            client = TelemetryClient()
            client._get_config()
            assert client.config == TelemetryConfig(
                ingestion_url="http://localhost:9999",
                disable_events=set(),
                **default_configs,
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
                    "disable_events": [],
                    "disable_sdks": ["mlflow-tracing"],
                    "batch_size": 100,
                    **default_configs,
                }
            ),
        )
        client = TelemetryClient()
        client._get_config()
        assert client._batch_size == 100


@pytest.mark.no_mock_telemetry_client
@pytest.mark.skipif(is_windows(), reason="This test only passes on non-Windows")
def test_get_config_disable_non_windows():
    default_configs = {
        "batch_time_interval_seconds": 30,
        "retryable_error_codes": {429, 500},
        "stop_on_error_codes": {400, 401, 403, 404},
    }
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
                    **default_configs,
                }
            ),
        )
        client = TelemetryClient()
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
                    **default_configs,
                }
            ),
        )
        client = TelemetryClient()
        client._get_config()
        assert client.config == TelemetryConfig(
            ingestion_url="http://localhost:9999",
            disable_events=set(),
            **default_configs,
        )


@pytest.mark.no_mock_telemetry_client
@pytest.mark.skipif(not is_windows(), reason="This test only passes on Windows")
def test_get_config_windows():
    default_configs = {
        "batch_time_interval_seconds": 30,
        "retryable_error_codes": {429, 500},
        "stop_on_error_codes": {400, 401, 403, 404},
    }
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
                    **default_configs,
                }
            ),
        )
        client = TelemetryClient()
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
                    **default_configs,
                }
            ),
        )
        client = TelemetryClient()
        client._get_config()
        assert client.config == TelemetryConfig(
            ingestion_url="http://localhost:9999",
            disable_events=set(),
            **default_configs,
        )


@pytest.mark.no_mock_telemetry_client
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
        client = TelemetryClient()
        assert client is not None
        client._config_thread.join(timeout=3)
        assert not client._config_thread.is_alive()
        assert client.config is None
        assert client._is_config_fetched is True
        assert client._is_stopped


@pytest.mark.no_mock_telemetry_client
@pytest.mark.parametrize("terminate", [True, False])
def test_records_not_dropped_when_fetching_config(mock_requests, terminate):
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
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
                    "batch_time_interval_seconds": 30,
                    "retryable_error_codes": {429, 500},
                    "stop_on_error_codes": {400, 401, 403, 404},
                }
            ),
        )
        client = TelemetryClient()
        client.add_record(record)
        assert len(client._pending_records) == 0
        assert client._queue.qsize() == 1
        time.sleep(1)
        assert len(mock_requests) == 1


@pytest.mark.no_mock_telemetry_client
@pytest.mark.parametrize("terminate", [True, False])
def test_records_not_processed_when_fetching_config_failed(mock_requests, terminate):
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )

    def mock_requests_get(*args, **kwargs):
        time.sleep(1)
        return mock.Mock(status_code=403)

    with mock.patch("mlflow.telemetry.client.requests.get", side_effect=mock_requests_get):
        client = TelemetryClient()
        client.add_record(record)
        assert len(client._pending_records) == 0
        assert client._queue.qsize() == 1
        time.sleep(1)
        assert len(mock_requests) == 0

        # clean up
        client._join_threads()


@pytest.mark.parametrize("error_code", [400, 401, 403, 404, 412, 500, 502, 503, 504])
def test_config_fetch_no_retry(mock_requests, error_code):
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )

    def mock_requests_get(*args, **kwargs):
        time.sleep(1)
        return mock.Mock(status_code=error_code)

    with mock.patch("mlflow.telemetry.client.requests.get", side_effect=mock_requests_get):
        client = TelemetryClient()
        client.add_record(record)
        assert len(client._pending_records) == 0
        # wait for config to be fetched
        client._config_thread.join()
        client.flush(terminate=True)
        assert len(mock_requests) == 0
        mock_requests.clear()
        # clean up
        client._join_threads()
        assert get_telemetry_client() is None


def test_warning_suppression_in_shutdown(recwarn, mock_telemetry_client: TelemetryClient):
    def flush_mock(*args, **kwargs):
        warnings.warn("test warning")

    with mock.patch.object(mock_telemetry_client, "flush", flush_mock):
        mock_telemetry_client._at_exit_callback()
        assert len(recwarn) == 0

    assert len(mock_telemetry_client._consumer_threads) == 0


@pytest.mark.parametrize("tracking_uri_scheme", ["databricks", "databricks-uc", "uc"])
@pytest.mark.parametrize("terminate", [True, False])
def test_databricks_tracking_uri_scheme(mock_requests, tracking_uri_scheme, terminate):
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )

    with _use_tracking_uri(f"{tracking_uri_scheme}://profile_name"):
        client = TelemetryClient()
        client.add_record(record)
        client.flush(terminate=terminate)
        assert len(mock_requests) == 0
        # clean up
        client._join_threads()
        assert get_telemetry_client() is None


@pytest.mark.no_mock_telemetry_client
def test_disable_events(mock_requests):
    with mock.patch("mlflow.telemetry.client.requests.get") as mock_requests_get:
        mock_requests_get.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                    "disable_events": [CreateLoggedModelEvent.name],
                    "disable_sdks": [],
                    "batch_time_interval_seconds": 30,
                    "retryable_error_codes": [429, 500],
                    "stop_on_error_codes": [400, 401, 403, 404],
                }
            ),
        )
        client = TelemetryClient()

        with mock.patch("mlflow.telemetry.track.get_telemetry_client", return_value=client):
            mlflow.create_external_model(name="model")
            mlflow.initialize_logged_model(name="model", tags={"key": "value"})
            mlflow.pyfunc.log_model(name="model", python_model=lambda x: x, input_example=["a"])
            client.flush()
            assert len(mock_requests) == 0

            with mlflow.start_run():
                pass
            validate_telemetry_record(
                client, mock_requests, CreateRunEvent.name, check_params=False
            )
