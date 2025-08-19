import logging
import threading
import time
import warnings
from unittest import mock

import pytest

import mlflow
from mlflow.telemetry.client import (
    BATCH_SIZE,
    BATCH_TIME_INTERVAL_SECONDS,
    MAX_QUEUE_SIZE,
    MAX_WORKERS,
    TelemetryClient,
    get_telemetry_client,
)
from mlflow.telemetry.events import CreateLoggedModelEvent, CreateRunEvent, ImportMlflowEvent
from mlflow.telemetry.schemas import Record, SourceSDK, Status
from mlflow.utils.os import is_windows
from mlflow.version import IS_TRACING_SDK_ONLY, VERSION

from tests.telemetry.helper_functions import validate_telemetry_record

if not IS_TRACING_SDK_ONLY:
    from mlflow.tracking._tracking_service.utils import _use_tracking_uri


def test_telemetry_client_initialization(mock_telemetry_client: TelemetryClient, mock_requests):
    """Test that TelemetryClient initializes correctly."""
    assert mock_telemetry_client.info is not None
    assert mock_telemetry_client._queue.maxsize == MAX_QUEUE_SIZE
    assert mock_telemetry_client._max_workers == MAX_WORKERS
    assert mock_telemetry_client.is_active
    assert mock_telemetry_client._batch_size == BATCH_SIZE
    assert mock_telemetry_client._batch_time_interval == BATCH_TIME_INTERVAL_SECONDS
    # wait for the import record to be sent
    time.sleep(1)
    assert len(mock_requests) == 1
    validate_telemetry_record(mock_telemetry_client, mock_requests, ImportMlflowEvent.name)


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

    received_record = mock_requests[1]
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

    # 1 import record + 5 test records
    assert len(mock_requests) == 6


def test_flush_functionality(mock_telemetry_client: TelemetryClient, mock_requests):
    """Test that flush properly sends pending records."""
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record)

    mock_telemetry_client.flush()

    # 1 import record + 1 test record
    assert len(mock_requests) == 2


def test_first_record_sent(mock_telemetry_client: TelemetryClient, mock_requests):
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
        duration_ms=0,
    )
    mock_telemetry_client.add_record(record)
    mock_telemetry_client.flush()
    # 1 import record + 1 test record
    assert len(mock_requests) == 2
    validate_telemetry_record(mock_telemetry_client, mock_requests, record.event_name)


def test_client_shutdown(mock_telemetry_client: TelemetryClient, mock_requests):
    for _ in range(100):
        record = Record(
            event_name="test_event",
            timestamp_ns=time.time_ns(),
            status=Status.SUCCESS,
        )
        mock_telemetry_client.add_record(record)

    start_time = time.time()
    mock_telemetry_client.flush(terminate=True)
    end_time = time.time()
    assert end_time - start_time < 0.1
    # remaining records are dropped
    assert len(mock_requests) <= 1

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
def test_telemetry_collection_stopped_on_error(mock_requests, mock_telemetry_client, url):
    mock_telemetry_client.config.ingestion_url = url

    # Add a record - should not crash
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record)

    mock_telemetry_client.flush(terminate=True)

    assert mock_telemetry_client._is_stopped is True
    assert mock_telemetry_client.is_active is False
    requests_count = len(mock_requests)
    assert requests_count <= 1

    # add record after stopping should be no-op
    mock_telemetry_client.add_record(record)
    mock_telemetry_client.flush(terminate=True)
    assert len(mock_requests) == requests_count


@pytest.mark.parametrize("error_code", [429, 500])
@pytest.mark.parametrize("terminate", [True, False])
def test_telemetry_retry_on_error(error_code, terminate):
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )

    class MockPostTracker:
        def __init__(self):
            self.count = 0
            self.responses = []

        def mock_post(self, url, json=None, **kwargs):
            self.count += 1
            if self.count < 3:
                return mock.Mock(status_code=error_code)
            else:
                self.responses.extend(json["records"])
                return mock.Mock(status_code=200)

    tracker = MockPostTracker()

    with mock.patch("requests.post", side_effect=tracker.mock_post):
        telemetry_client = TelemetryClient()
        telemetry_client.add_record(record)
        start_time = time.time()
        telemetry_client.flush(terminate=terminate)
        duration = time.time() - start_time
        if terminate:
            assert duration < 1.5
        else:
            assert duration < 2.5

        if terminate:
            assert tracker.responses == []
        else:
            assert len(tracker.responses) == 2
            assert tracker.responses[1]["data"]["event_name"] == record.event_name

        telemetry_client._clean_up()


@pytest.mark.parametrize("error_type", [ConnectionError, TimeoutError])
@pytest.mark.parametrize("terminate", [True, False])
def test_telemetry_retry_on_request_error(error_type, terminate):
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )

    class MockPostTracker:
        def __init__(self):
            self.count = 0
            self.responses = []

        def mock_post(self, url, json=None, **kwargs):
            self.count += 1
            if self.count < 3:
                raise error_type()
            else:
                self.responses.extend(json["records"])
                return mock.Mock(status_code=200)

    tracker = MockPostTracker()

    with mock.patch("requests.post", side_effect=tracker.mock_post):
        telemetry_client = TelemetryClient()
        telemetry_client.add_record(record)
        start_time = time.time()
        telemetry_client.flush(terminate=terminate)
        duration = time.time() - start_time
        if terminate:
            assert duration < 1.5
        else:
            assert duration < 2.5

    # no retry when terminating
    if terminate:
        assert tracker.responses == []
    else:
        assert len(tracker.responses) == 2
        assert tracker.responses[1]["data"]["event_name"] == record.event_name

    telemetry_client._clean_up()


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

    # No records should be sent since the client is stopped
    assert len(mock_requests) <= 1


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
    # 1 import record + 15 test records
    assert len(mock_requests) == 16


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
    received_record = mock_requests[1]
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


def test_max_workers_setup(monkeypatch):
    monkeypatch.setattr("mlflow.telemetry.client.MAX_WORKERS", 8)
    telemetry_client = TelemetryClient()
    assert telemetry_client._max_workers == 8
    telemetry_client.activate()
    # Test that correct number of threads are created
    assert len(telemetry_client._consumer_threads) == 8

    # Verify thread names
    for i, thread in enumerate(telemetry_client._consumer_threads):
        assert thread.name == f"MLflowTelemetryConsumer-{i}"
        assert thread.daemon is True
    telemetry_client._clean_up()


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
    # 1 import record + 1 test record
    assert len(mock_requests) == 2

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
    # 1 import record + 5 test records
    assert len(mock_requests) == 6

    # Capture output after consumer thread has processed all records
    captured = capsys.readouterr()

    # Verify consumer thread produced no stderr output
    assert captured.err == ""

    # Log from main thread after processing - this should be captured
    logger.info("MAIN THREAD LOG AFTER PROCESSING")
    captured_after = capsys.readouterr()
    assert "MAIN THREAD LOG AFTER PROCESSING" in captured_after.err


def test_batch_time_interval(mock_requests, monkeypatch):
    """Test that batching respects time interval configuration."""
    monkeypatch.setattr("mlflow.telemetry.client.BATCH_TIME_INTERVAL_SECONDS", 1)
    telemetry_client = TelemetryClient()

    assert telemetry_client._batch_time_interval == 1

    # Add first record
    record1 = Record(
        event_name="test_event_1",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    telemetry_client.add_record(record1)
    assert len(telemetry_client._pending_records) == 1

    # Should not send immediately since batch size is not reached
    events = {req["data"]["event_name"] for req in mock_requests}
    assert "test_event_1" not in events

    # Add second record before time interval
    record2 = Record(
        event_name="test_event_2",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    telemetry_client.add_record(record2)
    assert len(telemetry_client._pending_records) == 2

    # Wait for time interval to pass
    time.sleep(1.5)
    assert len(telemetry_client._pending_records) == 0
    # records are sent due to time interval
    events = {req["data"]["event_name"] for req in mock_requests}
    assert "test_event_1" in events
    assert "test_event_2" in events

    record3 = Record(
        event_name="test_event_3",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    telemetry_client.add_record(record3)
    telemetry_client.flush()

    # Verify all records were sent
    event_names = {req["data"]["event_name"] for req in mock_requests}
    assert event_names == {ImportMlflowEvent.name, "test_event_1", "test_event_2", "test_event_3"}
    telemetry_client._clean_up()


def test_set_telemetry_client_non_blocking():
    start_time = time.time()
    client = TelemetryClient()
    assert time.time() - start_time < 1
    assert client is not None
    time.sleep(1.1)
    assert not any(thread.name.startswith("GetTelemetryConfig") for thread in threading.enumerate())
    client._clean_up()


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
        client = TelemetryClient()
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
        client = TelemetryClient()
        client._get_config()
        assert client.config.ingestion_url == "http://localhost:9999"
        assert client.config.disable_events == set()
        client._clean_up()

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
        client = TelemetryClient()
        client._get_config()
        assert client.config.ingestion_url == "http://localhost:9999"
        assert client.config.disable_events == set()
        client._clean_up()
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
            assert client.config.ingestion_url == "http://localhost:9999"
            assert client.config.disable_events == set()
            client._clean_up()

        with mock.patch("mlflow.telemetry.client.get_source_sdk", return_value=SourceSDK.MLFLOW):
            client = TelemetryClient()
            client._get_config()
            assert client.config.ingestion_url == "http://localhost:9999"
            assert client.config.disable_events == set()


@pytest.mark.no_mock_requests_get
@pytest.mark.skipif(is_windows(), reason="This test only passes on non-Windows")
def test_get_config_disable_non_windows():
    with mock.patch("mlflow.telemetry.client.requests.get") as mock_requests_get:
        mock_requests_get.return_value = mock.Mock(
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
        client = TelemetryClient()
        client._get_config()
        assert client.config is None
        client._clean_up()

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
        client = TelemetryClient()
        client._get_config()
        assert client.config.ingestion_url == "http://localhost:9999"
        assert client.config.disable_events == set()
        client._clean_up()


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
                }
            ),
        )
        client = TelemetryClient()
        client._get_config()
        assert client.config.ingestion_url == "http://localhost:9999"
        assert client.config.disable_events == set()
        client._clean_up()


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
        client = TelemetryClient()
        assert client is not None
        client._config_thread.join(timeout=3)
        assert not client._config_thread.is_alive()
        assert client.config is None
        assert client._is_config_fetched is True
        assert client._is_stopped
        client._clean_up()


@pytest.mark.no_mock_requests_get
def test_records_not_dropped_when_fetching_config(mock_requests):
    assert len(mock_requests) == 0
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
        duration_ms=0,
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
        client = TelemetryClient()
        # wait for config to be fetched
        client._config_thread.join(timeout=3)
        client.add_record(record)
        assert len(client._pending_records) == 1
        client.flush()
        validate_telemetry_record(client, mock_requests, record.event_name)
        client._clean_up()


@pytest.mark.no_mock_requests_get
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
        assert len(client._pending_records) == 1
        # wait for config to be fetched
        client._config_thread.join()
        client.flush()
        events = [req["data"]["event_name"] for req in mock_requests]
        assert record.event_name not in events
        # clean up
        client._clean_up()
        assert get_telemetry_client() is None


def test_warning_suppression_in_shutdown(recwarn, mock_telemetry_client: TelemetryClient):
    def flush_mock(*args, **kwargs):
        warnings.warn("test warning")

    with mock.patch.object(mock_telemetry_client, "flush", flush_mock):
        mock_telemetry_client._at_exit_callback()
        assert len(recwarn) == 0


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
        client._clean_up()
        assert get_telemetry_client() is None


@pytest.mark.no_mock_requests_get
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
                }
            ),
        )
        client = TelemetryClient()

        with mock.patch("mlflow.telemetry.track.get_telemetry_client", return_value=client):
            mlflow.create_external_model(name="model")
            mlflow.initialize_logged_model(name="model", tags={"key": "value"})
            mlflow.pyfunc.log_model(name="model", python_model=lambda x: x, input_example=["a"])
            client.flush()
            assert len(mock_requests) == 1
            validate_telemetry_record(client, mock_requests, ImportMlflowEvent.name)

            with mlflow.start_run():
                pass
            validate_telemetry_record(
                client, mock_requests, CreateRunEvent.name, check_params=False
            )
        client._clean_up()
