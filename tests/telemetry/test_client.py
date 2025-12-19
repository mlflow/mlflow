import logging
import threading
import time
import warnings
from unittest import mock

import pytest

import mlflow
from mlflow.environment_variables import _MLFLOW_TELEMETRY_SESSION_ID
from mlflow.telemetry.client import (
    BATCH_SIZE,
    BATCH_TIME_INTERVAL_SECONDS,
    MAX_QUEUE_SIZE,
    MAX_WORKERS,
    TelemetryClient,
    get_telemetry_client,
)
from mlflow.telemetry.events import CreateLoggedModelEvent, CreateRunEvent
from mlflow.telemetry.schemas import Record, SourceSDK, Status
from mlflow.utils.os import is_windows
from mlflow.version import IS_TRACING_SDK_ONLY, VERSION

from tests.telemetry.helper_functions import validate_telemetry_record

if not IS_TRACING_SDK_ONLY:
    from mlflow.tracking._tracking_service.utils import _use_tracking_uri


def test_telemetry_client_initialization(mock_telemetry_client: TelemetryClient, mock_requests):
    assert mock_telemetry_client.info is not None
    assert mock_telemetry_client._queue.maxsize == MAX_QUEUE_SIZE
    assert mock_telemetry_client._max_workers == MAX_WORKERS
    assert mock_telemetry_client._batch_size == BATCH_SIZE
    assert mock_telemetry_client._batch_time_interval == BATCH_TIME_INTERVAL_SECONDS


def test_telemetry_client_session_id(
    mock_telemetry_client: TelemetryClient, mock_requests, monkeypatch
):
    monkeypatch.setenv(_MLFLOW_TELEMETRY_SESSION_ID.name, "test_session_id")
    with TelemetryClient() as telemetry_client:
        assert telemetry_client.info["session_id"] == "test_session_id"
    monkeypatch.delenv(_MLFLOW_TELEMETRY_SESSION_ID.name, raising=False)
    with TelemetryClient() as telemetry_client:
        assert telemetry_client.info["session_id"] != "test_session_id"


def test_add_record_and_send(mock_telemetry_client: TelemetryClient, mock_requests):
    # Create a test record
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )

    # Add record and wait for processing
    mock_telemetry_client.add_record(record)
    mock_telemetry_client.flush()
    received_record = [req for req in mock_requests if req["data"]["event_name"] == "test_event"][0]

    assert "data" in received_record
    assert "partition-key" in received_record

    data = received_record["data"]
    assert data["event_name"] == "test_event"
    assert data["status"] == "success"


def test_add_records_and_send(mock_telemetry_client: TelemetryClient, mock_requests):
    # Pre-populate pending_records with 200 records
    initial_records = [
        Record(
            event_name=f"initial_{i}",
            timestamp_ns=time.time_ns(),
            status=Status.SUCCESS,
        )
        for i in range(200)
    ]
    mock_telemetry_client.add_records(initial_records)

    # We haven't hit the batch size limit yet, so expect no records to be sent
    assert len(mock_telemetry_client._pending_records) == 200
    assert len(mock_requests) == 0

    # Add 1000 more records
    # Expected behavior:
    # - First 300 records fill to 500 -> send batch (200 + 300) to queue
    # - Next 500 records -> send batch to queue
    # - Last 200 records remain in pending (200 < 500)
    additional_records = [
        Record(
            event_name=f"additional_{i}",
            timestamp_ns=time.time_ns(),
            status=Status.SUCCESS,
        )
        for i in range(1000)
    ]
    mock_telemetry_client.add_records(additional_records)

    # Verify batching logic:
    # - 2 batches should be in the queue
    # - 200 records should remain in pending
    assert mock_telemetry_client._queue.qsize() == 2
    assert len(mock_telemetry_client._pending_records) == 200

    # Flush to process queue and send the remaining partial batch
    mock_telemetry_client.flush()

    # Verify all 1200 records were sent
    assert len(mock_requests) == 1200
    event_names = {req["data"]["event_name"] for req in mock_requests}
    assert all(f"initial_{i}" in event_names for i in range(200))
    assert all(f"additional_{i}" in event_names for i in range(1000))


def test_record_with_session_and_installation_id(
    mock_telemetry_client: TelemetryClient, mock_requests
):
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
        session_id="session_id_override",
        installation_id="installation_id_override",
    )
    mock_telemetry_client.add_record(record)
    mock_telemetry_client.flush()
    assert mock_requests[0]["data"]["session_id"] == "session_id_override"
    assert mock_requests[0]["data"]["installation_id"] == "installation_id_override"

    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record)
    mock_telemetry_client.flush()
    assert mock_requests[1]["data"]["session_id"] == mock_telemetry_client.info["session_id"]
    assert (
        mock_requests[1]["data"]["installation_id"] == mock_telemetry_client.info["installation_id"]
    )


def test_batch_processing(mock_telemetry_client: TelemetryClient, mock_requests):
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
    events = {req["data"]["event_name"] for req in mock_requests}
    assert all(event_name in events for event_name in [f"test_event_{i}" for i in range(5)])


def test_flush_functionality(mock_telemetry_client: TelemetryClient, mock_requests):
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record)

    mock_telemetry_client.flush()
    events = {req["data"]["event_name"] for req in mock_requests}
    assert record.event_name in events


def test_record_sent(mock_telemetry_client: TelemetryClient, mock_requests):
    record_1 = Record(
        event_name="test_event_1",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record_1)
    mock_telemetry_client.flush()

    assert len(mock_requests) == 1
    data = mock_requests[0]["data"]
    assert data["event_name"] == record_1.event_name
    assert data["status"] == "success"

    session_id = data.get("session_id")
    installation_id = data.get("installation_id")
    assert session_id is not None
    assert installation_id is not None

    record_2 = Record(
        event_name="test_event_2",
        timestamp_ns=time.time_ns(),
        status=Status.FAILURE,
    )
    record_3 = Record(
        event_name="test_event_3",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record_2)
    mock_telemetry_client.add_record(record_3)
    mock_telemetry_client.flush()
    assert len(mock_requests) == 3

    # all record should have the same session id and installation id
    assert {req["data"].get("session_id") for req in mock_requests} == {session_id}
    assert {req["data"].get("installation_id") for req in mock_requests} == {installation_id}


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
    events = {req["data"]["event_name"] for req in mock_requests}
    assert "test_event" not in events
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

    with (
        mock.patch("requests.post", side_effect=tracker.mock_post),
        TelemetryClient() as telemetry_client,
    ):
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
            assert record.event_name in [resp["data"]["event_name"] for resp in tracker.responses]


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

    with (
        mock.patch("requests.post", side_effect=tracker.mock_post),
        TelemetryClient() as telemetry_client,
    ):
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
        assert record.event_name in [resp["data"]["event_name"] for resp in tracker.responses]


def test_stop_event(mock_telemetry_client: TelemetryClient, mock_requests):
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
    events = {req["data"]["event_name"] for req in mock_requests}
    assert record.event_name not in events


def test_concurrent_record_addition(mock_telemetry_client: TelemetryClient, mock_requests):
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
    events = {req["data"]["event_name"] for req in mock_requests}
    assert all(
        event_name in events
        for event_name in [
            f"test_event_{thread_id}_{i}" for thread_id in range(3) for i in range(5)
        ]
    )


def test_telemetry_info_inclusion(mock_telemetry_client: TelemetryClient, mock_requests):
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
    )
    mock_telemetry_client.add_record(record)

    mock_telemetry_client.flush()

    # Verify telemetry info is included
    data = [req["data"] for req in mock_requests if req["data"]["event_name"] == "test_event"][0]

    # Check that telemetry info fields are present
    assert mock_telemetry_client.info.items() <= data.items()

    # Check that record fields are present
    assert data["event_name"] == "test_event"
    assert data["status"] == "success"


def test_partition_key(mock_telemetry_client: TelemetryClient, mock_requests):
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
    with TelemetryClient() as telemetry_client:
        assert telemetry_client._max_workers == 8
        telemetry_client.activate()
        # Test that correct number of threads are created
        assert len(telemetry_client._consumer_threads) == 8

        # Verify thread names
        for i, thread in enumerate(telemetry_client._consumer_threads):
            assert thread.name == f"MLflowTelemetryConsumer-{i}"
            assert thread.daemon is True


def test_log_suppression_in_consumer_thread(mock_requests, capsys, mock_telemetry_client):
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
    events = {req["data"]["event_name"] for req in mock_requests}
    assert record.event_name in events

    captured = capsys.readouterr()

    assert "TEST LOG FROM MAIN THREAD" in captured.err
    # Verify that the consumer thread log was suppressed
    assert "TEST LOG FROM CONSUMER THREAD" not in captured.err


def test_consumer_thread_no_stderr_output(mock_requests, capsys, mock_telemetry_client):
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
    events = {req["data"]["event_name"] for req in mock_requests}
    assert all(event_name in events for event_name in [f"test_event_{i}" for i in range(5)])

    # Capture output after consumer thread has processed all records
    captured = capsys.readouterr()

    # Verify consumer thread produced no stderr output
    assert captured.err == ""

    # Log from main thread after processing - this should be captured
    logger.info("MAIN THREAD LOG AFTER PROCESSING")
    captured_after = capsys.readouterr()
    assert "MAIN THREAD LOG AFTER PROCESSING" in captured_after.err


def test_batch_time_interval(mock_requests, monkeypatch):
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
    assert all(env in event_names for env in ["test_event_1", "test_event_2", "test_event_3"])


def test_set_telemetry_client_non_blocking():
    start_time = time.time()
    with TelemetryClient() as telemetry_client:
        assert time.time() - start_time < 1
        assert telemetry_client is not None
        time.sleep(1.1)
        assert not any(
            thread.name.startswith("GetTelemetryConfig") for thread in threading.enumerate()
        )


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
        with TelemetryClient() as telemetry_client:
            telemetry_client._get_config()
            assert telemetry_client.config.ingestion_url == "http://localhost:9999"
            assert telemetry_client.config.disable_events == set()

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
        with TelemetryClient() as telemetry_client:
            telemetry_client._get_config()
            assert telemetry_client.config.ingestion_url == "http://localhost:9999"
            assert telemetry_client.config.disable_events == set()

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
        with (
            mock.patch(
                "mlflow.telemetry.client.get_source_sdk", return_value=SourceSDK.MLFLOW_TRACING
            ),
            TelemetryClient() as telemetry_client,
        ):
            telemetry_client._get_config()
            assert telemetry_client.config is None

        with (
            mock.patch(
                "mlflow.telemetry.client.get_source_sdk", return_value=SourceSDK.MLFLOW_SKINNY
            ),
            TelemetryClient() as telemetry_client,
        ):
            telemetry_client._get_config()
            assert telemetry_client.config.ingestion_url == "http://localhost:9999"
            assert telemetry_client.config.disable_events == set()

        with (
            mock.patch("mlflow.telemetry.client.get_source_sdk", return_value=SourceSDK.MLFLOW),
            TelemetryClient() as telemetry_client,
        ):
            telemetry_client._get_config()
            assert telemetry_client.config.ingestion_url == "http://localhost:9999"
            assert telemetry_client.config.disable_events == set()


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
        with TelemetryClient() as telemetry_client:
            telemetry_client._get_config()
            assert telemetry_client.config is None

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
        with TelemetryClient() as telemetry_client:
            telemetry_client._get_config()
            assert telemetry_client.config.ingestion_url == "http://localhost:9999"
            assert telemetry_client.config.disable_events == set()


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
        with TelemetryClient() as telemetry_client:
            telemetry_client._get_config()
            assert telemetry_client.config is None

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
        with TelemetryClient() as telemetry_client:
            telemetry_client._get_config()
            assert telemetry_client.config.ingestion_url == "http://localhost:9999"
            assert telemetry_client.config.disable_events == set()


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
        with TelemetryClient() as telemetry_client:
            assert telemetry_client is not None
            telemetry_client.activate()
            telemetry_client._config_thread.join(timeout=3)
            assert not telemetry_client._config_thread.is_alive()
            assert telemetry_client.config is None
            assert telemetry_client._is_config_fetched is True
            assert telemetry_client._is_stopped


@pytest.mark.no_mock_requests_get
def test_records_not_dropped_when_fetching_config(mock_requests):
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
        with TelemetryClient() as telemetry_client:
            telemetry_client.activate()
            # wait for config to be fetched
            telemetry_client._config_thread.join(timeout=3)
            telemetry_client.add_record(record)
            telemetry_client.flush()
            validate_telemetry_record(
                telemetry_client, mock_requests, record.event_name, check_params=False
            )


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

    with (
        mock.patch("mlflow.telemetry.client.requests.get", side_effect=mock_requests_get),
        TelemetryClient() as telemetry_client,
    ):
        telemetry_client.add_record(record)
        telemetry_client.flush()
        events = [req["data"]["event_name"] for req in mock_requests]
        assert record.event_name not in events
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

    with (
        _use_tracking_uri(f"{tracking_uri_scheme}://profile_name"),
        TelemetryClient() as telemetry_client,
    ):
        telemetry_client.add_record(record)
        telemetry_client.flush(terminate=terminate)
        assert len(mock_requests) == 0
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
        with (
            TelemetryClient() as telemetry_client,
            mock.patch(
                "mlflow.telemetry.track.get_telemetry_client", return_value=telemetry_client
            ),
        ):
            telemetry_client.activate()
            telemetry_client._config_thread.join(timeout=1)
            mlflow.initialize_logged_model(name="model", tags={"key": "value"})
            telemetry_client.flush()
            assert len(mock_requests) == 0

            with mlflow.start_run():
                pass
            validate_telemetry_record(
                telemetry_client, mock_requests, CreateRunEvent.name, check_params=False
            )


@pytest.mark.no_mock_requests_get
def test_fetch_config_after_first_record():
    record = Record(
        event_name="test_event",
        timestamp_ns=time.time_ns(),
        status=Status.SUCCESS,
        duration_ms=0,
    )

    mock_response = mock.Mock(
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
    with mock.patch(
        "mlflow.telemetry.client.requests.get", return_value=mock_response
    ) as mock_requests_get:
        with TelemetryClient() as telemetry_client:
            assert telemetry_client._is_config_fetched is False
            telemetry_client.add_record(record)
            telemetry_client._config_thread.join(timeout=1)
            assert telemetry_client._is_config_fetched is True
        mock_requests_get.assert_called_once()
