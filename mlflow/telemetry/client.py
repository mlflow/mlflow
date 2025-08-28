import atexit
import random
import sys
import threading
import time
import uuid
import warnings
from dataclasses import asdict
from queue import Empty, Full, Queue

import requests

from mlflow.environment_variables import _MLFLOW_TELEMETRY_SESSION_ID
from mlflow.telemetry.constant import (
    BATCH_SIZE,
    BATCH_TIME_INTERVAL_SECONDS,
    MAX_QUEUE_SIZE,
    MAX_WORKERS,
    RETRYABLE_ERRORS,
    UNRECOVERABLE_ERRORS,
)
from mlflow.telemetry.events import ImportMlflowEvent
from mlflow.telemetry.schemas import Record, Status, TelemetryConfig, TelemetryInfo, get_source_sdk
from mlflow.telemetry.utils import _get_config_url, is_telemetry_disabled
from mlflow.utils.logging_utils import should_suppress_logs_in_thread, suppress_logs_in_thread


class TelemetryClient:
    def __init__(self):
        self.info = asdict(
            TelemetryInfo(session_id=_MLFLOW_TELEMETRY_SESSION_ID.get() or uuid.uuid4().hex)
        )
        self._queue: Queue[list[Record]] = Queue(maxsize=MAX_QUEUE_SIZE)
        self._lock = threading.RLock()
        self._max_workers = MAX_WORKERS

        self._is_stopped = False
        self._is_active = False
        self._atexit_callback_registered = False

        self._batch_size = BATCH_SIZE
        self._batch_time_interval = BATCH_TIME_INTERVAL_SECONDS
        self._pending_records: list[Record] = []
        self._last_batch_time = time.time()
        self._batch_lock = threading.Lock()

        # consumer threads for sending records
        self._consumer_threads = []
        self._is_config_fetched = False
        self.config = None
        self._fetch_config()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._clean_up()

    def _fetch_config(self):
        def _fetch():
            try:
                self._get_config()
                if self.config is None:
                    self._is_stopped = True
                    _set_telemetry_client(None)
                else:
                    # send the import record immediately after config is fetched
                    # do not add if config is None
                    self.add_record(
                        Record(
                            event_name=ImportMlflowEvent.name,
                            timestamp_ns=time.time_ns(),
                            status=Status.SUCCESS,
                            duration_ms=0,
                        ),
                        send_immediately=True,
                    )
                self._is_config_fetched = True
            except Exception:
                self._is_stopped = True
                self._is_config_fetched = True
                _set_telemetry_client(None)

        self._config_thread = threading.Thread(
            target=_fetch,
            name="GetTelemetryConfig",
            daemon=True,
        )
        self._config_thread.start()

    def _get_config(self):
        """
        Get the config for the given MLflow version.
        """
        mlflow_version = self.info["mlflow_version"]
        if config_url := _get_config_url(mlflow_version):
            try:
                response = requests.get(config_url, timeout=1)
                if response.status_code != 200:
                    return
                config = response.json()
                if (
                    config.get("mlflow_version") != mlflow_version
                    or config.get("disable_telemetry") is True
                    or config.get("ingestion_url") is None
                ):
                    return

                if get_source_sdk().value in config.get("disable_sdks", []):
                    return

                if sys.platform in config.get("disable_os", []):
                    return

                rollout_percentage = config.get("rollout_percentage", 100)
                if random.randint(0, 100) > rollout_percentage:
                    return

                self.config = TelemetryConfig(
                    ingestion_url=config["ingestion_url"],
                    disable_events=set(config.get("disable_events", [])),
                )
            except Exception:
                return

    def add_record(self, record: Record, send_immediately: bool = False):
        """
        Add a record to be batched and sent to the telemetry server.
        """
        if not self.is_active:
            self.activate()

        if self._is_stopped:
            return

        with self._batch_lock:
            self._pending_records.append(record)

            # Only send if we've reached the batch size;
            # time-based sending is handled by the consumer thread.
            if send_immediately or len(self._pending_records) >= self._batch_size:
                self._send_batch()

    def _send_batch(self):
        """Send the current batch of records."""
        if not self._pending_records:
            return

        self._last_batch_time = time.time()

        try:
            self._queue.put(self._pending_records, block=False)
            self._pending_records = []
        except Full:
            # TODO: record this case
            pass

    def _process_records(self, records: list[Record], request_timeout: float = 1):
        """Process a batch of telemetry records."""
        try:
            self._update_backend_store()
            if self.info["tracking_uri_scheme"] in ["databricks", "databricks-uc", "uc"]:
                self._is_stopped = True
                # set config to None to allow consumer thread drop records in the queue
                self.config = None
                self.is_active = False
                _set_telemetry_client(None)
                return

            records = [
                {
                    "data": self.info | record.to_dict(),
                    # use random uuid as partition key to make sure records are
                    # distributed evenly across shards
                    "partition-key": uuid.uuid4().hex,
                }
                for record in records
            ]
            # changing this value can affect total time for processing records
            # the total time = request_timeout * max_attempts + sleep_time * (max_attempts - 1)
            max_attempts = 3
            sleep_time = 1
            for i in range(max_attempts):
                should_retry = False
                response = None
                try:
                    response = requests.post(
                        self.config.ingestion_url,
                        json={"records": records},
                        headers={"Content-Type": "application/json"},
                        timeout=request_timeout,
                    )
                    should_retry = response.status_code in RETRYABLE_ERRORS
                except (ConnectionError, TimeoutError):
                    should_retry = True
                # NB: DO NOT retry when terminating
                # otherwise this increases shutdown overhead significantly
                if self._is_stopped:
                    return
                if i < max_attempts - 1 and should_retry:
                    # we do not use exponential backoff to avoid increasing
                    # the processing time significantly
                    time.sleep(sleep_time)
                elif response and response.status_code in UNRECOVERABLE_ERRORS:
                    self._is_stopped = True
                    self.is_active = False
                    # this is executed in the consumer thread, so
                    # we cannot join the thread here, but this should
                    # be enough to stop the telemetry collection
                    return
                else:
                    return
        except Exception:
            pass

    def _consumer(self) -> None:
        """Individual consumer that processes records from the queue."""
        # suppress logs in the consumer thread to avoid emitting any irrelevant
        # logs during telemetry collection.
        should_suppress_logs_in_thread.set(True)

        while not self._is_config_fetched:
            time.sleep(0.1)

        while self.config and not self._is_stopped:
            try:
                records = self._queue.get(timeout=1)
            except Empty:
                # check if batch time interval has passed and send data if needed
                if time.time() - self._last_batch_time >= self._batch_time_interval:
                    self._last_batch_time = time.time()
                    with self._batch_lock:
                        if self._pending_records:
                            self._send_batch()
                continue

            self._process_records(records)
            self._queue.task_done()

        # clear the queue if config is None
        while self.config is None and not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Empty:
                break
        # drop remaining records when terminating to avoid
        # causing any overhead

    def activate(self) -> None:
        """Activate the async queue to accept and handle incoming tasks."""
        with self._lock:
            if self.is_active:
                return

            self._set_up_threads()

            # Callback to ensure remaining tasks are processed before program exit
            if not self._atexit_callback_registered:
                # This works in jupyter notebook
                atexit.register(self._at_exit_callback)
                self._atexit_callback_registered = True

            self.is_active = True

    @property
    def is_active(self) -> bool:
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        self._is_active = value

    def _set_up_threads(self) -> None:
        """Set up multiple consumer threads."""
        with self._lock:
            # Start multiple consumer threads
            for i in range(self._max_workers):
                consumer_thread = threading.Thread(
                    target=self._consumer,
                    name=f"MLflowTelemetryConsumer-{i}",
                    daemon=True,
                )
                consumer_thread.start()
                self._consumer_threads.append(consumer_thread)

    def _at_exit_callback(self) -> None:
        """Callback function executed when the program is exiting."""
        try:
            # Suppress logs/warnings during shutdown
            # NB: this doesn't suppress log not emitted by mlflow
            with suppress_logs_in_thread(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.flush(terminate=True)
        except Exception:
            pass

    def flush(self, terminate=False) -> None:
        """
        Flush the async telemetry queue.

        Args:
            terminate: If True, shut down the telemetry threads after flushing.
        """
        if not self.is_active:
            return

        if terminate:
            # Full shutdown for termination - signal stop and exit immediately
            self._is_stopped = True
            self.is_active = False

        # non-terminating flush is only used in tests
        else:
            self._config_thread.join(timeout=1)

            # Send any pending records before flushing
            with self._batch_lock:
                if self._pending_records and self.config and not self._is_stopped:
                    self._send_batch()
            # For non-terminating flush, just wait for queue to empty
            try:
                self._queue.join()
            except Exception:
                pass

    def _update_backend_store(self):
        """
        Backend store might be changed after mlflow is imported, we should use this
        method to update the backend store info at sending telemetry step.
        """
        try:
            # import here to avoid circular import
            from mlflow.tracking._tracking_service.utils import _get_tracking_scheme

            self.info["tracking_uri_scheme"] = _get_tracking_scheme()
        except Exception:
            pass

    def _clean_up(self):
        """Join all threads"""
        self.flush(terminate=True)
        for thread in self._consumer_threads:
            if thread.is_alive():
                thread.join(timeout=1)


_MLFLOW_TELEMETRY_CLIENT = None
_client_lock = threading.Lock()


def set_telemetry_client():
    if is_telemetry_disabled():
        # set to None again so this function can be used to
        # re-initialize the telemetry client
        _set_telemetry_client(None)
    else:
        try:
            _set_telemetry_client(TelemetryClient())
        except Exception:
            _set_telemetry_client(None)


def _set_telemetry_client(value: TelemetryClient | None):
    global _MLFLOW_TELEMETRY_CLIENT
    with _client_lock:
        _MLFLOW_TELEMETRY_CLIENT = value
        if value:
            _MLFLOW_TELEMETRY_SESSION_ID.set(value.info["session_id"])
        else:
            _MLFLOW_TELEMETRY_SESSION_ID.unset()


def get_telemetry_client() -> TelemetryClient | None:
    with _client_lock:
        return _MLFLOW_TELEMETRY_CLIENT
