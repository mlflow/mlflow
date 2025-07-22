import atexit
import random
import sys
import threading
import time
import uuid
import warnings
from dataclasses import asdict
from queue import Empty, Full, Queue
from typing import Optional

import requests

from mlflow.telemetry.constant import (
    BATCH_SIZE,
    BATCH_TIME_INTERVAL_SECONDS,
    CONFIG_RETRYABLE_ERRORS,
    MAX_QUEUE_SIZE,
    MAX_WORKERS,
    RETRYABLE_ERRORS,
    STOP_COLLECTION_ERRORS,
)
from mlflow.telemetry.schemas import APIRecord, TelemetryConfig, TelemetryInfo, get_source_sdk
from mlflow.telemetry.utils import _get_config_url, is_telemetry_disabled
from mlflow.utils.logging_utils import should_suppress_logs_in_thread, suppress_logs_in_thread
from mlflow.version import IS_TRACING_SDK_ONLY

try:
    from IPython import get_ipython

    IS_IPYTHON = get_ipython() is not None
except ImportError:
    IS_IPYTHON = False


class TelemetryClient:
    def __init__(self):
        self.info = asdict(TelemetryInfo())
        self._queue: Queue[list[APIRecord]] = Queue(maxsize=MAX_QUEUE_SIZE)
        self._lock = threading.RLock()
        self._max_workers = MAX_WORKERS

        self._is_stopped = False
        self._is_active = False
        self._atexit_callback_registered = False

        self._batch_size = BATCH_SIZE
        self._batch_time_interval = BATCH_TIME_INTERVAL_SECONDS
        self._pending_records: list[APIRecord] = []
        self._last_batch_time = time.time()
        self._batch_lock = threading.Lock()

        # consumer threads for sending records
        self._consumer_threads = []
        self._is_config_fetched = False
        self.config = None
        self._fetch_config()

    def _fetch_config(self):
        def _fetch():
            try:
                self._get_config()
                if self.config is None:
                    self._is_stopped = True
                    _set_telemetry_client(None)
                else:
                    # If any telemetry records are generated before the config is loaded,
                    # filter them by the condition defined in the config before exporting.
                    with self._batch_lock:
                        if self._pending_records:
                            self._drop_disabled_records()
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
                for i in range(3):
                    response = requests.get(config_url, timeout=1)
                    if response.status_code == 200:
                        break
                    if response.status_code in CONFIG_RETRYABLE_ERRORS:
                        time.sleep(2**i)
                    else:
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
                    disable_api_map=config.get("disable_api_map", {}),
                )
            except Exception:
                return

    def _drop_disabled_records(self):
        """
        Drop invalid records that are disabled by the config.
        """
        if self.config:
            self._pending_records = [
                record
                for record in self._pending_records
                if record.api_name not in self.config.disable_api_map.get(record.api_module, [])
            ]

    def add_record(self, record: APIRecord):
        """
        Add a record to be batched and sent to the telemetry server.
        """
        if not self.is_active:
            self.activate()

        if self._is_stopped:
            return

        with self._batch_lock:
            self._pending_records.append(record)

            # Only send immediately if we've reached the batch size,
            # time-based sending is handled by the batch checker thread
            if len(self._pending_records) >= self._batch_size:
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

    def _process_records(self, records: list[APIRecord], timeout: float = 3):
        """Process a batch of telemetry records."""
        try:
            self._update_backend_store()
            records = [
                {
                    "data": self.info | record.to_dict(),
                    # use random uuid as partition key to make sure records are
                    # distributed evenly across shards
                    "partition-key": uuid.uuid4().hex,
                }
                for record in records
            ]
            max_retries = 3
            for i in range(max_retries):
                response = requests.post(
                    self.config.ingestion_url,
                    json={"records": records},
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )
                # if this is executed when terminating, we should not retry
                if self._is_stopped:
                    return
                if response.status_code in STOP_COLLECTION_ERRORS:
                    self._is_stopped = True
                    self.is_active = False
                    # this is executed in the consumer thread, so
                    # we cannot join the thread here, but this should
                    # be enough to stop the telemetry collection
                    return
                if response.status_code in RETRYABLE_ERRORS:
                    time.sleep(2**i)
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

        # process remaining records when terminating
        if self.config and self._pending_records:
            with self._batch_lock:
                if self._pending_records:
                    self._process_records(self._pending_records, timeout=1)
                    self._pending_records = []

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

            self._config_thread.join(timeout=1)

            # Wait for threads to finish with a timeout
            # The timeout for jupyter notebook needs to be higher
            timeout = 3 if IS_IPYTHON else 2
            avg_timeout_per_thread = (
                timeout / len(self._consumer_threads) if self._consumer_threads else 0
            )
            for thread in self._consumer_threads:
                if thread.is_alive():
                    thread.join(timeout=avg_timeout_per_thread)

        # non-terminating flush is only used in tests
        else:
            # Send any pending records before flushing
            with self._batch_lock:
                if self._pending_records and self.config and not self._is_stopped:
                    self._drop_disabled_records()
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
        if not IS_TRACING_SDK_ONLY:
            try:
                # import here to avoid circular import
                from mlflow.tracking._tracking_service.utils import _get_tracking_scheme

                self.info["backend_store_scheme"] = _get_tracking_scheme()
            except Exception:
                pass


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


def get_telemetry_client() -> Optional[TelemetryClient]:
    with _client_lock:
        return _MLFLOW_TELEMETRY_CLIENT
