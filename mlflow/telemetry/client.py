import atexit
import threading
import time
import uuid
from dataclasses import asdict
from queue import Empty, Full, Queue
from typing import Optional

import requests

from mlflow.telemetry.constant import (
    BATCH_SIZE,
    BATCH_TIME_INTERVAL_SECONDS,
    MAX_QUEUE_SIZE,
    MAX_WORKERS,
)
from mlflow.telemetry.schemas import APIRecord, TelemetryConfig, TelemetryInfo
from mlflow.telemetry.utils import _get_config, is_telemetry_disabled
from mlflow.utils.logging_utils import should_suppress_logs_in_thread, suppress_logs_in_thread
from mlflow.version import IS_TRACING_SDK_ONLY


class TelemetryClient:
    def __init__(self, config: TelemetryConfig):
        self.info = asdict(TelemetryInfo())
        self.config = config
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

    def _process_records(self, records: list[APIRecord]):
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
            # TODO: add retry logic
            response = requests.post(
                self.config.telemetry_url,
                json={"records": records},
                headers={"Content-Type": "application/json"},
                timeout=3,
            )
            if response.status_code != 200:
                pass
        except Exception:
            pass

    def _consumer(self) -> None:
        """Individual consumer that processes records from the queue."""
        # suppress logs in the consumer thread to avoid emitting any irrelevant
        # logs during telemetry collection.
        should_suppress_logs_in_thread.set(True)

        while not self._is_stopped:
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

    def activate(self) -> None:
        """Activate the async queue to accept and handle incoming tasks."""
        with self._lock:
            if self.is_active:
                return

            self._set_up_threads()

            # Callback to ensure remaining tasks are processed before program exit
            # TODO: make sure this works in jupyter notebook
            if not self._atexit_callback_registered:
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
        self.flush(terminate=True)

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

            # process pending records directly before exiting
            with self._batch_lock, suppress_logs_in_thread():
                if self._pending_records:
                    self._process_records(self._pending_records)
                self._pending_records = []

            # Wait for threads to finish with a timeout
            avg_timeout_per_thread = (
                1 / len(self._consumer_threads) if self._consumer_threads else 0
            )
            for thread in self._consumer_threads:
                if thread.is_alive():
                    thread.join(timeout=avg_timeout_per_thread)

        else:
            # Send any pending records before flushing
            with self._batch_lock:
                if self._pending_records:
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
            # import here to avoid circular import
            from mlflow.tracking._tracking_service.utils import _get_tracking_scheme

            self.info["backend_store_scheme"] = _get_tracking_scheme()


_MLFLOW_TELEMETRY_CLIENT = None


def set_telemetry_client():
    global _MLFLOW_TELEMETRY_CLIENT

    if is_telemetry_disabled():
        # set to None again so this function can be used to
        # re-initialize the telemetry client
        _MLFLOW_TELEMETRY_CLIENT = None
    else:

        def _init():
            global _MLFLOW_TELEMETRY_CLIENT

            if config := _get_config():
                _MLFLOW_TELEMETRY_CLIENT = TelemetryClient(config=config)

        thread = threading.Thread(target=_init, name="GetTelemetryConfig", daemon=True)
        thread.start()
        thread.join(timeout=3)


def get_telemetry_client() -> Optional[TelemetryClient]:
    return _MLFLOW_TELEMETRY_CLIENT
