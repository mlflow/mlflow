import atexit
import json
import logging
import threading
import time
from dataclasses import asdict
from queue import Empty, Full, Queue
from typing import Optional

import requests

from mlflow.telemetry.constant import (
    BATCH_SIZE,
    BATCH_TIME_INTERVAL_SECONDS,
    MAX_QUEUE_SIZE,
    MAX_WORKERS,
    TELEMETRY_URL,
)
from mlflow.telemetry.schemas import APIRecord, TelemetryInfo, TelemetryRecord
from mlflow.telemetry.utils import is_telemetry_disabled

_logger = logging.getLogger(__name__)


class TelemetryClient:
    def __init__(self):
        self.info = TelemetryInfo()
        self.telemetry_url = TELEMETRY_URL
        self._queue: Queue[list[APIRecord]] = Queue(maxsize=MAX_QUEUE_SIZE)
        self._lock = threading.RLock()
        self._max_workers = MAX_WORKERS

        self._is_stopped = False
        self._is_active = False
        self._atexit_callback_registered = False

        # Track consumer threads
        self._consumer_threads = []

        self._batch_size = BATCH_SIZE
        self._batch_time_interval = BATCH_TIME_INTERVAL_SECONDS
        self._pending_records: list[TelemetryRecord] = []
        self._last_batch_time = time.time()
        self._batch_lock = threading.Lock()

    def add_record(self, record: APIRecord):
        """
        Add a record to be batched and sent to the telemetry server.
        """
        if not self.is_active:
            self.activate()

        if self._is_stopped:
            _logger.debug("Telemetry is stopped, skipping adding record")
            return

        data = self._generate_telemetry_record(record)
        with self._batch_lock:
            self._pending_records.append(data)

            should_send = (
                len(self._pending_records) >= self._batch_size
                or (time.time() - self._last_batch_time) >= self._batch_time_interval
            )

            if should_send:
                self._send_batch()

    def _send_batch(self):
        """Send the current batch of records."""
        if not self._pending_records:
            return

        # Create a copy of the current batch and clear the pending list
        self._last_batch_time = time.time()

        try:
            self._queue.put(self._pending_records, block=False)
            self._pending_records = []
        except Full:
            # TODO: record this case
            _logger.debug("Telemetry queue is full, skipping sending data.")

    def _process_records(self, records: list[TelemetryRecord]):
        """Process a batch of telemetry records."""
        try:
            data = {
                "records": [
                    {"data": record.data, "partition-key": record.partition_key}
                    for record in records
                ]
            }
            response = requests.post(
                self.telemetry_url,
                json=data,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code != 200:
                _logger.debug(
                    f"Failed to send telemetry records. Status code: {response.status_code}, "
                    f"Response: {response.text}"
                )
        except Exception as e:
            _logger.debug(
                f"Failed to process telemetry records. Error: {e}.",
                exc_info=True,
            )

    def _consumer(self) -> None:
        """Individual consumer that processes records from the queue."""
        while not self._is_stopped:
            try:
                records = self._queue.get(timeout=1)
            except Empty:
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
        _logger.debug(
            "Flushing the async telemetry queue before program exit. This may take a while..."
        )
        self.flush(terminate=True)

    def flush(self, terminate=False) -> None:
        """
        Flush the async telemetry queue.

        Args:
            terminate: If True, shut down the telemetry threads after flushing.
        """
        if not self.is_active:
            return

        # Send any pending records before flushing
        with self._batch_lock:
            if self._pending_records:
                self._send_batch()

        if terminate:
            # Full shutdown for termination - signal stop and exit immediately
            self._is_stopped = True
            self.is_active = False
            _logger.debug(
                f"Telemetry shutdown complete, dropping {self._queue.qsize()} pending records"
            )
        else:
            # For non-terminating flush, just wait for queue to empty
            try:
                self._queue.join()
            except Exception as e:
                _logger.debug(f"Error waiting for queue to drain: {e}")

    def _update_backend_store(self):
        """
        Backend store might be changed after mlflow is imported, we should use this
        method to update the backend store info at sending telemetry step.
        """
        # import here to avoid circular import
        from mlflow.tracking._tracking_service.utils import _get_store

        tracking_store = _get_store()
        self.info.backend_store = tracking_store.__class__.__name__

    def _generate_telemetry_record(self, record: APIRecord) -> TelemetryRecord:
        self._update_backend_store()
        telemetry_info = asdict(self.info)
        # TODO: update partition key
        return TelemetryRecord(
            data=json.dumps(telemetry_info | asdict(record)), partition_key="test"
        )


_MLFLOW_TELEMETRY_CLIENT = None


def set_telemetry_client():
    global _MLFLOW_TELEMETRY_CLIENT

    if is_telemetry_disabled():
        _logger.debug("MLflow Telemetry is disabled")
        # set to None again so this function can be used to
        # re-initialize the telemetry client
        _MLFLOW_TELEMETRY_CLIENT = None
    else:
        _MLFLOW_TELEMETRY_CLIENT = TelemetryClient()


def get_telemetry_client() -> Optional[TelemetryClient]:
    return _MLFLOW_TELEMETRY_CLIENT
