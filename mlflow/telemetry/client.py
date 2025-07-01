import atexit
import json
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from queue import Empty, Full, Queue
from typing import Optional

import requests

from mlflow.environment_variables import (
    _MLFLOW_TELEMETRY_BATCH_SIZE,
    _MLFLOW_TELEMETRY_BATCH_TIME_INTERVAL,
    _MLFLOW_TELEMETRY_MAX_QUEUE_SIZE,
    _MLFLOW_TELEMETRY_MAX_WORKERS,
)
from mlflow.telemetry.schemas import APIRecord, TelemetryInfo, TelemetryRecord
from mlflow.telemetry.utils import is_telemetry_disabled

_logger = logging.getLogger(__name__)
# TODO: update this url with a custom domain
TELEMETRY_URL = "https://twqhrx9tai.execute-api.us-west-2.amazonaws.com/test/telemetry"


class TelemetryClient:
    def __init__(self):
        self.info = TelemetryInfo()
        self.telemetry_url = TELEMETRY_URL
        self._queue: Queue[list[APIRecord]] = Queue(maxsize=_MLFLOW_TELEMETRY_MAX_QUEUE_SIZE.get())
        self._lock = threading.RLock()
        self._max_workers = _MLFLOW_TELEMETRY_MAX_WORKERS.get()

        # Thread event that indicates the queue should stop processing tasks
        self._stop_event = threading.Event()
        self._is_active = False
        self._atexit_callback_registered = False

        self._active_tasks = set()

        self._batch_size = _MLFLOW_TELEMETRY_BATCH_SIZE.get()
        self._batch_time_interval = _MLFLOW_TELEMETRY_BATCH_TIME_INTERVAL.get()
        self._pending_records: list[TelemetryRecord] = []
        self._last_batch_time = time.time()
        self._batch_lock = threading.Lock()

    def add_record(self, record: APIRecord):
        """
        Add a record to be batched and sent to the telemetry server.
        """
        if not self.is_active():
            self.activate()

        # If stop event is set, don't add new records
        if self._stop_event.is_set():
            _logger.debug("Telemetry is stopped, skipping adding record")
            return

        with self._batch_lock:
            data = self._generate_telemetry_record(record)
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
        batch_records = self._pending_records.copy()
        self._pending_records.clear()
        self._last_batch_time = time.time()

        try:
            self._queue.put(batch_records, block=False)
        except Full:
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

    def _consumer_loop(self) -> None:
        while not self._stop_event.is_set():
            self._dispatch_task()

        # Drain remaining tasks when stopping
        while not self._queue.empty():
            self._dispatch_task()

    def _dispatch_task(self) -> None:
        """Dispatch a task from the queue to the worker thread pool."""
        if len(self._active_tasks) >= self._max_workers:
            _, self._active_tasks = wait(self._active_tasks, return_when=FIRST_COMPLETED)

        try:
            records = self._queue.get(timeout=1)
        except Empty:
            return

        def _handle(records):
            self._process_records(records)
            self._queue.task_done()

        try:
            future = self._worker_threadpool.submit(_handle, records)
            self._active_tasks.add(future)
        except Exception as e:
            # In case it fails to submit the task to the worker thread pool
            # such as interpreter shutdown, handle the task in this thread
            _logger.debug(
                f"Failed to submit task to worker thread pool. Error: {e}",
                exc_info=True,
            )
            _handle(records)

    def activate(self) -> None:
        """Activate the async queue to accept and handle incoming tasks."""
        with self._lock:
            if self._is_active:
                return

            self._set_up_threads()

            # Callback to ensure remaining tasks are processed before program exit
            if not self._atexit_callback_registered:
                atexit.register(self._at_exit_callback)
                self._atexit_callback_registered = True

            self._is_active = True

    def is_active(self) -> bool:
        return self._is_active

    def _set_up_threads(self) -> None:
        """Set up the consumer and worker threads."""
        with self._lock:
            self._worker_threadpool = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="MlflowTelemetryWorker",
            )
            self._consumer_thread = threading.Thread(
                target=self._consumer_loop,
                name="MLflowTelemetryConsumer",
                daemon=True,
            )
            self._consumer_thread.start()

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
        if not self.is_active():
            return

        try:
            # Send any pending records before flushing
            with self._batch_lock:
                if self._pending_records:
                    self._send_batch()

            self._stop_event.set()

            try:
                self._consumer_thread.join(timeout=30)
            except Exception as e:
                _logger.debug(f"Error waiting for consumer thread: {e}")

            try:
                self._queue.join()
            except Exception as e:
                _logger.debug(f"Error waiting for queue to drain: {e}")

            try:
                self._worker_threadpool.shutdown(wait=True, timeout=30)
            except Exception as e:
                _logger.debug(f"Error shutting down worker thread pool: {e}")

            self._is_active = False

            # Restart threads to listen to incoming requests after flushing, if not terminating
            if not terminate:
                self._stop_event.clear()
                self.activate()

        except Exception as e:
            _logger.debug(f"Error during telemetry flush: {e}", exc_info=True)
            # Ensure we mark as inactive even if there was an error
            self._is_active = False

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
