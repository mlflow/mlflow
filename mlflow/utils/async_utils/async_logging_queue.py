"""
Defines an AsyncLoggingQueue that provides async fashion logging of metrics/tags/params using
 queue based approach.
"""

import atexit
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

from mlflow.utils.async_utils.run_batch import RunBatch
from mlflow.utils.async_utils.run_operations import RunOperations

_logger = logging.getLogger(__name__)

# Keeping max_workers=1 so that there are no two threads
_RUN_DATA_LOGGING_THREADPOOL = ThreadPoolExecutor(max_workers=1)

_RUN_BATCH_PROCESSING_STATUS_CHECK_THREADPOOL = ThreadPoolExecutor(max_workers=1)


class AsyncLoggingQueue:
    """
    This is a queue based run data processor that queues incoming batches and processes them using
    single worker thread.
    """

    def __init__(self, processing_func):
        self._running_runs = {}  # Dict[str, Queue]
        self._processing_func = processing_func
        self.enqueued_watermark = -1
        self.processed_watermark = -1
        self._queue_consumer = threading.Event()
        self._lock = threading.RLock()
        self.continue_to_process_data = True
        self.run_data_process_thread = _RUN_DATA_LOGGING_THREADPOOL.submit(
            self._log_run_data
        )  # concurrent.futures.Future[self._log_run_data]

        atexit.register(self._at_exit_callback)

    def _add_run(self, run_id) -> Queue:
        self._lock.acquire()
        if not self._running_runs.get(run_id, None):
            self._running_runs[run_id] = Queue()
        self._lock.release()
        return self._running_runs[run_id]

    def _at_exit_callback(self):
        try:
            # Stop the data processing thread
            self._lock.acquire()
            self.continue_to_process_data = False
            self._lock.release()
            # Waits till queue is drained.
            self.run_data_process_thread.result()
            _RUN_DATA_LOGGING_THREADPOOL.shutdown(wait=False)
            _RUN_BATCH_PROCESSING_STATUS_CHECK_THREADPOOL.shutdown(wait=False)
        except Exception as e:
            _logger.error(f"Error while callback from atexit in _at_exit_callback: {e}")

    def _log_run_data(self):
        try:
            while self.continue_to_process_data:
                self._queue_consumer.wait()
                self._process_run_data()
        except Exception as e:
            _logger.error(f"Exception inside the thread: {e}")
            raise

    def _process_run_data(self):
        for run_id, run_queue in self._running_runs.items():
            try:
                while not run_queue.empty():
                    run_batch = run_queue.get(timeout=1)
                    try:
                        if run_batch.is_empty():
                            continue
                        self._processing_func(
                            run_id=run_id,
                            metrics=run_batch.metrics,
                            params=run_batch.params,
                            tags=run_batch.tags,
                        )
                        self.processed_watermark = run_batch.id
                        # Signal the batch processing is done.
                        run_batch.event.set()
                        _logger.debug(
                            f"run_id: {run_id}, Processed watermark: {self.processed_watermark}"
                        )
                    except Exception as e:  # Importing MlflowException gives circular reference
                        # / module load error, need to figure out why.
                        _logger.error(f"Failed to log run data: Exception: {e}")
                        run_batch.exception = e
                        run_batch.event.set()
            except Empty:
                # Ignore empty queue exception
                pass

    def _get_next_id(self):
        self._lock.acquire()
        next_id = self.enqueued_watermark + 1
        self.enqueued_watermark = next_id
        self._lock.release()
        return next_id

    def _wait_for_batch(self, batch):
        batch.event.wait()
        if batch.exception:
            raise batch.exception

    def log_batch_async(self, run_id, params, tags, metrics) -> RunOperations:
        run_queue = self._add_run(run_id=run_id)

        batch = RunBatch(
            id=self._get_next_id(),
            run_id=run_id,
            params=params,
            tags=tags,
            metrics=metrics,
            event=threading.Event(),
        )

        if batch.is_empty():
            return RunOperations()

        run_queue.put(batch)
        _logger.debug(f"run_id: {run_id}, Enqueued watermark: {batch.id}")

        # Signal for consumer to start consuming.
        self._queue_consumer.set()

        operation_future = _RUN_BATCH_PROCESSING_STATUS_CHECK_THREADPOOL.submit(
            self._wait_for_batch, batch
        )
        return RunOperations(operation_futures=[operation_future])
