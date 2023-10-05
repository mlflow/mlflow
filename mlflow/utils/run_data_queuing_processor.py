"""
Defines an RunDataQueuingProcessor that provides batching, queueing, and
asynchronous execution capabilities for a subset of MLflow Tracking logging operations
 log_metrics/log_params/set_tags
"""

import atexit
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from mlflow.utils.pending_run_batches import PendingRunBatches, RunBatch

_logger = logging.getLogger(__name__)

# Keeping max_workers=1 so that there are no two threads
_RUN_DATA_LOGGING_THREADPOOL = ThreadPoolExecutor(max_workers=1)


class RunDataQueuingProcessor:
    """
    This is a queue based run data processor that queues incoming batches and processes them using
    single worker thread.
    """

    def __init__(self, processing_func):
        self._running_runs = {}  # Dict[str, PendingRunBatches]
        self._processing_func = processing_func
        self._lock = threading.Lock()
        self.continue_to_process_data = True
        self.run_data_process_thread = _RUN_DATA_LOGGING_THREADPOOL.submit(
            self._log_run_data
        )  # concurrent.futures.Future[self._log_run_data]

        atexit.register(self._at_exit_callback)
        self.processed_watermark = -1

    def _add_run(self, run_id) -> PendingRunBatches:
        self._lock.acquire()
        if not self._running_runs.get(run_id, None):
            self._running_runs[run_id] = PendingRunBatches(run_id=run_id)
        self._lock.release()
        return self._running_runs[run_id]

    def _has_more_data_to_process(self) -> bool:
        for _, pending_batches in self._running_runs.items():
            if not pending_batches.is_empty():
                return True
        return False

    def _at_exit_callback(self):
        try:
            # Stop the data processing thread
            self.continue_to_process_data = False
            # need better way to decide this timeout
            self.run_data_process_thread.result(timeout=60)
            _RUN_DATA_LOGGING_THREADPOOL.shutdown(wait=True)
        except Exception as e:
            _logger.error(f"Error while callback from atexit in _process_at_exit: {e}")

    def _log_run_data(self):
        try:
            while self.continue_to_process_data or self._has_more_data_to_process():
                self._process_run_data()
                time.sleep(5)
        except Exception as e:
            _logger.error(f"Exception inside the thread: {e}")
            raise

    def _process_run_data(self):
        for run_id, pending_items in self._running_runs.items():
            (yet_to_process_watermark, params, tags, metrics) = pending_items.get_next_batch(
                max_tags=100, max_metrics=250, max_params=200
            )  # need to get these values from env variable etc.

            while len(params) > 0 or len(tags) > 0 or len(metrics) > 0:
                try:
                    self._processing_func(run_id=run_id, metrics=metrics, params=params, tags=tags)

                    self.processed_watermark = yet_to_process_watermark
                    _logger.info(
                        f"run_id: {run_id}, Processed watermark: {self.processed_watermark}"
                    )
                    # get next batch
                    (
                        yet_to_process_watermark,
                        params,
                        tags,
                        metrics,
                    ) = pending_items.get_next_batch(max_tags=100, max_metrics=250, max_params=200)
                except Exception as e:
                    _logger.error(e)
                    raise

    def log_batch_async(self, run_id, params, tags, metrics):
        pending_run_batches = self._running_runs.get(run_id, None)
        if not pending_run_batches:
            pending_run_batches = self._add_run(run_id=run_id)

        batch = RunBatch(params=params, tags=tags, metrics=metrics)
        enqueue_id = pending_run_batches.append(batch)
        _logger.info(f"Enqueued watermark: {enqueue_id}")
