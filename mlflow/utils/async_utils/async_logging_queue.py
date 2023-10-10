"""
Defines an AsyncLoggingQueue that provides async fashion logging of metrics/tags/params using
 queue based approach.
"""

import atexit
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag
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

    def __init__(self, processing_func: callable) -> None:
        """
        Initializes an AsyncLoggingQueue object.

        Args:
            processing_func (callable): A function that will be called to process each item in
             the queue.
        """
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

    def _add_run(self, run_id: str) -> Queue:
        """
        Adds a new run to the running runs map.

        Args:
            run_id (str): The ID of the run to add.

        Returns:
            Queue: The queue associated with the given run ID.
        """
        self._lock.acquire()
        if not self._running_runs.get(run_id, None):
            self._running_runs[run_id] = Queue()
        self._lock.release()
        return self._running_runs[run_id]

    def _at_exit_callback(self) -> None:
        """
        Callback function to be executed when the program is exiting.
        Stops the data processing thread and waits for the
        queue to be drained.
        Finally, shuts down the thread pools used for data logging and batch processing status
          check.
        """
        try:
            # Stop the data processing thread
            self._lock.acquire()
            self.continue_to_process_data = False
            self._lock.release()
            self._queue_consumer.set()
            # Waits till queue is drained.
            self.run_data_process_thread.result()
            _RUN_DATA_LOGGING_THREADPOOL.shutdown(wait=False)
            _RUN_BATCH_PROCESSING_STATUS_CHECK_THREADPOOL.shutdown(wait=False)
        except Exception as e:
            _logger.error(f"Error while callback from atexit in _at_exit_callback: {e}")

    def _log_run_data(self) -> None:
        """
        Continuously processes run data from the logging queue until `continue_to_process_data`
         is False.
        If an exception is raised during processing, it is logged and re-raised.
        """
        try:
            while self.continue_to_process_data:
                self._queue_consumer.wait()
                self._process_run_data()
        except Exception as e:
            _logger.error(f"Exception inside the thread: {e}")
            raise

    def _process_run_data(self) -> None:
        """
        Process the run data in the running runs queues.

        For each run in the running runs queues, this method retrieves the next batch of run data
         from the queue and processes it by calling the `_processing_func` method with the run ID,
           metrics, parameters, and tags in the batch.
        If the batch is empty, it is skipped. After processing the batch, the processed watermark
         is updated and the batch event is set.
        If an exception occurs during processing, the exception is logged and the batch event is set
        with the exception. If the queue is empty, it is ignored.

        Returns: None
        """
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

    def _get_next_id(self) -> int:
        """
        Returns the next available ID for an item in the queue. This method is thread-safe.
        """
        self._lock.acquire()
        next_id = self.enqueued_watermark + 1
        self.enqueued_watermark = next_id
        self._lock.release()
        return next_id

    def _wait_for_batch(self, batch: RunBatch) -> None:
        """
        Wait for the given batch to be processed by the logging thread.

        Args:
            batch: The batch to wait for.

        Raises:
            Exception: If an exception occurred while processing the batch.
        """
        batch.event.wait()
        if batch.exception:
            raise batch.exception

    def log_batch_async(
        self, run_id: str, params: [Param], tags: [RunTag], metrics: [Metric]
    ) -> RunOperations:
        """
        Asynchronously logs a batch of run data (parameters, tags, and metrics) for a given run ID.

        Args:
            run_id (str): The ID of the run to log data for.
            params (list[mlflow.entities.Param]): A list of parameters to log for the run.
            tags (list[mlflow.entities.RunTag]): A list of tags to log for the run.
            metrics (list[mlflow.entities.Metric]): A list of metrics to log for the run.

        Returns:
            mlflow.utils.async_utils.RunOperations: An object that encapsulates the
              asynchronous operation of logging the batch of run data.
              The object contains a list of `concurrent.futures.Future` objects that can be used
                to check the status of the operation and retrieve any exceptions
            that occurred during the operation.
        """
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
