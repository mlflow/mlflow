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
from mlflow.utils.async_logging.run_batch import RunBatch
from mlflow.utils.async_logging.run_operations import RunOperations

_logger = logging.getLogger(__name__)


class AsyncLoggingQueue:
    """
    This is a queue based run data processor that queues incoming batches and processes them using
    single worker thread.
    """

    def __init__(self, logging_func: callable([str, [Metric], [Param], [RunTag]])) -> None:
        """
        Initializes an AsyncLoggingQueue object.

        Args:
            logging_func: A callable function that takes in four arguments: a string
                representing the run_id, a list of Metric objects,
                a list of Param objects, and a list of RunTag objects.
        """
        self._queue = Queue()
        self._logging_func = logging_func
        self._continue_to_log_data = threading.Event()
        # Keeping max_workers=1 so that there are no two threads
        self._BATCH_LOGGING_THREADPOOL = ThreadPoolExecutor(max_workers=1)

        self._BATCH_STATUS_CHECK_THREADPOOL = ThreadPoolExecutor(max_workers=10)

        self._run_data_logging_thread = self._BATCH_LOGGING_THREADPOOL.submit(
            self._logging_loop
        )  # concurrent.futures.Future[self._logging_loop]

        atexit.register(self._at_exit_callback)

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
            self._continue_to_log_data.set()
            # Waits till queue is drained.
            self._run_data_logging_thread.result()
            self._BATCH_LOGGING_THREADPOOL.shutdown(wait=False)
            self._BATCH_STATUS_CHECK_THREADPOOL.shutdown(wait=False)
        except Exception as e:
            _logger.error(f"Encountered error while trying to finish logging: {e}")

    def _logging_loop(self) -> None:
        """
        Continuously logs run data until `self._continue_to_process_data` is set to False.
        If an exception occurs during logging, a `MlflowException` is raised.
        """
        try:
            while not self._continue_to_log_data.is_set():
                self._log_run_data()
        except Exception as e:
            from mlflow.exceptions import MlflowException

            raise MlflowException(f"Exception inside the run data logging thread: {e}")

    def _log_run_data(self) -> None:
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
        run_batch = None  # type: RunBatch
        try:
            run_batch = self._queue.get(timeout=1)
        except Empty:
            # Ignore empty queue exception
            return
        try:
            self._logging_func(
                run_id=run_batch.run_id,
                metrics=run_batch.metrics,
                params=run_batch.params,
                tags=run_batch.tags,
            )

            # Signal the batch processing is done.
            run_batch.completion_event.set()

        except Exception as e:
            _logger.error(f"Run Id {run_batch.run_id}: Failed to log run data: Exception: {e}")
            run_batch.exception = e
            run_batch.completion_event.set()

    def _wait_for_batch(self, batch: RunBatch) -> None:
        """
        Wait for the given batch to be processed by the logging thread.

        Args:
            batch: The batch to wait for.

        Raises:
            Exception: If an exception occurred while processing the batch.
        """
        batch.completion_event.wait()
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
        batch = RunBatch(
            run_id=run_id,
            params=params,
            tags=tags,
            metrics=metrics,
            completion_event=threading.Event(),
        )

        self._queue.put(batch)

        operation_future = self._BATCH_STATUS_CHECK_THREADPOOL.submit(self._wait_for_batch, batch)
        return RunOperations(operation_futures=[operation_future])
