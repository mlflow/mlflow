"""
Defines an AsyncLoggingQueue that provides async fashion logging of metrics/tags/params using
queue based approach.
"""

import atexit
import enum
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from typing import Callable

from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag
from mlflow.environment_variables import (
    MLFLOW_ASYNC_LOGGING_BUFFERING_SECONDS,
    MLFLOW_ASYNC_LOGGING_THREADPOOL_SIZE,
)
from mlflow.utils.async_logging.run_batch import RunBatch
from mlflow.utils.async_logging.run_operations import RunOperations

_logger = logging.getLogger(__name__)


ASYNC_LOGGING_WORKER_THREAD_PREFIX = "MLflowBatchLoggingWorkerPool"
ASYNC_LOGGING_STATUS_CHECK_THREAD_PREFIX = "MLflowAsyncLoggingStatusCheck"


class QueueStatus(enum.Enum):
    """Status of the async queue"""

    # The queue is listening to new data and logging enqueued data to MLflow.
    ACTIVE = 1
    # The queue is not listening to new data, but still logging enqueued data to MLflow.
    TEAR_DOWN = 2
    # The queue is neither listening to new data or logging enqueued data to MLflow.
    IDLE = 3


_MAX_ITEMS_PER_BATCH = 1000
_MAX_PARAMS_PER_BATCH = 100
_MAX_TAGS_PER_BATCH = 100


class AsyncLoggingQueue:
    """
    This is a queue based run data processor that queues incoming batches and processes them using
    single worker thread.
    """

    def __init__(
        self, logging_func: Callable[[str, list[Metric], list[Param], list[RunTag]], None]
    ) -> None:
        """Initializes an AsyncLoggingQueue object.

        Args:
            logging_func: A callable function that takes in four arguments: a string
                representing the run_id, a list of Metric objects,
                a list of Param objects, and a list of RunTag objects.
        """
        self._queue = Queue()
        self._lock = threading.RLock()
        self._logging_func = logging_func

        self._stop_data_logging_thread_event = threading.Event()
        self._status = QueueStatus.IDLE

    def _at_exit_callback(self) -> None:
        """Callback function to be executed when the program is exiting.

        Stops the data processing thread and waits for the queue to be drained. Finally, shuts down
        the thread pools used for data logging and batch processing status check.
        """
        try:
            # Stop the data processing thread
            self._stop_data_logging_thread_event.set()
            # Waits till logging queue is drained.
            self._batch_logging_thread.join()
            self._batch_logging_worker_threadpool.shutdown(wait=True)
            self._batch_status_check_threadpool.shutdown(wait=True)
        except Exception as e:
            _logger.error(f"Encountered error while trying to finish logging: {e}")

    def end_async_logging(self) -> None:
        with self._lock:
            # Stop the data processing thread.
            self._stop_data_logging_thread_event.set()
            # Waits till logging queue is drained.
            self._batch_logging_thread.join()
            # Set the status to tear down. The worker threads will still process
            # the remaining data.
            self._status = QueueStatus.TEAR_DOWN
            # Clear the status to avoid blocking next logging.
            self._stop_data_logging_thread_event.clear()

    def shut_down_async_logging(self) -> None:
        """
        Shut down the async logging queue and wait for the queue to be drained.
        Use this method if the async logging should be terminated.
        """
        self.end_async_logging()
        self._batch_logging_worker_threadpool.shutdown(wait=True)
        self._batch_status_check_threadpool.shutdown(wait=True)
        self._status = QueueStatus.IDLE

    def flush(self) -> None:
        """
        Flush the async logging queue and restart thread to listen
        to incoming data after flushing.

        Calling this method will flush the queue to ensure all the data are logged.
        """
        self.shut_down_async_logging()
        # Reinitialize the logging thread and set the status to active.
        self.activate()

    def _logging_loop(self) -> None:
        """
        Continuously logs run data until `self._continue_to_process_data` is set to False.
        If an exception occurs during logging, a `MlflowException` is raised.
        """
        try:
            while not self._stop_data_logging_thread_event.is_set():
                self._log_run_data()
            # Drain the queue after the stop event is set.
            while not self._queue.empty():
                self._log_run_data()
        except Exception as e:
            from mlflow.exceptions import MlflowException

            raise MlflowException(f"Exception inside the run data logging thread: {e}")

    def _fetch_batch_from_queue(self) -> list[RunBatch]:
        """Fetches a batch of run data from the queue.

        Returns:
            RunBatch: A batch of run data.
        """
        batches = []
        if self._queue.empty():
            return batches
        queue_size = self._queue.qsize()  # Estimate the queue's size.
        merged_batch = self._queue.get()
        for i in range(queue_size - 1):
            if self._queue.empty():
                # `queue_size` is an estimate, so we need to check if the queue is empty.
                break
            batch = self._queue.get()

            if (
                merged_batch.run_id != batch.run_id
                or (
                    len(merged_batch.metrics + merged_batch.params + merged_batch.tags)
                    + len(batch.metrics + batch.params + batch.tags)
                )
                >= _MAX_ITEMS_PER_BATCH
                or len(merged_batch.params) + len(batch.params) >= _MAX_PARAMS_PER_BATCH
                or len(merged_batch.tags) + len(batch.tags) >= _MAX_TAGS_PER_BATCH
            ):
                # Make a new batch if the run_id is different or the batch is full.
                batches.append(merged_batch)
                merged_batch = batch
            else:
                merged_batch.add_child_batch(batch)
                merged_batch.params.extend(batch.params)
                merged_batch.tags.extend(batch.tags)
                merged_batch.metrics.extend(batch.metrics)

        batches.append(merged_batch)
        return batches

    def _log_run_data(self) -> None:
        """Process the run data in the running runs queues.

        For each run in the running runs queues, this method retrieves the next batch of run data
        from the queue and processes it by calling the `_processing_func` method with the run ID,
        metrics, parameters, and tags in the batch. If the batch is empty, it is skipped. After
        processing the batch, the processed watermark is updated and the batch event is set.
        If an exception occurs during processing, the exception is logged and the batch event is set
        with the exception. If the queue is empty, it is ignored.

        Returns: None
        """
        async_logging_buffer_seconds = MLFLOW_ASYNC_LOGGING_BUFFERING_SECONDS.get()
        try:
            if async_logging_buffer_seconds:
                self._stop_data_logging_thread_event.wait(async_logging_buffer_seconds)
                run_batches = self._fetch_batch_from_queue()
            else:
                run_batches = [self._queue.get(timeout=1)]
        except Empty:
            # Ignore empty queue exception
            return

        def logging_func(run_batch):
            try:
                self._logging_func(
                    run_id=run_batch.run_id,
                    metrics=run_batch.metrics,
                    params=run_batch.params,
                    tags=run_batch.tags,
                )
            except Exception as e:
                _logger.error(f"Run Id {run_batch.run_id}: Failed to log run data: Exception: {e}")
                run_batch.exception = e
            finally:
                run_batch.complete()

        for run_batch in run_batches:
            try:
                self._batch_logging_worker_threadpool.submit(logging_func, run_batch)
            except Exception as e:
                _logger.error(
                    f"Failed to submit batch for logging: {e}. Usually this means you are not "
                    "shutting down MLflow properly before exiting. Please make sure you are using "
                    "context manager, e.g., `with mlflow.start_run():` or call `mlflow.end_run()`"
                    "explicitly to terminate MLflow logging before exiting."
                )
                run_batch.exception = e
                run_batch.complete()

    def _wait_for_batch(self, batch: RunBatch) -> None:
        """Wait for the given batch to be processed by the logging thread.

        Args:
            batch: The batch to wait for.

        Raises:
            Exception: If an exception occurred while processing the batch.
        """
        batch.completion_event.wait()
        if batch.exception:
            raise batch.exception

    def __getstate__(self):
        """Return the state of the object for pickling.

        This method is called by the `pickle` module when the object is being pickled. It returns a
        dictionary containing the object's state, with non-picklable attributes removed.

        Returns:
            dict: A dictionary containing the object's state.
        """
        state = self.__dict__.copy()
        del state["_queue"]
        del state["_lock"]
        del state["_status"]

        if "_run_data_logging_thread" in state:
            del state["_run_data_logging_thread"]
        if "_stop_data_logging_thread_event" in state:
            del state["_stop_data_logging_thread_event"]
        if "_batch_logging_thread" in state:
            del state["_batch_logging_thread"]
        if "_batch_logging_worker_threadpool" in state:
            del state["_batch_logging_worker_threadpool"]
        if "_batch_status_check_threadpool" in state:
            del state["_batch_status_check_threadpool"]

        return state

    def __setstate__(self, state):
        """Set the state of the object from a given state dictionary.

        It pops back the removed non-picklable attributes from `self.__getstate__()`.

        Args:
            state (dict): A dictionary containing the state of the object.

        Returns:
            None
        """
        self.__dict__.update(state)
        self._queue = Queue()
        self._lock = threading.RLock()
        self._status = QueueStatus.IDLE
        self._batch_logging_thread = None
        self._batch_logging_worker_threadpool = None
        self._batch_status_check_threadpool = None
        self._stop_data_logging_thread_event = threading.Event()

    def log_batch_async(
        self, run_id: str, params: list[Param], tags: list[RunTag], metrics: list[Metric]
    ) -> RunOperations:
        """Asynchronously logs a batch of run data (parameters, tags, and metrics).

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
        from mlflow import MlflowException

        if not self.is_active():
            raise MlflowException("AsyncLoggingQueue is not activated.")
        batch = RunBatch(
            run_id=run_id,
            params=params,
            tags=tags,
            metrics=metrics,
            completion_event=threading.Event(),
        )
        self._queue.put(batch)
        operation_future = self._batch_status_check_threadpool.submit(self._wait_for_batch, batch)
        return RunOperations(operation_futures=[operation_future])

    def is_active(self) -> bool:
        return self._status == QueueStatus.ACTIVE

    def is_idle(self) -> bool:
        return self._status == QueueStatus.IDLE

    def _set_up_logging_thread(self) -> None:
        """
        Sets up the logging thread.

        This method shouldn't be called directly without shutting down the async
        logging first if an existing async logging exists, otherwise it might
        hang the program.
        """
        with self._lock:
            self._batch_logging_thread = threading.Thread(
                target=self._logging_loop,
                name="MLflowAsyncLoggingLoop",
                daemon=True,
            )
            self._batch_logging_worker_threadpool = ThreadPoolExecutor(
                max_workers=MLFLOW_ASYNC_LOGGING_THREADPOOL_SIZE.get() or 10,
                thread_name_prefix=ASYNC_LOGGING_WORKER_THREAD_PREFIX,
            )

            self._batch_status_check_threadpool = ThreadPoolExecutor(
                max_workers=MLFLOW_ASYNC_LOGGING_THREADPOOL_SIZE.get() or 10,
                thread_name_prefix=ASYNC_LOGGING_STATUS_CHECK_THREAD_PREFIX,
            )

            self._batch_logging_thread.start()

    def activate(self) -> None:
        """Activates the async logging queue

        1. Initializes queue draining thread.
        2. Initializes threads for checking the status of logged batch.
        3. Registering an atexit callback to ensure that any remaining log data
            is flushed before the program exits.

        If the queue is already activated, this method does nothing.
        """
        with self._lock:
            if self.is_active():
                return

            self._set_up_logging_thread()
            atexit.register(self._at_exit_callback)

            self._status = QueueStatus.ACTIVE
