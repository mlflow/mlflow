"""
Defines an AsyncArtifactsLoggingQueue that provides async fashion artifact writes using
queue based approach.
"""

import atexit
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from typing import TYPE_CHECKING, Union

from mlflow.utils.async_logging.run_artifact import RunArtifact
from mlflow.utils.async_logging.run_operations import RunOperations

if TYPE_CHECKING:
    import PIL

_logger = logging.getLogger(__name__)


class AsyncArtifactsLoggingQueue:
    """
    This is a queue based run data processor that queue incoming data and process it using a single
    worker thread. This class is used to process artifacts saving in async fashion.

    Args:
        logging_func: A callable function that takes in two arguments: a string
            representing the run_id and a list of artifact paths to log.
    """

    def __init__(
        self, artifact_logging_func: callable([str, str, Union["PIL.Image.Image"]])
    ) -> None:
        self._queue = Queue()
        self._lock = threading.RLock()
        self._artifact_logging_func = artifact_logging_func

        self._stop_data_logging_thread_event = threading.Event()
        self._is_activated = False

    def _at_exit_callback(self) -> None:
        """Callback function to be executed when the program is exiting.

        Stops the data processing thread and waits for the queue to be drained. Finally, shuts down
        the thread pools used for data logging and artifact processing status check.
        """
        try:
            # Stop the data processing thread
            self._stop_data_logging_thread_event.set()
            # Waits till logging queue is drained.
            self._artifact_logging_thread.join()
            self._artifact_logging_worker_threadpool.shutdown(wait=True)
            self._artifact_status_check_threadpool.shutdown(wait=True)
        except Exception as e:
            _logger.error(f"Encountered error while trying to finish logging: {e}")

    def flush(self) -> None:
        """Flush the async logging queue.

        Calling this method will flush the queue to ensure all the data are logged.
        """
        # Stop the data processing thread.
        self._stop_data_logging_thread_event.set()
        # Waits till logging queue is drained.
        self._artifact_logging_thread.join()
        self._artifact_logging_worker_threadpool.shutdown(wait=True)
        self._artifact_status_check_threadpool.shutdown(wait=True)

        # Restart the thread to listen to incoming data after flushing.
        self._stop_data_logging_thread_event.clear()
        self._set_up_logging_thread()

    def _logging_loop(self) -> None:
        """
        Continuously logs run data until `self._continue_to_process_data` is set to False.
        If an exception occurs during logging, a `MlflowException` is raised.
        """
        try:
            while not self._stop_data_logging_thread_event.is_set():
                self._log_artifact()
            # Drain the queue after the stop event is set.
            while not self._queue.empty():
                self._log_artifact()
        except Exception as e:
            from mlflow.exceptions import MlflowException

            raise MlflowException(f"Exception inside the run data logging thread: {e}")

    def _log_artifact(self) -> None:
        """Process the run's artifacts in the running runs queues.

        For each run in the running runs queues, this method retrieves the next artifact of run
        from the queue and processes it by calling the `_artifact_logging_func` method with the run
        ID and artifact. If the artifact is empty, it is skipped. After processing the artifact,
        the processed watermark is updated and the artifact event is set.
        If an exception occurs during processing, the exception is logged and the artifact event
        is set with the exception. If the queue is empty, it is ignored.
        """
        run_artifacts = None  # type: RunArtifact
        try:
            run_artifacts = self._queue.get(timeout=1)
        except Empty:
            # Ignore empty queue exception
            return

        def logging_func(run_artifacts):
            try:
                self._artifact_logging_func(
                    filename=run_artifacts.filename,
                    artifact_path=run_artifacts.artifact_path,
                    artifact=run_artifacts.artifact,
                )

                # Signal the artifact processing is done.
                run_artifacts.completion_event.set()

            except Exception as e:
                _logger.error(f"Failed to log artifact {run_artifacts.filename}. Exception: {e}")
                run_artifacts.exception = e
                run_artifacts.completion_event.set()

        self._artifact_logging_worker_threadpool.submit(logging_func, run_artifacts)

    def _wait_for_artifact(self, artifacts: RunArtifact) -> None:
        """Wait for given artifacts to be processed by the logging thread.

        Args:
            artifacts: The artifacts to wait for.

        Raises:
            Exception: If an exception occurred while processing the artifact.
        """
        artifacts.completion_event.wait()
        if artifacts.exception:
            raise artifacts.exception

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
        del state["_is_activated"]

        if "_stop_data_logging_thread_event" in state:
            del state["_stop_data_logging_thread_event"]
        if "_artifact_logging_thread" in state:
            del state["_artifact_logging_thread"]
        if "_artifact_logging_worker_threadpool" in state:
            del state["_artifact_logging_worker_threadpool"]
        if "_artifact_status_check_threadpool" in state:
            del state["_artifact_status_check_threadpool"]

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
        self._is_activated = False
        self._artifact_logging_thread = None
        self._artifact_logging_worker_threadpool = None
        self._artifact_status_check_threadpool = None
        self._stop_data_logging_thread_event = threading.Event()

    def log_artifacts_async(self, filename, artifact_path, artifact) -> RunOperations:
        """Asynchronously logs runs artifacts.

        Args:
            filename: Filename of the artifact to be logged.
            artifact_path: Directory within the run's artifact directory in which to log the
                artifact.
            artifact: The artifact to be logged.

        Returns:
            mlflow.utils.async_utils.RunOperations: An object that encapsulates the
                asynchronous operation of logging the artifact of run data.
                The object contains a list of `concurrent.futures.Future` objects that can be used
                to check the status of the operation and retrieve any exceptions
                that occurred during the operation.
        """
        from mlflow import MlflowException

        if not self._is_activated:
            raise MlflowException("AsyncArtifactsLoggingQueue is not activated.")
        artifacts = RunArtifact(
            filename=filename,
            artifact_path=artifact_path,
            artifact=artifact,
            completion_event=threading.Event(),
        )
        self._queue.put(artifacts)
        operation_future = self._artifact_status_check_threadpool.submit(
            self._wait_for_artifact, artifacts
        )
        return RunOperations(operation_futures=[operation_future])

    def is_active(self) -> bool:
        return self._is_activated

    def _set_up_logging_thread(self) -> None:
        """Sets up the logging thread.

        If the logging thread is already set up, this method does nothing.
        """
        with self._lock:
            self._artifact_logging_thread = threading.Thread(
                target=self._logging_loop,
                name="MLflowAsyncArtifactsLoggingLoop",
                daemon=True,
            )
            self._artifact_logging_worker_threadpool = ThreadPoolExecutor(
                max_workers=5,
                thread_name_prefix="MLflowArtifactsLoggingWorkerPool",
            )

            self._artifact_status_check_threadpool = ThreadPoolExecutor(
                max_workers=5,
                thread_name_prefix="MLflowAsyncArtifactsLoggingStatusCheck",
            )
            self._artifact_logging_thread.start()

    def activate(self) -> None:
        """Activates the async logging queue

        1. Initializes queue draining thread.
        2. Initializes threads for checking the status of logged artifact.
        3. Registering an atexit callback to ensure that any remaining log data
            is flushed before the program exits.

        If the queue is already activated, this method does nothing.
        """
        with self._lock:
            if self._is_activated:
                return

            self._set_up_logging_thread()
            atexit.register(self._at_exit_callback)

            self._is_activated = True
