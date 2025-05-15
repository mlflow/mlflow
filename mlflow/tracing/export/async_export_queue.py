import atexit
import logging
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from queue import Empty, Queue
from queue import Full as queue_Full
from typing import Callable, Sequence

from mlflow.environment_variables import (
    MLFLOW_ASYNC_TRACE_LOGGING_MAX_QUEUE_SIZE,
    MLFLOW_ASYNC_TRACE_LOGGING_MAX_WORKERS,
)

_logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A dataclass to represent a simple task."""

    handler: Callable
    args: Sequence
    error_msg: str = ""

    def handle(self) -> None:
        """Handle the task execution. This method must not raise any exception."""
        try:
            self.handler(*self.args)
        except Exception as e:
            _logger.warning(
                f"{self.error_msg} Error: {e}.",
                exc_info=_logger.isEnabledFor(logging.DEBUG),
            )


class AsyncTraceExportQueue:
    """A queue-based asynchronous tracing export processor."""

    def __init__(self):
        self._queue: Queue[Task] = Queue(maxsize=MLFLOW_ASYNC_TRACE_LOGGING_MAX_QUEUE_SIZE.get())
        self._lock = threading.RLock()
        self._max_workers = MLFLOW_ASYNC_TRACE_LOGGING_MAX_WORKERS.get()

        # Thread event that indicates the queue should stop processing tasks
        self._stop_event = threading.Event()
        self._is_active = False
        self._atexit_callback_registered = False

        self._active_tasks = set()

    def put(self, task: Task):
        """Put a new task to the queue for processing."""
        if not self.is_active():
            self.activate()

        # If stop event is set, wait for the queue to be drained before putting the task
        if self._stop_event.is_set():
            self._stop_event.wait()

        try:
            # Do not block if the queue is full, it will block the main application
            self._queue.put(task, block=False)
        except queue_Full:
            _logger.warning(
                "Trace export queue is full, trace will be discarded. "
                "Consider increasing the queue size or number of workers."
            )

    def _consumer_loop(self) -> None:
        while not self._stop_event.is_set():
            self._dispatch_task()

        # Drain remaining tasks when stopping
        while not self._queue.empty():
            self._dispatch_task()

    def _dispatch_task(self) -> None:
        """Dispatch a task from the queue to the worker thread pool."""
        # NB: Monitor number of active tasks being processed by the workers. If the all
        #   workers are busy, wait for one of them to finish before draining a new task
        #   from the queue. This is because ThreadPoolExecutor does not have a built-in
        #   mechanism to limit the number of pending tasks in the internal queue.
        #   This ruins the purpose of having a size bound for self._queue, because the
        #   TPE's internal queue can grow indefinitely and potentially run out of memory.
        #   Therefore, we should only dispatch a new task when there is a worker available,
        #   and pend the new tasks in the self._queue which has a size bound.
        if len(self._active_tasks) >= self._max_workers:
            _, self._active_tasks = wait(self._active_tasks, return_when=FIRST_COMPLETED)

        try:
            task = self._queue.get(timeout=1)
        except Empty:
            return

        def _handle(task):
            task.handle()
            self._queue.task_done()

        try:
            future = self._worker_threadpool.submit(_handle, task)
            self._active_tasks.add(future)
        except Exception as e:
            # In case it fails to submit the task to the worker thread pool
            # such as interpreter shutdown, handle the task in this thread
            _logger.debug(
                f"Failed to submit task to worker thread pool. Error: {e}",
                exc_info=True,
            )
            _handle(task)

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
                thread_name_prefix="MlflowTraceLoggingWorker",
            )
            self._consumer_thread = threading.Thread(
                target=self._consumer_loop,
                name="MLflowTraceLoggingConsumer",
                daemon=True,
            )
            self._consumer_thread.start()

    def _at_exit_callback(self) -> None:
        """Callback function executed when the program is exiting."""
        try:
            _logger.info(
                "Flushing the async trace logging queue before program exit. "
                "This may take a while..."
            )
            self.flush(terminate=True)
        except Exception as e:
            _logger.error(f"Error while finishing trace export requests: {e}")

    def flush(self, terminate=False) -> None:
        """
        Flush the async logging queue.

        Args:
            terminate: If True, shut down the logging threads after flushing.
        """
        if not self.is_active():
            return

        self._stop_event.set()
        self._consumer_thread.join()

        # Wait for all tasks to be processed
        self._queue.join()

        self._worker_threadpool.shutdown(wait=True)
        self._is_active = False
        # Restart threads to listen to incoming requests after flushing, if not terminating
        if not terminate:
            self._stop_event.clear()
            self.activate()
