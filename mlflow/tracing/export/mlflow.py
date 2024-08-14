import atexit
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from typing import Callable, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.trace import Trace
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_LOGGING
from mlflow.tracing.constant import TraceTagKey
from mlflow.tracing.display import get_display_handler
from mlflow.tracing.display.display_handler import IPythonTraceDisplayHandler
from mlflow.tracing.fluent import TRACE_BUFFER
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import maybe_get_request_id
from mlflow.tracking.client import MlflowClient

_logger = logging.getLogger(__name__)


class MlflowSpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to MLflow.

    MLflow backend (will) only support logging the complete trace, not incremental updates
    for spans, so this exporter is designed to aggregate the spans into traces in memory.
    Therefore, this only works within a single process application and not intended to work
    in a distributed environment. For the same reason, this exporter should only be used with
    SimpleSpanProcessor.

    If we want to support distributed tracing, we should first implement an incremental trace
    logging in MLflow backend, then we can get rid of the in-memory trace aggregation.

    :meta private:
    """

    def __init__(
        self,
        client: Optional[MlflowClient] = None,
        display_handler: Optional[IPythonTraceDisplayHandler] = None,
    ):
        self._client = client or MlflowClient()
        self._display_handler = display_handler or get_display_handler()
        self._trace_manager = InMemoryTraceManager.get_instance()
        self._async_queue = AsyncTraceExportQueue(self._client)

    def export(self, root_spans: Sequence[ReadableSpan]):
        """
        Export the spans to MLflow backend.

        Args:
            root_spans: A sequence of OpenTelemetry ReadableSpan objects to be exported.
                Only root spans for each trace are passed to this method.
        """
        for span in root_spans:
            if span._parent is not None:
                _logger.debug("Received a non-root span. Skipping export.")
                continue

            trace = self._trace_manager.pop_trace(span.context.trace_id)
            if trace is None:
                _logger.debug(f"TraceInfo for span {span} not found. Skipping export.")
                continue

            # Add the trace to the in-memory buffer
            TRACE_BUFFER[trace.info.request_id] = trace
            # Add evaluation trace to the in-memory buffer with eval_request_id key
            if eval_request_id := trace.info.tags.get(TraceTagKey.EVAL_REQUEST_ID):
                TRACE_BUFFER[eval_request_id] = trace

            if not maybe_get_request_id(is_evaluate=True):
                # Display the trace in the UI if the trace is not generated from within
                # an MLflow model evaluation context
                self._display_handler.display_traces([trace])

            self._log_trace(trace)

    def _log_trace(self, trace: Trace):
        """Log the trace to MLflow backend."""
        upload_tag_task = Task(
            handler=self._client._upload_trace_spans_as_tag,
            args=(trace.info, trace.data),
            error_msg="Failed to log trace spans as tag to MLflow backend.",
        )

        upload_trace_data_task = Task(
            handler=self._client._upload_trace_data,
            args=(trace.info, trace.data),
            error_msg="Failed to log trace to MLflow backend.",
        )

        upload_ended_trace_info_task = Task(
            handler=self._client._upload_ended_trace_info,
            args=(trace.info,),
            error_msg="Failed to log trace to MLflow backend.",
        )

        if MLFLOW_ENABLE_ASYNC_LOGGING.get():
            self._async_queue.put(upload_tag_task)
            self._async_queue.put(upload_trace_data_task)
            self._async_queue.put(upload_ended_trace_info_task)
        else:
            upload_tag_task.handle()
            upload_trace_data_task.handle()
            upload_ended_trace_info_task.handle()


class Task:
    """A dataclass to represent a task to be processed by the async trace export queue."""

    def __init__(self, handler: Callable, args: Sequence, error_msg: str = ""):
        self._handler = handler
        self._args = args
        self._error_msg = error_msg

    def handle(self):
        """Handle the task. This method must not raise any exception."""
        try:
            self._handler(*self._args)
        except Exception as e:
            _logger.warning(
                f"{self._error_msg} Error: {e}. For full traceback, set logging level to debug.",
                exc_info=_logger.isEnabledFor(logging.DEBUG),
            )


class AsyncTraceExportQueue:
    """
    This is a queue based run data processor that queue incoming data and process it using a single
    worker thread. This class is used to process traces saving in async fashion.
    """

    def __init__(self, client) -> None:
        self._queue: Queue[Task] = Queue()
        self._client = client
        self._lock = threading.RLock()

        # A thread event that indicates the logging queue stop processing task.
        self._stop_event = threading.Event()
        self._is_activated = False
        self._unprocessed_tasks = set()
        self._atexit_callback_registered = False

    def put(self, task: Task) -> None:
        """Put a new task to the queue for processing."""
        if not self.is_active():
            self.activate()

        # If stop event is set, we should wait for the queue to be drained before putting the task.
        if self._stop_event.is_set():
            self._stop_event.wait()

        self._unprocessed_tasks.add(task)
        self._queue.put(task)

    def _logging_loop(self) -> None:
        """
        Continuously process incoming tasks in the queue until the stop event is set.
        When the stop event is set, the loop will drain the remaining tasks in the queue.
        """
        while not self._stop_event.is_set():
            self._handle_task()
        while not self._queue.empty() or self._unprocessed_tasks:
            self._handle_task()

    def _handle_task(self) -> None:
        """Process the given task in the running runs queues."""
        try:
            task = self._queue.get(timeout=1)
        except Empty:
            return

        def _handle(task):
            task.handle()
            self._unprocessed_tasks.discard(task)

        self._trace_logging_worker_threadpool.submit(_handle, task)

    def activate(self) -> None:
        """Activates the async logging queue"""
        with self._lock:
            if self._is_activated:
                return

            self._set_up_logging_thread()
            # Registering an atexit callback to ensure that any remaining log data
            # is flushed before the program exits.
            if not self._atexit_callback_registered:
                atexit.register(self._at_exit_callback)
                self._atexit_callback_registered = True
            self._is_activated = True

    def is_active(self) -> bool:
        return self._is_activated

    def _set_up_logging_thread(self) -> None:
        """Sets up the logging thread."""
        with self._lock:
            self._trace_logging_thread = threading.Thread(
                target=self._logging_loop,
                name="MLflowTraceLoggingLoop",
                daemon=True,
            )
            self._trace_logging_worker_threadpool = ThreadPoolExecutor(
                max_workers=5,
                thread_name_prefix="MLflowTraceLoggingWorkerPool",
            )
            self._trace_logging_thread.start()

    def _at_exit_callback(self) -> None:
        """Callback function to be executed when the program is exiting."""
        try:
            self.flush(terminate=True)
        except Exception as e:
            _logger.error(f"Encountered error while trying to finish logging: {e}")

    def flush(self, terminate=False) -> None:
        """
        Flush the async logging queue.

        Args:
            terminate: If True, shut down the logging threads after flushing.
        """
        if not self.is_active():
            return

        self._stop_event.set()
        self._trace_logging_thread.join()
        self._trace_logging_worker_threadpool.shutdown(wait=True)
        self._is_activated = False

        # Restart the thread to listen to incoming data after flushing.
        if not terminate:
            self._stop_event.clear()
            self.activate()
