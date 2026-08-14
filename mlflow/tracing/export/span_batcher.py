import atexit
import logging
import threading
from collections import defaultdict
from queue import Queue
from typing import Callable

from mlflow.entities.span import Span
from mlflow.environment_variables import (
    MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS,
    MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE,
)
from mlflow.tracing.export.async_export_queue import AsyncTraceExportQueue, Task

_logger = logging.getLogger(__name__)


class SpanBatcher:
    """
    Queue based batching processor for span export to Databricks Unity Catalog table.

    Exposes two configuration knobs
    - Max span batch size: The maximum number of spans to export in a single batch.
    - Max interval: The maximum interval in milliseconds between two batches.
    When one of two conditions is met, the batch is exported.
    """

    def __init__(
        self, async_task_queue: AsyncTraceExportQueue, log_spans_func: Callable[[list[Span]], None]
    ):
        self._max_span_batch_size = MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE.get()
        self._max_interval_ms = MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS.get()

        self._span_queue = Queue()
        self._async_task_handler = async_task_queue
        self._log_spans_func = log_spans_func
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

        # Batch size = 1 means no batching, so we don't need to setup the worker thread.
        if self._max_span_batch_size >= 1:
            self._worker = threading.Thread(
                name="MLflowSpanBatcherWorker",
                daemon=True,
                target=self._worker_loop,
            )
            self._worker_awaken = threading.Event()
            self._worker.start()
            atexit.register(self.shutdown)

        _logger.debug(
            "Async trace logging is configured with batch size "
            f"{self._max_span_batch_size} and max interval {self._max_interval_ms}ms"
        )

    def add_span(self, location: str, span: Span):
        if self._max_span_batch_size <= 1:
            self._export(location, [span])
            return

        if self._stop_event.is_set():
            return

        self._span_queue.put((location, span))
        if self._span_queue.qsize() >= self._max_span_batch_size:
            # Trigger the immediate export when the batch is full
            self._worker_awaken.set()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            # sleep_interrupted is True when the export is triggered by the batch size limit.
            # If this is False, the interval has expired so we should export the current batch
            # even if the batch size is not reached.
            sleep_interrupted = self._worker_awaken.wait(self._max_interval_ms / 1000)
            if self._stop_event.is_set():
                break
            self._consume_batch(flush_all=not sleep_interrupted)
            self._worker_awaken.clear()

        self._consume_batch(flush_all=True)

    def _consume_batch(self, flush_all: bool = False):
        with self._lock:
            while (
                self._span_queue.qsize() >= self._max_span_batch_size
                # Export all remaining spans in the queue if necessary
                or (flush_all and not self._span_queue.empty())
            ):
                # Spans in the queue can have multiple locations. Since the backend API only support
                # logging spans to a single location, we need to group spans by location and export
                # them separately.
                location_to_spans = defaultdict(list)
                for location, span in [
                    self._span_queue.get()
                    for _ in range(min(self._max_span_batch_size, self._span_queue.qsize()))
                ]:
                    location_to_spans[location].append(span)

                for location, spans in location_to_spans.items():
                    self._export(location, spans)

    def _export(self, location: str, spans: list[Span]):
        _logger.debug(f"Exporting a span batch with {len(spans)} spans to {location}")

        self._async_task_handler.put(
            Task(
                handler=self._log_spans_func,
                args=(location, spans),
                error_msg="Failed to export batch of spans.",
            )
        )

    def shutdown(self):
        if self._stop_event.is_set():
            return

        try:
            self._stop_event.set()
            self._worker_awaken.set()
            self._worker.join()
        except Exception as e:
            _logger.debug(f"Error while shutting down span batcher: {e}")
