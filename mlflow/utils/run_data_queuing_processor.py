"""
Defines an MlflowAutologgingQueueingClient developer API that provides batching, queueing, and
asynchronous execution capabilities for a subset of MLflow Tracking logging operations used most
frequently by autologging operations.

TODO(dbczumar): Migrate request batching, queueing, and async execution support from
MlflowAutologgingQueueingClient to MlflowClient in order to provide broader benefits to end users.
Remove this developer API.
"""

import atexit
import logging
import threading
import time
from collections import deque, namedtuple
from concurrent.futures import ThreadPoolExecutor

from mlflow.entities import Metric, Param, RunTag
from mlflow.utils.validation import (
    MAX_METRICS_PER_BATCH,
    MAX_PARAM_VAL_LENGTH,
    MAX_TAG_VAL_LENGTH,
)

_logger = logging.getLogger(__name__)

_PendingCreateRun = namedtuple(
    "_PendingCreateRun", ["experiment_id", "start_time", "tags", "run_name"]
)
_PendingSetTerminated = namedtuple("_PendingSetTerminated", ["status", "end_time"])

# Keeping max_workers=1 so that there are no two threads
_RUN_DATA_LOGGING_THREADPOOL = ThreadPoolExecutor(max_workers=1)


class Item:
    def __init__(self, id, value) -> None:
        self.id = id
        self.value = value
        self.item_type = type(value).__name__


class RunItems:
    def __init__(self, run_id) -> None:
        self.run_id = run_id
        self.params = deque()
        self.tags = deque()
        self.metrics = deque()
        self.enqueue_id = 0
        self.lock = threading.Lock()

    def _get_next_id(self):
        self.lock.acquire()
        next_id = self.enqueue_id + 1
        self.enqueue_id = next_id
        self.lock.release()
        return next_id

    def append(self, value):
        if isinstance(value, Metric):
            self.metrics.append(Item(id=self._get_next_id(), value=value))
        elif isinstance(value, RunTag):
            self.tags.append(Item(id=self._get_next_id(), value=value))
        elif isinstance(value, Param):
            self.params.append(Item(id=self._get_next_id(), value=value))
        else:
            raise Exception("Invalid data type specified")

    def get_next_batch(
        self,
        max_tags=MAX_TAG_VAL_LENGTH,
        max_params=MAX_PARAM_VAL_LENGTH,
        max_metrics=MAX_METRICS_PER_BATCH,
    ):
        param_count = 0
        params_batch = []
        while len(self.params) > 0 and param_count + 1 <= max_params:
            params_batch.append(self.params.popleft())

        tag_count = 0
        tags_batch = []
        while len(self.tags) > 0 and tag_count + 1 <= max_tags:
            tags_batch.append(self.tags.popleft())

        metric_count = 0
        max_metrics = MAX_METRICS_PER_BATCH - param_count - tag_count
        metrics_batch = []
        while len(self.metrics) > 0 and metric_count + 1 <= max_metrics:
            metrics_batch.append(self.metrics.popleft())

        return (params_batch, tags_batch, metrics_batch)


class RunDataQueuingProcessor:
    """
    Efficiently implements a subset of MLflow Tracking's  `MlflowClient` and fluent APIs to provide
    automatic batching and async execution of run operations by way of queueing, as well as
    parameter / tag truncation for autologging use cases. Run operations defined by this client,
    such as `create_run` and `log_metrics`, enqueue data for future persistence to MLflow
    Tracking. Data is not persisted until the queue is flushed via the `flush()` method, which
    supports synchronous and asynchronous execution.

    MlflowAutologgingQueueingClient is not threadsafe; none of its APIs should be called
    concurrently.
    """

    def __init__(self, processing_func):
        self._pending_ops_by_run_id = {}
        self._pending_items = {}
        self._processing_func = processing_func
        self._lock = threading.Lock()
        self.run_log_thread_alive = True
        self.run_log_thread = _RUN_DATA_LOGGING_THREADPOOL.submit(self._log_run_data)
        atexit.register(self._process_at_exit)
        self.current_watermark = -1

    def _add_run_deque(self, run_id):
        self._lock.acquire()
        if not self._pending_items.get(run_id, None):
            self._pending_items[run_id] = RunItems(run_id=run_id)
        self._lock.release()

    def _process_at_exit(self):
        _logger.info("Inside RunDataQueuingProcessor._process_at_exit")
        self._lock.acquire()
        self.run_log_thread = False
        self._lock.release()
        self._process_run_data()

    def _log_run_data(self):
        try:
            while self.run_log_thread_alive:
                self._process_run_data()
                time.sleep(5)
        except Exception as e:
            _logger.error(e)
            raise

    def _process_run_data(self):
        for run_id, pending_items in self._pending_items.items():
            (params, tags, metrics) = pending_items.get_next_batch()
            params_arr = [p.value for p in params]
            tags_arr = [p.value for p in tags]
            metrics_arr = [p.value for p in metrics]
            max_params_id = 0 if len(params) == 0 else params[-1].id
            max_tags_id = 0 if len(tags) == 0 else tags[-1].id
            max_metrics_id = 0 if len(metrics) == 0 else metrics[-1].id

            while len(params) > 0 or len(tags) > 0 or len(metrics) > 0:
                try:
                    self._processing_func(
                        run_id=run_id, metrics=metrics_arr, params=params_arr, tags=tags_arr
                    )
                    self.current_watermark = max(max_metrics_id, max(max_params_id, max_tags_id))
                    _logger.info(f"run_id: {run_id}, current watermark: {self.current_watermark}")
                    (params, tags, metrics) = pending_items.get_next_batch()
                except Exception as e:
                    _logger.error(e)
                    raise

    def log_batch_async(self, run_id, params, tags, metrics):
        self._log_params(run_id=run_id, params=params)
        self._set_tags(run_id=run_id, tags=tags)
        self._log_metrics(run_id=run_id, metrics=metrics)

    def _log_params(self, run_id: str, params: [Param]) -> None:
        """
        Enqueues a collection of Parameters to be logged to the run specified by `run_id`.
        """
        if not params or len(params) == 0:
            return
        if not self._pending_items.get(run_id, None):
            self._add_run_deque(run_id=run_id)

        run_items = self._pending_items.get(run_id)
        assert run_items, "No deque found for specified run_id"

        for param in params:
            run_items.append(param)

    def _log_metrics(self, run_id: str, metrics: [Metric]) -> None:
        """
        Enqueues a collection of Metrics to be logged to the run specified by `run_id` at the
        step specified by `step`.
        """
        if not metrics or len(metrics) == 0:
            return
        if not self._pending_items.get(run_id, None):
            self._add_run_deque(run_id=run_id)

        run_items = self._pending_items.get(run_id)
        assert run_items, "No deque found for specified run_id"

        for metric in metrics:
            run_items.append(metric)

    def _set_tags(self, run_id: str, tags: [RunTag]) -> None:
        """
        Enqueues a collection of Tags to be logged to the run specified by `run_id`.
        """
        if not tags or len(tags) == 0:
            return
        if not self._pending_items.get(run_id, None):
            self._add_run_deque(run_id=run_id)

        run_items = self._pending_items.get(run_id)
        assert run_items, "No deque found for specified run_id"

        for tag in tags:
            run_items.append(tag)
