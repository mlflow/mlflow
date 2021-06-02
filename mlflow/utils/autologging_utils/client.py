import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from threading import RLock

from mlflow.entities import Experiment, Run, RunInfo, RunStatus, Param, RunTag, Metric
from mlflow.tracking.client import MlflowClient


class _RunMetadataQueue:

    def __init__(self, run_id):
        self._run_id = run_id
        self._params_queue = []
        self._tags_queue = []
        self._metrics_queue = []

    def enqueue(self, params=None, tags=None, metrics=None):
        self._params_queue += (params or [])
        self._tags_queue += (tags or [])
        self._metrics_queue += (metrics or [])

    def flush(self, mlflow_client):
        mlflow_client.log_batch(
            run_id=self._run_id,
            metrics=self._metrics_queue,
            params=self._params_queue,
            tags=self._tags_queue,
        )


class AsyncLoggingOperations:

    def __init__(self, runs_to_futures_map):
        self._runs_to_futures_map = runs_to_futures_map

    def result(self):
        for run_id, future in self._runs_to_futures_map.items():
            future.result()


class AutologgingBatchingClient:
    """
    Efficienty implements a subset of MLflow Tracking's  `MlflowClient` and fluent APIs to provide
    automatic request batching, parallel execution, and parameter / tag truncation for autologging
    use cases.
    """

    def __init__(self):
        self._client = MlflowClient()
        self._queues = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=4)

    def _get_or_create_metadata_queue(self, run_id):
        if run_id not in self._queues:
            self._queues[run_id] = _RunMetadataQueue(run_id=run_id)
        return self._queues[run_id]

    def create_run(
        self,
        experiment_id: str,
        start_time: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> Run:
        """
        Enqueues a CreateRun operation with the specified attributes.
        """
        return self._client.create_run(experiment_id, start_time, tags)

    def set_terminated(
        self, run_id: str, status: Optional[str] = None, end_time: Optional[int] = None
    ) -> None:
        """
        Enqueues an UpdateRun operation with the specified `status` and `end_time` attributes
        for the specified `run_id`.
        """
        return self._client.set_terminated(run_id, status, end_time)

    def log_params(self, run_id, params: Dict[str, Any]) -> None:
        """
        Enqueues a collection of Parameters to be logged to the run specified by `run_id`. 
        """
        queue = self._get_or_create_metadata_queue(run_id)
        params_arr = [Param(key, str(value)) for key, value in params.items()]
        queue.enqueue(params=params_arr)

    def log_metrics(self, run_id, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Enqueues a collection of Metrics to be logged to the run specified by `run_id` at the
        step specified by `step`.
        """
        queue = self._get_or_create_metadata_queue(run_id)
        timestamp = int(time.time() * 1000)
        metrics_arr = [Metric(key, value, timestamp, step or 0) for key, value in metrics.items()]
        queue.enqueue(metrics=metrics_arr)

    def set_tags(self, run_id, tags: Dict[str, Any]) -> None:
        """
        Enqueues a collection of Tags to be logged to the run specified by `run_id`. 
        """
        queue = self._get_or_create_metadata_queue(run_id)
        tags_arr = [RunTag(key, str(value)) for key, value in tags.items()]
        queue.enqueue(tags=tags_arr)

    def flush(self, synchronous=True):
        """
        Flushes all queued logging operations, resulting in the creation or mutation of runs
        and run metadata.

        :param synchronous: If `True`, logging operations are performed synchronously, and a result
                            is only returned once all operations are complete. If `False`
                            logging operations are performed asynchronously, and an
                            `AsyncLoggingOperations` object is returned that represents the ongoing
                            logging operations.
        :return: `None` if `synchronous` is `True`, or an instance of `AsyncLoggingOperations` if
                 `synchronous` is `False`.
        """
        if synchronous:
            for run_id, metadata_queue in self._queues.items():
                metadata_queue.flush(self._client)
        else:
            runs_to_futures_map = {}
            for run_id, metadata_queue in self._queues.items():
                future = self._thread_pool.submit(metadata_queue.flush, self._client)
                runs_to_futures_map[run_id] = future

            return AsyncLoggingOperations(runs_to_futures_map)


__all__ = [
    "AutologgingBatchingClient",
]
