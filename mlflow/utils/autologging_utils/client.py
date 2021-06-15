import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from threading import RLock

from mlflow.entities import Experiment, Run, RunInfo, RunStatus, Param, RunTag, Metric
from mlflow.tracking.client import MlflowClient
from mlflow.utils import chunk_list, _truncate_dict
from mlflow.utils.validation import (
    MAX_ENTITIES_PER_BATCH,
    MAX_ENTITY_KEY_LENGTH,
    MAX_TAG_VAL_LENGTH,
    MAX_PARAM_VAL_LENGTH,
    MAX_PARAMS_TAGS_PER_BATCH,
    MAX_METRICS_PER_BATCH,
)


_PendingCreateRun = namedtuple("_PendingCreateRun", ["experiment_id", "start_time", "tags"])
_PendingSetTerminated = namedtuple("_PendingSetTerminated", ["status", "end_time"])


class PendingRunId:
    pass


class _PendingRunOperations:

    def __init__(self):
        self.create_run = None
        self.set_terminated = None
        self.params_queue = []
        self.tags_queue = []
        self.metrics_queue = []

    def add(self, params=None, tags=None, metrics=None, create_run=None, set_terminated=None):
        if create_run:
            self.create_run = create_run
        if set_terminated:
            self.set_terminated = set_terminated

        self.params_queue += (params or [])
        self.tags_queue += (tags or [])
        self.metrics_queue += (metrics or [])


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
        self._pending_ops_by_run_id = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=4)

    def _get_pending_operations(self, run_id):
        if run_id not in self._pending_ops_by_run_id:
            self._pending_ops_by_run_id[run_id] = _PendingRunOperations()
        return self._pending_ops_by_run_id[run_id]

    def create_run(
        self,
        experiment_id: str,
        start_time: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> Run:
        """
        Enqueues a CreateRun operation with the specified attributes.
        """
        tags = tags or {}
        tags = _truncate_dict(tags, max_key_length=MAX_ENTITY_KEY_LENGTH, max_value_length=MAX_TAG_VAL_LENGTH)
        run_id = PendingRunId()
        self._get_pending_operations(run_id).add(
            create_run=_PendingCreateRun(
                experiment_id=experiment_id,
                start_time=start_time,
                tags=[RunTag(key, str(value)) for key, value in tags.items()],
            )
        )
        return run_id

    def set_terminated(
        self, run_id: Union[str, PendingRunId], status: Optional[str] = None, end_time: Optional[int] = None
    ) -> None:
        """
        Enqueues an UpdateRun operation with the specified `status` and `end_time` attributes
        for the specified `run_id`.
        """
        self._get_pending_operations(run_id).add(
            set_terminated=_PendingSetTerminated(
                status=status,
                end_time=end_time,
            )
        )

    def log_params(self, run_id: Union[str, PendingRunId], params: Dict[str, Any]) -> None:
        """
        Enqueues a collection of Parameters to be logged to the run specified by `run_id`.
        """
        params = _truncate_dict(params, max_key_length=MAX_ENTITY_KEY_LENGTH, max_value_length=MAX_PARAM_VAL_LENGTH) 
        params_arr = [Param(key, str(value)) for key, value in params.items()]
        self._get_pending_operations(run_id).add(params=params_arr)


    def log_metrics(self, run_id: Union[str, PendingRunId], metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Enqueues a collection of Metrics to be logged to the run specified by `run_id` at the
        step specified by `step`.
        """
        metrics = _truncate_dict(metrics, max_key_length=MAX_ENTITY_KEY_LENGTH) 
        timestamp = int(time.time() * 1000)
        metrics_arr = [Metric(key, value, timestamp, step or 0) for key, value in metrics.items()]
        self._get_pending_operations(run_id).add(metrics=metrics_arr)

    def set_tags(self, run_id: Union[str, PendingRunId], tags: Dict[str, Any]) -> None:
        """
        Enqueues a collection of Tags to be logged to the run specified by `run_id`.
        """
        tags = _truncate_dict(tags, max_key_length=MAX_ENTITY_KEY_LENGTH, max_value_length=MAX_TAG_VAL_LENGTH) 
        tags_arr = [RunTag(key, str(value)) for key, value in tags.items()]
        self._get_pending_operations(run_id).add(tags=tags_arr)

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
            for run_id, pending_operations in self._pending_ops_by_run_id.items():
                self._flush_pending_operations(
                    run_id=run_id,
                    pending_operations=pending_operations,
                    synchronous=True
                )
        else:
            runs_to_futures_map = {}
            for run_id, pending_operations in self._pending_ops_by_run_id.items():
                future = self._thread_pool.submit(
                    self._flush_pending_operations,
                    run_id=run_id,
                    pending_operations=pending_operations,
                    synchronous=False,
                )
                runs_to_futures_map[run_id] = future

            return AsyncLoggingOperations(runs_to_futures_map)

    def _flush_pending_operations(self, run_id, pending_operations, synchronous):
        if pending_operations.create_run:
            create_run_tags = pending_operations.create_run.tags
            num_additional_tags_to_include_during_creation = MAX_ENTITIES_PER_BATCH - len(create_run_tags)
            if num_additional_tags_to_include_during_creation > 0:
                create_run_tags.extend(pending_operations.tags_queue[:num_additional_tags_to_include_during_creation])
                pending_operations.tags_queue = pending_operations.tags_queue[num_additional_tags_to_include_during_creation:]

            run_id = self._client.create_run(
                experiment_id=pending_operations.create_run.experiment_id,
                start_time=pending_operations.create_run.start_time,
                tags=create_run_tags,
            )
        else:
            assert not isinstance(run_id, PendingRunId)

        param_batches_to_log = chunk_list(
            pending_operations.params_queue,
            chunk_size=MAX_PARAMS_TAGS_PER_BATCH,
        )
        tag_batches_to_log = chunk_list(
            pending_operations.tags_queue,
            chunk_size=MAX_PARAMS_TAGS_PER_BATCH,
        )

        for params_batch, tags_batch in zip_longest(
            param_batches_to_log, tag_batches_to_log, fillvalue=[]
        ):
            metrics_batch_size = min(
                MAX_ENTITIES_PER_BATCH - len(params_batch) - len(tags_batch),
                MAX_METRICS_PER_BATCH,
            )
            metrics_batch = pending_operations.metrics_queue[:metrics_batch_size]
            pending_operations.metrics_queue = pending_operations.metrics_queue[metrics_batch_size:]

            self._client.log_batch(
                run_id=run_id,
                metrics=metrics_batch,
                params=params_batch,
                tags=tags_batch,
            )

        for metrics_batch in chunk_list(pending_operations.metrics_queue, chunk_size=MAX_METRICS_PER_BATCH):
            self._client.log_batch(
                run_id=run_id,
                metrics=metrics_batch,
            )

        if pending_operations.set_terminated:
            self._client.set_terminated(
                run_id=run_id,
                status=pending_operations.set_terminated.status,
                end_time=pending_operations.set_terminated.end_time,
            )


__all__ = [
    "AutologgingBatchingClient",
]
