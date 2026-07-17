import concurrent.futures
from threading import RLock

from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient

_metrics_queue_lock = RLock()
_metrics_queue = []
_thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="MlflowMetricsQueue"
)

_MAX_METRIC_QUEUE_SIZE = 500


def _assoc_list_to_map(lst):
    """
    Convert an association list to a dictionary.
    """
    d = {}
    for run_id, metric in lst:
        d[run_id] = d[run_id] + [metric] if run_id in d else [metric]
    return d


def flush_metrics_queue():
    """Flush the metric queue and log contents in batches to MLflow.

    Queue is divided into batches according to run id.
    """
    try:
        # Multiple queue flushes may be scheduled simultaneously on different threads
        # (e.g., if the queue is at its flush threshold and several more items
        # are added before a flush occurs). For correctness and efficiency, only one such
        # flush operation should proceed; all others are redundant and should be dropped
        acquired_lock = _metrics_queue_lock.acquire(blocking=False)
        if acquired_lock:
            # For thread safety and to avoid modifying a list while iterating over it, we record a
            # separate list of the items being flushed and remove each one from the metric queue,
            # rather than clearing the metric queue or reassigning it (clearing / reassigning is
            # dangerous because we don't block threads from adding to the queue while a flush is
            # in progress)
            snapshot = _metrics_queue[:]
            for item in snapshot:
                _metrics_queue.remove(item)

            # Only create MlflowClient if there are metrics to log
            if snapshot:
                client = MlflowClient()
                metrics_by_run = _assoc_list_to_map(snapshot)
                for run_id, metrics in metrics_by_run.items():
                    client.log_batch(run_id, metrics=metrics, params=[], tags=[])
    finally:
        if acquired_lock:
            _metrics_queue_lock.release()


def add_to_metrics_queue(key, value, step, time, run_id):
    """Add a metric to the metric queue.

    Flush the queue if it exceeds max size.

    Args:
        key: string, the metrics key,
        value: float, the metrics value.
        step: int, the step of current metric.
        time: int, the timestamp of current metric.
        run_id: string, the run id of the associated mlflow run.
    """
    met = Metric(key=key, value=value, timestamp=time, step=step)
    _metrics_queue.append((run_id, met))
    if len(_metrics_queue) > _MAX_METRIC_QUEUE_SIZE:
        _thread_pool.submit(flush_metrics_queue)
