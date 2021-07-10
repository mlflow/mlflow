import time
import atexit
import mlflow
from mlflow.entities import Metric
import concurrent.futures
from threading import RLock
from mlflow.utils.autologging_utils import (
    autologging_integration,
    safe_patch,
    exception_safe_function,
    ExceptionSafeClass,
    PatchFunction,
    try_mlflow_log,
    log_fn_args_as_params,
    batch_metrics_logger,
)
from mlflow.utils.annotations import experimental
from tensorboardX import SummaryWriter

FLAVOR_NAME = "tbx_autolog"

_metric_queue = []
_MAX_METRIC_QUEUE_SIZE = 500
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_metric_queue_lock = RLock()


def _add_to_queue(key, value, step, time, run_id):
    """
    Add a metric to the metric queue. Flush the queue if it exceeds
    max size.
    """
    met = Metric(key=key, value=value, timestamp=time, step=step)
    _metric_queue.append((run_id, met))
    if len(_metric_queue) > _MAX_METRIC_QUEUE_SIZE:
        _thread_pool.submit(_flush_queue)


def _log_scalar(arg1, arg2, arg3):
    _add_to_queue(
        key=arg1,
        value=arg2,
        step=arg3,
        time=int(time.time() * 1000),
        run_id=mlflow.active_run().info.run_id
    )

def _assoc_list_to_map(lst):
    """
    Convert an association list to a dictionary.
    """
    d = {}
    for run_id, metric in lst:
        d[run_id] = d[run_id] + [metric] if run_id in d else [metric]
    return d

def _flush_queue():
    """
    Flush the metric queue and log contents in batches to MLflow.
    Queue is divided into batches according to run id.
    """
    try:
        # Multiple queue flushes may be scheduled simultaneously on different threads
        # (e.g., if the queue is at its flush threshold and several more items
        # are added before a flush occurs). For correctness and efficiency, only one such
        # flush operation should proceed; all others are redundant and should be dropped
        acquired_lock = _metric_queue_lock.acquire(blocking=False)
        if acquired_lock:
            client = mlflow.tracking.MlflowClient()
            # For thread safety and to avoid modifying a list while iterating over it, we record a
            # separate list of the items being flushed and remove each one from the metric queue,
            # rather than clearing the metric queue or reassigning it (clearing / reassigning is
            # dangerous because we don't block threads from adding to the queue while a flush is
            # in progress)
            snapshot = _metric_queue[:]
            for item in snapshot:
                _metric_queue.remove(item)

            metrics_by_run = _assoc_list_to_map(snapshot)
            for run_id, metrics in metrics_by_run.items():
                try_mlflow_log(client.log_batch, run_id, metrics=metrics, params=[], tags=[])
    finally:
        if acquired_lock:
            _metric_queue_lock.release()

@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    disable=False,
    silent=False
):
    """
    Enables automatic logging from TensorboardX to MLflow.
    """

    atexit.register(_flush_queue)

    def add_scalar(original, self, arg1, arg2, arg3):
        arg1 = "autologged-" + arg1
        _log_scalar(arg1, arg2, arg3)
        return original(self, arg1, arg2, arg3)

    non_managed = [
        (SummaryWriter, "add_scalar", add_scalar),
    ]

    for p in non_managed:
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Safe patching &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        safe_patch(FLAVOR_NAME, *p)