import time
from contextlib import contextmanager

import mlflow
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from mlflow.utils.autologging_utils.metrics_queue import (
    add_to_metrics_queue,
    flush_metrics_queue,
)

DISABLED = False


@contextmanager
def disable_pytorch_autologging():
    global DISABLED
    old_value = DISABLED
    DISABLED = True
    try:
        yield
    finally:
        DISABLED = old_value


def patched_add_hparams(original, self, hparam_dict, metric_dict, *args, **kwargs):
    """use a synchronous call here since this is going to get called very infrequently."""

    run = mlflow.active_run()

    if not DISABLED and run is not None and hparam_dict:
        run_id = run.info.run_id
        # str() is required by mlflow :(
        params_arr = [Param(key, str(value)) for key, value in hparam_dict.items()]
        metrics_arr = [
            Metric(key, value, int(time.time() * 1000), 0) for key, value in metric_dict.items()
        ]
        MlflowClient().log_batch(run_id=run_id, metrics=metrics_arr, params=params_arr, tags=[])

    return original(self, hparam_dict, metric_dict, *args, **kwargs)


def patched_add_event(original, self, event, *args, mlflow_log_every_n_step, **kwargs):
    run = mlflow.active_run()
    if (
        not DISABLED
        and run is not None
        and event.WhichOneof("what") == "summary"
        and mlflow_log_every_n_step
    ):
        summary = event.summary
        global_step = args[0] if len(args) > 0 else kwargs.get("global_step", None)
        global_step = global_step or 0
        for v in summary.value:
            if v.HasField("simple_value"):
                if global_step % mlflow_log_every_n_step == 0:
                    add_to_metrics_queue(
                        key=v.tag,
                        value=v.simple_value,
                        step=global_step,
                        time=int((event.wall_time or time.time()) * 1000),
                        run_id=run.info.run_id,
                    )

    return original(self, event, *args, **kwargs)


def patched_add_summary(original, self, *args, **kwargs):
    flush_metrics_queue()
    return original(self, *args, **kwargs)
