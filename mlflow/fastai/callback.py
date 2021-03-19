import numpy as np

import mlflow.tracking
from mlflow.utils.autologging_utils import try_mlflow_log
from mlflow.fastai import log_model

from fastai.callback.core import Callback


# Move outside, because it cannot be pickled. Besides, ExceptionSafeClass was giving some issues
class __MLflowFastaiCallback(Callback):
    from fastai.learner import Recorder
    from fastai.callback.all import TrackerCallback

    """
    Callback for auto-logging metrics and parameters.
    Records model structural information as params when training begins
    """
    remove_on_fetch, run_before, run_after = True, TrackerCallback, Recorder

    def __init__(self, metrics_logger, log_models):
        super().__init__()
        self.metrics_logger = metrics_logger
        self.log_models = log_models

    def after_epoch(self):
        """
        Log loss and other metrics values after each epoch
        """

        # Do not record in case of predicting
        if self.learn.y is None:
            return

        metrics = self.recorder.log
        metrics = dict(zip(self.recorder.metric_names, metrics))

        keys = list(metrics.keys())
        i = 0
        while i < len(metrics):
            key = keys[i]
            try:
                float(metrics[key])
                i += 1
            except (ValueError, TypeError):
                del metrics[key]
                del keys[i]

        self.metrics_logger.record_metrics(metrics, step=metrics["epoch"])

    def before_fit(self):
        from fastai.callback.all import ParamScheduler

        # Do not record in case of predicting
        if self.learn.y is None:
            return

        try_mlflow_log(mlflow.log_param, "opt_func", self.opt_func.__name__)

        params_not_to_log = []
        for cb in self.cbs:
            if isinstance(cb, ParamScheduler):
                params_not_to_log = list(cb.scheds.keys())
                for param, f in cb.scheds.items():
                    values = []
                    for step in np.linspace(0, 1, num=100, endpoint=False):
                        values.append(f(step))
                    values = np.array(values)
                    try_mlflow_log(mlflow.log_param, param + "_min", np.min(values, 0))
                    try_mlflow_log(mlflow.log_param, param + "_max", np.max(values, 0))
                    try_mlflow_log(mlflow.log_param, param + "_init", values[0])
                    try_mlflow_log(mlflow.log_param, param + "_final", values[-1])
                break

        for param in self.opt.hypers[0]:
            if param not in params_not_to_log:
                try_mlflow_log(mlflow.log_param, param, [h[param] for h in self.opt.hypers])

        if hasattr(self.opt, "true_wd"):
            try_mlflow_log(mlflow.log_param, "true_wd", self.opt.true_wd)

        if hasattr(self.opt, "bn_wd"):
            try_mlflow_log(mlflow.log_param, "bn_wd", self.opt.bn_wd)

        if hasattr(self.opt, "train_bn"):
            try_mlflow_log(mlflow.log_param, "train_bn", self.opt.train_bn)

    def after_train(self):
        if self.log_models:
            try_mlflow_log(log_model, self.learn, artifact_path="model")
