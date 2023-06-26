import numpy as np
import os
import tempfile
from functools import partial
import matplotlib.pyplot as plt
import logging

import mlflow.tracking
from mlflow.utils.autologging_utils import ExceptionSafeClass, get_autologging_config
from mlflow.fastai import log_model

from fastai.callback.core import Callback

_logger = logging.getLogger(__name__)


# Move outside, because it cannot be pickled. Besides, ExceptionSafeClass was giving some issues
class __MlflowFastaiCallback(Callback, metaclass=ExceptionSafeClass):
    """
    Callback for auto-logging metrics and parameters.
    Records model structural information as params when training begins.
    """

    from fastai.learner import Recorder
    from fastai.callback.all import TrackerCallback

    remove_on_fetch, run_before, run_after = True, TrackerCallback, Recorder

    def __init__(self, metrics_logger, log_models, is_fine_tune=False):
        super().__init__()
        self.metrics_logger = metrics_logger
        self.log_models = log_models
        self.is_fine_tune = is_fine_tune
        self.freeze_prefix = ""

    def after_epoch(self):
        """Log loss and other metrics values after each epoch"""

        def _is_float(x):
            try:
                float(x)
                return True
            except (ValueError, TypeError):
                return False

        # Do not record in case of predicting
        if hasattr(self, "lr_finder") or hasattr(self, "gather_preds"):
            return
        # Remove non-float metrics and record the rest.
        metrics = self.recorder.log
        metrics = {k: v for k, v in zip(self.recorder.metric_names, metrics) if _is_float(v)}
        self.metrics_logger.record_metrics(metrics, step=metrics["epoch"])

    def before_fit(self):
        from fastai.callback.all import ParamScheduler

        # Do not record in case of predicting or lr_finder
        if hasattr(self, "lr_finder") or hasattr(self, "gather_preds"):
            return

        if self.is_fine_tune and len(self.opt.param_lists) == 1:
            _logger.warning(
                "Using `fine_tune` with model which cannot be frozen."
                " Current model have only one param group which makes it impossible to freeze."
                " Because of this it will record some fitting params twice (overriding exception)"
            )

        frozen = self.opt.frozen_idx != 0
        if frozen and self.is_fine_tune:
            self.freeze_prefix = "freeze_"
            mlflow.log_param("frozen_idx", self.opt.frozen_idx)
        else:
            self.freeze_prefix = ""

        # Extract function name when `opt_func` is partial function
        if isinstance(self.opt_func, partial):
            mlflow.log_param(
                self.freeze_prefix + "opt_func",
                self.opt_func.keywords["opt"].__name__,
            )
        else:
            mlflow.log_param(self.freeze_prefix + "opt_func", self.opt_func.__name__)

        params_not_to_log = []
        for cb in self.cbs:
            if isinstance(cb, ParamScheduler):
                params_not_to_log = list(cb.scheds.keys())
                for param, f in cb.scheds.items():
                    values = []
                    for step in np.linspace(0, 1, num=100, endpoint=False):
                        values.append(f(step))
                    values = np.array(values)

                    # Log params main values from scheduling
                    mlflow.log_param(self.freeze_prefix + param + "_min", np.min(values, 0))
                    mlflow.log_param(self.freeze_prefix + param + "_max", np.max(values, 0))
                    mlflow.log_param(self.freeze_prefix + param + "_init", values[0])
                    mlflow.log_param(self.freeze_prefix + param + "_final", values[-1])

                    # Plot and save image of scheduling
                    fig = plt.figure()
                    plt.plot(values)
                    plt.ylabel(param)

                    with tempfile.TemporaryDirectory() as tempdir:
                        scheds_file = os.path.join(tempdir, self.freeze_prefix + param + ".png")
                        plt.savefig(scheds_file)
                        plt.close(fig)
                        mlflow.log_artifact(local_path=scheds_file)
                break

        for param in self.opt.hypers[0]:
            if param not in params_not_to_log:
                mlflow.log_param(self.freeze_prefix + param, [h[param] for h in self.opt.hypers])

        if hasattr(self.opt, "true_wd"):
            mlflow.log_param(self.freeze_prefix + "true_wd", self.opt.true_wd)

        if hasattr(self.opt, "bn_wd"):
            mlflow.log_param(self.freeze_prefix + "bn_wd", self.opt.bn_wd)

        if hasattr(self.opt, "train_bn"):
            mlflow.log_param(self.freeze_prefix + "train_bn", self.opt.train_bn)

    def after_fit(self):
        from fastai.callback.all import SaveModelCallback

        # Do not log model in case of predicting
        if hasattr(self, "lr_finder") or hasattr(self, "gather_preds"):
            return

        # Workaround to log model from SaveModelCallback
        # Use this till able to set order between SaveModelCallback and EarlyStoppingCallback
        for cb in self.cbs:
            if isinstance(cb, SaveModelCallback):
                cb("after_fit")

        if self.log_models:
            registered_model_name = get_autologging_config(
                mlflow.fastai.FLAVOR_NAME, "registered_model_name", None
            )
            log_model(
                self.learn, artifact_path="model", registered_model_name=registered_model_name
            )
