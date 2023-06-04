from packaging.version import Version
import logging
import mlflow.pytorch
import os
import tempfile
import warnings

from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.pytorch import _pytorch_autolog
from mlflow.utils.autologging_utils import (
    ExceptionSafeAbstractClass,
    BatchMetricsLogger,
    MlflowAutologgingQueueingClient,
    get_autologging_config,
)

logging.basicConfig(level=logging.ERROR)
MIN_REQ_VERSION = Version(_ML_PACKAGE_VERSIONS["pytorch-lightning"]["autologging"]["minimum"])
MAX_REQ_VERSION = Version(_ML_PACKAGE_VERSIONS["pytorch-lightning"]["autologging"]["maximum"])

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


# The following are the downsides of using PyTorch Lightning's built-in MlflowLogger.
# 1. MlflowLogger doesn't provide a mechanism to store an entire model into mlflow.
#    Only model checkpoint is saved.
# 2. For storing the model into mlflow `mlflow.pytorch` library is used
# and the library expects `mlflow` object to be instantiated.
# In case of MlflowLogger, Run management is completely controlled by the class and
# hence mlflow object needs to be reinstantiated by setting
# tracking uri, experiment_id and run_id which may lead to a race condition.
# TODO: Replace __MlflowPLCallback with Pytorch Lightning's built-in MlflowLogger
# once the above mentioned issues have been addressed


_pl_version = Version(pl.__version__)
if _pl_version < Version("1.5.0"):
    # pylint: disable-next=ungrouped-imports
    from pytorch_lightning.core.memory import ModelSummary
else:
    # pylint: disable-next=ungrouped-imports
    from pytorch_lightning.utilities.model_summary import ModelSummary


def _get_optimizer_name(optimizer):
    """
    In pytorch-lightining 1.1.0, `LightningOptimizer` was introduced:
    https://github.com/PyTorchLightning/pytorch-lightning/pull/4658

    If a user sets `enable_pl_optimizer` to True when instantiating a `Trainer` object,
    each optimizer will be wrapped by `LightningOptimizer`:
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.html
    #pytorch_lightning.trainer.trainer.Trainer.params.enable_pl_optimizer
    """
    if Version(pl.__version__) < Version("1.1.0"):
        return optimizer.__class__.__name__
    else:
        # pylint: disable-next=ungrouped-imports
        from pytorch_lightning.core.optimizer import LightningOptimizer

        return (
            optimizer._optimizer.__class__.__name__
            if isinstance(optimizer, LightningOptimizer)
            else optimizer.__class__.__name__
        )


class __MLflowPLCallback(pl.Callback, metaclass=ExceptionSafeAbstractClass):
    """
    Callback for auto-logging metrics and parameters.
    """

    def __init__(
        self, client, metrics_logger, run_id, log_models, log_every_n_epoch, log_every_n_step
    ):
        if log_every_n_step and _pl_version < Version("1.1.0"):
            raise MlflowException(
                "log_every_n_step is only supported for PyTorch-Lightning >= 1.1.0"
            )
        self.early_stopping = False
        self.client = client
        self.metrics_logger = metrics_logger
        self.run_id = run_id
        self.log_models = log_models
        self.log_every_n_epoch = log_every_n_epoch
        self.log_every_n_step = log_every_n_step
        # Sets for tracking which metrics are logged on steps and which are logged on epochs
        self._step_metrics = set()
        self._epoch_metrics = set()

    def _log_metrics(self, trainer, step, metric_items):
        # pytorch-lightning runs a few steps of validation in the beginning of training
        # as a sanity check to catch bugs without having to wait for the training routine
        # to complete. During this check, we should skip logging metrics.
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#num-sanity-val-steps
        sanity_checking = (
            # `running_sanity_check` has been renamed to `sanity_checking`:
            # https://github.com/PyTorchLightning/pytorch-lightning/pull/9209
            trainer.sanity_checking
            if Version(pl.__version__) > Version("1.4.5")
            else trainer.running_sanity_check
        )
        if sanity_checking:
            return

        # Cast metric value as  float before passing into logger.
        metrics = {x[0]: float(x[1]) for x in metric_items}
        self.metrics_logger.record_metrics(metrics, step)

    def _log_epoch_metrics(self, trainer, pl_module):
        # `trainer.callback_metrics` contains both training and validation metrics
        # and includes metrics logged on steps and epochs.
        # If we have logged any metrics on a step basis in mlflow, we exclude these from the
        # epoch level metrics to prevent mixing epoch and step based values.
        metric_items = [
            (name, val)
            for (name, val) in trainer.callback_metrics.items()
            if name not in self._step_metrics
        ]
        # Record which metrics are logged on epochs, so we don't try to log these on steps
        self._epoch_metrics.update(name for (name, _) in metric_items)
        if (pl_module.current_epoch + 1) % self.log_every_n_epoch == 0:
            self._log_metrics(trainer, pl_module.current_epoch, metric_items)

    _pl_version = Version(pl.__version__)

    # In pytorch-lightning >= 1.4.0, validation is run inside the training epoch and
    # `trainer.callback_metrics` contains both training and validation metrics of the
    # current training epoch when `on_train_epoch_end` is called:
    # https://github.com/PyTorchLightning/pytorch-lightning/pull/7357
    if _pl_version >= Version("1.4.0dev"):

        @rank_zero_only
        def on_train_epoch_end(
            self, trainer, pl_module, *args
        ):  # pylint: disable=signature-differs,arguments-differ,unused-argument
            self._log_epoch_metrics(trainer, pl_module)

    # In pytorch-lightning >= 1.2.0, logging metrics in `on_epoch_end` results in duplicate
    # metrics records because `on_epoch_end` is called after both train and validation
    # epochs (related PR: https://github.com/PyTorchLightning/pytorch-lightning/pull/5986)
    # As a workaround, use `on_train_epoch_end` and `on_validation_epoch_end` instead
    # in pytorch-lightning >= 1.2.0.
    elif _pl_version >= Version("1.2.0"):
        # NB: Override `on_train_epoch_end` with an additional `*args` parameter for
        # compatibility with versions of pytorch-lightning <= 1.2.0, which required an
        # `outputs` argument that was not used and is no longer defined in
        # pytorch-lightning >= 1.3.0

        @rank_zero_only
        def on_train_epoch_end(
            self, trainer, pl_module, *args
        ):  # pylint: disable=signature-differs,arguments-differ,unused-argument
            """
            Log loss and other metrics values after each train epoch

            :param trainer: pytorch lightning trainer instance
            :param pl_module: pytorch lightning base module
            """
            # If validation loop is enabled (meaning `validation_step` is overridden),
            # log metrics in `on_validaion_epoch_end` to avoid logging the same metrics
            # records twice
            if not trainer.enable_validation:
                self._log_epoch_metrics(trainer, pl_module)

        @rank_zero_only
        def on_validation_epoch_end(self, trainer, pl_module):
            """
            Log loss and other metrics values after each validation epoch

            :param trainer: pytorch lightning trainer instance
            :param pl_module: pytorch lightning base module
            """
            self._log_epoch_metrics(trainer, pl_module)

    else:

        @rank_zero_only
        def on_epoch_end(self, trainer, pl_module):
            """
            Log loss and other metrics values after each epoch

            :param trainer: pytorch lightning trainer instance
            :param pl_module: pytorch lightning base module
            """
            self._log_epoch_metrics(trainer, pl_module)

    @rank_zero_only
    def on_train_batch_end(
        self, trainer, pl_module, *args
    ):  # pylint: disable=signature-differs,arguments-differ,unused-argument
        """
        Log metric values after each step

        :param trainer: pytorch lightning trainer instance
        :param pl_module: pytorch lightning base module
        """
        if not self.log_every_n_step:
            return
        # When logging at the end of a batch step, we only want to log metrics that are logged
        # on steps. For forked metrics (metrics logged on both steps and epochs), we exclude the
        # metric with the non-forked name (eg. "loss" when we have "loss", "loss_step" and
        # "loss_epoch") so that this is only logged on epochs. We also record which metrics
        # we've logged per step, so we can later exclude these from metrics logged on epochs.
        metrics = _get_step_metrics(trainer)
        metric_items = [
            (name, val)
            for (name, val) in metrics.items()
            if (name not in self._epoch_metrics) and (f"{name}_step" not in metrics.keys())
        ]
        self._step_metrics.update(name for (name, _) in metric_items)
        step = trainer.global_step
        if (step + 1) % self.log_every_n_step == 0:
            self._log_metrics(trainer, step, metric_items)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """
        Logs Optimizer related metrics when the train begins

        :param trainer: pytorch lightning trainer instance
        :param pl_module: pytorch lightning base module
        """
        self.client.set_tags(self.run_id, {"Mode": "training"})

        params = {"epochs": trainer.max_epochs}

        # TODO For logging optimizer params - Following scenarios are to revisited.
        # 1. In the current scenario, only the first optimizer details are logged.
        #    Code to be enhanced to log params when multiple optimizers are used.
        # 2. mlflow.log_params is used to store optimizer default values into mlflow.
        #    The keys in default dictionary are too short, Ex: (lr - learning_rate).
        #    Efficient mapping technique needs to be introduced
        #    to rename the optimizer parameters based on keys in default dictionary.

        if hasattr(trainer, "optimizers"):
            optimizer = trainer.optimizers[0]
            params["optimizer_name"] = _get_optimizer_name(optimizer)

            if hasattr(optimizer, "defaults"):
                params.update(optimizer.defaults)

        self.client.log_params(self.run_id, params)
        self.client.flush(synchronous=True)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        """
        Logs the model checkpoint into mlflow - models folder on the training end

        :param trainer: pytorch lightning trainer instance
        :param pl_module: pytorch lightning base module
        """
        # manually flush any remaining metadata from training
        self.metrics_logger.flush()
        self.client.flush(synchronous=True)

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        """
        Logs accuracy and other relevant metrics on the testing end

        :param trainer: pytorch lightning trainer instance
        :param pl_module: pytorch lightning base module
        """
        self.client.set_tags(self.run_id, {"Mode": "testing"})
        self.client.flush(synchronous=True)

        self.metrics_logger.record_metrics(
            {key: float(value) for key, value in trainer.callback_metrics.items()}
        )
        self.metrics_logger.flush()


# PyTorch-Lightning refactored the LoggerConnector class in version 1.4.0 and made metrics
# update on demand. Prior to this, the metrics from the current step were not available to
# callbacks immediately, so the view of metrics was off by one step.
# To avoid this problem, we access the metrics via the logger_connector for older versions.
if _pl_version >= Version("1.4.0"):

    def _get_step_metrics(trainer):
        return trainer.callback_metrics

else:

    def _get_step_metrics(trainer):
        return trainer.logger_connector.cached_results.get_latest_batch_log_metrics()


def _log_early_stop_params(early_stop_callback, client, run_id):
    """
    Logs early stopping configuration parameters to MLflow.

    :param early_stop_callback: The early stopping callback instance used during training.
    :param client: An `MlflowAutologgingQueueingClient` instance used for MLflow logging.
    :param run_id: The ID of the MLflow Run to which to log configuration parameters.
    """
    client.log_params(
        run_id,
        {
            p: getattr(early_stop_callback, p)
            for p in ["monitor", "mode", "patience", "min_delta", "stopped_epoch"]
            if hasattr(early_stop_callback, p)
        },
    )


def _log_early_stop_metrics(early_stop_callback, client, run_id):
    """
    Logs early stopping behavior results (e.g. stopped epoch) as metrics to MLflow.

    :param early_stop_callback: The early stopping callback instance used during training.
    :param client: An `MlflowAutologgingQueueingClient` instance used for MLflow logging.
    :param run_id: The ID of the MLflow Run to which to log configuration parameters.
    """
    if early_stop_callback.stopped_epoch == 0:
        return

    metrics = {
        "stopped_epoch": early_stop_callback.stopped_epoch,
        "restored_epoch": early_stop_callback.stopped_epoch - max(1, early_stop_callback.patience),
    }

    if hasattr(early_stop_callback, "best_score"):
        metrics["best_score"] = float(early_stop_callback.best_score)

    if hasattr(early_stop_callback, "wait_count"):
        metrics["wait_count"] = early_stop_callback.wait_count

    client.log_metrics(run_id, metrics)


def patched_fit(original, self, *args, **kwargs):
    """
    A patched implementation of `pytorch_lightning.Trainer.fit` which enables logging the
    following parameters, metrics and artifacts:

    - Training epochs
    - Optimizer parameters
    - `EarlyStoppingCallback`_ parameters
    - Metrics stored in `trainer.callback_metrics`
    - Model checkpoints
    - Trained model

    .. _EarlyStoppingCallback:
        https://pytorch-lightning.readthedocs.io/en/latest/early_stopping.html
    """
    if not MIN_REQ_VERSION <= _pl_version <= MAX_REQ_VERSION:
        warnings.warn(
            (
                "Autologging is known to be compatible with pytorch-lightning versions between "
                "%s and %s and may not succeed with packages "
                "outside this range."
            )
            % (MIN_REQ_VERSION, MAX_REQ_VERSION)
        )

    with _pytorch_autolog.disable_pytorch_autologging():
        run_id = mlflow.active_run().info.run_id
        tracking_uri = mlflow.get_tracking_uri()
        client = MlflowAutologgingQueueingClient(tracking_uri)
        metrics_logger = BatchMetricsLogger(run_id, tracking_uri)

        log_models = get_autologging_config(mlflow.pytorch.FLAVOR_NAME, "log_models", True)
        log_every_n_epoch = get_autologging_config(
            mlflow.pytorch.FLAVOR_NAME, "log_every_n_epoch", 1
        )
        log_every_n_step = get_autologging_config(
            mlflow.pytorch.FLAVOR_NAME, "log_every_n_step", None
        )

        early_stop_callback = None
        for callback in self.callbacks:
            if isinstance(callback, pl.callbacks.early_stopping.EarlyStopping):
                early_stop_callback = callback
                _log_early_stop_params(early_stop_callback, client, run_id)

        if not any(isinstance(callbacks, __MLflowPLCallback) for callbacks in self.callbacks):
            self.callbacks += [
                __MLflowPLCallback(
                    client, metrics_logger, run_id, log_models, log_every_n_epoch, log_every_n_step
                )
            ]

        client.flush(synchronous=False)

        result = original(self, *args, **kwargs)

        if early_stop_callback is not None:
            _log_early_stop_metrics(early_stop_callback, client, run_id)

        if Version(pl.__version__) < Version("1.4.0"):
            # pylint: disable-next=unexpected-keyword-arg
            summary = str(ModelSummary(self.model, mode="full"))
        else:
            summary = str(ModelSummary(self.model, max_depth=-1))

        with tempfile.TemporaryDirectory() as tempdir:
            summary_file = os.path.join(tempdir, "model_summary.txt")
            with open(summary_file, "w") as f:
                f.write(summary)

            mlflow.log_artifact(local_path=summary_file)

        if log_models:
            registered_model_name = get_autologging_config(
                mlflow.pytorch.FLAVOR_NAME, "registered_model_name", None
            )
            mlflow.pytorch.log_model(
                pytorch_model=self.model,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )

            if early_stop_callback is not None and self.checkpoint_callback.best_model_path:
                mlflow.log_artifact(
                    local_path=self.checkpoint_callback.best_model_path,
                    artifact_path="restored_model_checkpoint",
                )

        client.flush(synchronous=True)

    return result
