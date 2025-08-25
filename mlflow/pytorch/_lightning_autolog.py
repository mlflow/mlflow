import logging
import os
import tempfile
import warnings

from packaging.version import Version

import mlflow.pytorch
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.tracking.fluent import _initialize_logged_model
from mlflow.utils.autologging_utils import (
    BatchMetricsLogger,
    ExceptionSafeAbstractClass,
    MlflowAutologgingQueueingClient,
    disable_autologging,
    get_autologging_config,
)
from mlflow.utils.checkpoint_utils import MlflowModelCheckpointCallbackBase

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

_logger = logging.getLogger(__name__)

_pl_version = Version(pl.__version__)
if _pl_version < Version("1.5.0"):
    from pytorch_lightning.core.memory import ModelSummary
else:
    from pytorch_lightning.utilities.model_summary import ModelSummary


def _get_optimizer_name(optimizer):
    """
    In pytorch-lightning 1.1.0, `LightningOptimizer` was introduced:
    https://github.com/PyTorchLightning/pytorch-lightning/pull/4658

    If a user sets `enable_pl_optimizer` to True when instantiating a `Trainer` object,
    each optimizer will be wrapped by `LightningOptimizer`:
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.html
    #pytorch_lightning.trainer.trainer.Trainer.params.enable_pl_optimizer
    """
    if Version(pl.__version__) < Version("1.1.0"):
        return optimizer.__class__.__name__
    else:
        from pytorch_lightning.core.optimizer import LightningOptimizer

        return (
            optimizer._optimizer.__class__.__name__
            if isinstance(optimizer, LightningOptimizer)
            else optimizer.__class__.__name__
        )


class __MlflowPLCallback(pl.Callback, metaclass=ExceptionSafeAbstractClass):
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
        self._global_steps_per_training_step = 1
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
        def on_train_epoch_end(self, trainer, pl_module, *args):
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
        def on_train_epoch_end(self, trainer, pl_module, *args):
            """
            Log loss and other metrics values after each train epoch

            Args:
                trainer: pytorch lightning trainer instance
                pl_module: pytorch lightning base module
                args: additional positional arguments
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

            Args:
                trainer: pytorch lightning trainer instance
                pl_module: pytorch lightning base module
            """
            self._log_epoch_metrics(trainer, pl_module)

    else:

        @rank_zero_only
        def on_epoch_end(self, trainer, pl_module):
            """
            Log loss and other metrics values after each epoch

            Args:
                trainer: pytorch lightning trainer instance
                pl_module: pytorch lightning base module
            """
            self._log_epoch_metrics(trainer, pl_module)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, *args):
        """
        Log metric values after each step

        Args:
            trainer: pytorch lightning trainer instance
            pl_module: pytorch lightning base module
            args: additional positional arguments
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
        if ((step // self._global_steps_per_training_step) + 1) % self.log_every_n_step == 0:
            self._log_metrics(trainer, step, metric_items)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """
        Logs Optimizer related metrics when the train begins

        Args:
            trainer: pytorch lightning trainer instance
            pl_module: pytorch lightning base module
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
            # Lightning >= 1.6.0 increments the global step every time an optimizer is stepped.
            # We assume every optimizer will be stepped in each training step.
            if _pl_version >= Version("1.6.0"):
                self._global_steps_per_training_step = len(trainer.optimizers)
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


        Args:
            trainer: pytorch lightning trainer instance
            pl_module: pytorch lightning base module
        """
        # manually flush any remaining metadata from training
        self.metrics_logger.flush()
        self.client.flush(synchronous=True)

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        """
        Logs accuracy and other relevant metrics on the testing end

        Args:
            trainer: pytorch lightning trainer instance
            pl_module: pytorch lightning base module
        """
        self.client.set_tags(self.run_id, {"Mode": "testing"})
        self.client.flush(synchronous=True)

        self.metrics_logger.record_metrics(
            {key: float(value) for key, value in trainer.callback_metrics.items()}
        )
        self.metrics_logger.flush()


class MlflowModelCheckpointCallback(pl.Callback, MlflowModelCheckpointCallbackBase):
    """Callback for auto-logging pytorch-lightning model checkpoints to MLflow.
    This callback implementation only supports pytorch-lightning >= 1.6.0.

    Args:
        monitor: In automatic model checkpointing, the metric name to monitor if
            you set `model_checkpoint_save_best_only` to True.
        save_best_only: If True, automatic model checkpointing only saves when
            the model is considered the "best" model according to the quantity
            monitored and previous checkpoint model is overwritten.
        mode: one of {"min", "max"}. In automatic model checkpointing,
            if save_best_only=True, the decision to overwrite the current save file is made
            based on either the maximization or the minimization of the monitored quantity.
        save_weights_only: In automatic model checkpointing, if True, then
            only the model's weights will be saved. Otherwise, the optimizer states,
            lr-scheduler states, etc are added in the checkpoint too.
        save_freq: `"epoch"` or integer. When using `"epoch"`, the callback
            saves the model after each epoch. When using integer, the callback
            saves the model at end of this many batches. Note that if the saving isn't
            aligned to epochs, the monitored metric may potentially be less reliable (it
            could reflect as little as 1 batch, since the metrics get reset
            every epoch). Defaults to `"epoch"`.

    .. code-block:: python
        :caption: Example

        import mlflow
        from mlflow.pytorch import MlflowModelCheckpointCallback
        from pytorch_lightning import Trainer

        mlflow.pytorch.autolog(checkpoint=True)

        model = MyLightningModuleNet()  # A custom-pytorch lightning model
        train_loader = create_train_dataset_loader()

        mlflow_checkpoint_callback = MlflowModelCheckpointCallback()

        trainer = Trainer(callbacks=[mlflow_checkpoint_callback])

        with mlflow.start_run() as run:
            trainer.fit(model, train_loader)

    """

    def __init__(
        self,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        save_freq="epoch",
    ):
        super().__init__(
            checkpoint_file_suffix=".pth",
            monitor=monitor,
            mode=mode,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            save_freq=save_freq,
        )
        self.trainer = None

    def save_checkpoint(self, filepath: str):
        # Note: `trainer.save_checkpoint` implementation contains invocation of
        # `self.strategy.barrier("Trainer.save_checkpoint")`,
        # in DDP training, this callback is only invoked in rank 0 process,
        # the `barrier` invocation causes deadlock,
        # so I implement `save_checkpoint` instead of
        # calling `trainer.save_checkpoint`.
        checkpoint = self.trainer._checkpoint_connector.dump_checkpoint(self.save_weights_only)
        self.trainer.strategy.save_checkpoint(checkpoint, filepath)

    @rank_zero_only
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.trainer = trainer

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx,
    ) -> None:
        if isinstance(self.save_freq, int) and (
            trainer.global_step > 0 and trainer.global_step % self.save_freq == 0
        ):
            self.check_and_save_checkpoint_if_needed(
                current_epoch=trainer.current_epoch,
                global_step=trainer.global_step,
                metric_dict={k: float(v) for k, v in trainer.callback_metrics.items()},
            )

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.save_freq == "epoch":
            self.check_and_save_checkpoint_if_needed(
                current_epoch=trainer.current_epoch,
                global_step=trainer.global_step,
                metric_dict={k: float(v) for k, v in trainer.callback_metrics.items()},
            )


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

    Args:
        early_stop_callback: The early stopping callback instance used during training.
        client: An `MlflowAutologgingQueueingClient` instance used for MLflow logging.
        run_id: The ID of the MLflow Run to which to log configuration parameters.
    """
    client.log_params(
        run_id,
        {
            p: getattr(early_stop_callback, p)
            for p in ["monitor", "mode", "patience", "min_delta", "stopped_epoch"]
            if hasattr(early_stop_callback, p)
        },
    )


def _log_early_stop_metrics(early_stop_callback, client, run_id, model_id=None):
    """
    Logs early stopping behavior results (e.g. stopped epoch) as metrics to MLflow.

    Args:
        early_stop_callback: The early stopping callback instance used during training.
        client: An `MlflowAutologgingQueueingClient` instance used for MLflow logging.
        run_id: The ID of the MLflow Run to which to log configuration parameters.
        model_id: The ID of the LoggedModel to which the metrics are associated.
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

    client.log_metrics(run_id, metrics, model_id=model_id)


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
            "Autologging is known to be compatible with pytorch-lightning versions between "
            f"{MIN_REQ_VERSION} and {MAX_REQ_VERSION} and may not succeed with packages "
            "outside this range."
        )

    with disable_autologging():
        run_id = mlflow.active_run().info.run_id
        tracking_uri = mlflow.get_tracking_uri()
        client = MlflowAutologgingQueueingClient(tracking_uri)

        log_models = get_autologging_config(mlflow.pytorch.FLAVOR_NAME, "log_models", True)
        model_id = None
        if log_models:
            model_id = _initialize_logged_model(
                name="model", flavor=mlflow.pytorch.FLAVOR_NAME
            ).model_id
        metrics_logger = BatchMetricsLogger(run_id, tracking_uri, model_id=model_id)

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

        if not any(isinstance(callbacks, __MlflowPLCallback) for callbacks in self.callbacks):
            self.callbacks += [
                __MlflowPLCallback(
                    client, metrics_logger, run_id, log_models, log_every_n_epoch, log_every_n_step
                )
            ]

        model_checkpoint = get_autologging_config(mlflow.pytorch.FLAVOR_NAME, "checkpoint", True)
        if model_checkpoint:
            # __MLflowModelCheckpoint only supports pytorch-lightning >= 1.6.0
            if _pl_version >= Version("1.6.0"):
                checkpoint_monitor = get_autologging_config(
                    mlflow.pytorch.FLAVOR_NAME, "checkpoint_monitor", "val_loss"
                )
                checkpoint_mode = get_autologging_config(
                    mlflow.pytorch.FLAVOR_NAME, "checkpoint_mode", "min"
                )
                checkpoint_save_best_only = get_autologging_config(
                    mlflow.pytorch.FLAVOR_NAME, "checkpoint_save_best_only", True
                )
                checkpoint_save_weights_only = get_autologging_config(
                    mlflow.pytorch.FLAVOR_NAME, "checkpoint_save_weights_only", False
                )
                checkpoint_save_freq = get_autologging_config(
                    mlflow.pytorch.FLAVOR_NAME, "checkpoint_save_freq", "epoch"
                )

                if not any(
                    isinstance(callbacks, MlflowModelCheckpointCallback)
                    for callbacks in self.callbacks
                ):
                    self.callbacks += [
                        MlflowModelCheckpointCallback(
                            monitor=checkpoint_monitor,
                            mode=checkpoint_mode,
                            save_best_only=checkpoint_save_best_only,
                            save_weights_only=checkpoint_save_weights_only,
                            save_freq=checkpoint_save_freq,
                        )
                    ]
            else:
                warnings.warn(
                    "Automatic model checkpointing is disabled because this feature only "
                    "supports pytorch-lightning >= 1.6.0."
                )

        client.flush(synchronous=False)

        result = original(self, *args, **kwargs)

        if early_stop_callback is not None:
            _log_early_stop_metrics(early_stop_callback, client, run_id, model_id=model_id)

        if Version(pl.__version__) < Version("1.4.0"):
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
                self.model,
                name="model",
                registered_model_name=registered_model_name,
                model_id=model_id,
            )

            if early_stop_callback is not None and self.checkpoint_callback.best_model_path:
                mlflow.log_artifact(
                    local_path=self.checkpoint_callback.best_model_path,
                    artifact_path="restored_model_checkpoint",
                )

        client.flush(synchronous=True)

    return result
