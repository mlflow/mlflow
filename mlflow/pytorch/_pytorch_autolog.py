import logging
import mlflow.pytorch
import os
import pytorch_lightning as pl
import shutil
import tempfile
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.utilities import rank_zero_only
from mlflow.utils.autologging_utils import try_mlflow_log, wrap_patch, batch_metrics_logger
from mlflow.utils.annotations import experimental
from mlflow.utils import gorilla

logging.basicConfig(level=logging.ERROR)

every_n_epoch = 1


# autolog module uses `try_mlflow_log` - mlflow utility to log param/metrics/artifacts into mlflow
# MlflowLogger(Pytorch Lightning's inbuild class),
# following are the downsides in using MlflowLogger.
# 1. MlflowLogger doesn't provide a mechanism to store an entire model into mlflow.
#    Only model checkpoint is saved.
# 2. For storing the model into mlflow `mlflow.pytorch` library is used
# and the library expects `mlflow` object to be instantiated.
# In case of MlflowLogger, Run management is completely controlled by the class and
# hence mlflow object needs to be reinstantiated by setting
# tracking uri, experiment_id and run_id which may lead to a race condition.
# TODO: Replace __MlflowPLCallback with Pytorch Lightning's built-in MlflowLogger
# once the above mentioned issues have been addressed


@rank_zero_only
@experimental
def _autolog(log_every_n_epoch=1, log_models=True):
    """
    Enable automatic logging from pytorch to MLflow.
    Logs loss and any other metrics specified in the fit
    function, and optimizer data as parameters. Model checkpoints
    are logged as artifacts and pytorch model is stored under `model` directory.

    MLflow will also log the parameters of the
    `EarlyStoppingCallback <https://pytorch-lightning.readthedocs.io/en/latest/early_stopping.html>`

    :param log_every_n_epoch: parameter to log metrics once in `n` epoch. By default, metrics
                       are logged after every epoch.
    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
    """
    global every_n_epoch
    every_n_epoch = log_every_n_epoch

    class __MetricsLoggerProvider:
        """
        This class is used to enable swapping .
        """

        def __init__(self):
            self.metrics_logger = None

        def swap_logger(self, new_metrics_logger):
            self.metrics_logger = new_metrics_logger

        def get_logger(self):
            return self.metrics_logger

    metrics_logger_provider = __MetricsLoggerProvider()

    def getPLCallback(log_models):
        class __MLflowPLCallback(pl.Callback):
            """
            Callback for auto-logging metrics and parameters.
            """

            def __init__(self):
                self.early_stopping = False

            def on_epoch_end(self, trainer, pl_module):
                """
                Log loss and other metrics values after each epoch

                :param trainer: pytorch lightning trainer instance
                :param pl_module: pytorch lightning base module
                """
                if (pl_module.current_epoch + 1) % every_n_epoch == 0:
                    cur_metrics = trainer.callback_metrics
                    # Cast metric value as  float before passing into logger.
                    metrics = dict(map(lambda x: (x[0], float(x[1])), cur_metrics.items()))

                    # Get the BatchMetricsLogger tied to the current mlflow run.
                    metrics_logger = metrics_logger_provider.get_logger()
                    metrics_logger.record_metrics(metrics, pl_module.current_epoch)

                for callback in trainer.callbacks:
                    if isinstance(callback, pl.callbacks.early_stopping.EarlyStopping):
                        self._early_stop_check(callback)

            def on_train_start(self, trainer, pl_module):
                """
                Logs Optimizer related metrics when the train begins

                :param trainer: pytorch lightning trainer instance
                :param pl_module: pytorch lightning base module
                """
                try_mlflow_log(mlflow.set_tag, "Mode", "training")
                try_mlflow_log(mlflow.log_param, "epochs", trainer.max_epochs)

                for callback in trainer.callbacks:
                    if isinstance(callback, pl.callbacks.early_stopping.EarlyStopping):
                        self.early_stopping = True
                        self._log_early_stop_params(callback)

                # TODO For logging optimizer params - Following scenarios are to revisited.
                # 1. In the current scenario, only the first optimizer details are logged.
                #    Code to be enhanced to log params when multiple optimizers are used.
                # 2. mlflow.log_params is used to store optimizer default values into mlflow.
                #    The keys in default dictionary are too short, Ex: (lr - learning_rate).
                #    Efficient mapping technique needs to be introduced
                #    to rename the optimizer parameters based on keys in default dictionary.

                if hasattr(trainer, "optimizers"):
                    optimizer = trainer.optimizers[0]
                    try_mlflow_log(mlflow.log_param, "optimizer_name", type(optimizer).__name__)

                    if hasattr(optimizer, "defaults"):
                        try_mlflow_log(mlflow.log_params, optimizer.defaults)

                summary = str(ModelSummary(pl_module, mode="full"))
                tempdir = tempfile.mkdtemp()
                try:
                    summary_file = os.path.join(tempdir, "model_summary.txt")
                    with open(summary_file, "w") as f:
                        f.write(summary)

                    try_mlflow_log(mlflow.log_artifact, local_path=summary_file)
                finally:
                    shutil.rmtree(tempdir)

            def on_train_end(self, trainer, pl_module):
                """
                Logs the model checkpoint into mlflow - models folder on the training end

                :param trainer: pytorch lightning trainer instance
                :param pl_module: pytorch lightning base module
                """
                if log_models:
                    mlflow.pytorch.log_model(pytorch_model=trainer.model, artifact_path="model")

                    if self.early_stopping and trainer.checkpoint_callback.best_model_path:
                        try_mlflow_log(
                            mlflow.log_artifact,
                            local_path=trainer.checkpoint_callback.best_model_path,
                            artifact_path="restored_model_checkpoint",
                        )

            def on_test_end(self, trainer, pl_module):
                """
                Logs accuracy and other relevant metrics on the testing end

                :param trainer: pytorch lightning trainer instance
                :param pl_module: pytorch lightning base module
                """
                try_mlflow_log(mlflow.set_tag, "Mode", "testing")
                for key, value in trainer.callback_metrics.items():
                    try_mlflow_log(mlflow.log_metric, key, float(value))

            @staticmethod
            def _log_early_stop_params(early_stop_obj):
                """
                Logs Early Stop parameters into mlflow

                :param early_stop_obj: Early stopping callback dict
                """
                if hasattr(early_stop_obj, "monitor"):
                    try_mlflow_log(mlflow.log_param, "monitor", early_stop_obj.monitor)

                if hasattr(early_stop_obj, "mode"):
                    try_mlflow_log(mlflow.log_param, "mode", early_stop_obj.mode)

                if hasattr(early_stop_obj, "patience"):
                    try_mlflow_log(mlflow.log_param, "patience", early_stop_obj.patience)

                if hasattr(early_stop_obj, "min_delta"):
                    try_mlflow_log(mlflow.log_param, "min_delta", early_stop_obj.min_delta)

                if hasattr(early_stop_obj, "stopped_epoch"):
                    try_mlflow_log(mlflow.log_param, "stopped_epoch", early_stop_obj.stopped_epoch)

            @staticmethod
            def _early_stop_check(early_stop_callback):
                """
                Logs all early stopping metrics

                :param early_stop_callback: Early stopping callback object
                """
                # Get the BatchMetricsLogger tied to the current mlflow run.
                metrics_logger = metrics_logger_provider.get_logger()

                if early_stop_callback.stopped_epoch != 0:
                    if hasattr(early_stop_callback, "stopped_epoch"):
                        metrics_logger.record_metrics(
                            {"stopped_epoch": early_stop_callback.stopped_epoch}
                        )
                        restored_epoch = early_stop_callback.stopped_epoch - max(
                            1, early_stop_callback.patience
                        )
                        metrics_logger.record_metrics({"restored_epoch": restored_epoch})

                    if hasattr(early_stop_callback, "best_score"):
                        metrics_logger.record_metrics(
                            {"best_score": float(early_stop_callback.best_score)}
                        )

                    if hasattr(early_stop_callback, "wait_count"):
                        metrics_logger.record_metrics(
                            {"wait_count": early_stop_callback.wait_count}
                        )

        return __MLflowPLCallback

    def _run_and_log_function(self, original, args, kwargs):
        """
        This method would be called from patched fit method and
        It adds the custom callback class into callback list.
        """
        if not mlflow.active_run():
            try_mlflow_log(mlflow.start_run)
            auto_end_run = True
        else:
            auto_end_run = False

        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            # Swap the BatchMetricsLogger set on metrics_logger_provider before each fit session.
            metrics_logger_provider.swap_logger(metrics_logger)
            __MLflowPLCallback = getPLCallback(log_models)
            if not any(isinstance(callbacks, __MLflowPLCallback) for callbacks in self.callbacks):
                self.callbacks += [__MLflowPLCallback()]
            result = original(self, *args, **kwargs)

        if auto_end_run:
            try_mlflow_log(mlflow.end_run)
        return result

    @gorilla.patch(pl.Trainer)
    def fit(self, *args, **kwargs):
        """
        Patching trainer.fit method to add autolog class into callback
        """
        original = gorilla.get_original_attribute(pl.Trainer, "fit")
        return _run_and_log_function(self, original, args, kwargs)

    wrap_patch(pl.Trainer, "fit", fit)
