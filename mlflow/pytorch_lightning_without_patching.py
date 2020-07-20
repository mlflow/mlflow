import logging
import mlflow
import os
import pytorch_lightning as pl
import shutil
import tempfile

from mlflow.utils.autologging_utils import try_mlflow_log

logging.basicConfig(level=logging.ERROR)


class __MLflowPLCallback(pl.Callback):
    """
    Enable automatic logging from Pytorch-Lightning to MLflow.
    Logs loss and any other metrics specified in the fit
    function, and optimizer data as parameters. Model checkpoints
    are logged as artifacts to a 'models' directory.

    MLflow will also log the parameters of the EarlyStopping callback
    """

    def __init__(self):
        super().__init__()

    def on_epoch_end(self, trainer, pl_module):
        """
        Log loss and other metrics values after each epoch
        """
        loss_metrics = trainer.callback_metrics

        for key, value in loss_metrics.items():
            if key != "epoch":
                try_mlflow_log(mlflow.log_metric, key, float(value))

    def on_train_start(self, trainer, pl_module):
        """
        Logs Optimizer related metrics when the train begins
        """
        if trainer.early_stop_callback:
            self._log_early_stop_params(trainer.early_stop_callback)

        if hasattr(trainer, "optimizers"):
            for optimizer in trainer.optimizers:
                try_mlflow_log(
                    mlflow.log_param, "optimizer_name", type(optimizer).__name__
                )
                if hasattr(optimizer, "defaults"):
                    defaults_dict = optimizer.defaults
                    if "lr" in defaults_dict:
                        try_mlflow_log(
                            mlflow.log_param, "learning_rate", defaults_dict["lr"]
                        )

                    if "eps" in defaults_dict:
                        try_mlflow_log(
                            mlflow.log_param, "epsilon", defaults_dict["eps"]
                        )

        summary = str(trainer.model.summarize("full"))
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
        """

        model_file_name = "model.ckpt"
        trainer.save_checkpoint(model_file_name)
        try_mlflow_log(
            mlflow.log_artifact, local_path=model_file_name, artifact_path="models"
        )

    def on_test_end(self, trainer, pl_module):
        """
        Logs accuracy and other relevant metrics on the testing end
        """
        metrics = trainer.callback_metrics

        for key, value in metrics.items():
            if key != "epoch":
                try_mlflow_log(mlflow.log_metric, key, float(value))

    def _log_early_stop_params(self, early_stop_obj):
        """
        Logs Early Stop parameters into mlflow
        """
        if hasattr(early_stop_obj, "monitor"):
            try_mlflow_log(mlflow.log_param, "monitor", early_stop_obj.monitor)
        if hasattr(early_stop_obj, "mode"):
            try_mlflow_log(mlflow.log_param, "mode", early_stop_obj.mode)
        if hasattr(early_stop_obj, "patience"):
            try_mlflow_log(mlflow.log_param, "patience", float(early_stop_obj.patience))
        if hasattr(early_stop_obj, "min_delta"):
            try_mlflow_log(
                mlflow.log_param, "min_delta", float(early_stop_obj.min_delta)
            )
        if hasattr(early_stop_obj, "stopped_epoch"):
            try_mlflow_log(
                mlflow.log_param, "stopped_epoch", float(early_stop_obj.stopped_epoch)
            )
