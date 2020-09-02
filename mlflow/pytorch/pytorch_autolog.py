import gorilla
import logging
import mlflow.pytorch
import os
import pytorch_lightning as pl
import shutil
import tempfile
from pytorch_lightning.core.memory import ModelSummary

logging.basicConfig(level=logging.ERROR)


def autolog(log_every_n_iter=1, aggregation_step=None):
    global every_n_iter
    every_n_iter = log_every_n_iter

    global training_metrics

    class __MLflowPLCallback(pl.Callback):
        """
        Callback for auto-logging metrics and parameters.
        """

        def __init__(self):
            super().__init__()

        def on_epoch_end(self, trainer, pl_module):
            """
            Log loss and other metrics values after each epoch
            """
            global training_metrics
            if (pl_module.current_epoch - 1) % every_n_iter == 0:
                training_metrics = trainer.callback_metrics

                if aggregation_step:
                    metrics = dict(
                        (key, float(value)) for key, value in training_metrics.items()
                    )
                    trainer.logger.agg_and_log_metrics(
                        metrics=metrics, step=aggregation_step
                    )
                else:
                    for key, value in training_metrics.items():
                        trainer.logger.experiment.log_metric(
                            trainer.logger.run_id,
                            key,
                            float(value),
                            step=pl_module.current_epoch,
                        )

            if trainer.early_stop_callback:
                self._early_stop_check(trainer=trainer)

        def on_train_start(self, trainer, pl_module):
            """
            Logs Optimizer related metrics when the train begins
            """
            if trainer.early_stop_callback:
                self._log_early_stop_params(trainer, trainer.early_stop_callback)

            if hasattr(trainer, "optimizers"):
                for optimizer in trainer.optimizers:
                    trainer.logger.experiment.log_param(
                        trainer.logger.run_id,
                        "optimizer_name",
                        type(optimizer).__name__,
                    )
                    if hasattr(optimizer, "defaults"):
                        defaults_dict = optimizer.defaults
                        if "lr" in defaults_dict:
                            trainer.logger.experiment.log_param(
                                trainer.logger.run_id,
                                "learning_rate",
                                defaults_dict["lr"],
                            )

                        if "eps" in defaults_dict:
                            trainer.logger.experiment.log_param(
                                trainer.logger.run_id, "epsilon", defaults_dict["eps"]
                            )

            summary = str(ModelSummary(pl_module, mode="full"))
            tempdir = tempfile.mkdtemp()
            try:
                summary_file = os.path.join(tempdir, "model_summary.txt")
                with open(summary_file, "w") as f:
                    f.write(summary)

                trainer.logger.experiment.log_artifact(
                    trainer.logger.run_id, local_path=summary_file
                )
            finally:
                shutil.rmtree(tempdir)

        def on_train_end(self, trainer, pl_module):
            """
            Logs the model checkpoint into mlflow - models folder on the training end
            """

            tempdir = tempfile.mkdtemp()
            if os.path.exists(tempdir):
                shutil.rmtree(tempdir)

            try:
                mlflow.pytorch.save_model(trainer.model, path=tempdir)
                trainer.logger.experiment.log_artifact(
                    trainer.logger.run_id, local_path=tempdir, artifact_path="model"
                )
            finally:
                shutil.rmtree(tempdir)

            if (
                trainer.early_stop_callback
                and trainer.checkpoint_callback.best_model_path
            ):
                trainer.logger.experiment.log_artifact(
                    trainer.logger.run_id,
                    local_path=trainer.checkpoint_callback.best_model_path,
                    artifact_path="restored_model_checkpoint",
                )

        def on_test_end(self, trainer, pl_module):
            """
            Logs accuracy and other relevant metrics on the testing end
            """
            global training_metrics
            metrics = trainer.callback_metrics

            for key, value in metrics.items():
                if key not in training_metrics:
                    trainer.logger.experiment.log_metric(
                        trainer.logger.run_id, key, float(value)
                    )

        def _log_early_stop_params(self, trainer, early_stop_obj):
            """
            Logs Early Stop parameters into mlflow
            """
            if hasattr(early_stop_obj, "monitor"):
                trainer.logger.experiment.log_param(
                    trainer.logger.run_id, "monitor", early_stop_obj.monitor
                )
            if hasattr(early_stop_obj, "mode"):
                trainer.logger.experiment.log_param(
                    trainer.logger.run_id, "mode", early_stop_obj.mode
                )
            if hasattr(early_stop_obj, "patience"):
                trainer.logger.experiment.log_param(
                    trainer.logger.run_id, "patience", float(early_stop_obj.patience)
                )
            if hasattr(early_stop_obj, "min_delta"):
                trainer.logger.experiment.log_param(
                    trainer.logger.run_id, "min_delta", float(early_stop_obj.min_delta)
                )
            if hasattr(early_stop_obj, "stopped_epoch"):
                trainer.logger.experiment.log_param(
                    trainer.logger.run_id,
                    "stopped_epoch",
                    float(early_stop_obj.stopped_epoch),
                )

        def _early_stop_check(self, trainer):
            """
            Logs all early stopping metrics
            """
            if trainer.early_stop_callback.stopped_epoch != 0:

                if hasattr(trainer.early_stop_callback, "stopped_epoch"):
                    trainer.logger.experiment.log_metric(
                        trainer.logger.run_id,
                        "Stopped_Epoch",
                        trainer.early_stop_callback.stopped_epoch,
                    )
                if hasattr(trainer.early_stop_callback, "best_score"):
                    trainer.logger.experiment.log_metric(
                        trainer.logger.run_id,
                        "Best_Score",
                        float(trainer.early_stop_callback.best_score),
                    )
                if hasattr(trainer.early_stop_callback, "wait_count"):
                    trainer.logger.experiment.log_metric(
                        trainer.logger.run_id,
                        "Wait_Count",
                        trainer.early_stop_callback.wait_count,
                    )
                restored_epoch = trainer.early_stop_callback.stopped_epoch - max(
                    1, trainer.early_stop_callback.patience
                )
                trainer.logger.experiment.log_metric(
                    trainer.logger.run_id, "Restored_Epoch", restored_epoch
                )

    def _run_and_log_function(self, original, args, kwargs):
        """
        This method would be called from patched fit method and
        It adds the custom callback class into callback list.
        """
        if not any(isinstance(callbacks, __MLflowPLCallback) for callbacks in self.callbacks):
            self.callbacks += [__MLflowPLCallback()]
        result = original(self, *args, **kwargs)
        return result

    @gorilla.patch(pl.Trainer)
    def fit(self, *args, **kwargs):
        """
        Patching trainer.fit method to add autolog class into callback
        """
        original = gorilla.get_original_attribute(pl.Trainer, "fit")
        return _run_and_log_function(self, original, args, kwargs)

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    gorilla.apply(gorilla.Patch(pl.Trainer, "fit", fit, settings=settings))
