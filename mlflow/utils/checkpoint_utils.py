import os
import shutil
import logging
from mlflow.exceptions import MlflowException
from mlflow.utils.file_utils import create_tmp_dir

from mlflow.utils.mlflow_tags import LATEST_CHECKPOINT_ARTIFACT_TAG_KEY
from mlflow.utils.autologging_utils import (
    ExceptionSafeAbstractClass,
)


_logger = logging.getLogger(__name__)


class MlflowModelCheckpointCallbackBase(metaclass=ExceptionSafeAbstractClass):

    def __init__(
        self,
        client,
        run_id,
        monitor,
        mode,
        save_best_only,
        save_weights_only,
        save_freq,
    ):
        """
        Args:
            client: An instance of `MlflowClient`.
            run_id: The id of the MLflow run which you want to log checkpoints to.
            monitor: In automatic model checkpointing, the metric name to monitor if
                you set `model_checkpoint_save_best_only` to True.
            save_best_only: If True, automatic model checkpointing only saves when
                the model is considered the "best" model according to the quantity
                monitored and previous checkpoint model is overwritten.
            mode: one of {"min", "max"}. In automatic model checkpointing,
                if save_best_only=True, the decision to overwrite the current save file is made
                based on either the maximization or the minimization of the monitored quantity.
            save_weights_only: In automatic model checkpointing, if True, then
                only the model’s weights will be saved. Otherwise, the optimizer states,
                lr-scheduler states, etc are added in the checkpoint too.
            save_freq: `"epoch"` or integer. When using `"epoch"`, the callback
                saves the model after each epoch. When using integer, the callback
                saves the model at end of this many batches. Note that if the saving isn't
                aligned to epochs, the monitored metric may potentially be less reliable (it
                could reflect as little as 1 batch, since the metrics get reset
                every epoch). Defaults to `"epoch"`.
        """
        self.client = client
        self.run_id = run_id
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.last_monitor_value = None

        if self.save_best_only:
            if self.monitor is None:
                raise MlflowException(
                    "If checkpoint 'save_best_only' config is set to True, you need to set "
                    "'monitor' config as well."
                )
            if self.mode not in ["min", "max"]:
                raise MlflowException(
                    "If checkpoint 'save_best_only' config is set to True, you need to set "
                    "'mode' config and available modes includes 'min' and 'max', but you set "
                    f"'mode' to '{self.mode}'."
                )

    def _is_new_checkpoint_better(self, new_monitor_value):
        if self.last_monitor_value is None:
            return True

        if self.mode == "min":
            return new_monitor_value <= self.last_monitor_value

        return new_monitor_value >= self.last_monitor_value

    def save_checkpoint(self, filepath: str):
        raise NotImplementedError()

    def check_and_save_checkpoint_if_needed(self, current_epoch, global_step, metric_dict):
        if self.save_best_only:
            if self.monitor not in metric_dict:
                _logger.warning(
                    "Checkpoint logging is skipped, because checkpoint 'save_best_only' config is "
                    "True, it requires to compare the monitored metric value, but the provided "
                    "monitored metric value is not available."
                )
                return

            new_monitor_value = metric_dict[self.monitor]
            if not self._is_new_checkpoint_better(new_monitor_value):
                # Current checkpoint is worse than last saved checkpoint,
                # so skip checkpointing.
                self.last_monitor_value = new_monitor_value
                return

            self.last_monitor_value = new_monitor_value

        if self.save_best_only:
            if self.save_weights_only:
                checkpoint_model_filename = "latest_checkpoint.weights.pth"
            else:
                checkpoint_model_filename = "latest_checkpoint.pth"
            checkpoint_metrics_filename = "latest_checkpoint_metrics.json"
            checkpoint_artifact_dir = "checkpoints"
        else:
            if self.save_freq == "epoch":
                sub_dir_name = f"epoch_{current_epoch}"
            else:
                sub_dir_name = f"global_step_{global_step}"

            if self.save_weights_only:
                checkpoint_model_filename = "checkpoint.weights.pth"
            else:
                checkpoint_model_filename = "checkpoint.pth"
            checkpoint_metrics_filename = "checkpoint_metrics.json"
            checkpoint_artifact_dir = f"checkpoints/{sub_dir_name}"

        self.client.set_tag(
            self.run_id,
            LATEST_CHECKPOINT_ARTIFACT_TAG_KEY,
            f"{checkpoint_artifact_dir}/{checkpoint_model_filename}",
        )

        self.client.log_dict(
            self.run_id,
            {**metric_dict, "epoch": current_epoch, "global_step": global_step},
            f"{checkpoint_artifact_dir}/{checkpoint_metrics_filename}",
        )

        tmp_dir = create_tmp_dir()
        try:
            tmp_model_save_path = os.path.join(tmp_dir, checkpoint_model_filename)

            self.save_checkpoint(tmp_model_save_path)
            self.client.log_artifact(self.run_id, tmp_model_save_path, checkpoint_artifact_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
