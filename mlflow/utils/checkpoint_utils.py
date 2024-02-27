import logging
import os
import posixpath

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.utils.autologging_utils import (
    ExceptionSafeAbstractClass,
)
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import LATEST_CHECKPOINT_ARTIFACT_TAG_KEY

_logger = logging.getLogger(__name__)


_CHECKPOINT_DIR = "checkpoints"
_CHECKPOINT_METRIC_FILENAME = "checkpoint_metrics.json"
_CHECKPOINT_MODEL_FILENAME = "checkpoint"
_LATEST_CHECKPOINT_PREFIX = "latest_"
_CHECKPOINT_EPOCH_PREFIX = "epoch_"
_CHECKPOINT_GLOBAL_STEP_PREFIX = "global_step_"
_WEIGHT_ONLY_CHECKPOINT_SUFFIX = ".weights"


class MlflowModelCheckpointCallbackBase(metaclass=ExceptionSafeAbstractClass):
    """Callback base class for automatic model checkpointing to MLflow.

    You must implement "save_checkpoint" method to save the model as the checkpoint file.
    and you must call `check_and_save_checkpoint_if_needed` method in relevant
    callback events to trigger automatic checkpointing.

    Args:
        checkpoint_file_suffix: checkpoint file suffix.
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

    def __init__(
        self,
        checkpoint_file_suffix,
        monitor,
        mode,
        save_best_only,
        save_weights_only,
        save_freq,
    ):
        self.checkpoint_file_suffix = checkpoint_file_suffix
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.last_monitor_value = None

        self.mlflow_tracking_uri = mlflow.get_tracking_uri()

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
            return new_monitor_value < self.last_monitor_value

        return new_monitor_value > self.last_monitor_value

    def save_checkpoint(self, filepath: str):
        raise NotImplementedError()

    def check_and_save_checkpoint_if_needed(self, current_epoch, global_step, metric_dict):
        # For distributed model training, trainer workers need to use the driver process
        # mlflow_tracking_uri.
        # Note that `self.mlflow_tracking_uri` value is assigned in the driver process
        # then it is pickled to trainer workers.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

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

        suffix = self.checkpoint_file_suffix

        if self.save_best_only:
            if self.save_weights_only:
                checkpoint_model_filename = (
                    f"{_LATEST_CHECKPOINT_PREFIX}{_CHECKPOINT_MODEL_FILENAME}"
                    f"{_WEIGHT_ONLY_CHECKPOINT_SUFFIX}{suffix}"
                )
            else:
                checkpoint_model_filename = (
                    f"{_LATEST_CHECKPOINT_PREFIX}{_CHECKPOINT_MODEL_FILENAME}{suffix}"
                )
            checkpoint_metrics_filename = (
                f"{_LATEST_CHECKPOINT_PREFIX}{_CHECKPOINT_METRIC_FILENAME}"
            )
            checkpoint_artifact_dir = _CHECKPOINT_DIR
        else:
            if self.save_freq == "epoch":
                sub_dir_name = f"{_CHECKPOINT_EPOCH_PREFIX}{current_epoch}"
            else:
                sub_dir_name = f"{_CHECKPOINT_GLOBAL_STEP_PREFIX}{global_step}"

            if self.save_weights_only:
                checkpoint_model_filename = (
                    f"{_CHECKPOINT_MODEL_FILENAME}{_WEIGHT_ONLY_CHECKPOINT_SUFFIX}{suffix}"
                )
            else:
                checkpoint_model_filename = f"{_CHECKPOINT_MODEL_FILENAME}{suffix}"
            checkpoint_metrics_filename = _CHECKPOINT_METRIC_FILENAME
            checkpoint_artifact_dir = f"{_CHECKPOINT_DIR}/{sub_dir_name}"

        mlflow.set_tag(
            LATEST_CHECKPOINT_ARTIFACT_TAG_KEY,
            f"{checkpoint_artifact_dir}/{checkpoint_model_filename}",
        )

        mlflow.log_dict(
            {**metric_dict, "epoch": current_epoch, "global_step": global_step},
            f"{checkpoint_artifact_dir}/{checkpoint_metrics_filename}",
        )

        with TempDir() as tmp_dir:
            tmp_model_save_path = os.path.join(tmp_dir.path(), checkpoint_model_filename)
            self.save_checkpoint(tmp_model_save_path)
            mlflow.log_artifact(tmp_model_save_path, checkpoint_artifact_dir)


def download_checkpoint_artifact(run_id=None, epoch=None, global_step=None, dst_path=None):
    from mlflow.client import MlflowClient
    from mlflow.utils.mlflow_tags import LATEST_CHECKPOINT_ARTIFACT_TAG_KEY

    client = MlflowClient()

    if run_id is None:
        run = mlflow.active_run()
        if run is None:
            raise MlflowException(
                "There is no active run, please provide the 'run_id' argument for "
                "'load_checkpoint' invocation."
            )
        run_id = run.info.run_id
    else:
        run = client.get_run(run_id)

    latest_checkpoint_artifact_path = run.data.tags.get(LATEST_CHECKPOINT_ARTIFACT_TAG_KEY)
    if latest_checkpoint_artifact_path is None:
        raise MlflowException("There is no logged checkpoint artifact in the current run.")

    checkpoint_filename = posixpath.basename(latest_checkpoint_artifact_path)

    if epoch is not None and global_step is not None:
        raise MlflowException(
            "Only one of 'epoch' and 'global_step' can be set for 'load_checkpoint'."
        )
    elif global_step is not None:
        checkpoint_artifact_path = (
            f"{_CHECKPOINT_DIR}/{_CHECKPOINT_GLOBAL_STEP_PREFIX}{global_step}/{checkpoint_filename}"
        )
    elif epoch is not None:
        checkpoint_artifact_path = (
            f"{_CHECKPOINT_DIR}/{_CHECKPOINT_EPOCH_PREFIX}{epoch}/{checkpoint_filename}"
        )
    else:
        checkpoint_artifact_path = latest_checkpoint_artifact_path

    return client.download_artifacts(run_id, checkpoint_artifact_path, dst_path=dst_path)
