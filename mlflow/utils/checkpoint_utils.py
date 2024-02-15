


class MlflowModelCheckpointCallback(metaclass=ExceptionSafeAbstractClass):
    """Callback for auto-logging pytorch-lightning model checkpoints to MLflow.
    This callback implementation only supports pytorch-lightning >= 1.6.0.
    """

    def __init__(
        self,
        client,
        run_id,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        save_freq="epoch",
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

    def _save_checkpoint_rank_zero_only(self, trainer: "pl.Trainer", filepath: str):
        checkpoint = trainer._checkpoint_connector.dump_checkpoint(self.save_weights_only)
        trainer.strategy.save_checkpoint(checkpoint, filepath)

    def _check_and_save_checkpoint_if_needed(self, trainer: "pl.Trainer"):
        current_epoch = trainer.current_epoch
        metric_dict = {k: float(v) for k, v in trainer.callback_metrics.items()}

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
                sub_dir_name = f"global_step_{trainer.global_step}"

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
            {**metric_dict, "epoch": current_epoch, "global_step": trainer.global_step},
            f"{checkpoint_artifact_dir}/{checkpoint_metrics_filename}",
        )

        tmp_dir = create_tmp_dir()
        try:
            tmp_model_save_path = os.path.join(tmp_dir, checkpoint_model_filename)
            # Note: `trainer.save_checkpoint` implementation contains invocation of
            # `self.strategy.barrier("Trainer.save_checkpoint")`,
            # in DDP training, this callback is only invoked in rank 0 process,
            # the `barrier` invocation causes deadlock,
            # so I implement `_save_checkpoint_rank_zero_only` instead of
            # `trainer.save_checkpoint`.
            self._save_checkpoint_rank_zero_only(
                trainer,
                tmp_model_save_path,
            )
            self.client.log_artifact(self.run_id, tmp_model_save_path, checkpoint_artifact_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

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
            self._check_and_save_checkpoint_if_needed(trainer)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.save_freq == "epoch":
            self._check_and_save_checkpoint_if_needed(trainer)
