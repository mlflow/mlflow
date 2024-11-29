from tensorflow import keras
from tensorflow.keras.callbacks import Callback

from mlflow import log_metrics, log_params, log_text
from mlflow.utils.autologging_utils import ExceptionSafeClass
from mlflow.utils.checkpoint_utils import MlflowModelCheckpointCallbackBase


class MlflowCallback(keras.callbacks.Callback, metaclass=ExceptionSafeClass):
    """Callback for logging Tensorflow training metrics to MLflow.

    This callback logs model information at training start, and logs training metrics every epoch or
    every n steps (defined by the user) to MLflow.

    Args:
        log_every_epoch: bool, If True, log metrics every epoch. If False, log metrics every n
            steps.
        log_every_n_steps: int, log metrics every n steps. If None, log metrics every epoch.
            Must be `None` if `log_every_epoch=True`.

    .. code-block:: python
        :caption: Example

        from tensorflow import keras
        import mlflow
        import numpy as np

        # Prepare data for a 2-class classification.
        data = tf.random.uniform([8, 28, 28, 3])
        label = tf.convert_to_tensor(np.random.randint(2, size=8))

        model = keras.Sequential(
            [
                keras.Input([28, 28, 3]),
                keras.layers.Flatten(),
                keras.layers.Dense(2),
            ]
        )

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(0.001),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

        with mlflow.start_run() as run:
            model.fit(
                data,
                label,
                batch_size=4,
                epochs=2,
                callbacks=[mlflow.keras.MlflowCallback(run)],
            )
    """

    def __init__(self, log_every_epoch=True, log_every_n_steps=None):
        self.log_every_epoch = log_every_epoch
        self.log_every_n_steps = log_every_n_steps

        if log_every_epoch and log_every_n_steps is not None:
            raise ValueError(
                "`log_every_n_steps` must be None if `log_every_epoch=True`, received "
                f"`log_every_epoch={log_every_epoch}` and `log_every_n_steps={log_every_n_steps}`."
            )
        if not log_every_epoch and log_every_n_steps is None:
            raise ValueError(
                "`log_every_n_steps` must be specified if `log_every_epoch=False`, received"
                "`log_every_n_steps=False` and `log_every_n_steps=None`."
            )

    def on_train_begin(self, logs=None):
        """Log model architecture and optimizer configuration when training begins."""
        config = self.model.optimizer.get_config()
        log_params({f"opt_{k}": v for k, v in config.items()})

        model_summary = []

        def print_fn(line, *args, **kwargs):
            model_summary.append(line)

        self.model.summary(print_fn=print_fn)
        summary = "\n".join(model_summary)
        log_text(summary, artifact_file="model_summary.txt")

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        if not self.log_every_epoch or logs is None:
            return
        log_metrics(logs, step=epoch, synchronous=False)

    def on_batch_end(self, batch, logs=None):
        """Log metrics at the end of each batch with user specified frequency."""
        if self.log_every_n_steps is None or logs is None:
            return
        current_iteration = int(self.model.optimizer.iterations.numpy())

        if current_iteration % self.log_every_n_steps == 0:
            log_metrics(logs, step=current_iteration, synchronous=False)

    def on_test_end(self, logs=None):
        """Log validation metrics at validation end."""
        if logs is None:
            return
        metrics = {"validation_" + k: v for k, v in logs.items()}
        log_metrics(metrics, synchronous=False)


class MlflowModelCheckpointCallback(Callback, MlflowModelCheckpointCallbackBase):
    """Callback for automatic Keras model checkpointing to MLflow.

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
            only the model’s weights will be saved. Otherwise, the optimizer states,
            lr-scheduler states, etc are added in the checkpoint too.
        save_freq: `"epoch"` or integer. When using `"epoch"`, the callback
            saves the model after each epoch. When using integer, the callback
            saves the model at end of this many batches. Note that if the saving isn't
            aligned to epochs, the monitored metric may potentially be less reliable (it
            could reflect as little as 1 batch, since the metrics get reset
            every epoch). Defaults to `"epoch"`.

    .. code-block:: python
        :caption: Example

        from tensorflow import keras
        import tensorflow as tf
        import mlflow
        import numpy as np
        from mlflow.tensorflow import MlflowModelCheckpointCallback

        # Prepare data for a 2-class classification.
        data = tf.random.uniform([8, 28, 28, 3])
        label = tf.convert_to_tensor(np.random.randint(2, size=8))

        model = keras.Sequential(
            [
                keras.Input([28, 28, 3]),
                keras.layers.Flatten(),
                keras.layers.Dense(2),
            ]
        )

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(0.001),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

        mlflow_checkpoint_callback = MlflowModelCheckpointCallback(
            monitor="sparse_categorical_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
        )

        with mlflow.start_run() as run:
            model.fit(
                data,
                label,
                batch_size=4,
                epochs=2,
                callbacks=[mlflow_checkpoint_callback],
            )
    """

    def __init__(
        self,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        save_freq="epoch",
    ):
        Callback.__init__(self)
        MlflowModelCheckpointCallbackBase.__init__(
            self,
            checkpoint_file_suffix=".h5",
            monitor=monitor,
            mode=mode,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            save_freq=save_freq,
        )
        self.trainer = None
        self.current_epoch = None
        self._last_batch_seen = 0
        self.global_step = 0
        self.global_step_last_saving = 0

    def save_checkpoint(self, filepath: str):
        if self.save_weights_only:
            self.model.save_weights(filepath, overwrite=True)
        else:
            self.model.save(filepath, overwrite=True)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        # Note that `on_train_batch_end` might be invoked by every N train steps,
        # (controlled by `steps_per_execution` argument in `model.compile` method).
        # the following logic is similar to
        # https://github.com/keras-team/keras/blob/e6e62405fa1b4444102601636d871610d91e5783/keras/callbacks/model_checkpoint.py#L212
        add_batches = batch + 1 if batch <= self._last_batch_seen else batch - self._last_batch_seen
        self._last_batch_seen = batch

        self.global_step += add_batches

        if isinstance(self.save_freq, int):
            if self.global_step - self.global_step_last_saving >= self.save_freq:
                self.check_and_save_checkpoint_if_needed(
                    current_epoch=self.current_epoch,
                    global_step=self.global_step,
                    metric_dict={k: float(v) for k, v in logs.items()},
                )
                self.global_step_last_saving = self.global_step

    def on_epoch_end(self, epoch, logs=None):
        if self.save_freq == "epoch":
            self.check_and_save_checkpoint_if_needed(
                current_epoch=self.current_epoch,
                global_step=self.global_step,
                metric_dict={k: float(v) for k, v in logs.items()},
            )
