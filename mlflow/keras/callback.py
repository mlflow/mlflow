"""Keras 3 callback to log information to MLflow."""

import keras

from mlflow import log_metrics, log_params, log_text
from mlflow.utils.autologging_utils import ExceptionSafeClass


class MlflowCallback(keras.callbacks.Callback, metaclass=ExceptionSafeClass):
    """Callback for logging Keras metrics/params/model/... to MLflow.

    This callback logs model metadata at training begins, and logs training metrics every epoch or
    every n steps (defined by the user) to MLflow.

    Args:
        log_every_epoch: bool, defaults to True. If True, log metrics every epoch. If False,
            log metrics every n steps.
        log_every_n_steps: int, defaults to None. If set, log metrics every n steps. If None,
            log metrics every epoch. Must be `None` if `log_every_epoch=True`.

    .. code-block:: python
        :caption: Example

        import keras
        import mlflow
        import numpy as np

        # Prepare data for a 2-class classification.
        data = np.random.uniform([8, 28, 28, 3])
        label = np.random.randint(2, size=8)
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
                callbacks=[mlflow.keras.MlflowCallback()],
            )
    """

    def __init__(self, log_every_epoch=True, log_every_n_steps=None, model_id=None):
        self.log_every_epoch = log_every_epoch
        self.log_every_n_steps = log_every_n_steps
        self.model_id = model_id

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
        log_params({f"optimizer_{k}": v for k, v in config.items()})

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
        log_metrics(logs, step=epoch, synchronous=False, model_id=self.model_id)

    def on_batch_end(self, batch, logs=None):
        """Log metrics at the end of each batch with user specified frequency."""
        if self.log_every_n_steps is None or logs is None:
            return
        current_iteration = int(self.model.optimizer.iterations.numpy())

        if current_iteration % self.log_every_n_steps == 0:
            log_metrics(logs, step=current_iteration, synchronous=False, model_id=self.model_id)

    def on_test_end(self, logs=None):
        """Log validation metrics at validation end."""
        if logs is None:
            return
        metrics = {"validation_" + k: v for k, v in logs.items()}
        log_metrics(metrics, synchronous=False, model_id=self.model_id)
