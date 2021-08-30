import logging
import mlflow.paddle
import os
import shutil
import tempfile
import paddle

from mlflow.utils.autologging_utils import (
    ExceptionSafeAbstractClass,
    BatchMetricsLogger,
    MlflowAutologgingQueueingClient,
    get_autologging_config,
)

logging.basicConfig(level=logging.ERROR)


def _get_optimizer_name(optimizer):
    return optimizer.__class__.__name__


class __MLflowPDCallback(paddle.callbacks.Callback, metaclass=ExceptionSafeAbstractClass):
    """
    Callback for auto-logging metrics and parameters.
    """

    def __init__(
        self, client, metrics_logger, run_id, log_models, log_every_n_epoch
    ):  # pylint: disable=super-init-not-called
        self.early_stopping = False
        self.client = client
        self.metrics_logger = metrics_logger
        self.run_id = run_id
        self.log_models = log_models
        self.log_every_n_epoch = log_every_n_epoch
        self.epoch = 0

    def _log_metrics(self, logs, current_epoch):
        metrics = {
            key: (metric[0] if isinstance(metric, list) else metric) for key, metric in logs.items()
        }
        self.metrics_logger.record_metrics(metrics, current_epoch)

    def on_epoch_end(self, epoch, logs=None):
        if self.model is not None and epoch % self.log_every_n_epoch == 0:
            self._log_metrics(logs, epoch)
            self.epoch = epoch

    def on_train_begin(self, logs=None):
        self.client.set_tags(self.run_id, {"Mode": "training"})
        params = {
            "optimizer_name": _get_optimizer_name(self.model._optimizer),
            "learning_rate": self.model._optimizer._learning_rate,
        }
        self.client.log_params(self.run_id, params)
        self.client.flush(synchronous=True)

    def on_train_end(self, logs=None):
        self.metrics_logger.flush()
        self.client.flush(synchronous=True)

    def on_eval_end(self, logs=None):
        eval_logs = {
            "eval_" + key: (metric[0] if isinstance(metric, list) else metric)
            for key, metric in logs.items()
        }
        self._log_metrics(eval_logs, self.epoch)


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
            for p in ["monitor", "patience", "min_delta", "stopped_epoch"]
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

    if early_stop_callback.best_weights is not None:
        metrics["best_weights"] = float(early_stop_callback.best_weights)

    if hasattr(early_stop_callback, "best_score"):
        metrics["best_score"] = float(early_stop_callback.best_score)

    if hasattr(early_stop_callback, "wait_count"):
        metrics["wait_count"] = early_stop_callback.wait_count

    client.log_metrics(run_id, metrics)


def patched_fit(original, self, *args, **kwargs):
    from paddle import callbacks

    run_id = mlflow.active_run().info.run_id
    tracking_uri = mlflow.get_tracking_uri()
    client = MlflowAutologgingQueueingClient(tracking_uri)
    metrics_logger = BatchMetricsLogger(run_id, tracking_uri)

    log_models = get_autologging_config(mlflow.paddle.FLAVOR_NAME, "log_models", True)
    log_every_n_epoch = get_autologging_config(mlflow.paddle.FLAVOR_NAME, "log_every_n_epoch", 1)

    early_stop_callback = None
    if "callbacks" in kwargs:
        train_callbacks = kwargs["callbacks"]
        for train_callback in train_callbacks:
            if isinstance(train_callback, callbacks.EarlyStopping):
                early_stop_callback = train_callback
                _log_early_stop_params(early_stop_callback, client, run_id)
        kwargs["callbacks"].append(
            __MLflowPDCallback(client, metrics_logger, run_id, log_models, log_every_n_epoch)
        )
    else:
        kwargs["callbacks"] = [
            __MLflowPDCallback(client, metrics_logger, run_id, log_models, log_every_n_epoch)
        ]
    client.flush(synchronous=False)

    result = original(self, *args, **kwargs)

    if early_stop_callback is not None:
        _log_early_stop_metrics(early_stop_callback, client, run_id)
    tempdir = tempfile.mkdtemp()
    try:
        summary_file = os.path.join(tempdir, "model_summary.txt")
        with open(summary_file, "w") as f:
            summary = str(self.summary())
            f.write(summary)
        mlflow.log_artifact(local_path=summary_file)
    finally:
        shutil.rmtree(tempdir)

    if log_models:
        mlflow.paddle.log_model(pd_model=self, artifact_path="model")

    client.flush(synchronous=True)

    return result
