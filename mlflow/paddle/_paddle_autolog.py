import paddle

import mlflow
from mlflow.utils.autologging_utils import (
    BatchMetricsLogger,
    ExceptionSafeAbstractClass,
    MlflowAutologgingQueueingClient,
    get_autologging_config,
)


class __MlflowPaddleCallback(paddle.callbacks.Callback, metaclass=ExceptionSafeAbstractClass):
    """Callback for auto-logging metrics and parameters."""

    def __init__(self, client, metrics_logger, run_id, log_models, log_every_n_epoch):
        super().__init__()
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
        params = {
            "optimizer_name": self.model._optimizer.__class__.__name__,
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

    Args:
        early_stop_callback: The early stopping callback instance used during training.
        client: An `MlflowAutologgingQueueingClient` instance used for MLflow logging.
        run_id: The ID of the MLflow Run to which to log configuration parameters.
    """
    client.log_params(
        run_id,
        {
            p: getattr(early_stop_callback, p)
            for p in ["monitor", "patience", "min_delta", "baseline"]
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
        model_id: The ID of the model metrics will be associated with.
    """
    if early_stop_callback.stopped_epoch == 0:
        return

    metrics = {
        "stopped_epoch": early_stop_callback.stopped_epoch,
        "best_value": early_stop_callback.best_value,
    }
    client.log_metrics(run_id, metrics, model_id=model_id)


def patched_fit(original, self, *args, **kwargs):
    run_id = mlflow.active_run().info.run_id
    tracking_uri = mlflow.get_tracking_uri()
    client = MlflowAutologgingQueueingClient(tracking_uri)
    log_models = get_autologging_config(mlflow.paddle.FLAVOR_NAME, "log_models", True)
    log_every_n_epoch = get_autologging_config(mlflow.paddle.FLAVOR_NAME, "log_every_n_epoch", 1)

    model_id = None
    if log_models:
        model_id = mlflow.initialize_logged_model("model").model_id
    metrics_logger = BatchMetricsLogger(run_id, tracking_uri, model_id=model_id)

    early_stop_callback = None
    mlflow_callback = __MlflowPaddleCallback(
        client, metrics_logger, run_id, log_models, log_every_n_epoch
    )
    if "callbacks" in kwargs:
        callbacks = kwargs["callbacks"]
        for callback in callbacks:
            if isinstance(callback, paddle.callbacks.EarlyStopping):
                early_stop_callback = callback
                _log_early_stop_params(early_stop_callback, client, run_id)
                break
        kwargs["callbacks"].append(mlflow_callback)
    else:
        kwargs["callbacks"] = [mlflow_callback]
    client.flush(synchronous=False)

    result = original(self, *args, **kwargs)

    if early_stop_callback is not None:
        _log_early_stop_metrics(early_stop_callback, client, run_id, model_id=model_id)

    mlflow.log_text(str(self.summary()), "model_summary.txt")

    if log_models:
        registered_model_name = get_autologging_config(
            mlflow.paddle.FLAVOR_NAME, "registered_model_name", None
        )
        mlflow.paddle.log_model(
            self, name="model", registered_model_name=registered_model_name, model_id=model_id
        )

    client.flush(synchronous=True)

    return result
