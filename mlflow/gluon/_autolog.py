from mxnet.gluon.contrib.estimator import EpochEnd, TrainBegin, TrainEnd
from mxnet.gluon.nn import HybridSequential

import mlflow
from mlflow.utils.autologging_utils import ExceptionSafeClass


class __MLflowGluonCallback(EpochEnd, TrainEnd, TrainBegin, metaclass=ExceptionSafeClass):
    def __init__(self, log_models, metrics_logger):
        self.log_models = log_models
        self._logger = metrics_logger
        self.current_epoch = 0

    def epoch_end(self, estimator, *args, **kwargs):
        logs = {}
        for metric in estimator.train_metrics:
            metric_name, metric_val = metric.get()
            logs[metric_name] = metric_val
        for metric in estimator.val_metrics:
            metric_name, metric_val = metric.get()
            logs[metric_name] = metric_val
        self._logger.record_metrics(logs, self.current_epoch)
        self.current_epoch += 1

    def train_begin(self, estimator, *args, **kwargs):
        mlflow.log_param("num_layers", len(estimator.net))
        if estimator.max_epoch is not None:
            mlflow.log_param("epochs", estimator.max_epoch)
        if estimator.max_batch is not None:
            mlflow.log_param("batches", estimator.max_batch)
        mlflow.log_param("optimizer_name", type(estimator.trainer.optimizer).__name__)
        if hasattr(estimator.trainer.optimizer, "lr"):
            mlflow.log_param("learning_rate", estimator.trainer.optimizer.lr)
        if hasattr(estimator.trainer.optimizer, "epsilon"):
            mlflow.log_param("epsilon", estimator.trainer.optimizer.epsilon)

    def train_end(self, estimator, *args, **kwargs):
        if isinstance(estimator.net, HybridSequential) and self.log_models:
            mlflow.gluon.log_model(estimator.net, artifact_path="model")
