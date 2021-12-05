from tensorflow.keras.callbacks import Callback, TensorBoard

import mlflow
from mlflow.utils.autologging_utils import ExceptionSafeClass


class _TensorBoard(TensorBoard, metaclass=ExceptionSafeClass):
    pass


class __MLflowTfKeras2Callback(Callback, metaclass=ExceptionSafeClass):
    """
    Callback for auto-logging parameters and metrics in TensorFlow >= 2.0.0.
    Records model structural information as params when training starts.
    """

    def __init__(self, log_models, metrics_logger, log_every_n_steps):
        super().__init__()
        self.log_models = log_models
        self.metrics_logger = metrics_logger
        self.log_every_n_steps = log_every_n_steps

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def on_train_begin(self, logs=None):  # pylint: disable=unused-argument
        config = self.model.optimizer.get_config()
        for attribute in config:
            mlflow.log_param("opt_" + attribute, config[attribute])

        sum_list = []
        self.model.summary(print_fn=sum_list.append)
        summary = "\n".join(sum_list)
        mlflow.log_text(summary, artifact_file="model_summary.txt")

    def on_epoch_end(self, epoch, logs=None):
        # NB: tf.Keras uses zero-indexing for epochs, while other TensorFlow Estimator
        # APIs (e.g., tf.Estimator) use one-indexing. Accordingly, the modular arithmetic
        # used here is slightly different from the arithmetic used in `_log_event`, which
        # provides  metric logging hooks for TensorFlow Estimator & other TensorFlow APIs
        if epoch % self.log_every_n_steps == 0:
            self.metrics_logger.record_metrics(logs, epoch)

    def on_train_end(self, logs=None):  # pylint: disable=unused-argument
        if self.log_models:
            mlflow.keras.log_model(self.model, artifact_path="model")
