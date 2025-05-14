import functools
import logging

import mlflow
from mlflow.models.evaluation import evaluate as model_evaluate
from mlflow.utils.uri import is_databricks_uri

_logger = logging.getLogger(__name__)


@functools.wraps(model_evaluate)
def evaluate(*args, **kwargs):
    tracking_uri = mlflow.get_tracking_uri()
    if is_databricks_uri(tracking_uri):
        _logger.warning(
            "The `mlflow.evaluate` API is deprecated in MLflow 3.0.0. MLflow provides "
            "new API for evaluating your models or applications.\n"
            " - To evaluate traditional ML or deep learning models, please use "
            "    `mlflow.models.evaluate` instead. It is 100% compatible with the old "
            "    `mlflow.evaluate` API.\n"
            " - To evaluate LLMs or GenAI applications, please use the new "
            "    `mlflow.genai.evaluate` API. It provides more powerful features and "
            "    easy interface for evaluating LLMs and GenAI applications.",
            DeprecationWarning,
        )
    return model_evaluate(*args, **kwargs)
