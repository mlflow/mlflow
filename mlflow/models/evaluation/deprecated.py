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
            "The `mlflow.evaluate` API has been deprecated as of MLflow 3.0.0. "
            "Please use these improved alternatives:\n\n"
            " - For traditional ML or deep learning models: Use `mlflow.models.evaluate`, "
            "which maintains full compatibility with the original `mlflow.evaluate` API.\n\n"
            " - For LLMs or GenAI applications: Use the new `mlflow.genai.evaluate` API, "
            "which offers enhanced features specifically designed for evaluating "
            "LLMs and GenAI applications.\n",
        )
    return model_evaluate(*args, **kwargs)
