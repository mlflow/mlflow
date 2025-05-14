import functools
import warnings

import mlflow
from mlflow.models.evaluation import evaluate as model_evaluate
from mlflow.utils.uri import is_databricks_uri


@functools.wraps(model_evaluate)
def evaluate(*args, **kwargs):
    tracking_uri = mlflow.get_tracking_uri()
    if is_databricks_uri(tracking_uri):
        warnings.warn(
            "The `mlflow.evaluate` API is deprecated in MLflow 3.0.0. MLflow provides "
            "new API for evaluating your models or applications."
            " - To evaluate traditional ML or deep learning models, please use "
            "    `mlflow.models.evaluate` instead. It is 100% compatible with the old "
            "    `mlflow.evaluate` API."
            " - To evaluate LLMs or GenAI applications, please use the new "
            "    `mlflow.genai.evaluate` API. It provides more powerful features and "
            "    easy interface for evaluating LLMs and GenAI applications.",
            DeprecationWarning,
        )
    return model_evaluate(*args, **kwargs)
