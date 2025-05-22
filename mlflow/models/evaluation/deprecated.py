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
            # TODO (B-Step62): Update this message to include mlflow.genai.evaluate
            # once we officially release the new API.
            "The `mlflow.evaluate` API has been deprecated as of MLflow 3.0.0. "
            "Please use `mlflow.models.evaluate` instead, which is fully compatible "
            "with the original `mlflow.evaluate` API.",
            FutureWarning,
        )
    return model_evaluate(*args, **kwargs)
