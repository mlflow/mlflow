import os
from contextlib import contextmanager

from mlflow.environment_variables import _MLFLOW_IS_SERVING_ENVIRONMENT


@contextmanager
def serving_environment():
    """
    Context manager that sets the `_MLFLOW_IS_SERVING_ENVIRONMENT` environment variable to `True`.
    """
    original_value = _MLFLOW_IS_SERVING_ENVIRONMENT.get_raw()
    try:
        _MLFLOW_IS_SERVING_ENVIRONMENT.set("true")
        yield
    finally:
        if original_value is not None:
            os.environ[_MLFLOW_IS_SERVING_ENVIRONMENT.name] = original_value
        else:
            del os.environ[_MLFLOW_IS_SERVING_ENVIRONMENT.name]
