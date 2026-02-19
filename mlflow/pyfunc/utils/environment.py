import os
from contextlib import contextmanager

from mlflow.environment_variables import _MLFLOW_IS_IN_SERVING_ENVIRONMENT


@contextmanager
def _simulate_serving_environment():
    """
    Some functions (e.g. validate_serving_input) replicate the data transformation logic
    that happens in the model serving environment to validate data before model deployment.
    This context manager can be used to simulate the serving environment for such functions.
    """
    original_value = _MLFLOW_IS_IN_SERVING_ENVIRONMENT.get_raw()
    try:
        _MLFLOW_IS_IN_SERVING_ENVIRONMENT.set("true")
        yield
    finally:
        if original_value is not None:
            os.environ[_MLFLOW_IS_IN_SERVING_ENVIRONMENT.name] = original_value
        else:
            del os.environ[_MLFLOW_IS_IN_SERVING_ENVIRONMENT.name]
