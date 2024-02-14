from mlflow.utils.async_logging import run_operations  # noqa: F401

from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_LOGGING
from mlflow.utils.annotations import experimental


def enable_async_logging(enable=True):
    """Enable or disable async logging globally."""

    MLFLOW_ENABLE_ASYNC_LOGGING.set(enable)
