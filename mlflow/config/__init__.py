from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_LOGGING


def enable_async_logging(enable=True):
    """Enable or disable async logging globally."""

    MLFLOW_ENABLE_ASYNC_LOGGING.set(enable)
