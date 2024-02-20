from mlflow.environment_variables import MLFLOW_ENABLE_SYNCHRONOUS_LOGGING


def log_synchronously(enable=True):
    """Enable or disable synchronous logging globally."""

    MLFLOW_ENABLE_SYNCHRONOUS_LOGGING.set(enable)
