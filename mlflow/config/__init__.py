from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_LOGGING


def enable_async_logging(enable=True):
    """Enable or disable async logging globally for fluent API.

    `enable_async_logging` only affects fluent logging APIs, such as `mlflow.log_metric`,
    `mlflow.log_param`, etc. Client APIs, i.e., logging APIs in class
    :py:func:`mlflow.client.MlflowClient` are not affected.

    Args:
        enable: bool, if True, enable async logging. If False, disable async logging.

    .. code-block:: python
        :caption: Example

        import mlflow

        mlflow.config.enable_async_logging(True)

        with mlflow.start_run():
            mlflow.log_param("a", 1)  # This will be logged asynchronously

        mlflow.config.enable_async_logging(False)
        with mlflow.start_run():
            mlflow.log_param("a", 1)  # This will be logged synchronously
    """

    MLFLOW_ENABLE_ASYNC_LOGGING.set(enable)
