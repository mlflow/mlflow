from mlflow.base_exception import MLflowException


class ExecutionException(MLflowException):
    """Exception thrown when executing a project fails."""
    pass
