from functools import wraps
from typing import Any, Callable, TypeVar, cast

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import FEATURE_DISABLED

F = TypeVar("F", bound=Callable[..., Any])


def filestore_not_supported(func: F) -> F:
    """
    Decorator for FileStore methods that are not supported.

    This decorator wraps methods to raise a helpful error message when
    SQL-backend-only features are called on a FileStore instance.

    Returns:
        A wrapped function that raises MlflowException when called.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        raise MlflowException(
            f"{func.__name__} is not supported with FileStore. "
            f"This feature requires a SQL-based tracking backend "
            f"(e.g., SQLite, PostgreSQL, MySQL). Please configure MLflow "
            f"with a SQL backend using --backend-store-uri. "
            f"For SQLite setup instructions, see: "
            f"https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/#configure-server",
            error_code=FEATURE_DISABLED,
        )

    return cast(F, wrapper)
