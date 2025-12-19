"""
Utilities for MLflow GenAI judge tools.

This module contains utility functions and classes used across
different judge tool implementations.
"""

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
def create_page_token(offset: int) -> str:
    """
    Create a page token from an offset value.

    Args:
        offset: The byte offset for pagination

    Returns:
        String representation of the offset to use as a page token
    """
    return str(offset)


@experimental(version="3.4.0")
def parse_page_token(page_token: str | None) -> int:
    """
    Parse a page token to extract the offset value.

    Args:
        page_token: The page token string to parse, or None

    Returns:
        The offset value, or 0 if token is None

    Raises:
        MlflowException: If page_token is invalid
    """
    if page_token is None:
        return 0

    try:
        return int(page_token)
    except (ValueError, TypeError) as e:
        raise MlflowException(
            f"Invalid page_token '{page_token}': must be a valid integer",
            error_code=INVALID_PARAMETER_VALUE,
        ) from e
