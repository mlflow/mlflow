"""
Pagination utilities for MLflow GenAI judge tools.

This module provides utilities for handling pagination tokens and offsets
used across different judge tools that support paginated responses.
"""

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
        The offset value, or 0 if token is None or invalid
    """
    if not page_token:
        return 0

    try:
        return int(page_token)
    except (ValueError, TypeError):
        return 0
