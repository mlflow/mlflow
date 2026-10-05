"""Shared HTTP responses for MLflow exceptions."""

import logging

from starlette.responses import Response

from mlflow.exceptions import MlflowException

_logger = logging.getLogger(__name__)


def mlflow_exception_response(exc: MlflowException, *, handler_name: str) -> Response:
    """Return Flask's exact ``serialize_as_json()`` body."""
    status_code = exc.get_http_status_code()
    if status_code >= 500:
        is_debug = _logger.isEnabledFor(logging.DEBUG)
        msg = f"Error in {handler_name}: {exc}"
        if not is_debug:
            msg += ". Set MLFLOW_LOGGING_LEVEL=DEBUG for traceback."
        _logger.error(msg, exc_info=is_debug)
    return Response(
        content=exc.serialize_as_json(),
        media_type="application/json",
        status_code=status_code,
    )
