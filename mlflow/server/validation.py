from typing import List

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


def _validate_content_type(flask_request, allowed_content_types: List[str]):
    """
    Validates that the request content type is one of the allowed content types.

    Args:
        flask_request: Flask request object (flask.request)
        allowed_content_types: A list of allowed content types
    """
    if flask_request.method not in ["POST", "PUT"]:
        return

    if flask_request.content_type is None:
        raise MlflowException(
            message="Bad Request. Content-Type header is missing.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    # Remove any parameters e.g. "application/json; charset=utf-8" -> "application/json"
    content_type = flask_request.content_type.split(";")[0]
    if content_type not in allowed_content_types:
        message = f"Bad Request. Content-Type must be one of {allowed_content_types}."

        raise MlflowException(
            message=message,
            error_code=INVALID_PARAMETER_VALUE,
        )
