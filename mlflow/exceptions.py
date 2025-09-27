import json
import logging

from mlflow.protos.databricks_pb2 import (
    ABORTED,
    ALREADY_EXISTS,
    BAD_REQUEST,
    CANCELLED,
    CUSTOMER_UNAUTHORIZED,
    DATA_LOSS,
    DEADLINE_EXCEEDED,
    ENDPOINT_NOT_FOUND,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    NOT_FOUND,
    NOT_IMPLEMENTED,
    PERMISSION_DENIED,
    REQUEST_LIMIT_EXCEEDED,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_CONFLICT,
    RESOURCE_DOES_NOT_EXIST,
    RESOURCE_EXHAUSTED,
    TEMPORARILY_UNAVAILABLE,
    UNAUTHENTICATED,
    ErrorCode,
)

ERROR_CODE_TO_HTTP_STATUS = {
    ErrorCode.Name(INTERNAL_ERROR): 500,
    ErrorCode.Name(INVALID_STATE): 500,
    ErrorCode.Name(DATA_LOSS): 500,
    ErrorCode.Name(NOT_IMPLEMENTED): 501,
    ErrorCode.Name(TEMPORARILY_UNAVAILABLE): 503,
    ErrorCode.Name(DEADLINE_EXCEEDED): 504,
    ErrorCode.Name(REQUEST_LIMIT_EXCEEDED): 429,
    ErrorCode.Name(CANCELLED): 499,
    ErrorCode.Name(RESOURCE_EXHAUSTED): 429,
    ErrorCode.Name(ABORTED): 409,
    ErrorCode.Name(RESOURCE_CONFLICT): 409,
    ErrorCode.Name(ALREADY_EXISTS): 409,
    ErrorCode.Name(NOT_FOUND): 404,
    ErrorCode.Name(ENDPOINT_NOT_FOUND): 404,
    ErrorCode.Name(RESOURCE_DOES_NOT_EXIST): 404,
    ErrorCode.Name(PERMISSION_DENIED): 403,
    ErrorCode.Name(CUSTOMER_UNAUTHORIZED): 401,
    ErrorCode.Name(UNAUTHENTICATED): 401,
    ErrorCode.Name(BAD_REQUEST): 400,
    ErrorCode.Name(RESOURCE_ALREADY_EXISTS): 400,
    ErrorCode.Name(INVALID_PARAMETER_VALUE): 400,
}

HTTP_STATUS_TO_ERROR_CODE = {v: k for k, v in ERROR_CODE_TO_HTTP_STATUS.items()}
HTTP_STATUS_TO_ERROR_CODE[400] = ErrorCode.Name(BAD_REQUEST)
HTTP_STATUS_TO_ERROR_CODE[404] = ErrorCode.Name(ENDPOINT_NOT_FOUND)
HTTP_STATUS_TO_ERROR_CODE[500] = ErrorCode.Name(INTERNAL_ERROR)

_logger = logging.getLogger(__name__)


def get_error_code(http_status):
    return ErrorCode.Value(
        HTTP_STATUS_TO_ERROR_CODE.get(http_status, ErrorCode.Name(INTERNAL_ERROR))
    )


class MlflowException(Exception):
    """
    Generic exception thrown to surface failure information about external-facing operations.
    The error message associated with this exception may be exposed to clients in HTTP responses
    for debugging purposes. If the error text is sensitive, raise a generic `Exception` object
    instead.
    """

    def __init__(self, message, error_code=INTERNAL_ERROR, **kwargs):
        """
        Args:
            message: The message or exception describing the error that occurred. This will be
                included in the exception's serialized JSON representation.
            error_code: An appropriate error code for the error that occurred; it will be
                included in the exception's serialized JSON representation. This should
                be one of the codes listed in the `mlflow.protos.databricks_pb2` proto.
            kwargs: Additional key-value pairs to include in the serialized JSON representation
                of the MlflowException.
        """
        try:
            self.error_code = ErrorCode.Name(error_code)
        except (ValueError, TypeError):
            self.error_code = ErrorCode.Name(INTERNAL_ERROR)
        message = str(message)
        self.message = message
        self.json_kwargs = kwargs
        super().__init__(message)

    def serialize_as_json(self):
        exception_dict = {"error_code": self.error_code, "message": self.message}
        exception_dict.update(self.json_kwargs)
        return json.dumps(exception_dict)

    def get_http_status_code(self):
        return ERROR_CODE_TO_HTTP_STATUS.get(self.error_code, 500)

    @classmethod
    def invalid_parameter_value(cls, message, **kwargs):
        """Constructs an `MlflowException` object with the `INVALID_PARAMETER_VALUE` error code.

        Args:
            message: The message describing the error that occurred. This will be included in the
                exception's serialized JSON representation.
            kwargs: Additional key-value pairs to include in the serialized JSON representation
                of the MlflowException.
        """
        return cls(message, error_code=INVALID_PARAMETER_VALUE, **kwargs)


class RestException(MlflowException):
    """Exception thrown on non 200-level responses from the REST API"""

    def __init__(self, json):
        self.json = json

        error_code = json.get("error_code", ErrorCode.Name(INTERNAL_ERROR))
        message = "{}: {}".format(
            error_code,
            json["message"] if "message" in json else "Response: " + str(json),
        )

        try:
            super().__init__(message, error_code=ErrorCode.Value(error_code))
        except ValueError:
            try:
                # The `error_code` can be an http error code, in which case we convert it to the
                # corresponding `ErrorCode`.
                error_code = HTTP_STATUS_TO_ERROR_CODE[int(error_code)]
                super().__init__(message, error_code=ErrorCode.Value(error_code))
            except ValueError or KeyError:
                _logger.warning(
                    f"Received error code not recognized by MLflow: {error_code}, this may "
                    "indicate your request encountered an error before reaching MLflow server, "
                    "e.g., within a proxy server or authentication / authorization service."
                )
                super().__init__(message)

    def __reduce__(self):
        """
        Overriding `__reduce__` to make `RestException` instance pickle-able.
        """
        return RestException, (self.json,)


class ExecutionException(MlflowException):
    """Exception thrown when executing a project fails"""


class MissingConfigException(MlflowException):
    """Exception thrown when expected configuration file/directory not found"""


class InvalidUrlException(MlflowException):
    """Exception thrown when a http request fails to send due to an invalid URL"""


class _UnsupportedMultipartUploadException(MlflowException):
    """Exception thrown when multipart upload is unsupported by an artifact repository"""

    MESSAGE = "Multipart upload is not supported for the current artifact repository"

    def __init__(self):
        super().__init__(self.MESSAGE, error_code=NOT_IMPLEMENTED)


class MlflowTracingException(MlflowException):
    """
    Exception thrown from tracing logic

    Tracing logic should not block the main execution flow in general, hence this exception
    is used to distinguish tracing related errors and handle them properly.
    """

    def __init__(self, message, error_code=INTERNAL_ERROR):
        super().__init__(message, error_code=error_code)


class MlflowTraceDataException(MlflowTracingException):
    """Exception thrown for trace data related error"""

    def __init__(
        self, error_code: str, request_id: str | None = None, artifact_path: str | None = None
    ):
        if request_id:
            self.ctx = f"request_id={request_id}"
        elif artifact_path:
            self.ctx = f"path={artifact_path}"

        if error_code == NOT_FOUND:
            super().__init__(f"Trace data not found for {self.ctx}", error_code=error_code)
        elif error_code == INVALID_STATE:
            super().__init__(f"Trace data is corrupted for {self.ctx}", error_code=error_code)


class MlflowTraceDataNotFound(MlflowTraceDataException):
    """Exception thrown when trace data is not found"""

    def __init__(self, request_id: str | None = None, artifact_path: str | None = None):
        super().__init__(NOT_FOUND, request_id, artifact_path)


class MlflowTraceDataCorrupted(MlflowTraceDataException):
    """Exception thrown when trace data is corrupted"""

    def __init__(self, request_id: str | None = None, artifact_path: str | None = None):
        super().__init__(INVALID_STATE, request_id, artifact_path)


class MlflowNotImplementedException(MlflowException):
    """Exception thrown when a feature is not implemented"""

    def __init__(self, message=""):
        super().__init__(message, error_code=NOT_IMPLEMENTED)
