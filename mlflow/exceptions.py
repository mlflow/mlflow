import json

from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    NOT_IMPLEMENTED,
    TEMPORARILY_UNAVAILABLE,
    ENDPOINT_NOT_FOUND,
    PERMISSION_DENIED,
    CUSTOMER_UNAUTHORIZED,
    REQUEST_LIMIT_EXCEEDED,
    BAD_REQUEST,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    DATA_LOSS,
    DEADLINE_EXCEEDED,
    CANCELLED,
    RESOURCE_EXHAUSTED,
    ABORTED,
    NOT_FOUND,
    ALREADY_EXISTS,
    UNAUTHENTICATED,
    RESOURCE_CONFLICT,
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
        :param message: The message or exception describing the error that occurred. This will be
                        included in the exception's serialized JSON representation.
        :param error_code: An appropriate error code for the error that occurred; it will be
                           included in the exception's serialized JSON representation. This should
                           be one of the codes listed in the `mlflow.protos.databricks_pb2` proto.
        :param kwargs: Additional key-value pairs to include in the serialized JSON representation
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
        """
        Constructs an `MlflowException` object with the `INVALID_PARAMETER_VALUE` error code.

        :param message: The message describing the error that occurred. This will be included in the
                        exception's serialized JSON representation.
        :param kwargs: Additional key-value pairs to include in the serialized JSON representation
                       of the MlflowException.
        """
        return cls(message, error_code=INVALID_PARAMETER_VALUE, **kwargs)


class RestException(MlflowException):
    """Exception thrown on non 200-level responses from the REST API"""

    def __init__(self, json):
        error_code = json.get("error_code", ErrorCode.Name(INTERNAL_ERROR))
        message = "{}: {}".format(
            error_code,
            json["message"] if "message" in json else "Response: " + str(json),
        )
        super().__init__(message, error_code=ErrorCode.Value(error_code))
        self.json = json


class ExecutionException(MlflowException):
    """Exception thrown when executing a project fails"""

    pass


class MissingConfigException(MlflowException):
    """Exception thrown when expected configuration file/directory not found"""

    pass
