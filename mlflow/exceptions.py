import json

from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, ErrorCode


class MlflowException(Exception):
    """
    Generic exception thrown to surface failure information about external-facing operations.
    The error message associated with this exception may be exposed to clients in HTTP responses
    for debugging purposes. If the error text is sensitive, raise a generic `Exception` object
    instead.
    """
    def __init__(self, message, error_code=INTERNAL_ERROR):
        try:
            self.error_code = ErrorCode.Name(error_code)
        except (ValueError, TypeError):
            self.error_code = ErrorCode.Name(INTERNAL_ERROR)
        self.message = message
        super(MlflowException, self).__init__(message)

    def serialize_as_json(self):
        return json.dumps({'error_code': self.error_code, 'message': self.message})


class RestException(MlflowException):
    """Exception thrown on non 200-level responses from the REST API"""
    def __init__(self, json):
        error_code = json['error_code']
        message = error_code
        if 'message' in json:
            message = "%s: %s" % (error_code, json['message'])
        super(RestException, self).__init__(message, error_code=error_code)
        self.json = json


class IllegalArtifactPathError(MlflowException):
    """The artifact_path parameter was invalid."""


class ExecutionException(MlflowException):
    """Exception thrown when executing a project fails."""
    pass
