import json

from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, ErrorCode


class MlflowException(Exception):
    """
    Generic exception thrown to surface failure information about external-facing operations.
    The error message associated with this exception may be exposed to clients in HTTP responses
    for debugging purposes. If the error text is sensitive, raise a generic `Exception` object
    instead.
    """
    def __init__(self, message, error_code=INTERNAL_ERROR, **kwargs):
        """
        :param message: The message describing the error that occured. This will be included in the
                        exception's serialized JSON representation.
        :param error_code: An appropriate error code for the error that occured; it will be included
                           in the exception's serialized JSON representation. This should be one of
                           the codes listed in the `mlflow.protos.databricks_pb2` proto.
        :param kwargs: Additional key-value pairs to include in the serialized JSON representation
                       of the MlflowException.
        """
        try:
            self.error_code = ErrorCode.Name(error_code)
        except (ValueError, TypeError):
            self.error_code = ErrorCode.Name(INTERNAL_ERROR)
        self.message = message
        self.json_kwargs = kwargs
        super(MlflowException, self).__init__(message)

    def serialize_as_json(self):
        exception_dict = {'error_code': self.error_code, 'message': self.message}
        exception_dict.update(self.json_kwargs)
        return json.dumps(exception_dict)


class RestException(MlflowException):
    """Exception thrown on non 200-level responses from the REST API"""
    def __init__(self, json):
        error_code = json['error_code']
        message = error_code
        if 'message' in json:
            message = "%s: %s" % (error_code, json['message'])
        super(RestException, self).__init__(message, error_code=error_code)
        self.json = json


class ExecutionException(MlflowException):
    """Exception thrown when executing a project fails."""
    pass


class MissingConfigException(MlflowException):
    """Exception thrown when expected configuration file/directory not found"""
    pass
