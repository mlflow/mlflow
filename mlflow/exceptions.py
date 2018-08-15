class MlflowException(Exception):
    """Base exception in MLflow."""


class IllegalArtifactPathError(MlflowException):
    """The artifact_path parameter was invalid."""


class ExecutionException(MlflowException):
    """Exception thrown when executing a project fails."""
    pass


class DownloadException(MlflowException):
    """Exception thrown when downloading an artifact fails."""
    pass


class ShellCommandException(MlflowException):
    """Exception thrown when executing a shell command fails."""
    pass


class RestException(MlflowException):
    """Exception thrown on 400-level errors from the REST API."""
    def __init__(self, json):
        message = json['error_code']
        if 'message' in json:
            message = "%s: %s" % (message, json['message'])
        super(RestException, self).__init__(message)
        self.json = json
