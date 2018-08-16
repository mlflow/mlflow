class MlflowException(Exception):
    """Base exception in MLflow."""


class IllegalArtifactPathError(MlflowException):
    """The artifact_path parameter was invalid."""
