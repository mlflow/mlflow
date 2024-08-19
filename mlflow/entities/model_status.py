from enum import Enum


class ModelStatus(str, Enum):
    """Enum for status of an :py:class:`mlflow.entities.Model`."""

    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"
