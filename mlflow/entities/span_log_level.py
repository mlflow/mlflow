from __future__ import annotations

from enum import IntEnum

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class SpanLogLevel(IntEnum):
    """
    Log level (severity) for an MLflow trace span.

    The public tracing API accepts a :class:`SpanLogLevel` member or its
    string name (e.g. ``"INFO"``).
    """

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @classmethod
    def from_value(cls, value: SpanLogLevel | str) -> SpanLogLevel:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError:
                raise MlflowException(
                    f"Invalid SpanLogLevel name {value!r}. Expected one of "
                    f"{[m.name for m in cls]}.",
                    INVALID_PARAMETER_VALUE,
                ) from None
        raise MlflowException(
            f"SpanLogLevel must be a SpanLogLevel or str, got {type(value).__name__}.",
            INVALID_PARAMETER_VALUE,
        )
