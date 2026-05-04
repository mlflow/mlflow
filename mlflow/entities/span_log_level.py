from __future__ import annotations

from enum import IntEnum

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class SpanLogLevel(IntEnum):
    """
    Log level (severity) for an MLflow trace span.

    Numeric values match Python's :py:mod:`logging` module, so a value sourced
    from a standard logger (e.g. ``logger.getEffectiveLevel()``) can be passed
    directly into the MLflow tracing API.
    """

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @classmethod
    def from_value(cls, value: SpanLogLevel | int | str) -> SpanLogLevel:
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
        if isinstance(value, int) and not isinstance(value, bool):
            try:
                return cls(value)
            except ValueError:
                raise MlflowException(
                    f"Invalid SpanLogLevel value {value!r}. Expected one of "
                    f"{[int(m) for m in cls]}.",
                    INVALID_PARAMETER_VALUE,
                ) from None
        raise MlflowException(
            f"SpanLogLevel must be a SpanLogLevel, int, or str, got {type(value).__name__}.",
            INVALID_PARAMETER_VALUE,
        )
