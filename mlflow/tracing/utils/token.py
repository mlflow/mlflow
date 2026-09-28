from dataclasses import dataclass
from typing import TYPE_CHECKING

from mlflow.entities import LiveSpan

if TYPE_CHECKING:
    from mlflow.tracing.provider import SpanContextToken


@dataclass
class SpanWithToken:
    """
    A utility container to hold an MLflow span and its corresponding OpenTelemetry token.

    The token is a special object that is generated when setting a span as active within
    the Open Telemetry span context. This token is required when inactivate the span i.e.
    detaching the span from the context. In isolated tracer provider mode it is a tuple of
    (MLflow runtime token, optional global OTel token); otherwise a single OTel context token.
    """

    span: LiveSpan
    token: "SpanContextToken | None" = None
