import contextvars
from dataclasses import dataclass

from mlflow.entities import LiveSpan


@dataclass
class SpanWithToken:
    """
    A utility container to hold an MLflow span and its corresponding OpenTelemetry token.

    The token is a special object that is generated when setting a span as active within
    the Open Telemetry span context. This token is required when inactivate the span i.e.
    detaching the span from the context.
    """

    span: LiveSpan
    token: contextvars.Token | None = None
