from dataclasses import dataclass
from typing import Optional

from mlflow.utils.annotations import experimental


@experimental
@dataclass
class TraceDestination:
    """A configuration object for specifying the destination of trace data."""

    @property
    def type(self) -> str:
        """Type of the destination."""
        raise NotImplementedError


@experimental
@dataclass
class MlflowExperiment(TraceDestination):
    """
    A destination representing an MLflow experiment.

    By setting this destination in the :py:func:`mlflow.tracing.set_destination` function,
    MLflow will log traces to the specified experiment.

    Attributes:
        experiment_id: The ID of the experiment to log traces to. If not specified,
            the current active experiment will be used.
        tracking_uri: The tracking URI of the MLflow server to log traces to.
            If not specified, the current tracking URI will be used.
    """

    experiment_id: Optional[str] = None
    tracking_uri: Optional[str] = None

    @property
    def type(self) -> str:
        return "experiment"
