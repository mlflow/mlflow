from dataclasses import dataclass
from typing import Optional

from mlflow.exceptions import MlflowException
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


@experimental
@dataclass
class Databricks(TraceDestination):
    """
    A destination representing a Databricks tracing server.

    By setting this destination in the :py:func:`mlflow.tracing.set_destination` function,
    MLflow will log traces to the specified experiment.

    If neither experiment_id nor experiment_name is specified, an active experiment
    when traces are created will be used as the destination.
    If both are specified, they must refer to the same experiment.

    Attributes:
        experiment_id: The ID of the experiment to log traces to.
        experiment_name: The name of the experiment to log traces to.
    """

    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None

    def __post_init__(self):
        if self.experiment_id is not None:
            self.experiment_id = str(self.experiment_id)

        if self.experiment_name is not None:
            from mlflow.tracking._tracking_service.utils import _get_store

            # NB: Use store directly rather than fluent API to avoid dependency on MLflowClient
            experiment_id = _get_store().get_experiment_by_name(self.experiment_name).experiment_id
            if self.experiment_id is not None and self.experiment_id != experiment_id:
                raise MlflowException.invalid_parameter_value(
                    "experiment_id and experiment_name must refer to the same experiment"
                )
            self.experiment_id = experiment_id

    @property
    def type(self) -> str:
        return "databricks"
