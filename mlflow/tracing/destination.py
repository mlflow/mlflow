from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental


class UserTraceDestinationRegistry:
    def __init__(self):
        self._global_value = None
        self._context_local_value = ContextVar("mlflow_trace_destination", default=None)

    def get(self) -> TraceDestination | None:
        """First check the context-local value, then the global value."""
        if local_destination := self._context_local_value.get():
            return local_destination
        return self._global_value

    def set(self, value, context_local: bool = False):
        if context_local:
            self._context_local_value.set(value)
        else:
            self._global_value = value

    def reset(self):
        self._global_value = None
        self._context_local_value.set(None)


@experimental(version="2.21.0")
@dataclass
class TraceDestination:
    """A configuration object for specifying the destination of trace data."""

    @property
    def type(self) -> str:
        """Type of the destination."""
        raise NotImplementedError


@experimental(version="2.21.0")
@dataclass
class MlflowExperiment(TraceDestination):
    """
    A destination representing an MLflow experiment.

    By setting this destination in the :py:func:`mlflow.tracing.set_destination` function,
    MLflow will log traces to the specified experiment.

    Attributes:
        experiment_id: The ID of the experiment to log traces to. If not specified,
            the current active experiment will be used.
    """

    experiment_id: str | None = None

    @property
    def type(self) -> str:
        return "experiment"


@experimental(version="2.22.0")
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

    experiment_id: str | None = None
    experiment_name: str | None = None

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
