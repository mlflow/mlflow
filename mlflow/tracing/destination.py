"""
Trace destination classes are DEPRECATED. Use mlflow.entities.trace_location.TraceLocation instead.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass

import mlflow
from mlflow.entities.trace_location import (
    MlflowExperimentLocation,
    TraceLocationBase,
    UCSchemaLocation,
)
from mlflow.environment_variables import MLFLOW_TRACING_DESTINATION
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import deprecated

_logger = logging.getLogger(__name__)


class UserTraceDestinationRegistry:
    def __init__(self):
        self._global_value = None
        self._context_local_value = ContextVar("mlflow_trace_destination", default=None)

    def get(self) -> TraceLocationBase | None:
        """First check the context-local value, then the global value."""
        if local_destination := self._context_local_value.get():
            return local_destination
        return self._global_value or self._get_trace_location_from_env()

    def set(self, value, context_local: bool = False):
        if context_local:
            self._context_local_value.set(value)
        else:
            self._global_value = value

    def reset(self):
        self._global_value = None
        self._context_local_value.set(None)

    def _get_trace_location_from_env(self) -> TraceLocationBase | None:
        """
        Get trace location from `MLFLOW_TRACING_DESTINATION` environment variable.
        """
        if location := MLFLOW_TRACING_DESTINATION.get():
            match location.split("."):
                case [catalog_name, schema_name]:
                    if (
                        mlflow.get_tracking_uri() is None
                        or not mlflow.get_tracking_uri().startswith("databricks")
                    ):
                        mlflow.set_tracking_uri("databricks")
                        _logger.info(
                            "Automatically setting the tracking URI to `databricks` "
                            "because the tracing destination is set to Databricks."
                        )
                    return UCSchemaLocation(catalog_name, schema_name)
                case [experiment_id]:
                    return MlflowExperimentLocation(experiment_id)
                case _:
                    raise MlflowException.invalid_parameter_value(
                        f"Failed to parse trace location {location} rom MLFLOW_TRACING_DESTINATION "
                        "environment variable. Expected format: <catalog_name>.<schema_name> or "
                        "<experiment_id>"
                    )
        return None


@deprecated(since="3.5.0", alternative="mlflow.entities.trace_location.TraceLocation")
@dataclass
class TraceDestination:
    """A configuration object for specifying the destination of trace data."""

    @property
    def type(self) -> str:
        """Type of the destination."""
        raise NotImplementedError

    def to_location(self) -> TraceLocationBase:
        raise NotImplementedError


@deprecated(since="3.5.0", alternative="mlflow.entities.trace_location.MlflowExperimentLocation")
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

    def to_location(self) -> TraceLocationBase:
        return MlflowExperimentLocation(experiment_id=self.experiment_id)


@deprecated(since="3.5.0", alternative="mlflow.entities.trace_location.MlflowExperimentLocation")
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

    def to_location(self) -> TraceLocationBase:
        return MlflowExperimentLocation(experiment_id=self.experiment_id)
