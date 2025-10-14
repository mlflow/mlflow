"""
Trace enablement functionality for MLflow to enable tracing to Databricks Storage.
"""

import logging

import mlflow
from mlflow.entities.trace_location import UCSchemaLocation
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.uri import is_databricks_uri
from mlflow.version import IS_TRACING_SDK_ONLY

_logger = logging.getLogger(__name__)


@experimental(version="3.5.0")
def set_experiment_trace_location(
    location: UCSchemaLocation,
    experiment_id: str | None = None,
    sql_warehouse_id: str | None = None,
) -> UCSchemaLocation:
    """
    Configure the storage location for traces of an experiment.

    Unity Catalog tables for storing trace data will be created in the specified schema.
    When tracing is enabled, all traces for the specified experiment will be
    stored in the provided Unity Catalog schema.

    .. note::

        If the experiment is already linked to a storage location, this will raise an error.
        Use `mlflow.tracing.unset_experiment_trace_location` to remove the existing storage
        location first and then set a new one.

    Args:
        location: The storage location for experiment traces in Unity Catalog.
        experiment_id: The MLflow experiment ID to set the storage location for.
            If not specified, the current active experiment will be used.
        sql_warehouse_id: SQL warehouse ID for creating views and querying.
            If not specified, uses the value from MLFLOW_TRACING_SQL_WAREHOUSE_ID,
            fallback to the default SQL warehouse if the environment variable is not set.

    Returns:
        The UCSchemaLocation object representing the configured storage location, including
        the table names of the spans and logs tables.

    Example:

        .. code-block:: python

            import mlflow
            from mlflow.entities import UCSchemaLocation

            location = UCSchemaLocation(catalog_name="my_catalog", schema_name="my_schema")

            result = mlflow.tracing.set_experiment_trace_location(
                location=location,
                experiment_id="12345",
            )
            print(result.full_otel_spans_table_name)  # my_catalog.my_schema.otel_spans_table


            @mlflow.trace
            def add(x):
                return x + 1


            add(1)  # this writes the trace to the storage location set above

    """
    from mlflow.tracing.client import TracingClient
    from mlflow.tracking import get_tracking_uri
    from mlflow.tracking.fluent import _get_experiment_id

    if not is_databricks_uri(get_tracking_uri()):
        raise MlflowException(
            "The `set_experiment_trace_location` API is only supported on Databricks."
        )

    experiment_id = experiment_id or _get_experiment_id()
    if experiment_id is None:
        raise MlflowException.invalid_parameter_value(
            "Experiment ID is required to set storage location, either pass it as an argument or "
            "use `mlflow.set_experiment` to set the current experiment."
        )

    # Check if the experiment exists. In Databricks notebook, this `get_experiment` call triggers
    # a side effect to create the experiment for the notebook if it doesn't exist. This side effect
    # is convenient for users.
    if experiment_id and not IS_TRACING_SDK_ONLY:
        try:
            mlflow.get_experiment(str(experiment_id))
        except Exception as e:
            raise MlflowException.invalid_parameter_value(
                f"Could not find experiment with ID {experiment_id}. Please make sure the "
                "experiment exists before setting the storage location."
            ) from e

    uc_schema_location = TracingClient()._set_experiment_trace_location(
        location=location,
        experiment_id=experiment_id,
        sql_warehouse_id=sql_warehouse_id,
    )

    _logger.info(
        f"Successfully configured storage location for experiment `{experiment_id}` to "
        f"Databricks storage at {uc_schema_location}"
    )

    return uc_schema_location


@experimental(version="3.5.0")
def unset_experiment_trace_location(
    location: UCSchemaLocation,
    experiment_id: str | None = None,
) -> None:
    """
    Unset the storage location for traces of an experiment.

    This function removes the experiment storage location configuration,
    including the view and the experiment tag.

    Args:
        location: The storage location to unset.
        experiment_id: The MLflow experiment ID to unset the storage location for. If not provided,
            the current active experiment will be used.

    Example:

        .. code-block:: python

            import mlflow
            from mlflow.entities import UCSchemaLocation

            mlflow.tracing.unset_experiment_trace_location(
                location=UCSchemaLocation(catalog_name="my_catalog", schema_name="my_schema"),
                experiment_id="12345",
            )

    """
    from mlflow.tracing.client import TracingClient
    from mlflow.tracking import get_tracking_uri
    from mlflow.tracking.fluent import _get_experiment_id

    if not is_databricks_uri(get_tracking_uri()):
        raise MlflowException(
            "The `unset_experiment_trace_location` API is only supported on Databricks."
        )

    if not isinstance(location, UCSchemaLocation):
        raise MlflowException.invalid_parameter_value(
            "`location` must be an instance of `mlflow.entities.UCSchemaLocation`."
        )
    experiment_id = experiment_id or _get_experiment_id()
    if experiment_id is None:
        raise MlflowException.invalid_parameter_value(
            "Experiment ID is required to clear storage location, either pass it as an argument or "
            "use `mlflow.set_experiment` to set the current experiment."
        )
    TracingClient()._unset_experiment_trace_location(experiment_id, location)
    _logger.info(f"Successfully cleared storage location for experiment `{experiment_id}`")
