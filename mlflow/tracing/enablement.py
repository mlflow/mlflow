"""
Trace enablement functionality for MLflow to enable tracing to Databricks Storage.
"""

import logging

from mlflow.entities.trace_location import UCSchemaLocation
from mlflow.exceptions import MlflowException
from mlflow.tracing.client import TracingClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


@experimental(version="3.5.0")
def set_experiment_storage_location(
    location: UCSchemaLocation,
    experiment_id: str | None = None,
    sql_warehouse_id: str | None = None,
) -> UCSchemaLocation:
    """
    Set the experiment storage location.

    Args:
        location: The storage location for experiment traces in Unity Catalog.
        experiment_id: The MLflow experiment ID to set the storage location for.
            If not specified, the current active experiment will be used.
        sql_warehouse_id: SQL warehouse ID for creating views and querying.
            If not specified, uses the value from MLFLOW_TRACING_SQL_WAREHOUSE_ID.

    Returns:
        The UCSchemaLocation object representing the configured storage location.

    Example:
    .. code-block:: python

        import mlflow
        from mlflow.entities import UCSchemaLocation
        from mlflow.tracing.enablement import set_experiment_storage_location

        location = UCSchemaLocation(catalog_name="my_catalog", schema_name="my_schema")

        result = set_experiment_storage_location(
            location=location,
            experiment_id="12345",
        )
        print(result.full_otel_spans_table_name)  # my_catalog.my_schema.otel_spans_table


        @mlflow.trace
        def add(x):
            return x + 1


        add(1)  # this writes the trace to the storage location set above

    """
    experiment_id = experiment_id or _get_experiment_id()
    if experiment_id is None:
        raise MlflowException.invalid_parameter_value(
            "Experiment ID is required to set storage location, either pass it as an argument or "
            "use `mlflow.set_experiment` to set the current experiment."
        )

    uc_schema_location = TracingClient()._set_experiment_storage_location(
        uc_schema=location,
        experiment_id=experiment_id,
        sql_warehouse_id=sql_warehouse_id,
    )

    _logger.info(
        f"Successfully enabled tracing on experiment `{experiment_id}` to Databricks storage "
        f"at {uc_schema_location}"
    )

    return uc_schema_location


@experimental(version="3.5.0")
def clear_experiment_storage_location(
    location: UCSchemaLocation,
    experiment_id: str | None = None,
) -> None:
    """
    Clear the experiment storage location.

    This function removes the experiment storage location configuration,
    including the view and the experiment tag.

    Args:
        location: The storage location to clear.
        experiment_id: The MLflow experiment ID to clear the storage location for. If not provided,
            the current active experiment will be used.

    Example:
    .. code-block:: python

        import mlflow
        from mlflow.entities import UCSchemaLocation
        from mlflow.tracing.enablement import clear_experiment_storage_location

        clear_experiment_storage_location(
            location=UCSchemaLocation(catalog_name="my_catalog", schema_name="my_schema"),
            experiment_id="12345",
        )
    """
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
    TracingClient()._clear_experiment_storage_location(experiment_id, location.schema_location)
    _logger.info(f"Successfully cleared storage location for experiment `{experiment_id}`")
