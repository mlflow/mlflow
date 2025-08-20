from mlflow.utils.annotations import experimental

_ERROR_MSG = (
    "The `databricks-agents` package is required to use databricks trace archival. "
    "Please install it with `pip install databricks-agents`."
)


@experimental(version="3.3.0")
def enable_databricks_trace_archival(
    *,
    delta_table_fullname: str,
    experiment_id: str | None = None,
) -> None:
    """
    Enable archiving traces for an MLflow experiment to a Unity Catalog Delta table.

    Args:
        delta_table_fullname: The full name of the Unity Catalog Delta table to archive traces to.
        experiment_id: The MLflow experiment ID to enable archival for.
            Default to the current active experiment.

    Example:

        .. code-block:: python

            from mlflow.tracing.archival import enable_databricks_trace_archival

            enable_databricks_trace_archival(
                delta_table_fullname="my_catalog.my_schema.my_prefix",
                experiment_id="12345",
            )
    """
    from mlflow.tracking.fluent import _get_experiment_id

    try:
        from databricks.agents.archive import enable_trace_archival
    except ImportError:
        raise ImportError(_ERROR_MSG)

    experiment_id = experiment_id or _get_experiment_id()

    enable_trace_archival(
        experiment_id=experiment_id,
        table_fullname=delta_table_fullname,
    )


@experimental(version="3.3.0")
def disable_databricks_trace_archival(*, experiment_id: str | None = None) -> None:
    """
    Disable archiving traces for an MLflow experiment to a Unity Catalog Delta table.

    Args:
        experiment_id: The MLflow experiment ID to disable archival for.

    Example:

        .. code-block:: python

            from mlflow.tracing.archival import disable_databricks_trace_archival

            disable_databricks_trace_archival(experiment_id="12345")
    """
    from mlflow.tracking.fluent import _get_experiment_id

    try:
        from databricks.agents.archive import disable_trace_archival
    except ImportError:
        raise ImportError(_ERROR_MSG)

    experiment_id = experiment_id or _get_experiment_id()

    disable_trace_archival(experiment_id=experiment_id)
