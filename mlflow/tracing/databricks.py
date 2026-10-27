from mlflow.exceptions import MlflowException
from mlflow.utils import databricks_utils
from mlflow.utils.uri import is_databricks_uri


def get_default_trace_catalog(schema_name: str = "default") -> str | None:
    """
    Return the Databricks workspace default catalog if traces can be stored in
    ``<catalog>.<schema_name>``, otherwise ``None``.

    The catalog is returned only if it is ISOLATED (bound to this workspace, so traces are not
    visible from other workspaces) and the current user has ``USE_CATALOG`` on it and
    ``USE_SCHEMA`` and ``CREATE_TABLE`` on the schema.

    Args:
        schema_name: The schema in the default catalog to store traces in.

    .. code-block:: python

        import mlflow
        from mlflow.entities.trace_location import UnityCatalog

        mlflow.set_tracking_uri("databricks")
        if catalog := mlflow.tracing.get_default_trace_catalog():
            mlflow.set_experiment("my-genai-app", trace_location=UnityCatalog(catalog, "default"))
    """
    from mlflow.tracking import get_tracking_uri

    tracking_uri = get_tracking_uri()
    if not is_databricks_uri(tracking_uri):
        raise MlflowException(
            "This function is only supported when the tracking URI is set to 'databricks'. "
            f"Current tracking URI: {tracking_uri}"
        )
    return databricks_utils.get_default_trace_catalog(tracking_uri, schema_name)


def set_databricks_monitoring_sql_warehouse_id(
    sql_warehouse_id: str, experiment_id: str | None = None
) -> None:
    """
    Set the SQL warehouse ID used for Databricks production monitoring on traces logged to the given
    MLflow experiment. This only has an effect for experiments with UC schema as trace location.

    Args:
        sql_warehouse_id: The SQL warehouse ID to use for monitoring.
        experiment_id: The MLflow experiment ID. If not provided, the current active experiment
            will be used.
    """
    from mlflow.entities import ExperimentTag
    from mlflow.tracking import get_tracking_uri
    from mlflow.tracking._tracking_service.utils import _get_store
    from mlflow.tracking.fluent import _get_experiment_id

    tracking_uri = get_tracking_uri()
    if not is_databricks_uri(tracking_uri):
        raise MlflowException(
            "This function is only supported when the tracking URI is set to 'databricks'. "
            f"Current tracking URI: {tracking_uri}"
        )

    resolved_experiment_id = experiment_id or _get_experiment_id()

    if not resolved_experiment_id:
        raise MlflowException(
            "No experiment ID provided and no active experiment found. "
            "Please provide an experiment_id or set an active experiment "
            "using mlflow.set_experiment()."
        )

    store = _get_store()
    store.set_experiment_tag(
        resolved_experiment_id,
        ExperimentTag("mlflow.monitoring.sqlWarehouseId", sql_warehouse_id),
    )
