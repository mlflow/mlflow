from mlflow.entities.trace_location import UnityCatalog
from mlflow.exceptions import MlflowException
from mlflow.utils import databricks_utils
from mlflow.utils.annotations import experimental
from mlflow.utils.uri import is_databricks_uri


@experimental(version="3.19.0")
def get_default_trace_location(schema_name: str = "default") -> UnityCatalog | None:
    """
    Return a trace location in the Databricks `workspace default catalog`_ if traces can be
    stored in ``<catalog>.<schema_name>``, otherwise ``None``. A workspace admin sets this
    catalog, and queries that don't specify a catalog use it.

    The location is returned only if the current user has ``USE_CATALOG`` on the catalog and
    ``USE_SCHEMA`` and ``CREATE_TABLE`` on the schema. ``None`` is also returned if the catalog
    or schema does not exist or is not accessible.

    Args:
        schema_name: The schema in the default catalog to store traces in.

    .. code-block:: python

        import mlflow

        mlflow.set_tracking_uri("databricks")
        if location := mlflow.tracing.get_default_trace_location():
            mlflow.set_experiment("my-genai-app", trace_location=location)

    .. _workspace default catalog: https://docs.databricks.com/aws/en/catalogs/default
    """
    from mlflow.tracking import get_tracking_uri

    tracking_uri = get_tracking_uri()
    if not is_databricks_uri(tracking_uri):
        raise MlflowException(
            "This function is only supported when the tracking URI is set to 'databricks'. "
            f"Current tracking URI: {tracking_uri}"
        )
    if catalog_name := databricks_utils.get_default_trace_catalog(tracking_uri, schema_name):
        return UnityCatalog(catalog_name=catalog_name, schema_name=schema_name)
    return None


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
