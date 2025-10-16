from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.uri import is_databricks_uri


@experimental(version="3.5.0")
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
