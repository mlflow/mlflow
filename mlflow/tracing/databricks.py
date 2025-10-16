from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.annotations import experimental


@experimental(version="3.6.0")
def set_databricks_monitoring_sql_warehouse_id(
    sql_warehouse_id: str, experiment_id: str | None = None
) -> None:
    """
    Set the SQL warehouse ID used for Databricks production monitoring on traces logged to the given
    MLflow experiment. This only has an effect for experiments with zerobus enabled.

    Args:
        sql_warehouse_id: The SQL warehouse ID to use for monitoring.
        experiment_id: The MLflow experiment ID. If not provided, the current active experiment
            will be used.
    """
    resolved_experiment_id = experiment_id or _get_experiment_id()

    if not resolved_experiment_id:
        raise MlflowException(
            "No experiment ID provided and no active experiment found. "
            "Please provide an experiment_id or set an active experiment "
            "using mlflow.set_experiment()."
        )

    client = MlflowClient()
    client.set_experiment_tag(
        resolved_experiment_id, "mlflow.monitoring.sqlWarehouseId", sql_warehouse_id
    )
