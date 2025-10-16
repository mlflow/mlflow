from unittest import mock

from mlflow.tracing.databricks import set_databricks_monitoring_sql_warehouse_id


def test_set_databricks_monitoring_sql_warehouse_id_with_explicit_experiment_id():
    mock_client = mock.MagicMock()
    with mock.patch("mlflow.tracing.databricks.MlflowClient", return_value=mock_client):
        set_databricks_monitoring_sql_warehouse_id(
            sql_warehouse_id="warehouse123", experiment_id="exp456"
        )
        mock_client.set_experiment_tag.assert_called_once_with(
            "exp456", "mlflow.monitoring.sqlWarehouseId", "warehouse123"
        )


def test_set_databricks_monitoring_sql_warehouse_id_with_default_experiment_id():
    mock_client = mock.MagicMock()
    with (
        mock.patch("mlflow.tracing.databricks.MlflowClient", return_value=mock_client),
        mock.patch("mlflow.tracing.databricks._get_experiment_id", return_value="default_exp"),
    ):
        set_databricks_monitoring_sql_warehouse_id(sql_warehouse_id="warehouse789")
        mock_client.set_experiment_tag.assert_called_once_with(
            "default_exp", "mlflow.monitoring.sqlWarehouseId", "warehouse789"
        )
