from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.tracing.databricks import set_databricks_monitoring_sql_warehouse_id


def test_set_databricks_monitoring_sql_warehouse_id_requires_databricks_tracking_uri():
    with mock.patch("mlflow.get_tracking_uri", return_value="file:///tmp"):
        with pytest.raises(MlflowException, match="only supported when the tracking URI"):
            set_databricks_monitoring_sql_warehouse_id(
                sql_warehouse_id="warehouse123", experiment_id="exp456"
            )


def test_set_databricks_monitoring_sql_warehouse_id_with_explicit_experiment_id():
    mock_store = mock.MagicMock()
    with (
        mock.patch("mlflow.tracking.get_tracking_uri", return_value="databricks"),
        mock.patch(
            "mlflow.tracking._tracking_service.utils._get_store",
            return_value=mock_store,
        ),
    ):
        set_databricks_monitoring_sql_warehouse_id(
            sql_warehouse_id="warehouse123", experiment_id="exp456"
        )
        mock_store.set_experiment_tag.assert_called_once()
        call_args = mock_store.set_experiment_tag.call_args
        assert call_args[0][0] == "exp456"
        assert call_args[0][1].key == "mlflow.monitoring.sqlWarehouseId"
        assert call_args[0][1].value == "warehouse123"


def test_set_databricks_monitoring_sql_warehouse_id_with_default_experiment_id():
    mock_store = mock.MagicMock()
    with (
        mock.patch("mlflow.tracking.get_tracking_uri", return_value="databricks"),
        mock.patch(
            "mlflow.tracking._tracking_service.utils._get_store",
            return_value=mock_store,
        ),
        mock.patch("mlflow.tracking.fluent._get_experiment_id", return_value="default_exp"),
    ):
        set_databricks_monitoring_sql_warehouse_id(sql_warehouse_id="warehouse789")
        mock_store.set_experiment_tag.assert_called_once()
        call_args = mock_store.set_experiment_tag.call_args
        assert call_args[0][0] == "default_exp"
        assert call_args[0][1].key == "mlflow.monitoring.sqlWarehouseId"
        assert call_args[0][1].value == "warehouse789"
