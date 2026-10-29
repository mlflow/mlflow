from unittest import mock

import pytest
from databricks.sdk.service.catalog import (
    CatalogInfo,
    CatalogIsolationMode,
    EffectivePermissionsList,
    EffectivePrivilege,
    EffectivePrivilegeAssignment,
    Privilege,
)

from mlflow.exceptions import MlflowException
from mlflow.tracing.databricks import (
    get_default_trace_catalog,
    set_databricks_monitoring_sql_warehouse_id,
)


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


def _mock_workspace_client(
    catalog_name="main",
    isolation_mode=CatalogIsolationMode.ISOLATED,
    catalog_privileges=("USE_CATALOG",),
    schema_privileges=("USE_SCHEMA", "CREATE_TABLE"),
):
    w = mock.MagicMock()
    w.settings.default_namespace.get.return_value.namespace.value = catalog_name
    w.catalogs.get.return_value = CatalogInfo(name=catalog_name, isolation_mode=isolation_mode)
    w.current_user.me.return_value.user_name = "user@example.com"
    privileges = {"catalog": catalog_privileges, "schema": schema_privileges}
    w.grants.get_effective.side_effect = lambda securable_type, full_name, principal: (
        EffectivePermissionsList(
            privilege_assignments=[
                EffectivePrivilegeAssignment(
                    principal=principal,
                    privileges=[
                        EffectivePrivilege(privilege=Privilege(p))
                        for p in privileges[securable_type]
                    ],
                )
            ]
        )
    )
    return w


@pytest.fixture
def databricks_tracking_uri():
    with (
        mock.patch("mlflow.tracking.get_tracking_uri", return_value="databricks"),
        mock.patch("mlflow.utils.databricks_utils.get_databricks_workspace_client_config"),
    ):
        yield


@pytest.mark.usefixtures("databricks_tracking_uri")
def test_get_default_trace_catalog_returns_catalog():
    w = _mock_workspace_client()
    with mock.patch("databricks.sdk.WorkspaceClient", return_value=w):
        assert get_default_trace_catalog() == "main"
    w.grants.get_effective.assert_has_calls([
        mock.call("catalog", "main", principal="user@example.com"),
        mock.call("schema", "main.default", principal="user@example.com"),
    ])


@pytest.mark.usefixtures("databricks_tracking_uri")
def test_get_default_trace_catalog_with_custom_schema():
    w = _mock_workspace_client()
    with mock.patch("databricks.sdk.WorkspaceClient", return_value=w):
        assert get_default_trace_catalog(schema_name="traces") == "main"
    w.grants.get_effective.assert_called_with("schema", "main.traces", principal="user@example.com")


@pytest.mark.usefixtures("databricks_tracking_uri")
def test_get_default_trace_catalog_accepts_all_privileges():
    w = _mock_workspace_client(
        catalog_privileges=("ALL_PRIVILEGES",), schema_privileges=("ALL_PRIVILEGES",)
    )
    with mock.patch("databricks.sdk.WorkspaceClient", return_value=w):
        assert get_default_trace_catalog() == "main"


@pytest.mark.usefixtures("databricks_tracking_uri")
@pytest.mark.parametrize(
    "kwargs",
    [
        {"catalog_name": ""},
        {"isolation_mode": CatalogIsolationMode.OPEN},
        {"catalog_privileges": ()},
        {"schema_privileges": ("USE_SCHEMA",)},
    ],
)
def test_get_default_trace_catalog_returns_none(kwargs):
    w = _mock_workspace_client(**kwargs)
    with mock.patch("databricks.sdk.WorkspaceClient", return_value=w):
        assert get_default_trace_catalog() is None


def test_get_default_trace_catalog_requires_databricks_tracking_uri():
    with mock.patch("mlflow.tracking.get_tracking_uri", return_value="file:///tmp"):
        with pytest.raises(MlflowException, match="only supported when the tracking URI"):
            get_default_trace_catalog()
