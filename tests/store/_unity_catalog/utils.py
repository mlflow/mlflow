from unittest import mock
import pytest

from mlflow.utils.rest_utils import MlflowHostCreds


_DATABRICKS_UC_REGISTRY_URI = "databricks-uc"
_DATABRICKS_TRACKING_URI = "databricks"
_REGISTRY_HOST_CREDS = MlflowHostCreds("https://hello-registry")
_TRACKING_HOST_CREDS = MlflowHostCreds("https://hello-tracking")


@pytest.fixture
def mock_databricks_uc_host_creds():
    def mock_host_creds(uri):
        if uri == _DATABRICKS_TRACKING_URI:
            return _TRACKING_HOST_CREDS
        elif uri == _DATABRICKS_UC_REGISTRY_URI:
            return _REGISTRY_HOST_CREDS
        raise Exception(f"Got unexpected store URI {uri}")

    with mock.patch(
        "mlflow.store._unity_catalog.registry.rest_store.get_databricks_host_creds",
        side_effect=mock_host_creds,
    ):
        yield
