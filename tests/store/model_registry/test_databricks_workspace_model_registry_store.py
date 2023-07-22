import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.model_registry.databricks_workspace_model_registry_rest_store import (
    DatabricksWorkspaceModelRegistryRestStore,
)
from mlflow.utils.rest_utils import MlflowHostCreds


@pytest.fixture
def creds():
    return MlflowHostCreds("https://hello")


@pytest.fixture
def store(creds):
    return DatabricksWorkspaceModelRegistryRestStore(lambda: creds)


def _expected_unsupported_method_error_message(method):
    return f"Method '{method}' is unsupported for models in the Workspace Model Registry"


def test_workspace_model_registry_alias_apis_unsupported(store):
    with pytest.raises(
        MlflowException,
        match=_expected_unsupported_method_error_message("set_registered_model_alias"),
    ):
        store.set_registered_model_alias(name="mycoolmodel", alias="myalias", version=1)
    with pytest.raises(
        MlflowException,
        match=_expected_unsupported_method_error_message("delete_registered_model_alias"),
    ):
        store.delete_registered_model_alias(name="mycoolmodel", alias="myalias")
    with pytest.raises(
        MlflowException,
        match=_expected_unsupported_method_error_message("get_model_version_by_alias"),
    ):
        store.get_model_version_by_alias(name="mycoolmodel", alias="myalias")
