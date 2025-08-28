from unittest import mock

import pytest

from mlflow.entities.model_registry import ModelVersion, ModelVersionTag
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


def test_copy_model_version_regular_path(store):
    """Test copy_model_version when dst_name is not a Unity Catalog name (doesn't have 3 parts)"""
    src_mv = ModelVersion(
        name="test_model",
        version="1",
        creation_timestamp=1234567890,
        run_id="test_run_id",
        source="runs:/test_run_id/model",
        description="test description",
        tags=[ModelVersionTag("key", "value")],
        run_link="test_link",
        model_id="test_model_id",
    )

    # Mock the parent class method
    with mock.patch.object(store.__class__.__bases__[0], "copy_model_version") as mock_parent_copy:
        mock_parent_copy.return_value = ModelVersion(
            name="new_model",
            version="1",
            creation_timestamp=1234567890,
            run_id="test_run_id",
            source="models:/test_model/1",
            description="test description",
            tags=[ModelVersionTag("key", "value")],
            run_link="test_link",
            model_id="test_model_id",
        )

        result = store.copy_model_version(src_mv, "new_model")

        # Should call parent method
        mock_parent_copy.assert_called_once_with(src_mv, "new_model")
        assert result.name == "new_model"
        assert result.source == "models:/test_model/1"


def test_copy_model_version_unity_catalog_migration_success(store):
    """Test copy_model_version when dst_name is a Unity Catalog name (has 3 parts) - success case"""
    src_mv = ModelVersion(
        name="test_model",
        version="1",
        creation_timestamp=1234567890,
        run_id="test_run_id",
        source="runs:/test_run_id/model",
        description="test description",
        tags=[ModelVersionTag("key", "value")],
        run_link="test_link",
        model_id="test_model_id",
    )

    dst_name = "catalog.schema.model"

    # Mock mlflow.artifacts.download_artifacts
    with mock.patch("mlflow.artifacts.download_artifacts") as mock_download:
        mock_download.return_value = "/tmp/local_model_dir"

        # Mock UcModelRegistryStore
        with mock.patch(
            "mlflow.store._unity_catalog.registry.rest_store.UcModelRegistryStore"
        ) as mock_uc_store_class:
            mock_uc_store = mock.MagicMock()
            mock_uc_store_class.return_value = mock_uc_store

            # Mock the _create_model_version_with_optional_signature_validation method
            mock_uc_store._create_model_version_with_optional_signature_validation.return_value = (
                ModelVersion(
                    name=dst_name,
                    version="1",
                    creation_timestamp=1234567890,
                    run_id="test_run_id",
                    source="models:/test_model/1",
                    description="test description",
                    tags=[ModelVersionTag("key", "value")],
                    run_link="test_link",
                    model_id="test_model_id",
                )
            )

            result = store.copy_model_version(src_mv, dst_name)

            # Verify UcModelRegistryStore was created with correct parameters
            mock_uc_store_class.assert_called_once_with(
                store_uri="databricks-uc", tracking_uri="databricks"
            )

            # Verify download_artifacts was called with correct parameters
            mock_download.assert_called_once_with(
                artifact_uri="models:/test_model/1", tracking_uri="databricks"
            )

            # Verify the UC store method was called with correct parameters
            mock_uc_store._create_model_version_with_optional_signature_validation.assert_called_once_with(
                name=dst_name,
                source="models:/test_model/1",
                run_id="test_run_id",
                local_model_path="/tmp/local_model_dir",
                model_id="test_model_id",
                bypass_signature_validation=False,
            )

            assert result.name == dst_name
            assert result.source == "models:/test_model/1"


def test_copy_model_version_unity_catalog_migration_download_failure(store):
    """Test copy_model_version when dst_name is a Unity Catalog name but download fails"""
    src_mv = ModelVersion(
        name="test_model",
        version="1",
        creation_timestamp=1234567890,
        run_id="test_run_id",
        source="runs:/test_run_id/model",
        description="test description",
        tags=[ModelVersionTag("key", "value")],
        run_link="test_link",
        model_id="test_model_id",
    )

    dst_name = "catalog.schema.model"

    # Mock mlflow.artifacts.download_artifacts to raise an exception
    with mock.patch("mlflow.artifacts.download_artifacts") as mock_download:
        mock_download.side_effect = Exception("Download failed")

        with pytest.raises(MlflowException, match="Unable to download model test_model version 1"):
            store.copy_model_version(src_mv, dst_name)

        # Verify download_artifacts was called
        mock_download.assert_called_once_with(
            artifact_uri="models:/test_model/1", tracking_uri="databricks"
        )
