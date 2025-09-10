from unittest import mock

import pytest

from mlflow.entities.model_registry import ModelVersion, ModelVersionTag
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry.databricks_workspace_model_registry_rest_store import (
    DatabricksWorkspaceModelRegistryRestStore,
    _extract_workspace_id_from_run_link,
)


@pytest.fixture
def store():
    return DatabricksWorkspaceModelRegistryRestStore("databricks", "databricks")


@pytest.fixture
def sample_model_version() -> ModelVersion:
    """Common ModelVersion object for testing copy_model_version"""
    return ModelVersion(
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


def _expected_unsupported_method_error_message(method):
    return f"Method '{method}' is unsupported for models in the Workspace Model Registry"


@pytest.mark.parametrize(
    ("run_link", "expected_workspace_id"),
    [
        # Valid cases
        # valid workspace ID
        (
            "https://workspace.databricks.com/?o=10002#mlflow/experiments/test-exp-id/runs/runid",
            "10002",
        ),
        # valid workspace ID with artifact path
        (
            "https://workspace.databricks.com/?o=12345#mlflow/experiments/test-exp-id/runs/runid/artifactPath/model",
            "12345",
        ),
        # different Databricks domain
        (
            "https://mycompany.cloud.databricks.com/?o=98765#mlflow/experiments/test-exp-id/runs/runid",
            "98765",
        ),
        # multiple query parameters
        (
            "https://workspace.databricks.com/?param1=value1&o=67890&param2=value2#mlflow/experiments/test-exp-id/runs/runid",
            "67890",
        ),
        # valid URL without hash fragment
        ("https://workspace.databricks.com/?o=12345", "12345"),
        # Invalid cases
        # no workspace ID parameter
        (
            "https://workspace.databricks.com/#mlflow/experiments/test-exp-id/runs/runid",
            None,
        ),
        # empty workspace ID parameter
        (
            "https://workspace.databricks.com/?o=#mlflow/experiments/test-exp-id/runs/runid",
            None,
        ),
        # non-numeric workspace ID
        (
            "https://workspace.databricks.com/?o=abc123#mlflow/experiments/test-exp-id/runs/runid",
            None,
        ),
        # workspace ID with spaces
        (
            "https://workspace.databricks.com/?o=12 34#mlflow/experiments/test-exp-id/runs/runid",
            None,
        ),
        # negative workspace ID
        (
            "https://workspace.databricks.com/?o=-123#mlflow/experiments/test-exp-id/runs/runid",
            None,
        ),
        # float workspace ID
        (
            "https://workspace.databricks.com/?o=123.45#mlflow/experiments/test-exp-id/runs/runid",
            None,
        ),
        # malformed URL missing equals sign
        (
            "https://workspace.databricks.com/?o12345#mlflow/experiments/test-exp-id/runs/runid",
            None,
        ),
        # empty string input
        ("", None),
        # None input
        (None, None),
    ],
)
def test_extract_workspace_id_from_run_link(
    run_link: str | None, expected_workspace_id: str | None
):
    assert _extract_workspace_id_from_run_link(run_link) == expected_workspace_id


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


def test_copy_model_version_regular_path(store, sample_model_version):
    """Test copy_model_version when dst_name is not a Unity Catalog name (doesn't have 3 parts)"""

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

        result = store.copy_model_version(sample_model_version, "new_model")

        # Should call parent method
        mock_parent_copy.assert_called_once_with(sample_model_version, "new_model")
        assert result.name == "new_model"
        assert result.source == "models:/test_model/1"


def test_copy_model_version_unity_catalog_success(store, sample_model_version):
    """Test copy_model_version when dst_name is a Unity Catalog name (has 3 parts) - successes"""

    dst_name = "catalog.schema.model"

    # Mock multiple dependencies in a single context manager
    with (
        mock.patch(
            "mlflow.artifacts.download_artifacts", return_value="/tmp/local_model_dir"
        ) as mock_download,
        mock.patch(
            "mlflow.store.model_registry.databricks_workspace_model_registry_rest_store.UcModelRegistryStore"
        ) as mock_uc_store_class,
    ):
        mock_uc_store = mock.MagicMock()
        mock_uc_store_class.return_value = mock_uc_store

        # Mock create_registered_model to succeed (model doesn't exist)
        mock_uc_store.create_registered_model.return_value = mock.MagicMock(name=dst_name)

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

        result = store.copy_model_version(sample_model_version, dst_name)

        # Verify UcModelRegistryStore was created with correct parameters
        mock_uc_store_class.assert_called_once_with(
            store_uri="databricks-uc", tracking_uri="databricks"
        )

        # Verify download_artifacts was called with correct parameters
        mock_download.assert_called_once_with(
            artifact_uri="models:/test_model/1", tracking_uri="databricks"
        )

        # Verify create_registered_model was called
        mock_uc_store.create_registered_model.assert_called_once_with(dst_name)

        # Verify the UC store method was called with correct parameters
        mock_uc_store._create_model_version_with_optional_signature_validation.assert_called_once_with(
            name=dst_name,
            source="models:/test_model/1",
            run_id="test_run_id",
            local_model_path="/tmp/local_model_dir",
            model_id="test_model_id",
            bypass_signature_validation=False,
            source_workspace_id=None,
        )

        assert result.name == dst_name
        assert result.source == "models:/test_model/1"


def test_copy_model_version_unity_catalog_migration_download_failure(store, sample_model_version):
    """Test copy_model_version when dst_name is a Unity Catalog name but download fails"""

    dst_name = "catalog.schema.model"

    # Mock multiple dependencies in a single context manager
    with mock.patch(
        "mlflow.artifacts.download_artifacts", side_effect=Exception("Download failed")
    ) as mock_download:
        with pytest.raises(MlflowException, match="Unable to download model test_model version 1"):
            store.copy_model_version(sample_model_version, dst_name)

        # Verify download_artifacts was called
        mock_download.assert_called_once_with(
            artifact_uri="models:/test_model/1", tracking_uri="databricks"
        )


def test_copy_model_version_unity_catalog_registered_model_already_exists(
    store, sample_model_version
):
    """Test copy_model_version when Unity Catalog registered model already exists"""

    dst_name = "catalog.schema.existing_model"

    # Mock multiple dependencies in a single context manager
    with (
        mock.patch(
            "mlflow.artifacts.download_artifacts", return_value="/tmp/local_model_dir"
        ) as mock_download,
        mock.patch(
            "mlflow.store.model_registry.databricks_workspace_model_registry_rest_store.UcModelRegistryStore"
        ) as mock_uc_store_class,
    ):
        mock_uc_store = mock.MagicMock()
        mock_uc_store_class.return_value = mock_uc_store

        # Mock create_registered_model to raise RESOURCE_ALREADY_EXISTS error
        from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS

        # Create a mock exception with the correct error_code
        mock_exception = MlflowException("Model already exists", error_code=RESOURCE_ALREADY_EXISTS)
        mock_uc_store.create_registered_model.side_effect = mock_exception

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

        result = store.copy_model_version(sample_model_version, dst_name)

        # Verify UcModelRegistryStore was created with correct parameters
        mock_uc_store_class.assert_called_once_with(
            store_uri="databricks-uc", tracking_uri="databricks"
        )

        # Verify download_artifacts was called with correct parameters
        mock_download.assert_called_once_with(
            artifact_uri="models:/test_model/1", tracking_uri="databricks"
        )

        # Verify create_registered_model was called
        mock_uc_store.create_registered_model.assert_called_once_with(dst_name)

        # Verify the UC store method was called with correct parameters
        mock_uc_store._create_model_version_with_optional_signature_validation.assert_called_once_with(
            name=dst_name,
            source="models:/test_model/1",
            run_id="test_run_id",
            local_model_path="/tmp/local_model_dir",
            model_id="test_model_id",
            bypass_signature_validation=False,
            source_workspace_id=None,
        )

        assert result.name == dst_name
        assert result.source == "models:/test_model/1"


def test_copy_model_version_unity_catalog_registered_model_creation_failure(
    store, sample_model_version
):
    """Test copy_model_version when UC model creation fails w/ non-RESOURCE_ALREADY_EXISTS error"""

    dst_name = "catalog.schema.failing_model"

    # Mock multiple dependencies in a single context manager
    with (
        mock.patch(
            "mlflow.artifacts.download_artifacts", return_value="/tmp/local_model_dir"
        ) as mock_download,
        mock.patch(
            "mlflow.store.model_registry.databricks_workspace_model_registry_rest_store.UcModelRegistryStore"
        ) as mock_uc_store_class,
    ):
        mock_uc_store = mock.MagicMock()
        mock_uc_store_class.return_value = mock_uc_store
        mock_uc_store.create_registered_model.side_effect = MlflowException(
            "Permission denied", error_code="PERMISSION_DENIED"
        )

        # The method should re-raise the exception
        with pytest.raises(MlflowException, match="Permission denied"):
            store.copy_model_version(sample_model_version, dst_name)

        # Verify UcModelRegistryStore was created with correct parameters
        mock_uc_store_class.assert_called_once_with(
            store_uri="databricks-uc", tracking_uri="databricks"
        )

        # Verify download_artifacts was called with correct parameters
        mock_download.assert_called_once_with(
            artifact_uri="models:/test_model/1", tracking_uri="databricks"
        )

        # Verify create_registered_model was called
        mock_uc_store.create_registered_model.assert_called_once_with(dst_name)

        # Verify the UC store method was NOT called since creation failed
        mock_uc_store._create_model_version_with_optional_signature_validation.assert_not_called()


def test_copy_model_version_unity_catalog_signature_validation_bypass(
    store, sample_model_version, monkeypatch
):
    """Test copy_model_version signature validation bypass via environment variable"""
    dst_name = "catalog.schema.model"
    # Mock multiple dependencies in a single context manager
    with (
        mock.patch("mlflow.artifacts.download_artifacts", return_value="/tmp/local_model_dir"),
        mock.patch(
            "mlflow.store.model_registry.databricks_workspace_model_registry_rest_store.UcModelRegistryStore"
        ) as mock_uc_store_class,
    ):
        mock_uc_store = mock.MagicMock()
        mock_uc_store_class.return_value = mock_uc_store

        # Mock create_registered_model to succeed
        mock_uc_store.create_registered_model.return_value = mock.MagicMock(name=dst_name)

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

        # Mock environment variable to enable signature validation bypass
        monkeypatch.setenv("MLFLOW_SKIP_SIGNATURE_CHECK_FOR_UC_REGISTRY_MIGRATION", "True")
        store.copy_model_version(sample_model_version, dst_name)

        # Verify the UC store method was called with bypass_signature_validation=True
        mock_uc_store._create_model_version_with_optional_signature_validation.assert_called_once_with(
            name=dst_name,
            source="models:/test_model/1",
            run_id="test_run_id",
            local_model_path="/tmp/local_model_dir",
            model_id="test_model_id",
            bypass_signature_validation=True,
            source_workspace_id=None,
        )
