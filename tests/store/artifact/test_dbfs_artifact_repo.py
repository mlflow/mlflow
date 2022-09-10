import pytest
from unittest import mock

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.dbfs_artifact_repo import (
    dbfs_artifact_repo_factory,
    DbfsRestArtifactRepository,
)
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository


@pytest.mark.parametrize(
    "artifact_uri, uri_at_init",
    [("dbfs:/path", "file:///dbfs/path"), ("dbfs://databricks/path", "file:///dbfs/path")],
)
def test_dbfs_artifact_repo_factory_local_repo(artifact_uri, uri_at_init):
    with mock.patch(
        "mlflow.utils.databricks_utils.is_dbfs_fuse_available", return_value=True
    ), mock.patch(
        "mlflow.store.artifact.dbfs_artifact_repo.LocalArtifactRepository", autospec=True
    ) as mock_repo:
        repo = dbfs_artifact_repo_factory(artifact_uri)
        assert isinstance(repo, LocalArtifactRepository)
        mock_repo.assert_called_once_with(uri_at_init)


@pytest.mark.parametrize(
    "artifact_uri",
    [
        "dbfs://someProfile@databricks/path",
        "dbfs://somewhere:else@databricks/path",
        # Model registry paths should use the REST artifact repo, both when communicating
        # with the current workspace (authority component = "databricks") and other workspaces
        # (authority component = "someProfile@databricks"), as model registry paths cannot
        # be accessed via the local filesystem (via FUSE)
        "dbfs://databricks/databricks/mlflow-registry/abcdefg123/path",
        "dbfs://someProfile@databricks/mlflow-registry/abcdefg123/path",
        "dbfs://somewhere:else@databricks/mlflow-registry/abcdefg123/path",
        "dbfs:/databricks/mlflow-registry/abcdefg123/path",
    ],
)
def test_dbfs_artifact_repo_factory_dbfs_rest_repo(artifact_uri):
    with mock.patch(
        "mlflow.utils.databricks_utils.is_dbfs_fuse_available", return_value=True
    ), mock.patch(
        "mlflow.store.artifact.dbfs_artifact_repo.DbfsRestArtifactRepository", autospec=True
    ) as mock_repo:
        repo = dbfs_artifact_repo_factory(artifact_uri)
        assert isinstance(repo, DbfsRestArtifactRepository)
        mock_repo.assert_called_once_with(artifact_uri)


@pytest.mark.parametrize(
    "artifact_uri",
    [
        ("dbfs:/databricks/mlflow-tracking/experiment/1/run/2"),
        ("dbfs://@databricks/databricks/mlflow-tracking/experiment/1/run/2"),
        ("dbfs://someProfile@databricks/databricks/mlflow-tracking/experiment/1/run/2"),
    ],
)
def test_dbfs_artifact_repo_factory_acled_paths(artifact_uri):
    repo_pkg_path = "mlflow.store.artifact.databricks_artifact_repo"
    with mock.patch(
        "mlflow.utils.databricks_utils.is_dbfs_fuse_available", return_value=True
    ), mock.patch(
        "mlflow.store.artifact.dbfs_artifact_repo.DatabricksArtifactRepository", autospec=True
    ) as mock_repo, mock.patch(
        repo_pkg_path + ".get_databricks_host_creds", return_value=None
    ), mock.patch(
        repo_pkg_path + ".DatabricksArtifactRepository._get_run_artifact_root",
        return_value="whatever",
    ):
        repo = dbfs_artifact_repo_factory(artifact_uri)
        assert isinstance(repo, DatabricksArtifactRepository)
        mock_repo.assert_called_once_with(artifact_uri)


@pytest.mark.parametrize(
    "artifact_uri", [("notdbfs:/path"), ("dbfs://some:where@notdatabricks/path")]
)
def test_dbfs_artifact_repo_factory_errors(artifact_uri):
    with pytest.raises(MlflowException, match="DBFS URI must be of the form dbfs"):
        dbfs_artifact_repo_factory(artifact_uri)
