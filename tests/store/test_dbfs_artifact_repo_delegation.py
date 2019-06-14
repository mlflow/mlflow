import os
import mock
import pytest

from mlflow.store.dbfs_artifact_repo import DbfsArtifactRepository
from mlflow.store.dbfs_fuse_artifact_repo import DbfsFuseArtifactRepository
from mlflow.store.dbfs_artifact_repo import DbfsRestArtifactRepository

from mlflow.utils.rest_utils import MlflowHostCreds


@pytest.fixture()
def host_creds_mock():
    with mock.patch('mlflow.store.dbfs_artifact_repo._get_host_creds_from_default_store') \
            as get_creds_mock:
        get_creds_mock.return_value = lambda: MlflowHostCreds('http://host')
        yield


@mock.patch('mlflow.utils.databricks_utils.is_dbfs_fuse_available')
def test_dbfs_artifact_repo_delegates_to_correct_repo(
        is_dbfs_fuse_available, host_creds_mock):  # pylint: disable=unused-argument
    is_dbfs_fuse_available.return_value = True
    artifact_uri = "dbfs:/my/absolute/dbfs/path"
    repo = DbfsArtifactRepository(artifact_uri)
    child_repo = repo.repo
    assert isinstance(child_repo, DbfsFuseArtifactRepository)
    assert child_repo.artifact_dir == os.path.join(
        os.path.sep, "dbfs", "my", "absolute", "dbfs", "path")
    with mock.patch.dict(os.environ, {'MLFLOW_ENABLE_DBFS_FUSE_ARTIFACT_REPO': 'false'}):
        fuse_disabled_repo = DbfsArtifactRepository(artifact_uri)
    assert isinstance(fuse_disabled_repo.repo, DbfsRestArtifactRepository)
    assert fuse_disabled_repo.repo.artifact_uri == artifact_uri
    is_dbfs_fuse_available.return_value = False
    rest_repo = DbfsArtifactRepository(artifact_uri)
    assert isinstance(rest_repo.repo, DbfsRestArtifactRepository)
    assert rest_repo.repo.artifact_uri == artifact_uri
