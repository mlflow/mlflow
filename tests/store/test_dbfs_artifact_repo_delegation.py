import os
import mock

from mlflow.store.dbfs_artifact_repo import DbfsArtifactRepository
from mlflow.store.dbfs_fuse_artifact_repo import DbfsFuseArtifactRepository
from mlflow.store.dbfs_artifact_repo import DbfsRestArtifactRepository

from mlflow.utils.rest_utils import MlflowHostCreds


@mock.patch('mlflow.utils.databricks_utils.is_dbfs_fuse_available')
def test_dbfs_artifact_repo_delegates_to_correct_repo(is_in_databricks_notebook):
    is_in_databricks_notebook.return_value = True
    artifact_uri = "dbfs:/my/absolute/dbfs/path"
    repo = DbfsArtifactRepository(artifact_uri)
    child_repo = repo.repo
    assert isinstance(child_repo, DbfsFuseArtifactRepository)
    assert child_repo.artifact_dir == os.path.join(
        os.path.sep, "dbfs", "my", "absolute", "dbfs", "path")
    is_in_databricks_notebook.return_value = False
    with mock.patch('mlflow.store.dbfs_artifact_repo._get_host_creds_from_default_store') \
            as get_creds_mock:
        get_creds_mock.return_value = lambda: MlflowHostCreds('http://host')
        repo = DbfsArtifactRepository(artifact_uri)
    child_repo = repo.repo
    assert isinstance(child_repo, DbfsRestArtifactRepository)
    assert child_repo.artifact_uri == artifact_uri
