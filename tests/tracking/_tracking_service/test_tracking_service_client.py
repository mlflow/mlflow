import pytest
import mock

from mlflow.entities import Run, RunInfo
from mlflow.tracking._tracking_service.client import TrackingServiceClient
from mlflow.store.artifact.dbfs_artifact_repo import DbfsRestArtifactRepository

@pytest.fixture
def mock_store():
    with mock.patch("mlflow.tracking._trackingZ_service.utils._get_store") as mock_get_store:
        yield mock_get_store.return_value


def newTrackingServiceClient():
    return TrackingServiceClient("databricks://scope:key")


@pytest.mark.parametrize("artifact_uri, databricks_uri, uri_for_repo", [
    ('dbfs:/path', 'databricks://profile', 'dbfs://profile@databricks/path'),
    ('dbfs:/path', 'databricks://scope/key', 'dbfs://scope:key@databricks/path'),
    ('runs:/path', 'databricks://scope/key', 'runs://scope:key@databricks/path'),
    ('models:/path', 'databricks://scope/key', 'models://scope:key@databricks/path'),
    # unaffected uri cases
    ('dbfs://profile@databricks/path', 'databricks://scope/key', 'dbfs://profile@databricks/path'),
    ('dbfs://profile@databricks/path', 'databricks://profile2', 'dbfs://profile@databricks/path'),
    ('s3:/path', 'databricks://profile', 's3:/path'),
    ('ftp://user:pass@host/path', 'databricks://profile', 'ftp://user:pass@host/path'),
])
def test_get_artifact_repo(artifact_uri, databricks_uri, uri_for_repo):
    with mock.patch('mlflow.tracking._tracking_service.client.TrackingServiceClient.get_run',  # or can mock the store
                    return_value=Run(RunInfo('uuid', 'expr_id', 'userid', 'status', 0, 10,
                                             'active', artifact_uri=artifact_uri), None)), \
            mock.patch('mlflow.tracking._tracking_service.client.get_artifact_repository',
                       return_value=None) as get_repo_mock:
        client = TrackingServiceClient(databricks_uri)
        client._get_artifact_repo('some-run-id')
        get_repo_mock.assert_called_once_with(uri_for_repo)
