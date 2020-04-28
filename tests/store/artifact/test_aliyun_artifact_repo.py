# pylint: disable=redefined-outer-name
import os
import mock
import pytest
from functools import partial

import oss2

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.aliyun_artifact_repo import AliyunArtifactRepository


@pytest.fixture
def oss_bucket_mock():
    # Make sure that our environment variable aren't set to actually access Aliyun
    old_endpoint_url = os.environ.get('MLFLOW_OSS_ENDPOINT_URL')
    if old_endpoint_url is not None:
        del os.environ['MLFLOW_OSS_ENDPOINT_URL']
    old_access_key_id = os.environ.get('MLFLOW_OSS_KEY_ID')
    if old_access_key_id is not None:
        del os.environ['MLFLOW_OSS_KEY_ID']
    old_access_key_secret = os.environ.get('MLFLOW_OSS_KEY_SECRET')
    if old_access_key_secret is not None:
        del os.environ['MLFLOW_OSS_KEY_SECRET']

    yield mock.MagicMock(autospec=oss2.Bucket)
    
    if old_endpoint_url is not None:
        os.environ['MLFLOW_OSS_ENDPOINT_URL'] = old_endpoint_url
    if old_access_key_id is not None:
        os.environ['MLFLOW_OSS_KEY_ID'] = old_access_key_id
    if old_access_key_secret is not None:
        os.environ['MLFLOW_OSS_KEY_SECRET'] = old_access_key_secret


def test_artifact_uri_factory(oss_bucket_mock):
    # pylint: disable=unused-argument
    # We pass in the oss_bucket_mock here to clear environment variables, but we don't use it;
    # We do need to set up a fake access key for the code to run though
    os.environ['MLFLOW_OSS_ENDPOINT_URL'] = ''
    os.environ['MLFLOW_OSS_KEY_ID'] = ''
    os.environ['MLFLOW_OSS_KEY_SECRET'] = ''
    repo = get_artifact_repository("oss://test_bucket/some/path")
    assert isinstance(repo, AliyunArtifactRepository)
    del os.environ['MLFLOW_OSS_ENDPOINT_URL']
    del os.environ['MLFLOW_OSS_KEY_ID']
    del os.environ['MLFLOW_OSS_KEY_SECRET']


def test_log_artifact(oss_bucket_mock, tmpdir):
    repo = AliyunArtifactRepository("oss://test_bucket/some/path", oss_bucket_mock)
    repo._get_oss_bucket = oss_bucket_mock

    d = tmpdir.mkdir("data")
    f = d.join("test.txt")
    f.write("hello world!")
    fpath = d + '/test.txt'
    fpath = fpath.strpath
    repo.log_artifact(fpath)
    repo.oss_bucket.put_object_from_file.assert_called_with('some/path/test.txt', fpath)


def test_log_artifacts(oss_bucket_mock, tmpdir):
    repo = AliyunArtifactRepository("oss://test_bucket/some/path", oss_bucket_mock)
    repo._get_oss_bucket = oss_bucket_mock
    
    subd = tmpdir.mkdir("data").mkdir("subdir")
    subd.join("a.txt").write("A")
    subd.join("b.txt").write("B")
    subd.join("c.txt").write("C")

    repo.log_artifacts(subd.strpath)
    repo.oss_bucket.put_object_from_file.assert_has_calls([
                mock.call('some/path/a.txt', os.path.normpath('%s/a.txt' % subd.strpath)),
                mock.call('some/path/b.txt', os.path.normpath('%s/b.txt' % subd.strpath)),
                mock.call('some/path/c.txt', os.path.normpath('%s/c.txt' % subd.strpath))
            ], any_order=True)
    
def test_list_artifacts_empty(oss_bucket_mock):
    repo = AliyunArtifactRepository("oss://test_bucket/some/path", oss_bucket_mock)
    repo._get_oss_bucket = repo.oss_bucket
    assert repo.list_artifacts() == []



def test_list_artifacts(oss_bucket_mock):
    import oss2.models
    artifact_root_path = "experiment_id/run_id/"
    repo = AliyunArtifactRepository("oss://test_bucket/" + artifact_root_path, oss_bucket_mock)
    repo._get_oss_bucket = repo.oss_bucket
    MockSimplifiedObjectInfo = mock.MagicMock(autospec=oss2.models.SimplifiedObjectInfo)
    file_path = 'file'
    obj_mock = oss2.models.SimplifiedObjectInfo(key=artifact_root_path + file_path, last_modified='123', size=1, etag=None, type=None, storage_class=None)
    dir_name = "model"
    dir_mock = oss2.models.SimplifiedObjectInfo(key=artifact_root_path + dir_name + "/", last_modified=None, size=None, etag=None, type=None, storage_class=None)

    mock_results = mock.MagicMock(autospec=oss2.models.ListObjectsResult)
    mock_results.object_list = [obj_mock, dir_mock]
    mock_results.prefix_list = []
    repo.oss_bucket.list_objects.return_value = mock_results

    artifacts = repo.list_artifacts(path=None)

    assert len(artifacts) == 2
    assert artifacts[0].path == file_path
    assert artifacts[0].is_dir is False
    assert artifacts[0].file_size == obj_mock.size
    assert artifacts[1].path == dir_name
    assert artifacts[1].is_dir is True
    assert artifacts[1].file_size is None

def test_list_artifacts_with_subdir(oss_bucket_mock):
    import oss2.models
    artifact_root_path = "experiment_id/run_id/"
    repo = AliyunArtifactRepository("oss://test_bucket/" + artifact_root_path, oss_bucket_mock)
    repo._get_oss_bucket = repo.oss_bucket
    MockSimplifiedObjectInfo = mock.MagicMock(autospec=oss2.models.SimplifiedObjectInfo)
    # list artifacts at sub directory level
    dir_name = "model"
    file_path = dir_name + "/" + 'model.pb'
    obj_mock = oss2.models.SimplifiedObjectInfo(key=artifact_root_path + file_path, last_modified='123', size=1, etag=None, type=None, storage_class=None)

    subdir_name = dir_name + "/" + 'variables'
    subdir_mock = oss2.models.SimplifiedObjectInfo(key=artifact_root_path + subdir_name + "/", last_modified=None, size=None, etag=None, type=None, storage_class=None)

    mock_results = mock.MagicMock(autospec=oss2.models.ListObjectsResult)
    mock_results.object_list = [obj_mock, subdir_mock]
    repo.oss_bucket.list_objects.return_value = mock_results

    artifacts = repo.list_artifacts(path=dir_name)
    assert len(artifacts) == 2
    assert artifacts[0].path == file_path
    assert artifacts[0].is_dir is False
    assert artifacts[0].file_size == obj_mock.size
    assert artifacts[1].path == subdir_name
    assert artifacts[1].is_dir is True
    assert artifacts[1].file_size is None

def test_download_file_artifact(oss_bucket_mock, tmpdir):
    repo = AliyunArtifactRepository("oss://test_bucket/some/path", oss_bucket_mock)

    def mkfile(fname, temp=''):
        fname = os.path.basename(fname)
        f = tmpdir.join(fname)
        f.write("hello world!")
    repo.oss_bucket.get_object_to_file.side_effect = mkfile

    local_file_path = repo.download_artifacts("test.txt")

    assert os.path.exists(os.path.join(tmpdir.strpath, "test.txt"))
    repo.oss_bucket.get_object_to_file.assert_called_with('some/path/test.txt', local_file_path)
    download_calls = repo.oss_bucket.get_object_to_file.call_args_list
    assert len(download_calls) == 1
    download_path_arg = download_calls[0][0][0]
    assert "test.txt" in download_path_arg
