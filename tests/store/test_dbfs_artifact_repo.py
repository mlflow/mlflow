# -*- coding: utf-8 -*-
import json

import pytest
import mock
from mock import Mock

from mlflow.exceptions import IllegalArtifactPathError, MlflowException
from mlflow.store.dbfs_artifact_repo import DbfsArtifactRepository


@pytest.fixture()
def dbfs_artifact_repo():
    return DbfsArtifactRepository('dbfs:/test/', {})

TEST_FILE_1_CONTENT = bytes("Hello 🍆🍔")
TEST_FILE_2_CONTENT = bytes("World 🍆🍔🍆")
TEST_FILE_3_CONTENT = bytes("¡🍆🍆🍔🍆🍆!")


@pytest.fixture()
def test_file(tmpdir):
    p = tmpdir.join("test.txt")
    with open(p.strpath, 'wb') as f:
        f.write(TEST_FILE_1_CONTENT)
    return p


@pytest.fixture()
def test_dir(tmpdir):
    with open(tmpdir.mkdir('subdir').join('test.txt').strpath, 'wb') as f:
        f.write(TEST_FILE_2_CONTENT)
    with open(tmpdir.join('test.txt').strpath, 'wb') as f:
        f.write(bytes(TEST_FILE_3_CONTENT))
    return tmpdir


LIST_ARTIFACTS_RESPONSE = {
    'files': [{
        'path': '/test/a.txt',
        'is_dir': False,
        'file_size': 100,
    }, {
        'path': '/test/dir',
        'is_dir': True,
        'file_size': 0,
    }]
}


class TestDbfsArtifactRepository(object):
    def test_init_validation_and_cleaning(self):
        repo = DbfsArtifactRepository('dbfs:/test/', {})
        assert repo.artifact_uri == 'dbfs:/test'
        with pytest.raises(MlflowException):
            DbfsArtifactRepository('s3://test', {})

    @pytest.mark.parametrize("artifact_path,expected_endpoint", [
        (None, '/dbfs/test/test.txt'),
        ('output', '/dbfs/test/output/test.txt'),
    ])
    def test_log_artifact(self, dbfs_artifact_repo, test_file, artifact_path, expected_endpoint):
        with mock.patch('mlflow.store.dbfs_artifact_repo.http_request') as http_request_mock:
            endpoints = []
            data = []

            def my_http_request(**kwargs):
                endpoints.append(kwargs['endpoint'])
                data.append(kwargs['data'].read())
            http_request_mock.side_effect = my_http_request
            dbfs_artifact_repo.log_artifact(test_file.strpath, artifact_path)
            assert endpoints == [expected_endpoint]
            assert data == [TEST_FILE_1_CONTENT]

    def test_log_artifact_empty(self, dbfs_artifact_repo, test_file):
        with pytest.raises(IllegalArtifactPathError):
            dbfs_artifact_repo.log_artifact(test_file.strpath, '')

    @pytest.mark.parametrize("artifact_path", [
        None,
        '',  # should behave like '/' and exclude base name of logged_dir
        # We should add '.',
    ])
    def test_log_artifacts(self, dbfs_artifact_repo, test_dir, artifact_path):
        with mock.patch('mlflow.store.dbfs_artifact_repo.http_request') as http_request_mock:
            endpoints = []
            data = []

            def my_http_request(**kwargs):
                endpoints.append(kwargs['endpoint'])
                data.append(kwargs['data'].read())
            http_request_mock.side_effect = my_http_request
            dbfs_artifact_repo.log_artifacts(test_dir.strpath, artifact_path)
            basename = test_dir.basename
            assert set(endpoints) == {
                '/dbfs/test/%s/subdir/test.txt' % basename,
                '/dbfs/test/%s/test.txt' % basename
            }
            assert set(data) == {
                TEST_FILE_2_CONTENT,
                TEST_FILE_3_CONTENT,
            }

    @pytest.mark.parametrize("artifact_path,expected_endpoints", [
        ('a', {'/dbfs/test/a/subdir/test.txt', '/dbfs/test/a/test.txt'}),
        ('a/', {'/dbfs/test/a/subdir/test.txt', '/dbfs/test/a/test.txt'}),
        ('/', {'/dbfs/test/subdir/test.txt', '/dbfs/test/test.txt'}),
    ])
    def test_log_artifacts_with_artifact_path(self, dbfs_artifact_repo, test_dir, artifact_path,
                                              expected_endpoints):
        with mock.patch('mlflow.store.dbfs_artifact_repo.http_request') as http_request_mock:
            endpoints = []

            def my_http_request(**kwargs):
                endpoints.append(kwargs['endpoint'])
            http_request_mock.side_effect = my_http_request
            dbfs_artifact_repo.log_artifacts(test_dir.strpath, artifact_path)
            assert set(endpoints) == expected_endpoints

    def test_list_artifacts(self, dbfs_artifact_repo):
        with mock.patch('mlflow.store.dbfs_artifact_repo.http_request') as http_request_mock:
            http_request_mock.return_value.text = json.dumps(LIST_ARTIFACTS_RESPONSE)
            artifacts = dbfs_artifact_repo.list_artifacts()
            assert len(artifacts) == 2
            assert artifacts[0].path == 'a.txt'
            assert artifacts[0].is_dir is False
            assert artifacts[0].file_size == 100
            assert artifacts[1].path == 'dir'
            assert artifacts[1].is_dir is True
            assert artifacts[1].file_size is None

    def test_download_artifacts(self, dbfs_artifact_repo):
        with mock.patch('mlflow.store.dbfs_artifact_repo._dbfs_is_dir') as is_dir_mock,\
                mock.patch('mlflow.store.dbfs_artifact_repo._dbfs_list_api') as list_mock, \
                mock.patch('mlflow.store.dbfs_artifact_repo._dbfs_download') as download_mock:
            is_dir_mock.side_effect = [
                True,
                False,
                True,
            ]
            list_mock.side_effect = [
                Mock(text=json.dumps(LIST_ARTIFACTS_RESPONSE)),
                Mock(text='{}')  # this call is for listing `/dir`.
            ]
            dbfs_artifact_repo.download_artifacts('/')
            assert list_mock.call_count == 2
            assert download_mock.call_count == 1
            _, kwargs = download_mock.call_args
            assert kwargs['endpoint'] == '/dbfs/test/a.txt'
