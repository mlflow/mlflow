# -*- coding: utf-8 -*-
import json

import pytest
import mock
from mock import Mock

from mlflow.exceptions import IllegalArtifactPathError, MlflowException
from mlflow.store.dbfs_artifact_repo import DbfsArtifactRepository
from mlflow.utils.rest_utils import MlflowHostCreds


@pytest.fixture()
def dbfs_artifact_repo():
    return DbfsArtifactRepository('dbfs:/test/', lambda: MlflowHostCreds('http://host'))


TEST_FILE_1_CONTENT = u"Hello 🍆🍔".encode("utf-8")
TEST_FILE_2_CONTENT = u"World 🍆🍔🍆".encode("utf-8")
TEST_FILE_3_CONTENT = u"¡🍆🍆🍔🍆🍆!".encode("utf-8")

DBFS_ARTIFACT_REPOSITORY_PACKAGE = 'mlflow.store.dbfs_artifact_repo.DbfsArtifactRepository'


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

LIST_ARTIFACTS_SINGLE_FILE_RESPONSE = {
    'files': [{
        'path': '/test/a.txt',
        'is_dir': False,
        'file_size': 0,
    }]
}


class TestDbfsArtifactRepository(object):
    def test_init_validation_and_cleaning(self):
        repo = DbfsArtifactRepository('dbfs:/test/', lambda: MlflowHostCreds('http://host'))
        assert repo.artifact_uri == 'dbfs:/test'
        with pytest.raises(MlflowException):
            DbfsArtifactRepository('s3://test', lambda: MlflowHostCreds('http://host'))

    @pytest.mark.parametrize("artifact_path,expected_endpoint", [
        (None, '/dbfs/test/test.txt'),
        ('output', '/dbfs/test/output/test.txt'),
    ])
    def test_log_artifact(self, dbfs_artifact_repo, test_file, artifact_path, expected_endpoint):
        with mock.patch('mlflow.utils.rest_utils.http_request') as http_request_mock:
            endpoints = []
            data = []

            def my_http_request(host_creds, **kwargs):  # pylint: disable=unused-argument
                endpoints.append(kwargs['endpoint'])
                data.append(kwargs['data'].read())
                return Mock(status_code=200)
            http_request_mock.side_effect = my_http_request
            dbfs_artifact_repo.log_artifact(test_file.strpath, artifact_path)
            assert endpoints == [expected_endpoint]
            assert data == [TEST_FILE_1_CONTENT]

    def test_log_artifact_empty(self, dbfs_artifact_repo, test_file):
        with pytest.raises(IllegalArtifactPathError):
            dbfs_artifact_repo.log_artifact(test_file.strpath, '')

    def test_log_artifact_error(self, dbfs_artifact_repo, test_file):
        with mock.patch('mlflow.utils.rest_utils.http_request') as http_request_mock:
            http_request_mock.return_value = Mock(status_code=409, text='')
            with pytest.raises(MlflowException):
                dbfs_artifact_repo.log_artifact(test_file.strpath)

    @pytest.mark.parametrize("artifact_path", [
        None,
        '',  # should behave like '/' and exclude base name of logged_dir
        # We should add '.',
    ])
    def test_log_artifacts(self, dbfs_artifact_repo, test_dir, artifact_path):
        with mock.patch('mlflow.utils.rest_utils.http_request') as http_request_mock:
            endpoints = []
            data = []

            def my_http_request(host_creds, **kwargs):  # pylint: disable=unused-argument
                endpoints.append(kwargs['endpoint'])
                data.append(kwargs['data'].read())
                return Mock(status_code=200)
            http_request_mock.side_effect = my_http_request
            dbfs_artifact_repo.log_artifacts(test_dir.strpath, artifact_path)
            assert set(endpoints) == {
                '/dbfs/test/subdir/test.txt',
                '/dbfs/test/test.txt'
            }
            assert set(data) == {
                TEST_FILE_2_CONTENT,
                TEST_FILE_3_CONTENT,
            }

    def test_log_artifacts_error(self, dbfs_artifact_repo, test_dir):
        with mock.patch('mlflow.utils.rest_utils.http_request') as http_request_mock:
            http_request_mock.return_value = Mock(status_code=409, text='')
            with pytest.raises(MlflowException):
                dbfs_artifact_repo.log_artifacts(test_dir.strpath)

    @pytest.mark.parametrize("artifact_path,expected_endpoints", [
        ('a', {'/dbfs/test/a/subdir/test.txt', '/dbfs/test/a/test.txt'}),
        ('a/', {'/dbfs/test/a/subdir/test.txt', '/dbfs/test/a/test.txt'}),
        ('/', {'/dbfs/test/subdir/test.txt', '/dbfs/test/test.txt'}),
    ])
    def test_log_artifacts_with_artifact_path(self, dbfs_artifact_repo, test_dir, artifact_path,
                                              expected_endpoints):
        with mock.patch('mlflow.utils.rest_utils.http_request') as http_request_mock:
            endpoints = []

            def my_http_request(host_creds, **kwargs):  # pylint: disable=unused-argument
                endpoints.append(kwargs['endpoint'])
                return Mock(status_code=200)
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
            # Calling list_artifacts() on a path that's a file should return an empty list
            http_request_mock.return_value.text = json.dumps(LIST_ARTIFACTS_SINGLE_FILE_RESPONSE)
            list_on_file = dbfs_artifact_repo.list_artifacts("a.txt")
            assert len(list_on_file) == 0

    def test_download_artifacts(self, dbfs_artifact_repo):
        with mock.patch(DBFS_ARTIFACT_REPOSITORY_PACKAGE + '._dbfs_is_dir') as is_dir_mock,\
                mock.patch(DBFS_ARTIFACT_REPOSITORY_PACKAGE + '._dbfs_list_api') as list_mock, \
                mock.patch(DBFS_ARTIFACT_REPOSITORY_PACKAGE + '._dbfs_download') as download_mock:
            is_dir_mock.side_effect = [
                True,
                False,
                True,
            ]
            list_mock.side_effect = [
                Mock(text=json.dumps(LIST_ARTIFACTS_RESPONSE)),
                Mock(text='{}'),  # this call is for listing `/dir`.
                Mock(text='{}')   # this call is for listing `/dir/a.txt`.
            ]
            dbfs_artifact_repo.download_artifacts('/')
            assert list_mock.call_count == 3
            assert download_mock.call_count == 2
            chronological_download_calls = list(download_mock.call_args_list)
            # Calls are in reverse chronological order by default
            chronological_download_calls.reverse()
            _, kwargs_call_1 = chronological_download_calls[0]
            _, kwargs_call_2 = chronological_download_calls[1]
            assert kwargs_call_1['endpoint'] == '/dbfs/test/dir'
            assert kwargs_call_2['endpoint'] == '/dbfs/test/a.txt'
