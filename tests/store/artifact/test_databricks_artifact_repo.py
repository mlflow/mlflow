# -*- coding: utf-8 -*-
import os

import pytest
import mock
from unittest.mock import ANY

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.dbfs_artifact_repo import DatabricksArtifactRepository
from mlflow.protos.service_pb2 import ListArtifacts, FileInfo
from mlflow.protos.databricks_artifacts_pb2 import GetCredentialsForWrite, GetCredentialsForRead, \
    ArtifactCredentialType, ArtifactCredentialInfo


@pytest.fixture()
def databricks_artifact_repo():
    return get_artifact_repository('dbfs:/databricks/mlflow-tracking/MOCK-EXP/MOCK-RUN/artifact')


DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE = 'mlflow.store.artifact.databricks_artifact_repo'
DATABRICKS_ARTIFACT_REPOSITORY = DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE + ".DatabricksArtifactRepository"

TEST_FILE_1_CONTENT = u"Hello 🍆🍔".encode("utf-8")
TEST_FILE_2_CONTENT = u"World 🍆🍔🍆".encode("utf-8")
TEST_FILE_3_CONTENT = u"¡🍆🍆🍔🍆🍆!".encode("utf-8")


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
    with open(tmpdir.join('empty-file').strpath, 'w'):
        pass
    return tmpdir


LIST_ARTIFACTS_PROTO_RESPONSE = [FileInfo(path='test/a.txt', is_dir=False, file_size=100),
                                 FileInfo(path='test/dir', is_dir=True, file_size=0)]

LIST_ARTIFACTS_SINGLE_FILE_PROTO_RESPONSE = [FileInfo(path='a.txt', is_dir=False, file_size=0)]
MOCK_AZURE_SIGNED_URI = "this_is_a_mock_sas_for_azure"
MOCK_RUN_ID = 'MOCK-RUN'


class TestDatabricksArtifactRepository(object):
    def test_init_validation_and_cleaning(self):
        repo = get_artifact_repository('dbfs:/databricks/mlflow-tracking/EXP/RUN/artifact')
        assert repo.artifact_uri == 'dbfs:/databricks/mlflow-tracking/EXP/RUN/artifact'
        with pytest.raises(MlflowException):
            DatabricksArtifactRepository('s3://test')
        with pytest.raises(MlflowException):
            DatabricksArtifactRepository('dbfs:/databricks/mlflow/EXP/RUN/artifact')

    @pytest.mark.parametrize("artifact_path,expected_location", [
        (None, 'test.txt'),
        ('output', 'output/test.txt'),
        ('', 'test.txt'),
    ])
    def test_log_artifact(self, databricks_artifact_repo, test_file, artifact_path, expected_location):
        with mock.patch(DATABRICKS_ARTIFACT_REPOSITORY + '._get_write_credentials') as write_credentials_mock, \
                mock.patch(DATABRICKS_ARTIFACT_REPOSITORY + '._azure_upload_file') as azure_upload_mock:
            mock_credentials = ArtifactCredentialInfo(signed_uri=MOCK_AZURE_SIGNED_URI,
                                                      type=ArtifactCredentialType.AZURE_SAS_URI)
            write_credentials_response_proto = GetCredentialsForWrite.Response(
                credentials=mock_credentials)
            write_credentials_mock.return_value = write_credentials_response_proto
            azure_upload_mock.return_value = None
            databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
            write_credentials_mock.assert_called_with(MOCK_RUN_ID, expected_location)
            azure_upload_mock.assert_called_with(mock_credentials, test_file.strpath)

    @pytest.mark.parametrize("artifact_path", [
        None,
        'output/',
        '',
    ])
    def test_log_artifacts(self, databricks_artifact_repo, test_dir, artifact_path):
        with mock.patch(DATABRICKS_ARTIFACT_REPOSITORY + '.log_artifact') as log_artifact_mock:
            log_artifact_mock.return_value = None
            databricks_artifact_repo.log_artifacts(test_dir.strpath, artifact_path)
            artifact_path = artifact_path or ''
            calls = [mock.call(os.path.join(test_dir.strpath, 'empty-file'), os.path.join(artifact_path, '')),
                     mock.call(os.path.join(test_dir.strpath, 'test.txt'), os.path.join(artifact_path, '')),
                     mock.call(os.path.join(test_dir.strpath, 'subdir/test.txt'),
                               os.path.join(artifact_path, 'subdir'))]
            log_artifact_mock.assert_has_calls(calls)

    def test_list_artifacts(self, databricks_artifact_repo):
        with mock.patch(DATABRICKS_ARTIFACT_REPOSITORY + '._call_endpoint') as call_endpoint_mock:
            list_artifact_response_proto = ListArtifacts.Response(root_uri='', files=LIST_ARTIFACTS_PROTO_RESPONSE)
            call_endpoint_mock.return_value = list_artifact_response_proto
            artifacts = databricks_artifact_repo.list_artifacts('test/')
            assert len(artifacts) == 2
            assert artifacts[0].path == 'test/a.txt'
            assert artifacts[0].is_dir is False
            assert artifacts[0].file_size == 100
            assert artifacts[1].path == 'test/dir'
            assert artifacts[1].is_dir is True
            assert artifacts[1].file_size is 0

            # Calling list_artifacts() on a path that's a file should return an empty list
            list_artifact_response_proto = ListArtifacts.Response(root_uri='',
                                                                  files=LIST_ARTIFACTS_SINGLE_FILE_PROTO_RESPONSE)
            call_endpoint_mock.return_value = list_artifact_response_proto
            artifacts = databricks_artifact_repo.list_artifacts('a.txt')
            assert len(artifacts) == 0

    @pytest.mark.parametrize("remote_file_path, local_path", [
        ('test_file.txt', ''),
        ('test_file.txt', None),
        ('output/test_file', None),
    ])
    def test_databricks_download_file(self, databricks_artifact_repo, remote_file_path, local_path):
        with mock.patch(DATABRICKS_ARTIFACT_REPOSITORY + '._get_read_credentials') as read_credentials_mock, \
                mock.patch(DATABRICKS_ARTIFACT_REPOSITORY + '.list_artifacts') as get_list_mock, \
                mock.patch(DATABRICKS_ARTIFACT_REPOSITORY + '._azure_download_file') as azure_download_mock:
            mock_credentials = ArtifactCredentialInfo(signed_uri=MOCK_AZURE_SIGNED_URI,
                                                      type=ArtifactCredentialType.AZURE_SAS_URI)
            read_credentials_response_proto = GetCredentialsForRead.Response(
                credentials=mock_credentials)
            read_credentials_mock.return_value = read_credentials_response_proto
            azure_download_mock.return_value = None
            get_list_mock.return_value = []
            databricks_artifact_repo.download_artifacts(remote_file_path, local_path)
            read_credentials_mock.assert_called_with(MOCK_RUN_ID, remote_file_path)
            azure_download_mock.assert_called_with(mock_credentials, ANY)
