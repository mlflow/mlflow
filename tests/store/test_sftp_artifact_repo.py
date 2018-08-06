from mock import MagicMock
import unittest

import pysftp

from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.store.sftp_artifact_repo import SFTPArtifactRepository


class SFTPArtifactRepositryTest(unittest.TestCase):
    def setUp(self):
        self.sftp = MagicMock(autospec=pysftp.Connection)

    def test_artifact_uri_factory(self):
        from paramiko.ssh_exception import SSHException
        self.assertRaises(
            SSHException,
            lambda: ArtifactRepository.from_artifact_uri(
                "sftp://user:pass@test_sftp:123/some/path"))

    def test_list_artifacts_empty(self):
        repo = SFTPArtifactRepository("sftp://test_sftp/some/path", self.sftp)
        self.sftp.listdir = MagicMock(return_value=[])
        assert repo.list_artifacts() == []
        self.sftp.listdir.assert_called_once_with("/some/path")

    def test_list_artifacts(self):
        artifact_root_path = "/experiment_id/run_id/"
        repo = SFTPArtifactRepository("sftp://test_sftp"+artifact_root_path, self.sftp)

        # mocked file structure
        #  |- file
        #  |- model
        #     |- model.pb

        file_path = "file"
        file_size = 678
        dir_path = "model"
        self.sftp.isdir = MagicMock(side_effect=lambda path: {
                (artifact_root_path+file_path): False,
                (artifact_root_path+dir_path): True
            }[path])
        self.sftp.listdir = MagicMock(return_value=[file_path, dir_path])

        file_stat = MagicMock()
        file_stat.configure_mock(st_size=file_size)
        self.sftp.stat = MagicMock(return_value=file_stat)

        artifacts = repo.list_artifacts(path=None)

        self.sftp.listdir.assert_called_once_with(artifact_root_path)
        self.sftp.stat.assert_called_once_with(artifact_root_path + file_path)

        assert len(artifacts) == 2
        assert artifacts[0].path == file_path
        assert artifacts[0].is_dir is False
        assert artifacts[0].file_size == file_size
        assert artifacts[1].path == dir_path
        assert artifacts[1].is_dir is True
        assert artifacts[1].file_size is None

    def test_list_artifacts_with_subdir(self):
        artifact_root_path = "/experiment_id/run_id/"
        repo = SFTPArtifactRepository("sftp://test_sftp"+artifact_root_path, self.sftp)

        # mocked file structure
        #  |- model
        #     |- model.pb
        #     |- variables
        dir_name = 'model'

        # list artifacts at sub directory level
        file_path = 'model.pb'
        file_size = 345
        subdir_name = 'variables'

        self.sftp.listdir = MagicMock(return_value=[file_path, subdir_name])

        self.sftp.isdir = MagicMock(side_effect=lambda path: {
                (artifact_root_path+dir_name+'/'+file_path): False,
                (artifact_root_path+dir_name+'/'+subdir_name): True
            }[path])

        file_stat = MagicMock()
        file_stat.configure_mock(st_size=file_size)
        self.sftp.stat = MagicMock(return_value=file_stat)

        artifacts = repo.list_artifacts(path=dir_name)

        self.sftp.listdir.assert_called_once_with(artifact_root_path + dir_name)
        self.sftp.stat.assert_called_once_with(artifact_root_path + dir_name + '/' + file_path)

        assert len(artifacts) == 2
        assert artifacts[0].path == dir_name + '/' + file_path
        assert artifacts[0].is_dir is False
        assert artifacts[0].file_size == file_size
        assert artifacts[1].path == dir_name + '/' + subdir_name
        assert artifacts[1].is_dir is True
        assert artifacts[1].file_size is None
