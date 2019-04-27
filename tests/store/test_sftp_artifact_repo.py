from mock import MagicMock
import pytest
from tempfile import NamedTemporaryFile
import pysftp
from mlflow.store.artifact_repository_registry import get_artifact_repository
from mlflow.store.sftp_artifact_repo import SFTPArtifactRepository
from mlflow.utils.file_utils import TempDir
import os


@pytest.fixture
def sftp_mock():
    return MagicMock(autospec=pysftp.Connection)


def test_artifact_uri_factory():
    from paramiko.ssh_exception import SSHException
    with pytest.raises(SSHException):
        get_artifact_repository("sftp://user:pass@test_sftp:123/some/path")


def test_list_artifacts_empty(sftp_mock):
    repo = SFTPArtifactRepository("sftp://test_sftp/some/path", sftp_mock)
    sftp_mock.listdir = MagicMock(return_value=[])
    assert repo.list_artifacts() == []
    sftp_mock.listdir.assert_called_once_with("/some/path")


def test_list_artifacts(sftp_mock):
    artifact_root_path = "/experiment_id/run_id/"
    repo = SFTPArtifactRepository("sftp://test_sftp"+artifact_root_path, sftp_mock)

    # mocked file structure
    #  |- file
    #  |- model
    #     |- model.pb

    file_path = "file"
    file_size = 678
    dir_path = "model"
    sftp_mock.isdir = MagicMock(side_effect=lambda path: {
            artifact_root_path: True,
            os.path.join(artifact_root_path, file_path): False,
            os.path.join(artifact_root_path, dir_path): True,
        }[path])
    sftp_mock.listdir = MagicMock(return_value=[file_path, dir_path])

    file_stat = MagicMock()
    file_stat.configure_mock(st_size=file_size)
    sftp_mock.stat = MagicMock(return_value=file_stat)

    artifacts = repo.list_artifacts(path=None)

    sftp_mock.listdir.assert_called_once_with(artifact_root_path)
    sftp_mock.stat.assert_called_once_with(artifact_root_path + file_path)

    assert len(artifacts) == 2
    assert artifacts[0].path == file_path
    assert artifacts[0].is_dir is False
    assert artifacts[0].file_size == file_size
    assert artifacts[1].path == dir_path
    assert artifacts[1].is_dir is True
    assert artifacts[1].file_size is None


def test_list_artifacts_with_subdir(sftp_mock):
    artifact_root_path = "/experiment_id/run_id/"
    repo = SFTPArtifactRepository("sftp://test_sftp"+artifact_root_path, sftp_mock)

    # mocked file structure
    #  |- model
    #     |- model.pb
    #     |- variables
    dir_name = 'model'

    # list artifacts at sub directory level
    file_path = 'model.pb'
    file_size = 345
    subdir_name = 'variables'

    sftp_mock.listdir = MagicMock(return_value=[file_path, subdir_name])

    sftp_mock.isdir = MagicMock(side_effect=lambda path: {
            os.path.join(artifact_root_path, dir_name): True,
            os.path.join(artifact_root_path, dir_name, file_path): False,
            os.path.join(artifact_root_path, dir_name, subdir_name): True,
        }[path])

    file_stat = MagicMock()
    file_stat.configure_mock(st_size=file_size)
    sftp_mock.stat = MagicMock(return_value=file_stat)

    artifacts = repo.list_artifacts(path=dir_name)

    sftp_mock.listdir.assert_called_once_with(artifact_root_path + dir_name)
    sftp_mock.stat.assert_called_once_with(artifact_root_path + dir_name + '/' + file_path)

    assert len(artifacts) == 2
    assert artifacts[0].path == os.path.join(dir_name, file_path)
    assert artifacts[0].is_dir is False
    assert artifacts[0].file_size == file_size
    assert artifacts[1].path == os.path.join(dir_name, subdir_name)
    assert artifacts[1].is_dir is True
    assert artifacts[1].file_size is None


@pytest.mark.requires_ssh
def test_log_artifact():
    for artifact_path in [None, "sub_dir", "very/nested/sub/dir"]:
        file_content = 'A simple test artifact\nThe artifact is located in: ' + str(artifact_path)
        with NamedTemporaryFile(mode="w") as local, TempDir() as remote:
            local.write(file_content)
            local.flush()

            sftp_path = "sftp://" + remote.path()
            store = SFTPArtifactRepository(sftp_path)
            store.log_artifact(local.name, artifact_path)

            remote_file = os.path.join(
                remote.path(),
                '.' if artifact_path is None else artifact_path,
                os.path.basename(local.name))
            assert os.path.isfile(remote_file)

            with open(remote_file, 'r') as remote_content:
                assert remote_content.read() == file_content


@pytest.mark.requires_ssh
def test_log_artifacts():
    for artifact_path in [None, "sub_dir", "very/nested/sub/dir"]:
        file_content_1 = 'A simple test artifact\nThe artifact is located in: ' + str(artifact_path)
        file_content_2 = os.urandom(300)

        file1 = "meta.yaml"
        directory = "saved_model"
        file2 = "sk_model.pickle"
        with TempDir() as local, TempDir() as remote:
            with open(os.path.join(local.path(), file1), "w") as f:
                f.write(file_content_1)
            os.mkdir(os.path.join(local.path(), directory))
            with open(os.path.join(local.path(), directory, file2), "wb") as f:
                f.write(file_content_2)

            sftp_path = "sftp://" + remote.path()
            store = SFTPArtifactRepository(sftp_path)
            store.log_artifacts(local.path(), artifact_path)

            remote_dir = os.path.join(
                remote.path(),
                '.' if artifact_path is None else artifact_path)
            assert os.path.isdir(remote_dir)
            assert os.path.isdir(os.path.join(remote_dir, directory))
            assert os.path.isfile(os.path.join(remote_dir, file1))
            assert os.path.isfile(os.path.join(remote_dir, directory, file2))

            with open(os.path.join(remote_dir, file1), 'r') as remote_content:
                assert remote_content.read() == file_content_1

            with open(os.path.join(remote_dir, directory, file2), 'rb') as remote_content:
                assert remote_content.read() == file_content_2
