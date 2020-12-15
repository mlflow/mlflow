# pylint: disable=redefined-outer-name
from unittest.mock import MagicMock
import pytest
import posixpath
import ftplib
from ftplib import FTP

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.ftp_artifact_repo import FTPArtifactRepository


@pytest.fixture
def ftp_mock():
    return MagicMock(autospec=FTP)


def test_artifact_uri_factory():
    repo = get_artifact_repository("ftp://user:pass@test_ftp:123/some/path")
    assert isinstance(repo, FTPArtifactRepository)


def test_list_artifacts_empty(ftp_mock):
    repo = FTPArtifactRepository("ftp://test_ftp/some/path")

    repo.get_ftp_client = MagicMock()
    call_mock = MagicMock(return_value=ftp_mock)
    repo.get_ftp_client.return_value = MagicMock(__enter__=call_mock)

    ftp_mock.nlst = MagicMock(return_value=[])
    assert repo.list_artifacts() == []
    ftp_mock.nlst.assert_called_once_with("/some/path")


def test_list_artifacts(ftp_mock):
    artifact_root_path = "/experiment_id/run_id/"
    repo = FTPArtifactRepository("ftp://test_ftp" + artifact_root_path)

    repo.get_ftp_client = MagicMock()
    call_mock = MagicMock(return_value=ftp_mock)
    repo.get_ftp_client.return_value = MagicMock(__enter__=call_mock)

    # mocked file structure
    #  |- file
    #  |- model
    #     |- model.pb

    file_path = "file"
    file_size = 678
    dir_path = "model"
    ftp_mock.cwd = MagicMock(side_effect=[None, ftplib.error_perm, None])
    ftp_mock.nlst = MagicMock(return_value=[file_path, dir_path])

    ftp_mock.size = MagicMock(return_value=file_size)

    artifacts = repo.list_artifacts(path=None)

    ftp_mock.nlst.assert_called_once_with(artifact_root_path)
    ftp_mock.size.assert_called_once_with(artifact_root_path + file_path)

    assert len(artifacts) == 2
    assert artifacts[0].path == file_path
    assert artifacts[0].is_dir is False
    assert artifacts[0].file_size == file_size
    assert artifacts[1].path == dir_path
    assert artifacts[1].is_dir is True
    assert artifacts[1].file_size is None


def test_list_artifacts_when_ftp_nlst_returns_absolute_paths(ftp_mock):
    artifact_root_path = "/experiment_id/run_id/"
    repo = FTPArtifactRepository("ftp://test_ftp" + artifact_root_path)

    repo.get_ftp_client = MagicMock()
    call_mock = MagicMock(return_value=ftp_mock)
    repo.get_ftp_client.return_value = MagicMock(__enter__=call_mock)

    # mocked file structure
    #  |- file
    #  |- model
    #     |- model.pb

    file_path = "file"
    dir_path = "model"
    file_size = 678
    ftp_mock.cwd = MagicMock(side_effect=[None, ftplib.error_perm, None])
    ftp_mock.nlst = MagicMock(
        return_value=[
            posixpath.join(artifact_root_path, file_path),
            posixpath.join(artifact_root_path, dir_path),
        ]
    )

    ftp_mock.size = MagicMock(return_value=file_size)

    artifacts = repo.list_artifacts(path=None)

    ftp_mock.nlst.assert_called_once_with(artifact_root_path)
    ftp_mock.size.assert_called_once_with(artifact_root_path + file_path)

    assert len(artifacts) == 2
    assert artifacts[0].path == file_path
    assert artifacts[0].is_dir is False
    assert artifacts[0].file_size == file_size
    assert artifacts[1].path == dir_path
    assert artifacts[1].is_dir is True
    assert artifacts[1].file_size is None


def test_list_artifacts_with_subdir(ftp_mock):
    artifact_root_path = "/experiment_id/run_id/"
    repo = FTPArtifactRepository("sftp://test_sftp" + artifact_root_path)

    repo.get_ftp_client = MagicMock()
    call_mock = MagicMock(return_value=ftp_mock)
    repo.get_ftp_client.return_value = MagicMock(__enter__=call_mock)

    # mocked file structure
    #  |- model
    #     |- model.pb
    #     |- variables
    dir_name = "model"

    # list artifacts at sub directory level
    file_path = "model.pb"
    file_size = 345
    subdir_name = "variables"

    ftp_mock.nlst = MagicMock(return_value=[file_path, subdir_name])

    ftp_mock.cwd = MagicMock(side_effect=[None, ftplib.error_perm, None])

    ftp_mock.size = MagicMock(return_value=file_size)

    artifacts = repo.list_artifacts(path=dir_name)

    ftp_mock.nlst.assert_called_once_with(artifact_root_path + dir_name)
    ftp_mock.size.assert_called_once_with(artifact_root_path + dir_name + "/" + file_path)

    assert len(artifacts) == 2
    assert artifacts[0].path == dir_name + "/" + file_path
    assert artifacts[0].is_dir is False
    assert artifacts[0].file_size == file_size
    assert artifacts[1].path == dir_name + "/" + subdir_name
    assert artifacts[1].is_dir is True
    assert artifacts[1].file_size is None


def test_log_artifact(ftp_mock, tmpdir):
    repo = FTPArtifactRepository("ftp://test_ftp/some/path")

    repo.get_ftp_client = MagicMock()
    call_mock = MagicMock(return_value=ftp_mock)
    repo.get_ftp_client.return_value = MagicMock(__enter__=call_mock)

    d = tmpdir.mkdir("data")
    f = d.join("test.txt")
    f.write("hello world!")
    fpath = d + "/test.txt"
    fpath = fpath.strpath

    ftp_mock.cwd = MagicMock(side_effect=[ftplib.error_perm, None])

    repo.log_artifact(fpath)

    ftp_mock.mkd.assert_called_once_with("/some/path")
    ftp_mock.cwd.assert_called_with("/some/path")
    ftp_mock.storbinary.assert_called_once()
    assert ftp_mock.storbinary.call_args_list[0][0][0] == "STOR test.txt"


def test_log_artifact_multiple_calls(ftp_mock, tmpdir):
    repo = FTPArtifactRepository("ftp://test_ftp/some/path")

    repo.get_ftp_client = MagicMock()
    call_mock = MagicMock(return_value=ftp_mock)
    repo.get_ftp_client.return_value = MagicMock(__enter__=call_mock)

    d = tmpdir.mkdir("data")
    file1 = d.join("test1.txt")
    file1.write("hello world!")
    fpath1 = d + "/test1.txt"
    fpath1 = fpath1.strpath

    file2 = d.join("test2.txt")
    file2.write("hello world!")
    fpath2 = d + "/test2.txt"
    fpath2 = fpath2.strpath

    ftp_mock.cwd = MagicMock(
        side_effect=[ftplib.error_perm, None, ftplib.error_perm, None, None, None]
    )

    repo.log_artifact(fpath1)
    ftp_mock.mkd.assert_called_once_with("/some/path")
    ftp_mock.cwd.assert_called_with("/some/path")
    ftp_mock.storbinary.assert_called()
    assert ftp_mock.storbinary.call_args_list[0][0][0] == "STOR test1.txt"
    ftp_mock.reset_mock()

    repo.log_artifact(fpath1, "subdir")
    ftp_mock.mkd.assert_called_once_with("/some/path/subdir")
    ftp_mock.cwd.assert_called_with("/some/path/subdir")
    ftp_mock.storbinary.assert_called()
    assert ftp_mock.storbinary.call_args_list[0][0][0] == "STOR test1.txt"
    ftp_mock.reset_mock()

    repo.log_artifact(fpath2)
    ftp_mock.mkd.assert_not_called()
    ftp_mock.cwd.assert_called_with("/some/path")
    ftp_mock.storbinary.assert_called()
    assert ftp_mock.storbinary.call_args_list[0][0][0] == "STOR test2.txt"


def __posixpath_parents(pathname, root):
    parents = [posixpath.dirname(pathname)]
    root = posixpath.normpath(root)
    while parents[-1] != "/" and parents[-1] != root:
        parents.append(posixpath.dirname(parents[-1]))
    return parents


@pytest.mark.parametrize("artifact_path", [None, "dir", "dir1/dir2"])
def test_log_artifacts(artifact_path, ftp_mock, tmpdir):
    # Setup FTP mock.
    dest_path_root = "/some/path"
    repo = FTPArtifactRepository("ftp://test_ftp" + dest_path_root)

    repo.get_ftp_client = MagicMock()
    call_mock = MagicMock(return_value=ftp_mock)
    repo.get_ftp_client.return_value = MagicMock(__enter__=call_mock)

    dirs_created = set([dest_path_root])
    files_created = set()
    cwd_history = ["/"]

    def mkd_mock(pathname):
        abs_pathname = posixpath.join(cwd_history[-1], pathname)
        if posixpath.dirname(abs_pathname) not in dirs_created:
            raise ftplib.error_perm
        dirs_created.add(abs_pathname)

    ftp_mock.mkd = MagicMock(side_effect=mkd_mock)

    def cwd_mock(pathname):
        abs_pathname = posixpath.join(cwd_history[-1], pathname)
        if abs_pathname not in dirs_created:
            raise ftplib.error_perm
        cwd_history.append(abs_pathname)

    ftp_mock.cwd = MagicMock(side_effect=cwd_mock)

    def storbinary_mock(cmd, _):
        head, basename = cmd.split(" ", 1)
        assert head == "STOR"
        assert "/" not in basename
        files_created.add(posixpath.join(cwd_history[-1], basename))

    ftp_mock.storbinary = MagicMock(side_effect=storbinary_mock)

    # Test
    subd = tmpdir.mkdir("data").mkdir("subdir")
    subd.join("a.txt").write("A")
    subd.join("b.txt").write("B")
    subd.join("c.txt").write("C")
    subd.mkdir("empty1")
    subsubd = subd.mkdir("subsubdir")
    subsubd.join("aa.txt").write("AA")
    subsubd.join("bb.txt").write("BB")
    subsubd.join("cc.txt").write("CC")
    subsubd.mkdir("empty2")

    dest_path = (
        dest_path_root if artifact_path is None else posixpath.join(dest_path_root, artifact_path)
    )
    dirs_expected = set(
        [
            dest_path,
            posixpath.join(dest_path, "empty1"),
            posixpath.join(dest_path, "subsubdir"),
            posixpath.join(dest_path, "subsubdir", "empty2"),
        ]
    )
    files_expected = set(
        [
            posixpath.join(dest_path, "a.txt"),
            posixpath.join(dest_path, "b.txt"),
            posixpath.join(dest_path, "c.txt"),
            posixpath.join(dest_path, "subsubdir/aa.txt"),
            posixpath.join(dest_path, "subsubdir/bb.txt"),
            posixpath.join(dest_path, "subsubdir/cc.txt"),
        ]
    )

    for dirs_expected_i in dirs_expected.copy():
        if dirs_expected_i != dest_path_root:
            dirs_expected |= set(__posixpath_parents(dirs_expected_i, root=dest_path_root))

    repo.log_artifacts(subd.strpath, artifact_path)
    assert dirs_created == dirs_expected
    assert files_created == files_expected


def test_download_artifacts_single(ftp_mock):
    repo = FTPArtifactRepository("ftp://test_ftp/some/path")

    repo.get_ftp_client = MagicMock()
    call_mock = MagicMock(return_value=ftp_mock)
    repo.get_ftp_client.return_value = MagicMock(__enter__=call_mock)

    ftp_mock.cwd = MagicMock(side_effect=ftplib.error_perm)

    repo.download_artifacts("test.txt")

    ftp_mock.retrbinary.assert_called_once()
    assert ftp_mock.retrbinary.call_args_list[0][0][0] == "RETR /some/path/test.txt"


def test_download_artifacts(ftp_mock):
    artifact_root_path = "/some/path"
    repo = FTPArtifactRepository("ftp://test_ftp" + artifact_root_path)

    repo.get_ftp_client = MagicMock()
    call_mock = MagicMock(return_value=ftp_mock)
    repo.get_ftp_client.return_value = MagicMock(__enter__=call_mock)

    # mocked file structure
    #  |- model
    #     |- model.pb
    #     |- empty_dir
    #     |- variables
    #        |- test.txt
    dir_path = posixpath.join(artifact_root_path, "model")

    # list artifacts at sub directory level
    model_file_path_sub = "model.pb"
    model_file_path_full = posixpath.join(dir_path, model_file_path_sub)
    empty_dir_name = "empty_dir"
    empty_dir_path = posixpath.join(dir_path, empty_dir_name)
    subdir_name = "variables"
    subdir_path_full = posixpath.join(dir_path, subdir_name)
    subfile_name = "test.txt"
    subfile_path_full = posixpath.join(artifact_root_path, subdir_path_full, subfile_name)

    is_dir_mapping = {
        dir_path: True,
        empty_dir_path: True,
        model_file_path_full: False,
        subdir_path_full: True,
        subfile_path_full: False,
    }

    is_dir_call_args = [
        dir_path,
        model_file_path_full,
        empty_dir_path,
        subdir_path_full,
        model_file_path_full,
        subdir_path_full,
        subfile_path_full,
        subfile_path_full,
    ]

    def cwd_side_effect(call_arg):
        if not is_dir_mapping[call_arg]:
            raise ftplib.error_perm

    ftp_mock.cwd = MagicMock(side_effect=cwd_side_effect)

    def nlst_side_effect(call_arg):
        if call_arg == dir_path:
            return [model_file_path_sub, subdir_name, empty_dir_name]
        elif call_arg == subdir_path_full:
            return [subfile_name]
        elif call_arg == empty_dir_path:
            return []
        else:
            raise Exception("should never call nlst for non-directories {}".format(call_arg))

    ftp_mock.nlst = MagicMock(side_effect=nlst_side_effect)
    repo.download_artifacts("model")

    cwd_call_args = [arg_entry[0][0] for arg_entry in ftp_mock.cwd.call_args_list]

    assert set(cwd_call_args) == set(is_dir_call_args)
    assert ftp_mock.nlst.call_count == 3
    assert ftp_mock.retrbinary.call_args_list[0][0][0] == "RETR " + model_file_path_full
    assert ftp_mock.retrbinary.call_args_list[1][0][0] == "RETR " + subfile_path_full


def test_log_artifact_reuse_ftp_client(ftp_mock, tmpdir):
    repo = FTPArtifactRepository("ftp://test_ftp/some/path")

    repo.get_ftp_client = MagicMock()
    call_mock = MagicMock(return_value=ftp_mock)
    repo.get_ftp_client.return_value = MagicMock(__enter__=call_mock)

    d = tmpdir.mkdir("data")
    file = d.join("test.txt")
    file.write("hello world!")
    fpath = file.strpath

    repo.log_artifact(fpath)
    repo.log_artifact(fpath, "subdir1/subdir2")
    repo.log_artifact(fpath, "subdir3")

    assert repo.get_ftp_client.call_count == 3
