import os
import sys
from tempfile import NamedTemporaryFile
from unittest import mock
from unittest.mock import call

import pyarrow
import pytest

from mlflow.entities import FileInfo
from mlflow.store.artifact.hdfs_artifact_repo import (
    HdfsArtifactRepository,
    _parse_extra_conf,
    _relative_path_remote,
    _resolve_base_path,
)
from mlflow.utils.file_utils import TempDir


@mock.patch(
    "mlflow.store.artifact.hdfs_artifact_repo.HadoopFileSystem", spec=pyarrow.fs.HadoopFileSystem
)
def test_log_artifact(hdfs_system_mock):
    repo = HdfsArtifactRepository("hdfs://host_name:8020/hdfs/path")

    with TempDir() as tmp_dir:
        local_file = tmp_dir.path("sample_file")
        with open(local_file, "w") as f:
            f.write("PyArrow Works")

        repo.log_artifact(local_file, "more_path/some")

        hdfs_system_mock.assert_called_once_with(
            extra_conf=None, host="hdfs://host_name", kerb_ticket=None, port=8020, user=None
        )

        upload_mock = hdfs_system_mock.return_value.open_output_stream
        upload_mock.assert_called_once_with("/hdfs/path/more_path/some/sample_file")


@mock.patch(
    "mlflow.store.artifact.hdfs_artifact_repo.HadoopFileSystem", spec=pyarrow.fs.HadoopFileSystem
)
def test_log_artifact_viewfs(hdfs_system_mock):
    repo = HdfsArtifactRepository("viewfs://host_name/mypath")

    with TempDir() as tmp_dir:
        local_file = tmp_dir.path("sample_file")
        with open(local_file, "w") as f:
            f.write("PyArrow Works")

        repo.log_artifact(local_file, "more_path/some")

        hdfs_system_mock.assert_called_once_with(
            extra_conf=None, host="viewfs://host_name", kerb_ticket=None, port=0, user=None
        )
        upload_mock = hdfs_system_mock.return_value.open_output_stream
        upload_mock.assert_called_once_with("/mypath/more_path/some/sample_file")


@mock.patch(
    "mlflow.store.artifact.hdfs_artifact_repo.HadoopFileSystem", spec=pyarrow.fs.HadoopFileSystem
)
def test_log_artifact_with_kerberos_setup(hdfs_system_mock, monkeypatch):
    if sys.platform == "win32":
        pytest.skip()
    monkeypatch.setenv("MLFLOW_KERBEROS_TICKET_CACHE", "/tmp/krb5cc_22222222")
    monkeypatch.setenv("MLFLOW_KERBEROS_USER", "some_kerberos_user")

    repo = HdfsArtifactRepository("hdfs:/some/maybe/path")

    with NamedTemporaryFile() as tmp_local_file:
        tmp_local_file.write(b"PyArrow Works")
        tmp_local_file.seek(0)

        repo.log_artifact(tmp_local_file.name, "test_hdfs/some/path")

        hdfs_system_mock.assert_called_once_with(
            extra_conf=None,
            host="default",
            kerb_ticket="/tmp/krb5cc_22222222",
            port=0,
            user="some_kerberos_user",
        )
        upload_mock = hdfs_system_mock.return_value.open_output_stream
        upload_mock.assert_called_once()


@mock.patch(
    "mlflow.store.artifact.hdfs_artifact_repo.HadoopFileSystem", spec=pyarrow.fs.HadoopFileSystem
)
def test_log_artifact_with_invalid_local_dir(_):
    repo = HdfsArtifactRepository("hdfs://host_name:8020/maybe/path")

    with pytest.raises(Exception, match="No such file or directory: '/not/existing/local/path'"):
        repo.log_artifact("/not/existing/local/path", "test_hdfs/some/path")


@mock.patch(
    "mlflow.store.artifact.hdfs_artifact_repo.HadoopFileSystem", spec=pyarrow.fs.HadoopFileSystem
)
def test_log_artifacts(hdfs_system_mock, monkeypatch):
    monkeypatch.setenv("MLFLOW_KERBEROS_TICKET_CACHE", "/tmp/krb5cc_22222222")
    monkeypatch.setenv("MLFLOW_KERBEROS_USER", "some_kerberos_user")

    repo = HdfsArtifactRepository("hdfs:/some_path/maybe/path")

    with TempDir() as root_dir:
        with open(root_dir.path("file_one.txt"), "w") as f:
            f.write("PyArrow Works once")

        os.mkdir(root_dir.path("subdir"))
        with open(root_dir.path("subdir/file_two.txt"), "w") as f:
            f.write("PyArrow Works two")

        repo.log_artifacts(root_dir._path)

        hdfs_system_mock.assert_called_once_with(
            extra_conf=None,
            host="default",
            kerb_ticket="/tmp/krb5cc_22222222",
            port=0,
            user="some_kerberos_user",
        )

        upload_mock = hdfs_system_mock.return_value.open_output_stream
        upload_mock.assert_has_calls(
            calls=[
                call("/some_path/maybe/path/file_one.txt"),
                call("/some_path/maybe/path/subdir/file_two.txt"),
            ],
            any_order=True,
        )


@mock.patch(
    "mlflow.store.artifact.hdfs_artifact_repo.HadoopFileSystem", spec=pyarrow.fs.HadoopFileSystem
)
def test_list_artifacts_root(hdfs_system_mock):
    repo = HdfsArtifactRepository("hdfs://host/some/path")

    expected = [FileInfo("model", True, 0)]

    hdfs_system_mock.return_value.get_file_info.side_effect = [
        pyarrow.fs.FileInfo(path="/some/path/", type=pyarrow.fs.FileType.Directory, size=0),
        [pyarrow.fs.FileInfo(path="/some/path/model", type=pyarrow.fs.FileType.Directory, size=0)],
    ]

    actual = repo.list_artifacts()

    assert actual == expected


@mock.patch(
    "mlflow.store.artifact.hdfs_artifact_repo.HadoopFileSystem", spec=pyarrow.fs.HadoopFileSystem
)
def test_list_artifacts_nested(hdfs_system_mock):
    repo = HdfsArtifactRepository("hdfs://host/some/path")

    expected = [
        FileInfo("model/conda.yaml", False, 33),
        FileInfo("model/model.pkl", False, 33),
        FileInfo("model/MLmodel", False, 33),
    ]

    hdfs_system_mock.return_value.get_file_info.side_effect = [
        pyarrow.fs.FileInfo(path="/some/path/model", type=pyarrow.fs.FileType.Directory, size=0),
        [
            pyarrow.fs.FileInfo(
                path="/some/path/model/conda.yaml", type=pyarrow.fs.FileType.File, size=33
            ),
            pyarrow.fs.FileInfo(
                path="/some/path/model/model.pkl", type=pyarrow.fs.FileType.File, size=33
            ),
            pyarrow.fs.FileInfo(
                path="/some/path/model/MLmodel", type=pyarrow.fs.FileType.File, size=33
            ),
        ],
    ]

    actual = repo.list_artifacts("model")

    assert actual == expected


@mock.patch(
    "mlflow.store.artifact.hdfs_artifact_repo.HadoopFileSystem", spec=pyarrow.fs.HadoopFileSystem
)
def test_list_artifacts_empty_hdfs_dir(hdfs_system_mock):
    hdfs_system_mock.return_value.get_file_info.return_value = pyarrow.fs.FileInfo(
        path="/some_path/maybe/path", type=pyarrow.fs.FileType.NotFound, size=0
    )

    repo = HdfsArtifactRepository("hdfs:/some_path/maybe/path")
    actual = repo.list_artifacts()
    assert actual == []


def test_resolve_path():
    assert _resolve_base_path("/dir/some/path", None) == "/dir/some/path"
    assert _resolve_base_path("/dir/some/path", "subdir/path") == "/dir/some/path/subdir/path"


def test_relative_path():
    assert _relative_path_remote("/dir/some", "/dir/some/path/file.txt") == "path/file.txt"
    assert _relative_path_remote("/dir/some", "/dir/some") is None


def test_parse_extra_conf():
    assert _parse_extra_conf("fs.permissions.umask-mode=022,some_other.extra.conf=abcd") == {
        "fs.permissions.umask-mode": "022",
        "some_other.extra.conf": "abcd",
    }
    assert _parse_extra_conf(None) is None

    with pytest.raises(ValueError, match="not enough values to unpack "):
        _parse_extra_conf("missing_equals_sign")


@mock.patch(
    "mlflow.store.artifact.hdfs_artifact_repo.HadoopFileSystem", spec=pyarrow.fs.HadoopFileSystem
)
def test_delete_artifacts(hdfs_system_mock):
    repo = HdfsArtifactRepository("hdfs:/some_path/maybe/path/")
    hdfs_system_mock.return_value.get_file_info.return_value = pyarrow.fs.FileInfo(
        path="/some_path/maybe/path/file.txt", type=pyarrow.fs.FileType.File, size=0
    )
    delete_mock = hdfs_system_mock.return_value.delete_file
    repo.delete_artifacts("file.ext")
    delete_mock.assert_called_once_with("/some_path/maybe/path/file.ext")
    hdfs_system_mock.return_value.get_file_info.return_value = pyarrow.fs.FileInfo(
        path="/some_path/maybe/path/artifacts", type=pyarrow.fs.FileType.Directory, size=0
    )
    delete_mock = hdfs_system_mock.return_value.delete_dir_contents
    repo.delete_artifacts("artifacts")
    delete_mock.assert_called_once_with("/some_path/maybe/path/artifacts")


@mock.patch(
    "mlflow.store.artifact.hdfs_artifact_repo.HadoopFileSystem", spec=pyarrow.fs.HadoopFileSystem
)
def test_is_directory_called_with_relative_path(hdfs_system_mock):
    repo = HdfsArtifactRepository("hdfs://host/some/path")

    get_file_info_mock = hdfs_system_mock.return_value.get_file_info
    get_file_info_mock.side_effect = [
        pyarrow.fs.FileInfo(path="/some/path/dir", type=pyarrow.fs.FileType.Directory, size=0),
    ]

    assert repo._is_directory("dir")
    get_file_info_mock.assert_called_once_with("/some/path/dir")
