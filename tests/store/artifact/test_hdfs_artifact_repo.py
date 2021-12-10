import os
import sys
from tempfile import NamedTemporaryFile
from unittest import mock

import pytest
from unittest.mock import call, mock_open
from pyarrow import HadoopFileSystem

from mlflow.entities import FileInfo
from mlflow.store.artifact.hdfs_artifact_repo import (
    HdfsArtifactRepository,
    _resolve_base_path,
    _relative_path_remote,
    _parse_extra_conf,
    _download_hdfs_file,
)
from mlflow.utils.file_utils import TempDir


@mock.patch("pyarrow.hdfs.HadoopFileSystem")
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

        open_mock = hdfs_system_mock.return_value.open
        open_mock.assert_called_once_with("/hdfs/path/more_path/some/sample_file", "wb")

        write_mock = open_mock.return_value.__enter__.return_value.write
        write_mock.assert_called_once_with(b"PyArrow Works")


@mock.patch("pyarrow.hdfs.HadoopFileSystem")
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

        open_mock = hdfs_system_mock.return_value.open
        open_mock.assert_called_once_with("/mypath/more_path/some/sample_file", "wb")

        write_mock = open_mock.return_value.__enter__.return_value.write
        write_mock.assert_called_once_with(b"PyArrow Works")


@mock.patch("pyarrow.hdfs.HadoopFileSystem")
def test_log_artifact_with_kerberos_setup(hdfs_system_mock):
    if sys.platform == "win32":
        pytest.skip()
    os.environ["MLFLOW_KERBEROS_TICKET_CACHE"] = "/tmp/krb5cc_22222222"
    os.environ["MLFLOW_KERBEROS_USER"] = "some_kerberos_user"

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

        # TODO: refactor this magic ...
        write_mock = hdfs_system_mock.return_value.open.return_value.__enter__.return_value.write
        write_mock.assert_called_once_with(b"PyArrow Works")


@mock.patch("pyarrow.hdfs.HadoopFileSystem")
def test_log_artifact_with_invalid_local_dir(_):
    repo = HdfsArtifactRepository("hdfs://host_name:8020/maybe/path")

    with pytest.raises(Exception, match="No such file or directory: '/not/existing/local/path'"):
        repo.log_artifact("/not/existing/local/path", "test_hdfs/some/path")


@mock.patch("pyarrow.hdfs.HadoopFileSystem")
def test_log_artifacts(hdfs_system_mock):
    os.environ["MLFLOW_KERBEROS_TICKET_CACHE"] = "/tmp/krb5cc_22222222"
    os.environ["MLFLOW_KERBEROS_USER"] = "some_kerberos_user"

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

        open_mock = hdfs_system_mock.return_value.open
        open_mock.assert_has_calls(
            calls=[
                call("/some_path/maybe/path/file_one.txt", "wb"),
                call("/some_path/maybe/path/subdir/file_two.txt", "wb"),
            ],
            any_order=True,
        )
        write_mock = open_mock.return_value.__enter__.return_value.write
        write_mock.assert_has_calls(
            calls=[call(b"PyArrow Works once"), call(b"PyArrow Works two")], any_order=True
        )


@mock.patch("pyarrow.hdfs.HadoopFileSystem")
def test_list_artifacts_root(hdfs_system_mock):
    repo = HdfsArtifactRepository("hdfs://host/some/path")

    expected = [FileInfo("model", True, 0)]

    hdfs_system_mock.return_value.ls.return_value = [
        {"kind": "directory", "name": "hdfs://host/some/path/model", "size": 0}
    ]

    actual = repo.list_artifacts()

    assert actual == expected


@mock.patch("pyarrow.hdfs.HadoopFileSystem")
def test_list_artifacts_nested(hdfs_system_mock):
    repo = HdfsArtifactRepository("hdfs:://host/some/path")

    expected = [
        FileInfo("model/conda.yaml", False, 33),
        FileInfo("model/model.pkl", False, 33),
        FileInfo("model/MLmodel", False, 33),
    ]

    hdfs_system_mock.return_value.ls.return_value = [
        {"kind": "file", "name": "hdfs://host/some/path/model/conda.yaml", "size": 33},
        {"kind": "file", "name": "hdfs://host/some/path/model/model.pkl", "size": 33},
        {"kind": "file", "name": "hdfs://host/some/path/model/MLmodel", "size": 33},
    ]

    actual = repo.list_artifacts("model")

    assert actual == expected


@mock.patch("pyarrow.hdfs.HadoopFileSystem", spec=HadoopFileSystem)
def test_list_artifacts_empty_hdfs_dir(hdfs_system_mock):
    hdfs_system_mock.return_value.exists.return_value = False

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


def test_download_artifacts():
    expected_data = b"hello"
    artifact_path = "test.txt"
    # mock hdfs
    hdfs = mock.Mock()
    hdfs.open = mock_open(read_data=expected_data)

    with TempDir() as tmp_dir:
        _download_hdfs_file(hdfs, artifact_path, os.path.join(tmp_dir.path(), artifact_path))
        with open(os.path.join(tmp_dir.path(), artifact_path), "rb") as fd:
            assert expected_data == fd.read()


@mock.patch("pyarrow.hdfs.HadoopFileSystem")
def test_delete_artifacts(hdfs_system_mock):
    delete_mock = hdfs_system_mock.return_value.delete
    repo = HdfsArtifactRepository("hdfs:/some_path/maybe/path")
    repo.delete_artifacts("artifacts")
    delete_mock.assert_called_once_with("/some_path/maybe/path/artifacts", recursive=True)
