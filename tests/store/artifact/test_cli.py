import json
import os
import posixpath

from unittest import mock

import mlflow
import mlflow.pyfunc
from mlflow.entities import FileInfo, Run, RunInfo
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.artifact.hdfs_artifact_repo import HdfsArtifactRepository
from mlflow.store.artifact.cli import _file_infos_to_json, archive_hdfs_artifacts
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir
from subprocess import Popen, STDOUT, PIPE
from click.testing import CliRunner


def test_file_info_to_json():
    file_infos = [
        FileInfo("/my/file", False, 123),
        FileInfo("/my/dir", True, None),
    ]
    info_str = _file_infos_to_json(file_infos)
    assert json.loads(info_str) == [
        {"path": "/my/file", "is_dir": False, "file_size": "123"},
        {"path": "/my/dir", "is_dir": True},
    ]


def test_download_from_uri():
    class TestArtifactRepo:
        def __init__(self, scheme):
            self.scheme = scheme

        def download_artifacts(self, artifact_path, **kwargs):  # pylint: disable=unused-argument
            return (self.scheme, artifact_path)

    def test_get_artifact_repository(artifact_uri):
        return TestArtifactRepo(artifact_uri)

    pairs = [
        ("path", ("", "path")),
        ("path/", ("path", "")),
        ("/path", ("/", "path")),
        ("/path/", ("/path", "")),
        ("path/to/dir", ("path/to", "dir")),
        ("file:", ("file:", "")),
        ("file:path", ("file:", "path")),
        ("file:path/", ("file:path", "")),
        ("file:path/to/dir", ("file:path/to", "dir")),
        ("file:/", ("file:///", "")),
        ("file:/path", ("file:///", "path")),
        ("file:/path/", ("file:///path", "")),
        ("file:/path/to/dir", ("file:///path/to", "dir")),
        ("file:///", ("file:///", "")),
        ("file:///path", ("file:///", "path")),
        ("file:///path/", ("file:///path", "")),
        ("file:///path/to/dir", ("file:///path/to", "dir")),
        ("s3://", ("s3:", "")),
        ("s3://path", ("s3://path", "")),  # path is netloc in this case
        ("s3://path/", ("s3://path/", "")),
        ("s3://path/to/", ("s3://path/to", "")),
        ("s3://path/to", ("s3://path/", "to")),
        ("s3://path/to/dir", ("s3://path/to", "dir")),
    ]
    with mock.patch(
        "mlflow.tracking.artifact_utils.get_artifact_repository"
    ) as get_artifact_repo_mock:
        get_artifact_repo_mock.side_effect = test_get_artifact_repository

        for uri, expected_result in pairs:
            actual_result = _download_artifact_from_uri(uri)
            assert expected_result == actual_result


def test_download_artifacts_from_uri():
    with mlflow.start_run() as run:
        with TempDir() as tmp:
            local_path = tmp.path("test")
            with open(local_path, "w") as f:
                f.write("test")
            mlflow.log_artifact(local_path, "test")
    command = ["mlflow", "artifacts", "download", "-u"]
    # Test with run uri
    run_uri = "runs:/{run_id}/test".format(run_id=run.info.run_id)
    actual_uri = posixpath.join(run.info.artifact_uri, "test")
    for uri in (run_uri, actual_uri):
        p = Popen(command + [uri], stdout=PIPE, stderr=STDOUT)
        output = p.stdout.readlines()
        downloaded_file_path = output[-1].strip()
        downloaded_file = os.listdir(downloaded_file_path)[0]
        with open(os.path.join(downloaded_file_path, downloaded_file), "r") as f:
            assert f.read() == "test"


def gen_mock_store(artifact_uri):
    mock_store = mock.Mock(spec=AbstractStore)
    mock_run = mock.Mock(spec=Run)
    mock_store.get_run.return_value = mock_run
    mock_runinfo = mock.Mock(spec=RunInfo)
    mock_run.info = mock_runinfo
    mock_runinfo.artifact_uri = artifact_uri
    return mock_store


@mock.patch("mlflow.store.artifact.cli._get_store")
def test_archive_hdfs_artifacts_non_hdfs_raises_error(mock_get_store):
    mock_store = gen_mock_store("ftp://ftpserver/to/myartifacts")
    mock_get_store.return_value = mock_store
    res = CliRunner().invoke(archive_hdfs_artifacts, ["-r", "42"])
    assert 1 == res.exit_code


@mock.patch("mlflow.store.artifact.cli._get_store")
@mock.patch("mlflow.store.artifact.cli.archive_artifacts")
@mock.patch("mlflow.store.artifact.cli.remove_folder")
def test_archive_hdfs_artifacts(mock_remove_folder, mock_archive_artifacts, mock_get_store):
    mock_store = gen_mock_store("hdfs:///path/to/myartifacts")
    mock_archive_artifacts.return_value = "har://hdfs-root/path/to/artifact.har"
    mock_get_store.return_value = mock_store
    print(mock_store.get_run("42").info.artifact_uri)
    res = CliRunner().invoke(archive_hdfs_artifacts, ["-r", "42"], catch_exceptions=False)
    assert 0 == res.exit_code
    mock_archive_artifacts.assert_called_once()
    call_args = mock_archive_artifacts.call_args[0]
    assert isinstance(call_args[0], HdfsArtifactRepository)
    assert "hdfs:///path/to" == call_args[1]
    mock_store.update_artifacts_location.assert_called_once_with(
        "42", "har://hdfs-root/path/to/artifact.har"
    )
    mock_remove_folder.assert_called_once_with("hdfs:///path/to/myartifacts")
