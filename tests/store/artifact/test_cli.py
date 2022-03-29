import json
import os
import posixpath
from subprocess import Popen, STDOUT, PIPE

import pytest
from unittest import mock

import mlflow
import mlflow.pyfunc
from mlflow.entities import FileInfo
from mlflow.store.artifact.cli import _file_infos_to_json
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir


@pytest.fixture()
def run_with_artifact():
    artifact_path = "test"
    artifact_content = "contents"
    with mlflow.start_run() as run:
        mlflow.log_text(artifact_content, artifact_path)

    return (run, artifact_path, artifact_content)


def test_file_info_to_json():
    file_infos = [
        FileInfo("/my/file", False, 123),
        FileInfo("/my/dir", True, None),
    ]
    info_str = _file_infos_to_json(file_infos)
    assert json.loads(info_str) == [
        {"path": "/my/file", "is_dir": False, "file_size": 123},
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


def test_download_artifacts_with_uri(run_with_artifact):
    run, artifact_path, artifact_content = run_with_artifact
    command = ["mlflow", "artifacts", "download", "-u"]
    run_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    actual_uri = posixpath.join(run.info.artifact_uri, artifact_path)
    for uri in (run_uri, actual_uri):
        p = Popen(command + [uri], stdout=PIPE, stderr=STDOUT)
        output = p.stdout.readlines()
        downloaded_file_path = output[-1].strip()
        downloaded_file = os.listdir(downloaded_file_path)[0]
        with open(os.path.join(downloaded_file_path, downloaded_file), "r") as f:
            assert f.read() == artifact_content


def test_download_artifacts_with_run_id_and_path(run_with_artifact):
    run, artifact_path, artifact_content = run_with_artifact
    command = ["mlflow", "artifacts", "download", "--run-id", run.info.run_id, "--artifact-path", artifact_path]
    p = Popen(command, stdout=PIPE, stderr=STDOUT)
    output = p.stdout.readlines()
    downloaded_file_path = output[-1].strip()
    downloaded_file = os.listdir(downloaded_file_path)[0]
    with open(os.path.join(downloaded_file_path, downloaded_file), "r") as f:
        assert f.read() == artifact_content


@pytest.mark.parametrize("dst_path", [None, "doesnt_exist_yet"])
def test_download_artifacts_with_dst_path(run_with_artifact, tmp_path, dst_path):
    run, artifact_path, artifact_content = run_with_artifact
    artifact_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    dst_path = dst_path or tmp_path
    
    command = ["mlflow", "artifacts", "download", "-u", artifact_uri, "-d", dst_path]
    p = Popen(command, stdout=PIPE, stderr=STDOUT)
    output = p.stdout.readlines()
    downloaded_file_path = output[-1].strip()
    assert downloaded_file_path == dst_path
