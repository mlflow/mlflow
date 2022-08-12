import json
import pathlib
from subprocess import Popen, STDOUT, PIPE

import pytest
from unittest import mock

import mlflow
import mlflow.pyfunc
from mlflow.entities import FileInfo
from mlflow.store.artifact.cli import _file_infos_to_json
from mlflow.tracking.artifact_utils import _download_artifact_from_uri


@pytest.fixture()
def run_with_artifact(tmp_path):
    artifact_path = "test"
    artifact_content = "content"
    local_path = tmp_path.joinpath("file.txt")
    local_path.write_text(artifact_content)
    with mlflow.start_run() as run:
        mlflow.log_artifact(local_path, artifact_path)

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


def _run_download_artifact_command(command):
    """
    :param command: An `mlflow artifacts` command list.
    :return: The downloaded artifact content.
    """
    p = Popen(command, stdout=PIPE, stderr=STDOUT)
    output = p.stdout.readlines()
    download_output_path = pathlib.Path(output[-1].strip().decode("utf-8"))
    downloaded_file = next(download_output_path.iterdir())
    return downloaded_file.read_text()


def test_download_artifacts_with_uri(run_with_artifact):
    run, artifact_path, artifact_content = run_with_artifact
    base_command = ["mlflow", "artifacts", "download", "-u"]
    run_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    actual_uri = str(pathlib.PurePosixPath(run.info.artifact_uri) / artifact_path)
    for uri in (run_uri, actual_uri):
        downloaded_content = _run_download_artifact_command(base_command + [uri])
        assert downloaded_content == artifact_content

    # Check for backwards compatibility with preexisting behavior in MLflow <= 1.24.0 where
    # specifying `artifact_uri` and `artifact_path` together did not throw an exception (unlike
    # `mlflow.artifacts.download_artifacts()`) and instead used `artifact_uri` while ignoring
    # `run_id` and `artifact_path`
    downloaded_content = _run_download_artifact_command(
        base_command + [uri] + ["--run-id", "bad", "--artifact-path", "bad"]
    )
    assert downloaded_content == artifact_content


def test_download_artifacts_with_run_id_and_path(run_with_artifact):
    run, artifact_path, artifact_content = run_with_artifact
    downloaded_content = _run_download_artifact_command(
        [
            "mlflow",
            "artifacts",
            "download",
            "--run-id",
            run.info.run_id,
            "--artifact-path",
            artifact_path,
        ]
    )
    assert downloaded_content == artifact_content


@pytest.mark.parametrize("dst_subdir_path", [None, "doesnt_exist_yet"])
def test_download_artifacts_with_dst_path(run_with_artifact, tmp_path, dst_subdir_path):
    run, artifact_path, _ = run_with_artifact
    artifact_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    dst_path = tmp_path / dst_subdir_path if dst_subdir_path else tmp_path

    command = ["mlflow", "artifacts", "download", "-u", artifact_uri, "-d", str(dst_path)]
    p = Popen(command, stdout=PIPE, stderr=STDOUT, text=True)
    p.wait()
    assert p.returncode == 0
    output = p.stdout.readlines()
    downloaded_file_path = output[-1].strip()
    assert downloaded_file_path.startswith(str(dst_path))
