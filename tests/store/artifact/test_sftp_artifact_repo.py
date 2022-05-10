import posixpath
import uuid
import subprocess
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import pytest

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.sftp_artifact_repo import SFTPArtifactRepository

URI = "sftp://user:pass@localhost:2222/upload"


class SFTP(NamedTuple):
    path: Path


@pytest.fixture(autouse=True, scope="module")
def sftp():
    image = "atmoz/sftp"
    subprocess.run(["docker", "pull", image], check=True)
    with tempfile.TemporaryDirectory() as t:
        tmpdir = Path(t).joinpath("upload")
        tmpdir.mkdir()
        tmpdir.chmod(0o0777)
        container = "mlflow-sftp"
        # Run an SFTP server in the background
        process = subprocess.Popen(
            [
                "docker",
                "run",
                "-p",
                "2222:22",
                "-v",
                f"{tmpdir}:/home/user/upload",
                "--name",
                container,
                "atmoz/sftp",
                "user:pass:::upload",
            ],
        )
        time.sleep(5)
        yield SFTP(tmpdir)
        # Stop and remove the container
        subprocess.run(["docker", "stop", container], check=True)
        subprocess.run(["docker", "rm", container], check=True)
        process.kill()


def rand_str():
    return uuid.uuid4().hex


def test_artifact_uri_factory():
    assert isinstance(get_artifact_repository(URI), SFTPArtifactRepository)


def test_list_artifacts_empty():
    artifact_subdir = rand_str()
    store = SFTPArtifactRepository(URI + "/" + artifact_subdir)
    assert store.list_artifacts() == []


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_list_artifacts(tmp_path, artifact_path):
    file_content_1 = rand_str()
    file_content_2 = rand_str()

    file_1 = tmp_path.joinpath(rand_str())
    subdir = tmp_path.joinpath(rand_str())
    subdir.mkdir()
    file_2 = subdir.joinpath(rand_str())
    file_1.write_text(file_content_1)
    file_2.write_text(file_content_2)

    artifact_subdir = rand_str()
    store = SFTPArtifactRepository(URI + "/" + artifact_subdir)
    store.log_artifacts(tmp_path, artifact_path)
    artifacts = store.list_artifacts(artifact_path)
    assert len(artifacts) == 2
    remote_file_1 = artifacts[0]
    assert remote_file_1.is_dir is False
    assert (
        remote_file_1.path == posixpath.join(artifact_path, file_1.name)
        if artifact_path
        else file_1.name
    )
    remote_subdir = artifacts[1]
    assert remote_file_1.file_size == file_1.stat().st_size
    assert remote_subdir.is_dir is True
    assert (
        remote_subdir.path == posixpath.join(artifact_path, subdir.name)
        if artifact_path
        else subdir.name
    )


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_log_artifact(tmp_path, sftp, artifact_path):
    file_content = "A simple test artifact\nThe artifact is located in: " + str(artifact_path)
    tmp_file = tmp_path.joinpath(rand_str())
    tmp_file.write_text(file_content)
    store = SFTPArtifactRepository(URI)
    store.log_artifact(tmp_file, artifact_path)
    remote_file = sftp.path.joinpath(artifact_path or ".", tmp_file.name)
    assert remote_file.read_text() == file_content


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_log_artifacts(tmp_path, sftp, artifact_path):
    file_content_1 = "A simple test artifact\nThe artifact is located in: " + str(artifact_path)
    file_content_2 = str(artifact_path)

    file_1 = tmp_path.joinpath(rand_str())
    subdir = tmp_path.joinpath(rand_str())
    subdir.mkdir()
    file_2 = subdir.joinpath(rand_str())
    file_1.write_text(file_content_1)
    file_2.write_text(file_content_2)

    store = SFTPArtifactRepository(URI)
    store.log_artifacts(tmp_path, artifact_path)

    remote_file_1 = sftp.path.joinpath(artifact_path or ".", file_1.name)
    assert remote_file_1.read_text() == file_content_1
    remote_file_2 = sftp.path.joinpath(artifact_path or ".", subdir.name, file_2.name)
    assert remote_file_2.read_text() == file_content_2


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_delete_artifact(tmp_path, sftp, artifact_path):
    file_content = "A simple test artifact\nThe artifact is located in: " + str(artifact_path)
    tmp_file = tmp_path.joinpath(rand_str())
    tmp_file.write_text(file_content)
    artifact_subdir = rand_str()
    store = SFTPArtifactRepository(URI + "/" + artifact_subdir)
    store.log_artifact(tmp_file, artifact_path)
    remote_dir = sftp.path.joinpath(artifact_subdir, artifact_path or ".")
    remote_file = remote_dir.joinpath(tmp_file.name)
    assert remote_file.read_text() == file_content
    artifact_path = Path(store.path, artifact_path or ".", tmp_file.name).resolve()
    store.delete_artifacts(str(artifact_path))
    assert not remote_file.exists()
    assert remote_dir.exists()


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_delete_artifacts(tmp_path, sftp, artifact_path):
    file_content_1 = "A simple test artifact\nThe artifact is located in: " + str(artifact_path)
    file_content_2 = str(artifact_path)

    file_1 = tmp_path.joinpath(rand_str())
    subdir = tmp_path.joinpath(rand_str())
    subdir.mkdir()
    file_2 = subdir.joinpath(rand_str())
    file_1.write_text(file_content_1)
    file_2.write_text(file_content_2)

    artifact_subdir = rand_str()
    store = SFTPArtifactRepository(URI + "/" + artifact_subdir)
    store.log_artifacts(tmp_path, artifact_path)

    remote_dir = sftp.path.joinpath(artifact_subdir, artifact_path or ".")
    remote_file_1 = remote_dir.joinpath(file_1.name)
    assert remote_file_1.read_text() == file_content_1
    remote_file_2 = remote_dir.joinpath(subdir.name, file_2.name)
    assert remote_file_2.read_text() == file_content_2

    artifact_dir = Path(store.path, artifact_path or ".").resolve()
    store.delete_artifacts(str(artifact_dir))

    assert not remote_dir.exists()
    assert not remote_file_1.exists()
    assert not remote_file_2.exists()
