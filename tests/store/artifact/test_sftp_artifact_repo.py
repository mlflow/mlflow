import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path, PosixPath
from typing import NamedTuple

import pytest

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.sftp_artifact_repo import SFTPArtifactRepository


pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="`atmoz/sftp` image cannot be used on Windows"
)


class SFTP(NamedTuple):
    path: Path
    uri: str


@pytest.fixture(autouse=True, scope="module")
def sftp():
    with tempfile.TemporaryDirectory() as t:
        # Launch an SFTP server in the background
        user = "user"
        password = "password"
        port = 2222
        container = "mlflow-sftp"
        directory = "upload"
        tmpdir = Path(t).joinpath(directory)
        tmpdir.mkdir()
        tmpdir.chmod(0o0777)
        process = subprocess.Popen(
            [
                "docker",
                "run",
                "-p",
                f"{port}:22",
                "-v",
                f"{tmpdir}:/home/{user}/{directory}",
                "--name",
                container,
                # https://hub.docker.com/r/atmoz/sftp
                "atmoz/sftp",
                f"{user}:{password}:{os.getuid()}:{os.getgid()}:::{directory}",
            ],
        )
        # Wait for the server to be ready
        for sleep_sec in (0, 1, 2, 4, 8):
            prc = subprocess.run(
                ["docker", "logs", "--tail", "5", container],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if "Server listening on" in prc.stdout:
                break
            time.sleep(sleep_sec)
        else:
            raise Exception(f"Failed to launch SFTP server: {prc.stdout}")

        uri = f"sftp://{user}:{password}@localhost:{port}/{directory}"
        yield SFTP(tmpdir, uri)

        # Stop and remove the container
        subprocess.run(["docker", "stop", container], check=True)
        subprocess.run(["docker", "rm", container], check=True)
        process.kill()


def rand_str():
    return uuid.uuid4().hex


def test_artifact_uri_factory(sftp):
    assert isinstance(get_artifact_repository(sftp.uri), SFTPArtifactRepository)


def test_list_artifacts_empty(sftp):
    artifact_subdir = rand_str()
    store = SFTPArtifactRepository(sftp.uri + "/" + artifact_subdir)
    assert store.list_artifacts() == []


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_list_artifacts(tmp_path, sftp, artifact_path):
    file_content1 = rand_str()
    file_content2 = rand_str()
    local_file1 = tmp_path.joinpath("file1")
    local_dir = tmp_path.joinpath("dir")
    local_dir.mkdir()
    local_file2 = local_dir.joinpath("file2")
    local_file1.write_text(file_content1)
    local_file2.write_text(file_content2)

    artifact_subdir = rand_str()
    store = SFTPArtifactRepository(sftp.uri + "/" + artifact_subdir)
    store.log_artifacts(tmp_path, artifact_path)

    artifacts = store.list_artifacts(artifact_path)
    assert len(artifacts) == 2
    remote_file1 = artifacts[0]
    assert remote_file1.is_dir is False
    assert (
        remote_file1.path == str(PosixPath(artifact_path, local_file1.name))
        if artifact_path
        else local_file1.name
    )
    remote_dir = artifacts[1]
    assert remote_file1.file_size == local_file1.stat().st_size
    assert remote_dir.is_dir is True
    assert (
        remote_dir.path == str(PosixPath(artifact_path, local_dir.name))
        if artifact_path
        else local_dir.name
    )


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_log_artifact(tmp_path, sftp, artifact_path):
    file_content = rand_str()
    local_file = tmp_path.joinpath("file")
    local_file.write_text(file_content)

    artifact_subdir = rand_str()
    store = SFTPArtifactRepository(sftp.uri + "/" + artifact_subdir)
    store.log_artifact(local_file, artifact_path)

    remote_file = sftp.path.joinpath(artifact_subdir, artifact_path or ".", local_file.name)
    assert remote_file.read_text() == file_content


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_log_artifacts(tmp_path, sftp, artifact_path):
    file_content1 = rand_str()
    file_content2 = rand_str()
    local_file1 = tmp_path.joinpath("file1")
    local_dir = tmp_path.joinpath("dir")
    local_dir.mkdir()
    local_file2 = local_dir.joinpath("file2")
    local_file1.write_text(file_content1)
    local_file2.write_text(file_content2)

    artifact_subdir = rand_str()
    store = SFTPArtifactRepository(sftp.uri + "/" + artifact_subdir)
    store.log_artifacts(tmp_path, artifact_path)

    remote_dir = sftp.path.joinpath(artifact_subdir, artifact_path or ".")
    remote_file1 = remote_dir.joinpath(local_file1.name)
    assert remote_file1.read_text() == file_content1
    remote_file2 = remote_dir.joinpath(local_dir.name, local_file2.name)
    assert remote_file2.read_text() == file_content2


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_delete_artifact(tmp_path, sftp, artifact_path):
    file_content = rand_str()
    local_file = tmp_path.joinpath("file")
    local_file.write_text(file_content)

    artifact_subdir = rand_str()
    store = SFTPArtifactRepository(sftp.uri + "/" + artifact_subdir)
    store.log_artifact(local_file, artifact_path)

    remote_dir = sftp.path.joinpath(artifact_subdir, artifact_path or ".")
    remote_file = remote_dir.joinpath(local_file.name)
    assert remote_file.read_text() == file_content

    artifact_path = Path(store.path, artifact_path or ".", local_file.name).resolve()
    store.delete_artifacts(str(artifact_path))
    assert not remote_file.exists()
    assert remote_dir.exists()


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_delete_artifacts(tmp_path, sftp, artifact_path):
    file_content1 = rand_str()
    file_content2 = rand_str()
    local_file1 = tmp_path.joinpath("file1")
    local_dir = tmp_path.joinpath("dir")
    local_dir.mkdir()
    local_file2 = local_dir.joinpath("file2")
    local_file1.write_text(file_content1)
    local_file2.write_text(file_content2)

    artifact_subdir = rand_str()
    store = SFTPArtifactRepository(sftp.uri + "/" + artifact_subdir)
    store.log_artifacts(tmp_path, artifact_path)

    remote_dir = sftp.path.joinpath(artifact_subdir, artifact_path or ".")
    remote_file1 = remote_dir.joinpath(local_file1.name)
    assert remote_file1.read_text() == file_content1
    remote_file2 = remote_dir.joinpath(local_dir.name, local_file2.name)
    assert remote_file2.read_text() == file_content2

    artifact_dir = Path(store.path, artifact_path or ".").resolve()
    store.delete_artifacts(str(artifact_dir))

    assert not remote_dir.exists()
    assert not remote_file1.exists()
    assert not remote_file2.exists()
