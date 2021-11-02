import os
from collections import namedtuple
import subprocess
import tempfile
import requests

import pytest

import mlflow
from tests.helper_functions import LOCALHOST, get_safe_port
from tests.tracking.integration_test_utils import _await_server_up_or_die


def _launch_server(backend_store_uri, artifacts_destination):
    port = get_safe_port()
    url = f"http://{LOCALHOST}:{port}"
    extra_cmd = ["--gunicorn-opts", "--log-level debug"] if os.name == "posix" else []
    cmd = [
        "mlflow",
        "server",
        "--backend-store-uri",
        backend_store_uri,
        "--default-artifact-root",
        f"{url}/api/2.0/mlflow-artifacts/artifacts",
        "--artifacts-destination",
        artifacts_destination,
        "--host",
        LOCALHOST,
        "--port",
        str(port),
        *extra_cmd,
    ]
    process = subprocess.Popen(cmd)
    _await_server_up_or_die(port)
    return url, process


ArtifactsServer = namedtuple(
    "ArtifactsServer", ["backend_store_uri", "artifacts_destination", "url", "process"]
)


@pytest.fixture(scope="module")
def artifacts_server():
    with tempfile.TemporaryDirectory() as tmpdir:
        backend_store_uri = f"{tmpdir}/mlruns"
        artifacts_destination = f"{tmpdir}/artifacts"
        url, process = _launch_server(backend_store_uri, artifacts_destination)
        yield ArtifactsServer(backend_store_uri, artifacts_destination, url, process)
        process.kill()


def read_file(path):
    with open(path) as f:
        return f.read()


def upload_file(url, local_path):
    with open(local_path, "rb") as f:
        requests.put(url, data=f).raise_for_status()


def download_file(url, local_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def test_rest_apis(tmpdir):
    backend_store_uri = f"{tmpdir}/mlruns"
    artifacts_destination = f"{tmpdir}/artifacts"
    url, _ = _launch_server(backend_store_uri, artifacts_destination)
    api_url = f"{url}/api/2.0/mlflow-artifacts/artifacts"

    # Upload artifact
    file_a = tmpdir.join("a.txt")
    file_a.write("0")
    upload_file(f"{api_url}/a.txt", file_a)
    assert os.path.exists(os.path.join(artifacts_destination, "a.txt"))
    assert read_file(os.path.join(artifacts_destination, "a.txt")) == "0"

    file_b = tmpdir.join("b.txt")
    file_b.write("1")
    upload_file(f"{api_url}/subdir/b.txt", file_b)
    assert os.path.join(artifacts_destination, "subdir", "b.txt")
    assert read_file(os.path.join(artifacts_destination, "subdir", "b.txt")) == "1"

    # Download artifact
    new_dir = tmpdir.mkdir("new")
    new_file_a = new_dir.join("a.txt")
    download_file(f"{api_url}/a.txt", new_file_a)
    assert read_file(new_file_a) == "0"

    new_file_b = new_dir.join("b.txt")
    download_file(f"{api_url}/subdir/b.txt", new_file_b)
    assert read_file(new_file_b) == "1"

    # List artifacts
    resp = requests.get(api_url)
    assert resp.json() == {
        "files": [
            {"path": "a.txt", "is_dir": False, "file_size": "1"},
            {"path": "subdir", "is_dir": True},
        ]
    }
    resp = requests.get(api_url, params={"path": "subdir"})
    assert resp.json() == {"files": [{"path": "b.txt", "is_dir": False, "file_size": "1"}]}


def test_log_artifact(artifacts_server, tmpdir):
    url = artifacts_server.url
    artifacts_destination = artifacts_server.artifacts_destination
    mlflow.set_tracking_uri(url)

    tmp_path = tmpdir.join("a.txt")
    tmp_path.write("0")

    # withtout `artifact_path`
    with mlflow.start_run() as run:
        mlflow.log_artifact(tmp_path)

    experiment_id = "0"
    run_artifact_root = os.path.join(
        artifacts_destination, experiment_id, run.info.run_id, "artifacts"
    )
    dest_path = os.path.join(run_artifact_root, tmp_path.basename)
    assert os.path.exists(dest_path)
    assert read_file(dest_path) == "0"

    # with `artifact_path`
    with mlflow.start_run() as run:
        mlflow.log_artifact(tmp_path, artifact_path="artifact_path")

    run_artifact_root = os.path.join(
        artifacts_destination, experiment_id, run.info.run_id, "artifacts"
    )
    dest_path = os.path.join(run_artifact_root, "artifact_path", tmp_path.basename)
    assert os.path.exists(dest_path)
    assert read_file(dest_path) == "0"


def test_log_artifacts(artifacts_server, tmpdir):
    url = artifacts_server.url
    mlflow.set_tracking_uri(url)

    tmpdir.join("a.txt").write("0")
    tmpdir.mkdir("subdir").join("b.txt").write("1")

    # without `artifact_path`
    with mlflow.start_run() as run:
        mlflow.log_artifacts(tmpdir)

    client = mlflow.tracking.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert artifacts == ["a.txt", "subdir"]
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "subdir")]
    assert artifacts == ["subdir/b.txt"]

    # with `artifact_path`
    with mlflow.start_run() as run:
        mlflow.log_artifacts(tmpdir, artifact_path="artifact_path")

    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert artifacts == ["artifact_path"]
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "artifact_path")]
    assert artifacts == ["artifact_path/a.txt", "artifact_path/subdir"]
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "artifact_path/subdir")]
    assert artifacts == ["artifact_path/subdir/b.txt"]


def test_list_artifacts(artifacts_server, tmpdir):
    url = artifacts_server.url
    mlflow.set_tracking_uri(url)

    tmp_path1 = tmpdir.join("a.txt")
    tmp_path1.write("0")
    tmp_path2 = tmpdir.join("b.txt")
    tmp_path2.write("1")
    client = mlflow.tracking.MlflowClient()
    with mlflow.start_run() as run:
        assert client.list_artifacts(run.info.run_id) == []
        mlflow.log_artifact(tmp_path1)
        mlflow.log_artifact(tmp_path2, "subdir")

    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert artifacts == ["a.txt", "subdir"]
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "subdir")]
    assert artifacts == ["subdir/b.txt"]


def test_download_artifacts(artifacts_server, tmpdir):
    url = artifacts_server.url
    mlflow.set_tracking_uri(url)

    tmp_path1 = tmpdir.join("a.txt")
    tmp_path1.write("0")
    tmp_path2 = tmpdir.join("b.txt")
    tmp_path2.write("1")
    with mlflow.start_run() as run:
        mlflow.log_artifact(tmp_path1)
        mlflow.log_artifact(tmp_path2, "subdir")

    client = mlflow.tracking.MlflowClient()
    dest_path = client.download_artifacts(run.info.run_id, "a.txt")
    assert read_file(dest_path) == "0"
    dest_path = client.download_artifacts(run.info.run_id, "subdir")
    assert os.listdir(dest_path) == ["b.txt"]
    assert read_file(os.path.join(dest_path, "b.txt")) == "1"
