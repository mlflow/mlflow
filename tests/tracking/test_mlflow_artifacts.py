import os
from collections import namedtuple
import subprocess
import tempfile
import requests
import pathlib
import pytest

import mlflow
from tests.helper_functions import LOCALHOST, get_safe_port
from tests.tracking.integration_test_utils import _await_server_up_or_die


def is_windows():
    return os.name == "nt"


def _launch_server(host, port, backend_store_uri, default_artifact_root, artifacts_destination):
    extra_cmd = [] if is_windows() else ["--gunicorn-opts", "--log-level debug"]
    cmd = [
        "mlflow",
        "server",
        "--host",
        host,
        "--port",
        str(port),
        "--backend-store-uri",
        backend_store_uri,
        "--serve-artifacts",
        "--default-artifact-root",
        default_artifact_root,
        "--artifacts-destination",
        artifacts_destination,
        *extra_cmd,
    ]
    process = subprocess.Popen(cmd)
    _await_server_up_or_die(port)
    return process


ArtifactsServer = namedtuple(
    "ArtifactsServer",
    ["backend_store_uri", "default_artifact_root", "artifacts_destination", "url", "process"],
)


@pytest.fixture(scope="module")
def artifacts_server():
    with tempfile.TemporaryDirectory() as tmpdir:
        port = get_safe_port()
        backend_store_uri = os.path.join(tmpdir, "mlruns")
        artifacts_destination = os.path.join(tmpdir, "mlartifacts")
        url = f"http://{LOCALHOST}:{port}"
        default_artifact_root = f"{url}/api/2.0/mlflow-artifacts/artifacts"
        uri_prefix = "file:///" if is_windows() else ""
        process = _launch_server(
            LOCALHOST,
            port,
            uri_prefix + backend_store_uri,
            default_artifact_root,
            uri_prefix + artifacts_destination,
        )
        yield ArtifactsServer(
            backend_store_uri, default_artifact_root, artifacts_destination, url, process
        )
        process.kill()


def read_file(path):
    with open(path) as f:
        return f.read()


def upload_file(path, url):
    with open(path, "rb") as f:
        requests.put(url, data=f).raise_for_status()


def download_file(url, local_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def test_mlflow_artifacts_rest_apis(artifacts_server, tmpdir):
    default_artifact_root = artifacts_server.default_artifact_root
    artifacts_destination = artifacts_server.artifacts_destination

    # Upload artifacts
    file_a = tmpdir.join("a.txt")
    file_a.write("0")
    upload_file(file_a, f"{default_artifact_root}/a.txt")
    assert os.path.exists(os.path.join(artifacts_destination, "a.txt"))
    assert read_file(os.path.join(artifacts_destination, "a.txt")) == "0"

    file_b = tmpdir.join("b.txt")
    file_b.write("1")
    upload_file(file_b, f"{default_artifact_root}/dir/b.txt")
    assert os.path.join(artifacts_destination, "dir", "b.txt")
    assert read_file(os.path.join(artifacts_destination, "dir", "b.txt")) == "1"

    # Download artifacts
    local_dir = tmpdir.mkdir("folder")
    local_path_a = local_dir.join("a.txt")
    download_file(f"{default_artifact_root}/a.txt", local_path_a)
    assert read_file(local_path_a) == "0"

    local_path_b = local_dir.join("b.txt")
    download_file(f"{default_artifact_root}/dir/b.txt", local_path_b)
    assert read_file(local_path_b) == "1"

    # List artifacts
    resp = requests.get(default_artifact_root)
    assert resp.json() == {
        "files": [
            {"path": "a.txt", "is_dir": False, "file_size": 1},
            {"path": "dir", "is_dir": True},
        ]
    }
    resp = requests.get(default_artifact_root, params={"path": "dir"})
    assert resp.json() == {"files": [{"path": "b.txt", "is_dir": False, "file_size": 1}]}


def test_log_artifact(artifacts_server, tmpdir):
    url = artifacts_server.url
    artifacts_destination = artifacts_server.artifacts_destination
    mlflow.set_tracking_uri(url)

    tmp_path = tmpdir.join("a.txt")
    tmp_path.write("0")

    with mlflow.start_run() as run:
        mlflow.log_artifact(tmp_path)

    experiment_id = "0"
    run_artifact_root = os.path.join(
        artifacts_destination, experiment_id, run.info.run_id, "artifacts"
    )
    dest_path = os.path.join(run_artifact_root, tmp_path.basename)
    assert os.path.exists(dest_path)
    assert read_file(dest_path) == "0"

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
    tmpdir.mkdir("dir").join("b.txt").write("1")

    with mlflow.start_run() as run:
        mlflow.log_artifacts(tmpdir)

    client = mlflow.tracking.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert sorted(artifacts) == ["a.txt", "dir"]
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "dir")]
    assert artifacts == ["dir/b.txt"]

    # With `artifact_path`
    with mlflow.start_run() as run:
        mlflow.log_artifacts(tmpdir, artifact_path="artifact_path")

    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert artifacts == ["artifact_path"]
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "artifact_path")]
    assert sorted(artifacts) == ["artifact_path/a.txt", "artifact_path/dir"]
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "artifact_path/dir")]
    assert artifacts == ["artifact_path/dir/b.txt"]


def test_list_artifacts(artifacts_server, tmpdir):
    url = artifacts_server.url
    mlflow.set_tracking_uri(url)

    tmp_path_a = tmpdir.join("a.txt")
    tmp_path_a.write("0")
    tmp_path_b = tmpdir.join("b.txt")
    tmp_path_b.write("1")
    client = mlflow.tracking.MlflowClient()
    with mlflow.start_run() as run:
        assert client.list_artifacts(run.info.run_id) == []
        mlflow.log_artifact(tmp_path_a)
        mlflow.log_artifact(tmp_path_b, "dir")

    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert sorted(artifacts) == ["a.txt", "dir"]
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "dir")]
    assert artifacts == ["dir/b.txt"]


def test_download_artifacts(artifacts_server, tmpdir):
    url = artifacts_server.url
    mlflow.set_tracking_uri(url)

    tmp_path_a = tmpdir.join("a.txt")
    tmp_path_a.write("0")
    tmp_path_b = tmpdir.join("b.txt")
    tmp_path_b.write("1")
    with mlflow.start_run() as run:
        mlflow.log_artifact(tmp_path_a)
        mlflow.log_artifact(tmp_path_b, "dir")

    client = mlflow.tracking.MlflowClient()
    dest_path = client.download_artifacts(run.info.run_id, "")
    assert sorted(os.listdir(dest_path)) == ["a.txt", "dir"]
    assert read_file(os.path.join(dest_path, "a.txt")) == "0"
    dest_path = client.download_artifacts(run.info.run_id, "dir")
    assert os.listdir(dest_path) == ["b.txt"]
    assert read_file(os.path.join(dest_path, "b.txt")) == "1"


def is_github_actions():
    return "GITHUB_ACTIONS" in os.environ


@pytest.mark.skipif(is_windows(), reason="This example doesn't work on Windows")
def test_mlflow_artifacts_example(tmpdir):
    root = pathlib.Path(mlflow.__file__).parents[1]
    # On GitHub Actions, remove generated images to save disk space
    rmi_option = "--rmi all" if is_github_actions() else ""
    cmd = f"""
set -ex
./build.sh
docker-compose run -v ${{PWD}}/example.py:/app/example.py client python example.py
docker-compose logs
docker-compose down {rmi_option} --volumes --remove-orphans
"""
    script_path = tmpdir.join("test.sh")
    script_path.write(cmd)
    subprocess.run(
        ["bash", script_path.strpath],
        check=True,
        cwd=os.path.join(root, "examples", "mlflow_artifacts"),
    )
