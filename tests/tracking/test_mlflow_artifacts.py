import cgi
import os
import pathlib
import subprocess
import tempfile
from collections import namedtuple
from io import BytesIO

import pytest
import requests

import mlflow
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.utils.os import is_windows

from tests.helper_functions import LOCALHOST, get_safe_port
from tests.tracking.integration_test_utils import _await_server_up_or_die


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
        backend_store_uri = f"sqlite:///{os.path.join(tmpdir, 'mlruns.db')}"
        artifacts_destination = os.path.join(tmpdir, "mlartifacts")
        url = f"http://{LOCALHOST}:{port}"
        default_artifact_root = f"{url}/api/2.0/mlflow-artifacts/artifacts"
        process = _launch_server(
            LOCALHOST,
            port,
            backend_store_uri,
            default_artifact_root,
            ("file:///" + artifacts_destination if is_windows() else artifacts_destination),
        )
        yield ArtifactsServer(
            backend_store_uri, default_artifact_root, artifacts_destination, url, process
        )
        process.kill()


def read_file(path):
    with open(path) as f:
        return f.read()


def upload_file(path, url, headers=None):
    with open(path, "rb") as f:
        requests.put(url, data=f, headers=headers).raise_for_status()


def download_file(url, local_path, headers=None):
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert "Content-Type" in r.headers
        assert "Content-Disposition" in r.headers
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return r


def test_mlflow_artifacts_rest_apis(artifacts_server, tmp_path):
    default_artifact_root = artifacts_server.default_artifact_root
    artifacts_destination = artifacts_server.artifacts_destination

    # Upload artifacts
    file_a = tmp_path.joinpath("a.txt")
    file_a.write_text("0")
    upload_file(file_a, f"{default_artifact_root}/a.txt")
    assert os.path.exists(os.path.join(artifacts_destination, "a.txt"))
    assert read_file(os.path.join(artifacts_destination, "a.txt")) == "0"

    file_b = tmp_path.joinpath("b.txt")
    file_b.write_text("1")
    upload_file(file_b, f"{default_artifact_root}/dir/b.txt")
    assert os.path.join(artifacts_destination, "dir", "b.txt")
    assert read_file(os.path.join(artifacts_destination, "dir", "b.txt")) == "1"

    # Download artifacts
    local_dir = tmp_path.joinpath("folder")
    local_dir.mkdir()
    local_path_a = local_dir.joinpath("a.txt")
    download_file(f"{default_artifact_root}/a.txt", local_path_a)
    assert read_file(local_path_a) == "0"

    local_path_b = local_dir.joinpath("b.txt")
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


def test_log_artifact(artifacts_server, tmp_path):
    url = artifacts_server.url
    artifacts_destination = artifacts_server.artifacts_destination
    mlflow.set_tracking_uri(url)

    tmp_path = tmp_path.joinpath("a.txt")
    tmp_path.write_text("0")

    with mlflow.start_run() as run:
        mlflow.log_artifact(tmp_path)

    experiment_id = "0"
    run_artifact_root = os.path.join(
        artifacts_destination, experiment_id, run.info.run_id, "artifacts"
    )
    dest_path = os.path.join(run_artifact_root, tmp_path.name)
    assert os.path.exists(dest_path)
    assert read_file(dest_path) == "0"

    with mlflow.start_run() as run:
        mlflow.log_artifact(tmp_path, artifact_path="artifact_path")

    run_artifact_root = os.path.join(
        artifacts_destination, experiment_id, run.info.run_id, "artifacts"
    )
    dest_path = os.path.join(run_artifact_root, "artifact_path", tmp_path.name)
    assert os.path.exists(dest_path)
    assert read_file(dest_path) == "0"


def test_log_artifacts(artifacts_server, tmp_path):
    url = artifacts_server.url
    mlflow.set_tracking_uri(url)

    tmp_path.joinpath("a.txt").write_text("0")
    d = tmp_path.joinpath("dir")
    d.mkdir()
    d.joinpath("b.txt").write_text("1")

    with mlflow.start_run() as run:
        mlflow.log_artifacts(tmp_path)

    client = MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert sorted(artifacts) == ["a.txt", "dir"]
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "dir")]
    assert artifacts == ["dir/b.txt"]

    # With `artifact_path`
    with mlflow.start_run() as run:
        mlflow.log_artifacts(tmp_path, artifact_path="artifact_path")

    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert artifacts == ["artifact_path"]
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "artifact_path")]
    assert sorted(artifacts) == ["artifact_path/a.txt", "artifact_path/dir"]
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "artifact_path/dir")]
    assert artifacts == ["artifact_path/dir/b.txt"]


def test_list_artifacts(artifacts_server, tmp_path):
    url = artifacts_server.url
    mlflow.set_tracking_uri(url)

    tmp_path_a = tmp_path.joinpath("a.txt")
    tmp_path_a.write_text("0")
    tmp_path_b = tmp_path.joinpath("b.txt")
    tmp_path_b.write_text("1")
    client = MlflowClient()
    with mlflow.start_run() as run:
        assert client.list_artifacts(run.info.run_id) == []
        mlflow.log_artifact(tmp_path_a)
        mlflow.log_artifact(tmp_path_b, "dir")

    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert sorted(artifacts) == ["a.txt", "dir"]
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "dir")]
    assert artifacts == ["dir/b.txt"]


def test_download_artifacts(artifacts_server, tmp_path):
    url = artifacts_server.url
    mlflow.set_tracking_uri(url)

    tmp_path_a = tmp_path.joinpath("a.txt")
    tmp_path_a.write_text("0")
    tmp_path_b = tmp_path.joinpath("b.txt")
    tmp_path_b.write_text("1")
    with mlflow.start_run() as run:
        mlflow.log_artifact(tmp_path_a)
        mlflow.log_artifact(tmp_path_b, "dir")

    dest_path = download_artifacts(run_id=run.info.run_id, artifact_path="")
    assert sorted(os.listdir(dest_path)) == ["a.txt", "dir"]
    assert read_file(os.path.join(dest_path, "a.txt")) == "0"
    dest_path = download_artifacts(run_id=run.info.run_id, artifact_path="dir")
    assert os.listdir(dest_path) == ["b.txt"]
    assert read_file(os.path.join(dest_path, "b.txt")) == "1"


def is_github_actions():
    return "GITHUB_ACTIONS" in os.environ


@pytest.mark.skipif(is_windows(), reason="This example doesn't work on Windows")
def test_mlflow_artifacts_example(tmp_path):
    root = pathlib.Path(mlflow.__file__).parents[1]
    # On GitHub Actions, remove generated images to save disk space
    rmi_option = "--rmi all" if is_github_actions() else ""
    cmd = f"""
err=0
trap 'err=1' ERR
./build.sh
docker compose run -v ${{PWD}}/example.py:/app/example.py client python example.py
docker compose logs
docker compose down {rmi_option} --volumes --remove-orphans
test $err = 0
"""
    script_path = tmp_path.joinpath("test.sh")
    script_path.write_text(cmd)
    subprocess.run(
        ["bash", script_path],
        check=True,
        cwd=os.path.join(root, "examples", "mlflow_artifacts"),
    )


def test_rest_tracking_api_list_artifacts_with_proxied_artifacts(artifacts_server, tmp_path):
    def list_artifacts_via_rest_api(url, run_id, path=None):
        if path:
            resp = requests.get(url, params={"run_id": run_id, "path": path})
        else:
            resp = requests.get(url, params={"run_id": run_id})
        resp.raise_for_status()
        return resp.json()

    url = artifacts_server.url
    mlflow.set_tracking_uri(url)
    api = f"{url}/api/2.0/mlflow/artifacts/list"

    tmp_path_a = tmp_path.joinpath("a.txt")
    tmp_path_a.write_text("0")
    tmp_path_b = tmp_path.joinpath("b.txt")
    tmp_path_b.write_text("1")
    mlflow.set_experiment("rest_list_api_test")
    with mlflow.start_run() as run:
        mlflow.log_artifact(tmp_path_a)
        mlflow.log_artifact(tmp_path_b, "dir")

    list_artifacts_response = list_artifacts_via_rest_api(url=api, run_id=run.info.run_id)
    assert list_artifacts_response.get("files") == [
        {"path": "a.txt", "is_dir": False, "file_size": 1},
        {"path": "dir", "is_dir": True},
    ]
    assert list_artifacts_response.get("root_uri") == run.info.artifact_uri

    nested_list_artifacts_response = list_artifacts_via_rest_api(
        url=api, run_id=run.info.run_id, path="dir"
    )
    assert nested_list_artifacts_response.get("files") == [
        {"path": "dir/b.txt", "is_dir": False, "file_size": 1},
    ]
    assert list_artifacts_response.get("root_uri") == run.info.artifact_uri


def test_rest_get_artifact_api_proxied_with_artifacts(artifacts_server, tmp_path):
    url = artifacts_server.url
    mlflow.set_tracking_uri(url)
    tmp_path_a = tmp_path.joinpath("a.txt")
    tmp_path_a.write_text("abcdefg")

    mlflow.set_experiment("rest_get_artifact_api_test")
    with mlflow.start_run() as run:
        mlflow.log_artifact(tmp_path_a)

    get_artifact_response = requests.get(
        url=f"{url}/get-artifact", params={"run_id": run.info.run_id, "path": "a.txt"}
    )
    get_artifact_response.raise_for_status()
    assert get_artifact_response.text == "abcdefg"


def test_rest_get_model_version_artifact_api_proxied_artifact_root(artifacts_server):
    url = artifacts_server.url
    artifact_file = pathlib.Path(artifacts_server.artifacts_destination, "a.txt")
    artifact_file.parent.mkdir(exist_ok=True, parents=True)
    artifact_file.write_text("abcdefg")

    name = "GetModelVersionTest"
    mlflow_client = MlflowClient(artifacts_server.backend_store_uri)
    mlflow_client.create_registered_model(name)
    # An artifact root with scheme http, https, or mlflow-artifacts is a proxied artifact root
    mlflow_client.create_model_version(name, "mlflow-artifacts:", 1)

    get_model_version_artifact_response = requests.get(
        url=f"{url}/model-versions/get-artifact",
        params={"name": name, "version": "1", "path": "a.txt"},
    )
    get_model_version_artifact_response.raise_for_status()
    assert get_model_version_artifact_response.text == "abcdefg"


@pytest.mark.parametrize(
    ("filename", "expected_mime_type"),
    [
        ("a.txt", "text/plain"),
        ("b.pkl", "application/octet-stream"),
        ("c.png", "image/png"),
        ("d.pdf", "application/pdf"),
        ("MLmodel", "text/plain"),
        ("mlproject", "text/plain"),
    ],
)
def test_mime_type_for_download_artifacts_api(
    artifacts_server, tmp_path, filename, expected_mime_type
):
    default_artifact_root = artifacts_server.default_artifact_root
    url = artifacts_server.url
    test_file = tmp_path.joinpath(filename)
    test_file.touch()
    upload_file(test_file, f"{default_artifact_root}/dir/{filename}")
    download_response = download_file(f"{default_artifact_root}/dir/{filename}", test_file)

    _, params = cgi.parse_header(download_response.headers["Content-Disposition"])
    assert params["filename"] == filename
    assert download_response.headers["Content-Type"] == expected_mime_type

    mlflow.set_tracking_uri(url)
    with mlflow.start_run() as run:
        mlflow.log_artifact(test_file)
    artifact_response = requests.get(
        url=f"{url}/get-artifact", params={"run_id": run.info.run_id, "path": filename}
    )
    artifact_response.raise_for_status()
    _, params = cgi.parse_header(artifact_response.headers["Content-Disposition"])
    assert params["filename"] == filename
    assert artifact_response.headers["Content-Type"] == expected_mime_type
    assert artifact_response.headers["X-Content-Type-Options"] == "nosniff"


def test_rest_get_artifact_api_log_image(artifacts_server):
    url = artifacts_server.url
    mlflow.set_tracking_uri(url)

    import numpy as np
    from PIL import Image

    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    with mlflow.start_run() as run:
        mlflow.log_image(image, key="dog", step=100, timestamp=100, synchronous=True)

    artifact_list_response = requests.get(
        url=f"{url}/ajax-api/2.0/mlflow/artifacts/list",
        params={"path": "images", "run_id": run.info.run_id},
    )
    artifact_list_response.raise_for_status()

    for file in artifact_list_response.json()["files"]:
        path = file["path"]
        get_artifact_response = requests.get(
            url=f"{url}/get-artifact", params={"run_id": run.info.run_id, "path": path}
        )
        get_artifact_response.raise_for_status()
        assert "dog%step%100%timestamp%100" in get_artifact_response.headers["Content-Disposition"]
        if path.endswith("png"):
            loaded_image = np.asarray(
                Image.open(BytesIO(get_artifact_response.content)), dtype=np.uint8
            )
            np.testing.assert_array_equal(loaded_image, image)


@pytest.mark.parametrize(
    ("filename", "requested_mime_type", "responded_mime_type"),
    [
        ("b.pkl", "text/html", "application/octet-stream"),
        ("c.png", "text/html", "image/png"),
        ("d.pdf", "text/html", "application/pdf"),
    ],
)
def test_server_overrides_requested_mime_type(
    artifacts_server, tmp_path, filename, requested_mime_type, responded_mime_type
):
    default_artifact_root = artifacts_server.default_artifact_root
    test_file = tmp_path.joinpath(filename)
    test_file.touch()
    upload_file(
        test_file,
        f"{default_artifact_root}/dir/{filename}",
    )
    download_response = download_file(
        f"{default_artifact_root}/dir/{filename}",
        test_file,
        headers={"Accept": requested_mime_type},
    )

    _, params = cgi.parse_header(download_response.headers["Content-Disposition"])
    assert params["filename"] == filename
    assert download_response.headers["Content-Type"] == responded_mime_type
