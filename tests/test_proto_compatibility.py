import os
import subprocess
import time
import logging
import signal
import uuid
import tempfile
from pathlib import Path

import requests
import pytest
import mlflow
from tests.helper_functions import get_safe_port

_logger = logging.getLogger(__name__)


def wait_until_tracking_server_ready(tracking_uri):
    for _ in range(10):
        try:
            resp = requests.get(f"{tracking_uri}/health")
            if resp.status_code == 200:
                _logger.info("Server is ready")
                break
        except requests.exceptions.ConnectionError:
            _logger.info("Waiting for server to start")
            time.sleep(3)
    else:
        raise Exception("Server failed to start")


def get_tracking_server_command(backend_store_uri):
    return [
        "mlflow",
        "server",
        "--backend-store-uri",
        backend_store_uri,
        "--default-artifact-root",
        "artifacts",
        "--host",
        "0.0.0.0",
    ]


@pytest.fixture(scope="module")
def docker_image():
    image = f"mlflow-{__name__}-{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        dockerfile = tmp_dir.joinpath("Dockerfile")
        dockerfile.write_text(
            """
FROM python:3.7
RUN pip install mlflow scikit-learn
RUN mlflow --version
"""
        )
        _logger.info(f"Building docker image {image}")
        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                image,
                "-f",
                dockerfile,
                tmp_dir,
            ],
            check=True,
        )

        yield image

        _logger.info(f"Removing docker image {image}")
        subprocess.run(["docker", "rmi", image], check=True)


@pytest.mark.skipif(os.name == "nt", reason="This test fails on Windows")
@pytest.mark.parametrize("backend_store_uri", ["./mlruns", "sqlite:///mlruns.db"])
def test_proto_compatibility(backend_store_uri, docker_image, tmp_path, monkeypatch):
    """
    This test checks proto compatibility between the latest version of MLflow on PyPI
    (mlflow-latest) and the development version of MLflow in the local repository (mlflow-dev) to
    prevent issues such as https://github.com/mlflow/mlflow/pull/6834.
    """
    monkeypatch.chdir(tmp_path)
    script = tmp_path.joinpath("script.py")
    script.write_text(
        r"""
import random
import sys
import mlflow
from sklearn.linear_model import LogisticRegression

tracking_uri = sys.argv[1]
mlflow.set_tracking_uri(tracking_uri)
for exp_index in range(2):
    mlflow.set_experiment(f"experiment_{exp_index}")
    for run_index in range(2):
        with mlflow.start_run():
            mlflow.log_param("param", random.random())
            mlflow.log_metric("metric", random.random())
            mlflow.set_tag("tag", random.random())
            mlflow.sklearn.log_model(
                LogisticRegression(),
                artifact_path="model",
                registered_model_name=(
                    f"model_{exp_index}_{run_index}"
                    if tracking_uri.startswith("sqlite")
                    else None
                ),
            )
"""
    )
    port = get_safe_port()
    tracking_server_command = get_tracking_server_command(backend_store_uri)
    cmd = " && ".join(
        [
            # Log data for testing
            f"python {script.name} {backend_store_uri}",
            # Modify permissions to allow pytest to remove files generated in the container
            "chmod -R 777 " + backend_store_uri.split("/")[-1],
            # Launch the tracking server
            " ".join(tracking_server_command),
        ]
    )
    is_db_uri = backend_store_uri.startswith("sqlite")
    with subprocess.Popen(
        [
            "docker",
            "run",
            "--rm",
            "-w",
            "/app",
            # Mount the temporary directory (= current working directory) to the container
            # to use the data logged in mlflow-latest later
            "-v",
            f"{tmp_path}:/app",
            "-p",
            f"{port}:5000",
            docker_image,
            "bash",
            "-c",
            f"{cmd}",
        ],
    ) as proc:
        tracking_uri = f"http://localhost:{port}"
        wait_until_tracking_server_ready(tracking_uri)
        client = mlflow.MlflowClient(tracking_uri)
        assert len(client.search_experiments()) == 3
        assert len(client.search_runs(experiment_ids=["1", "2"])) == 4
        if is_db_uri:
            assert len(client.search_registered_models()) == 4
            assert len(client.search_model_versions(filter_string="")) == 4
        proc.send_signal(signal.SIGINT)  # Terminate the tracking server

    # Ensure mlflow-dev can read the data logged in mlflow-latest
    if is_db_uri:
        subprocess.run(["mlflow", "db", "upgrade", backend_store_uri], check=True)
    port = get_safe_port()
    with subprocess.Popen([*tracking_server_command, "-p", str(port)]) as proc:
        tracking_uri = f"http://localhost:{port}"
        wait_until_tracking_server_ready(tracking_uri)
        client = mlflow.MlflowClient(tracking_uri)
        assert len(client.search_experiments()) == 3
        assert len(client.search_runs(experiment_ids=["1", "2"])) == 4
        if is_db_uri:
            assert len(client.search_registered_models()) == 4
            assert len(client.search_model_versions(filter_string="")) == 4
        proc.send_signal(signal.SIGINT)
