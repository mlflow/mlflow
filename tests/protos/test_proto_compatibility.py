import subprocess
import requests
import time
import logging
import signal

import pytest
import mlflow
from tests.helper_functions import get_safe_port

_logger = logging.getLogger(__name__)


def wait_until_tracking_server_ready(tracking_uri):
    for i in range(10):
        try:
            resp = requests.get(f"{tracking_uri}/health")
            if resp.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            _logger.info("Waiting for server to start...")
            time.sleep(2**i)
    else:
        raise Exception("Server failed to start")


class Popen(subprocess.Popen):
    def __exit__(self, exc_type, value, traceback):  # pylint: disable=arguments-differ
        self.send_signal(signal.SIGINT)
        super().__exit__(exc_type, value, traceback)


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


@pytest.mark.parametrize("backend_store_uri", ["./mlruns", "sqlite:///mlruns.db"])
def test_proto_compatibility(backend_store_uri, tmp_path, monkeypatch):
    """
    This test checks proto compatibility between the latest version of MLflow on PyPI and the
    development version of MLflow in the local repository to prevent issues such as
    https://github.com/mlflow/mlflow/pull/6834.
    """
    monkeypatch.chdir(tmp_path)
    script = tmp_path.joinpath("script.py")
    script.write_text(
        f"""
import random
import uuid
import mlflow
from sklearn.linear_model import LogisticRegression

mlflow.set_tracking_uri("{backend_store_uri}")
for _ in range(2):
    mlflow.set_experiment(uuid.uuid4().hex)
    for _ in range(2):
        with mlflow.start_run():
            mlflow.log_param("param", random.random())
            mlflow.log_metric("metric", random.random())
            mlflow.set_tag("tag", random.random())
            mlflow.sklearn.log_model(
                LogisticRegression(),
                artifact_path="model",
                registered_model_name=uuid.uuid4().hex,
            )
"""
    )
    port = get_safe_port()
    tracking_server_command = get_tracking_server_command(backend_store_uri)
    cmd = " && ".join(
        [
            "pip install mlflow scikit-learn",
            f"python {script.name}",
            # Modify permissions to allow the host to remove files generated in the container
            "chmod -R 777 " + backend_store_uri.split("/")[-1],
            " ".join(tracking_server_command),
        ]
    )
    with Popen(
        [
            "docker",
            "run",
            "--rm",
            "-w",
            "/app",
            "-v",
            f"{tmp_path}:/app",
            "-p",
            f"{port}:5000",
            "python:3.8",
            "bash",
            "-c",
            f"{cmd}",
        ],
    ):
        tracking_uri = f"http://localhost:{port}"
        wait_until_tracking_server_ready(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient(tracking_uri)
        assert len(client.search_experiments()) == 3
        assert len(client.search_runs(experiment_ids=["1", "2"])) == 4
        assert len(client.search_registered_models()) == 2
        assert len(client.search_model_versions(filter_string="")) == 2

    port = get_safe_port()
    with Popen(
        [
            *tracking_server_command,
            "-p",
            str(port),
        ]
    ):
        tracking_uri = f"http://localhost:{port}"
        wait_until_tracking_server_ready(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        assert len(mlflow.search_experiments()) == 3
        assert len(mlflow.search_runs(experiment_ids=["1", "2"])) == 4
        assert len(client.search_registered_models()) == 2
        assert len(client.search_model_versions(filter_string="")) == 2
