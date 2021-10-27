import os
from subprocess import Popen

import mlflow
from tests.helper_functions import LOCALHOST, get_safe_port
from tests.tracking.integration_test_utils import _await_server_up_or_die


def _launch_server(backend_store_uri, artifacts_destination):
    port = get_safe_port()
    cmd = [
        "mlflow",
        "server",
        "--backend-store-uri",
        backend_store_uri,
        "--artifacts-destination",
        artifacts_destination,
        "--host",
        LOCALHOST,
        "--port",
        str(port),
    ]
    process = Popen(cmd)
    _await_server_up_or_die(port)
    url = "http://{hostname}:{port}".format(hostname=LOCALHOST, port=port)
    return url, process


def read_file(path):
    with open(path) as f:
        return f.read()


def test_log_artifact(tmpdir):
    backend_store_uri = f"{tmpdir}/mlruns"
    artifacts_destination = f"{tmpdir}/artifacts"
    url, _ = _launch_server(
        backend_store_uri=backend_store_uri, artifacts_destination=artifacts_destination
    )
    mlflow.set_tracking_uri(url)
    experiment_name = "test"
    mlflow.create_experiment(
        experiment_name, artifact_location=f"{url}/api/2.0/mlflow-artifacts/artifacts"
    )
    mlflow.set_experiment(experiment_name)

    tmp_path = tmpdir.join("a.txt")
    tmp_path.write("1")
    with mlflow.start_run() as run:
        mlflow.log_artifact(tmp_path)

    dest_path = os.path.join(artifacts_destination, run.info.run_id, "artifacts", tmp_path.basename)
    assert read_file(dest_path) == "1"

