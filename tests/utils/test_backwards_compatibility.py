import os
import re
import subprocess
import time
import pytest

import mlflow
import mlflow.tracking
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.projects import _get_conda_bin_executable


# Test backwards compatibility with the latest mlflow release.
# Old client should work with new server and vice versa.
@pytest.mark.skip("Test hangs on build machine, runs locally.")
def test_backwards_compatibility():
    # create conda env
    _mlflow_conda_env(
        path="mlflow_latest_released.yml",
        additional_pip_deps=["mlflow"]
    )
    assert 0 == os.system("conda env create -n mlflow_release -f mlflow_latest_released.yml")
    pattern = re.compile(r".*Listening at\: (http\://127.0.0.1:[0-9]+).*")
    activate_conda_cmd = "source {activate} mlflow_release &&".format(
        activate=_get_conda_bin_executable("activate"))
    try:
        for conf in ("old server", "old client"):
            with TempDir(chdr=True):
                print("TESTING", conf)
                activate_server_env = activate_conda_cmd if conf == "old server" else ""
                server = subprocess.Popen(
                    ["bash", "-c", activate_server_env + "mlflow server -w 1 --port 0"],
                    stdout=subprocess.PIPE,
                    stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                try:
                    m = False
                    while not m:
                        time.sleep(1)
                        line = server.stderr.readline()
                        m = pattern.match(line)
                    url = m.group(1)
                    print("started mlflow server listening at ", url)
                    activate_client_env = activate_conda_cmd if conf == "old client" else ""
                    start_client_cmd = "MLFLOW_TRACKING_URI={url} python {src}".format(
                        url=url, src=__file__)
                    client = subprocess.Popen(
                        ["bash", "-c", activate_client_env + start_client_cmd],
                        stdout=subprocess.PIPE,
                        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    (stdout, stderr) = client.communicate(None)
                    print("STDOUT")
                    print(stdout)
                    print("STDERR")
                    print(stderr)
                    assert client.returncode == 0
                    mlflow.set_tracking_uri(url)
                    client = mlflow.tracking.MlflowClient()
                    assert len(client.list_experiments()) == 3

                finally:
                    server.kill()
    finally:
        # pass
        os.system("conda env remove -y -n mlflow_release")


if __name__ == '__main__':
    assert mlflow.get_tracking_uri().startswith("http://")
    from mlflow.version import VERSION
    print("mlflow version = " + VERSION)
    print("TRACKING URI = " + mlflow.get_tracking_uri())
    id1 = mlflow.create_experiment("test_experiment_1")
    mlflow.set_experiment("test_experiment_1")

    with mlflow.start_run():
        mlflow.log_param("a", "b")
        mlflow.log_metric("m", 1)

    id2 = mlflow.create_experiment("test_experiment_2")
    mlflow.set_experiment("test_experiment_2")
    with mlflow.start_run():
        mlflow.log_param("c", "d")
        mlflow.log_metric("m", 2)
        mlflow.set_tag("x", "y")

    with mlflow.start_run():
        mlflow.log_param("e", "f")
        mlflow.log_metric("m", 3)

    client = mlflow.tracking.MlflowClient()
    print(client.list_experiments())
    print("deleting experiment", id1, type(id1))
    client.delete_experiment(id1)
    print("deleting experiment", id2, type(id2))
    client.delete_experiment(id2)
    assert len(client.list_experiments()) == 1
    client.restore_experiment(id1)
    client.restore_experiment(id2)
    assert len(client.list_experiments()) == 3
    assert len(client.list_run_infos(experiment_id=id1)) == 1
    assert len(client.list_run_infos(experiment_id=id2)) == 2
    print(client.list_run_infos(experiment_id=id1))
    print(client.list_run_infos(experiment_id=id2))
