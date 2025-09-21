import os
import pytest
import requests
import signal
import subprocess
import time
from typing import Any


pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="MLflow job execution is not supported on Windows"
)


def simple_job_fun(x: int, y: int) -> dict[str, Any]:
    return {
        "a": x + y,
        "b": x * y,
    }


@pytest.fixture(scope="module")
def server_url(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("server_mod")
    backend_store_uri = f"sqlite:///{str(tmp_path / 'mlflow.db')}"

    server_proc = None
    try:
        server_proc = subprocess.Popen(
            [
                "mlflow", "server",
                "-h", "127.0.0.1",
                "-p", "6677",
                "--backend-store-uri", backend_store_uri,
            ],
            env={
                **os.environ,
                "PYTHONPATH": os.path.dirname(__file__),
                "_MLFLOW_JOB_FUNCTION_EXTRA_ALLOW_LIST": "test_endpoint.simple_job_fun",
            },
            start_new_session=True,  # new session & process group
        )
        time.sleep(10)  # wait for server to spin up
        yield "http://127.0.0.1:6677"
    finally:
        if server_proc is not None:
            # NOTE that we need to kill subprocesses
            # (uvicorn server / huey task runner)
            # so `killpg` is needed.
            os.killpg(server_proc.pid, signal.SIGTERM)


def test_job_endpoint(server_url):
    payload = {
        "function_fullname": "test_endpoint.simple_job_fun",
        "params": {"x": 3, "y": 4},
    }
    response = requests.post(f"{server_url}/ajax-api/3.0/job/submit", json=payload)
    job_id = response.json()["job_id"]
    time.sleep(1)  # wait for job completion
    response2 = requests.get(f"{server_url}/ajax-api/3.0/job/query/{job_id}")
    assert response2.json() == {'status': 'JobStatus.DONE', 'result': {'a': 7, 'b': 12}}


