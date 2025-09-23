import os
import signal
import subprocess
import sys
import time
from typing import Any

import pytest
import requests

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
    backend_store_uri = f"sqlite:///{tmp_path / 'mlflow.db'!s}"

    server_proc = None
    try:
        server_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "-h",
                "127.0.0.1",
                "-p",
                "6677",
                "--backend-store-uri",
                backend_store_uri,
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


def wait_job_finalize(server_url, job_id, timeout):
    beg_time = time.time()
    while time.time() - beg_time <= timeout:
        job_json = requests.get(f"{server_url}/ajax-api/3.0/jobs/{job_id}").json()
        if job_json["status"] in ["SUCCEEDED", "FAILED", "TIMEOUT"]:
            return
        time.sleep(0.5)
    raise TimeoutError("The job is not finalized within the timeout.")


def test_job_endpoint(server_url: str):
    payload = {
        "function_fullname": "test_endpoint.simple_job_fun",
        "params": {"x": 3, "y": 4},
    }
    response = requests.post(f"{server_url}/ajax-api/3.0/jobs/", json=payload)
    job_id = response.json()["job_id"]
    wait_job_finalize(server_url, job_id, 2)
    response2 = requests.get(f"{server_url}/ajax-api/3.0/jobs/{job_id}")
    job_json = response2.json()
    job_json.pop("creation_time")
    assert job_json == {
        "job_id": job_id,
        "function_fullname": "test_endpoint.simple_job_fun",
        "params": {"x": 3, "y": 4},
        "timeout": None,
        "status": "SUCCEEDED",
        "result": {"a": 7, "b": 12},
        "retry_count": 0,
    }
