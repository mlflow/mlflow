import os
import signal
import socket
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


def _wait_for_server_ready(port: int, timeout: int = 30) -> None:
    """Wait for the MLflow server to be ready to accept connections."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    # Server is accepting connections, give it a moment to fully initialize
                    time.sleep(1)
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Server failed to start on port {port} within {timeout} seconds")


@pytest.fixture(scope="module")
def server_url(tmp_path_factory):
    from tests.helper_functions import get_safe_port

    tmp_path = tmp_path_factory.mktemp("server_mod")
    backend_store_uri = f"sqlite:///{tmp_path / 'mlflow.db'!s}"

    # Get a dynamic port instead of using hardcoded 6677
    port = get_safe_port()

    server_proc = None
    try:
        # Construct PYTHONPATH to include both the test directory and the project root
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        pythonpath = f"{current_dir}:{project_root}"
        if "PYTHONPATH" in os.environ:
            pythonpath = f"{pythonpath}:{os.environ['PYTHONPATH']}"

        server_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "-h",
                "127.0.0.1",
                "-p",
                str(port),
                "--backend-store-uri",
                backend_store_uri,
            ],
            env={
                **os.environ,
                "PYTHONPATH": pythonpath,
                "_MLFLOW_JOB_FUNCTION_EXTRA_ALLOW_LIST": "test_endpoint.simple_job_fun",
            },
            start_new_session=True,  # new session & process group
        )

        # Wait for server to be ready instead of fixed sleep
        _wait_for_server_ready(port)

        yield f"http://127.0.0.1:{port}"
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
