import os
import signal
import subprocess
import sys
import time
from typing import Any

import pytest
import requests

import mlflow
from mlflow.server.jobs import job

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="MLflow job execution is not supported on Windows"
)


@job(max_workers=1)
def simple_job_fun(x: int, y: int) -> dict[str, Any]:
    return {
        "a": x + y,
        "b": x * y,
    }


@job(max_workers=1)
def job_assert_tracking_uri(server_url: str) -> None:
    assert mlflow.get_tracking_uri() == server_url


@pytest.fixture(scope="module")
def server_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    from tests.helper_functions import get_safe_port

    tmp_path = tmp_path_factory.mktemp("server_mod")
    backend_store_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    port = get_safe_port()
    with subprocess.Popen(
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
            "PYTHONPATH": os.path.dirname(__file__),
            "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "true",
            "_MLFLOW_ALLOWED_JOB_FUNCTION_LIST": (
                "test_endpoint.simple_job_fun,invalid_format_no_module,"
                "non_existent_module.some_function,os.non_existent_function,"
                "test_endpoint.job_assert_tracking_uri"
            ),
        },
        start_new_session=True,  # new session & process group
    ) as server_proc:
        try:
            # wait server up.
            deadline = time.time() + 15
            while time.time() < deadline:
                time.sleep(1)
                try:
                    resp = requests.get(f"http://127.0.0.1:{port}/health")
                except requests.ConnectionError:
                    continue
                if resp.status_code == 200:
                    break
            else:
                raise TimeoutError("Server did not report healthy within 15 seconds")
            yield f"http://127.0.0.1:{port}"
        finally:
            # NOTE that we need to kill subprocesses
            # (uvicorn server / huey task runner)
            # so `killpg` is needed.
            os.killpg(server_proc.pid, signal.SIGKILL)


def wait_job_finalize(server_url: str, job_id: str, timeout: float) -> None:
    beg_time = time.time()
    while time.time() - beg_time <= timeout:
        response = requests.get(f"{server_url}/ajax-api/3.0/jobs/{job_id}")
        response.raise_for_status()
        job_json = response.json()
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
    response.raise_for_status()
    job_id = response.json()["job_id"]
    wait_job_finalize(server_url, job_id, 10)
    response2 = requests.get(f"{server_url}/ajax-api/3.0/jobs/{job_id}")
    response2.raise_for_status()
    job_json = response2.json()
    job_json.pop("creation_time")
    job_json.pop("last_update_time")
    assert job_json == {
        "job_id": job_id,
        "function_fullname": "test_endpoint.simple_job_fun",
        "params": {"x": 3, "y": 4},
        "timeout": None,
        "status": "SUCCEEDED",
        "result": {"a": 7, "b": 12},
        "retry_count": 0,
    }


def test_job_endpoint_invalid_function_format(server_url: str):
    """Test that invalid function fullname format returns proper error"""
    payload = {
        "function_fullname": "invalid_format_no_module",
        "params": {"x": 3, "y": 4},
    }
    response = requests.post(f"{server_url}/ajax-api/3.0/jobs/", json=payload)
    assert response.status_code == 400
    error_json = response.json()
    assert "Invalid function fullname format" in error_json["detail"]


def test_job_endpoint_module_not_found(server_url: str):
    """Test that non-existent module returns proper error"""
    payload = {
        "function_fullname": "non_existent_module.some_function",
        "params": {"x": 3, "y": 4},
    }
    response = requests.post(f"{server_url}/ajax-api/3.0/jobs/", json=payload)
    assert response.status_code == 400
    error_json = response.json()
    assert "Module not found" in error_json["detail"]


def test_job_endpoint_function_not_found(server_url: str):
    """Test that non-existent function in existing module returns proper error"""
    payload = {
        "function_fullname": "os.non_existent_function",
        "params": {"x": 3, "y": 4},
    }
    response = requests.post(f"{server_url}/ajax-api/3.0/jobs/", json=payload)
    assert response.status_code == 400
    error_json = response.json()
    assert "Function not found" in error_json["detail"]


def test_job_endpoint_missing_parameters(server_url: str):
    """Test that proper error is returned when required function parameters are missing."""
    payload = {
        "function_fullname": "test_endpoint.simple_job_fun",
        "params": {"x": 3},  # Missing required parameter 'y'
    }
    response = requests.post(f"{server_url}/ajax-api/3.0/jobs/", json=payload)

    # Should return a 400 error with information about missing parameters
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Missing required parameters for function 'simple_job_fun': ['y']. "
        + "Expected parameters: ['x', 'y']"
    )


def test_job_tracking_uri(server_url: str):
    payload = {
        "function_fullname": "test_endpoint.job_assert_tracking_uri",
        "params": {"server_url": server_url},
    }
    response = requests.post(f"{server_url}/ajax-api/3.0/jobs/", json=payload)
    response.raise_for_status()
    job_id = response.json()["job_id"]
    wait_job_finalize(server_url, job_id, 10)
    response2 = requests.get(f"{server_url}/ajax-api/3.0/jobs/{job_id}")
    assert response2.json()["status"] == "SUCCEEDED"
