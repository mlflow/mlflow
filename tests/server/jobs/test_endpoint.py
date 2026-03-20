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


@job(name="simple_job_fun", max_workers=1)
def simple_job_fun(x: int, y: int, sleep_secs: int = 0) -> dict[str, Any]:
    if sleep_secs:
        time.sleep(sleep_secs)
    return {
        "a": x + y,
        "b": x * y,
    }


@job(name="job_assert_tracking_uri", max_workers=1)
def job_assert_tracking_uri(server_url: str) -> None:
    assert mlflow.get_tracking_uri() == server_url


@job(name="job_with_progress_tracking", max_workers=1)
def job_with_progress_tracking(sleep_secs: int = 1) -> str:
    from mlflow.server.jobs.progress import update_status_details

    update_status_details({"stage": "init", "progress": "0%"})
    time.sleep(sleep_secs)

    update_status_details({"stage": "processing", "progress": "50%"})
    time.sleep(sleep_secs)

    update_status_details({"stage": "done", "progress": "100%"})
    return "completed"


class Client:
    def __init__(self, server_url: str):
        self.server_url = server_url

    def submit_job(
        self, job_name: str, params: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        payload = {
            "job_name": job_name,
            "params": params,
            "timeout": timeout,
        }
        response = requests.post(f"{self.server_url}/ajax-api/3.0/jobs/", json=payload)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, payload: dict[str, Any]) -> requests.Response:
        return requests.post(f"{self.server_url}{path}", json=payload)

    def get_job(self, job_id: str) -> dict[str, Any]:
        response = requests.get(f"{self.server_url}/ajax-api/3.0/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        response = requests.patch(f"{self.server_url}/ajax-api/3.0/jobs/cancel/{job_id}")
        response.raise_for_status()
        return response.json()

    def wait_job(self, job_id: str, timeout: float = 10) -> dict[str, Any]:
        beg_time = time.time()
        while time.time() - beg_time <= timeout:
            job_json = self.get_job(job_id)
            if job_json["status"] in ["SUCCEEDED", "FAILED", "TIMEOUT"]:
                return job_json
            time.sleep(0.5)
        raise TimeoutError("The job is not finalized within the timeout.")

    def search_job(
        self,
        job_name: str | None = None,
        params: dict[str, Any] | None = None,
        statuses: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        response = self.post(
            "/ajax-api/3.0/jobs/search",
            payload={
                "job_name": job_name,
                "params": params,
                "statuses": statuses,
            },
        )
        response.raise_for_status()
        return response.json()["jobs"]


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Client:
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
            "_MLFLOW_SUPPORTED_JOB_FUNCTION_LIST": (
                "test_endpoint.simple_job_fun,"
                "test_endpoint.job_assert_tracking_uri,"
                "test_endpoint.job_with_progress_tracking"
            ),
            "_MLFLOW_ALLOWED_JOB_NAME_LIST": (
                "simple_job_fun,job_assert_tracking_uri,job_with_progress_tracking"
            ),
        },
        start_new_session=True,  # new session & process group
    ) as server_proc:
        try:
            time.sleep(10)  # wait the job runner up
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
            yield Client(f"http://127.0.0.1:{port}")
        finally:
            # NOTE that we need to kill subprocesses
            # (uvicorn server / huey task runner)
            # so `killpg` is needed.
            os.killpg(server_proc.pid, signal.SIGKILL)


def test_job_submit(client: Client):
    job_id = client.submit_job(
        job_name="simple_job_fun",
        params={"x": 3, "y": 4},
    )["job_id"]
    job_json = client.wait_job(job_id, timeout=30)
    job_json.pop("creation_time")
    job_json.pop("last_update_time")
    assert job_json == {
        "job_id": job_id,
        "job_name": "simple_job_fun",
        "params": {"x": 3, "y": 4},
        "timeout": None,
        "status": "SUCCEEDED",
        "result": {"a": 7, "b": 12},
        "retry_count": 0,
        "status_details": None,
    }


def test_job_cancel(client: Client):
    job_id = client.submit_job(
        job_name="simple_job_fun",
        params={"x": 3, "y": 4, "sleep_secs": 120},
    )["job_id"]
    deadline = time.time() + 20
    while time.time() < deadline:
        status = client.get_job(job_id)["status"]
        if status == "RUNNING":
            break
        time.sleep(0.5)
    else:
        raise TimeoutError(f"Job did not start running within 20 seconds, last status: {status}")

    client.cancel_job(job_id)

    job_json = client.get_job(job_id)
    job_json.pop("creation_time")
    job_json.pop("last_update_time")
    assert job_json == {
        "job_id": job_id,
        "job_name": "simple_job_fun",
        "params": {"x": 3, "y": 4, "sleep_secs": 120},
        "timeout": None,
        "status": "CANCELED",
        "result": None,
        "retry_count": 0,
        "status_details": None,
    }


def test_job_endpoint_invalid_job_name(client: Client):
    payload = {
        "job_name": "invalid_job_name",
        "params": {"x": 3, "y": 4},
    }
    response = client.post("/ajax-api/3.0/jobs/", payload=payload)
    assert response.status_code == 400
    error_json = response.json()
    assert "Invalid job name: invalid_job_name" in error_json["detail"]


def test_job_endpoint_missing_parameters(client: Client):
    payload = {
        "job_name": "simple_job_fun",
        "params": {"x": 3},  # Missing required parameter 'y'
    }
    response = client.post("/ajax-api/3.0/jobs/", payload=payload)

    # Should return a 400 error with information about missing parameters
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Missing required parameters for function 'simple_job_fun': ['y']. "
        + "Expected parameters: ['x', 'y', 'sleep_secs']"
    )


def test_job_tracking_uri(client: Client):
    job_id = client.submit_job(
        job_name="job_assert_tracking_uri",
        params={"server_url": client.server_url},
    )["job_id"]
    job_json = client.wait_job(job_id)
    assert job_json["status"] == "SUCCEEDED"


def test_job_endpoint_search(client: Client):
    job1_id = client.submit_job(
        job_name="simple_job_fun",
        params={"x": 7, "y": 4},
    )["job_id"]

    job2_id = client.submit_job(
        job_name="simple_job_fun",
        params={"x": 7, "y": 5},
    )["job_id"]

    job3_id = client.submit_job(
        job_name="simple_job_fun",
        params={"x": 4, "y": 5},
    )["job_id"]

    job4_id = client.submit_job(
        job_name="simple_job_fun",
        params={"x": 4, "y": 5, "sleep_secs": 5},
        timeout=2,
    )["job_id"]

    client.wait_job(job1_id)
    client.wait_job(job2_id)
    client.wait_job(job3_id)
    client.wait_job(job4_id)

    def extract_job_ids(jobs: list[dict[str, Any]]) -> list[str]:
        return [job_json["job_id"] for job_json in jobs]

    jobs = client.search_job(
        job_name="simple_job_fun",
        params={"x": 7},
    )
    assert extract_job_ids(jobs) == [job1_id, job2_id]

    jobs = client.search_job(
        job_name="bad_fun_name",
        params={"x": 7},
    )
    assert extract_job_ids(jobs) == []

    jobs = client.search_job(
        job_name="simple_job_fun",
        params={"x": 7, "y": 5},
    )
    assert extract_job_ids(jobs) == [job2_id]

    jobs = client.search_job(
        job_name="simple_job_fun",
        params={"y": 5},
    )
    assert extract_job_ids(jobs) == [job2_id, job3_id, job4_id]

    jobs = client.search_job(
        job_name="simple_job_fun",
        params={"y": 6},
    )
    assert extract_job_ids(jobs) == []

    jobs = client.search_job(
        job_name="simple_job_fun",
        params={"y": 5},
        statuses=["SUCCEEDED"],
    )
    assert extract_job_ids(jobs) == [job2_id, job3_id]

    jobs = client.search_job(
        job_name="simple_job_fun",
        params={"y": 5},
        statuses=["TIMEOUT"],
    )
    assert extract_job_ids(jobs) == [job4_id]

    jobs = client.search_job(
        job_name="simple_job_fun",
        params={"y": 5},
        statuses=["SUCCEEDED", "TIMEOUT"],
    )
    assert extract_job_ids(jobs) == [job2_id, job3_id, job4_id]

    response = client.post(
        "/ajax-api/3.0/jobs/search",
        payload={
            "job_name": "simple_job_fun",
            "statuses": ["BAD_STATUS"],
        },
    )
    assert response.status_code == 422
    assert (
        response.json()["detail"][0]["msg"]
        == "Input should be 'PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'TIMEOUT' or 'CANCELED'"
    )


def test_job_status_details_in_api_response(client: Client):
    job_id = client.submit_job(
        job_name="job_with_progress_tracking",
        params={"sleep_secs": 1},
    )["job_id"]

    job_json = client.wait_job(job_id, timeout=30)

    assert "status_details" in job_json
    assert job_json["status_details"] is not None
    assert job_json["status_details"].get("stage") == "done"
    assert job_json["status"] == "SUCCEEDED"
    assert job_json["result"] == "completed"
