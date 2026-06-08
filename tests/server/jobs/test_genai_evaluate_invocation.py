# End-to-end tests for `POST /ajax-api/3.0/mlflow/genai/evaluate/invoke`.

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Literal

import pytest
import requests

import mlflow
from mlflow.entities import RunStatus
from mlflow.genai.judges import make_judge
from mlflow.utils.mlflow_tags import (
    MLFLOW_GENAI_EVALUATE_JOB_ID,
    MLFLOW_RUN_TYPE,
    MLFLOW_RUN_TYPE_GENAI_EVALUATE,
)

from tests.helper_functions import get_safe_port

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="MLflow job execution is not supported on Windows"
)


class MockGatewayHandler(BaseHTTPRequestHandler):
    """Always returns ``{"result":"Yes","rationale":"Mock"}`` for any model."""

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        # Drain the body so the client doesn't see a connection error.
        self.rfile.read(content_length)

        response = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"result": "Yes", "rationale": "Mock"}),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class Client:
    def __init__(self, server_url: str):
        self.server_url = server_url

    def invoke_genai_evaluate(
        self,
        experiment_id: str,
        trace_ids: list[str],
        serialized_scorers: list[str],
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.server_url}/ajax-api/3.0/mlflow/genai/evaluate/invoke",
            json={
                "experiment_id": experiment_id,
                "trace_ids": trace_ids,
                "serialized_scorers": serialized_scorers,
            },
        )
        if not response.ok:
            raise AssertionError(
                f"invoke_genai_evaluate failed with status {response.status_code}: {response.text}"
            )
        return response.json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        response = requests.get(f"{self.server_url}/ajax-api/3.0/mlflow/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def wait_job(self, job_id: str, timeout: float = 60) -> dict[str, Any]:
        beg = time.time()
        while time.time() - beg <= timeout:
            job_json = self.get_job(job_id)
            if job_json["status"] in ["SUCCEEDED", "FAILED", "TIMEOUT"]:
                return job_json
            time.sleep(0.5)
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


@pytest.fixture(scope="module")
def mock_gateway_server():
    port = get_safe_port()
    server = HTTPServer(("127.0.0.1", port), MockGatewayHandler)
    thread = threading.Thread(name="genai-evaluate-mock-gateway", target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory, mock_gateway_server: str) -> Client:
    """Spin up an mlflow server with the new genai-evaluate job registered."""
    tmp_path = tmp_path_factory.mktemp("genai_evaluate_server")
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
            "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "true",
            "_MLFLOW_SUPPORTED_JOB_FUNCTION_LIST": (
                "mlflow.genai.evaluation.job.invoke_genai_evaluate_job"
            ),
            "_MLFLOW_ALLOWED_JOB_NAME_LIST": "invoke_genai_evaluate",
            "MLFLOW_GATEWAY_URI": mock_gateway_server,
        },
        start_new_session=True,
    ) as server_proc:
        try:
            # Server needs time for the job runner to start before we can submit.
            time.sleep(10)
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
            os.killpg(server_proc.pid, signal.SIGKILL)


@pytest.fixture
def experiment_with_traces(client: Client):
    """Create an experiment with 2 traces. Returns (experiment_id, trace_ids)."""
    mlflow.set_tracking_uri(client.server_url)
    experiment_id = mlflow.create_experiment(f"genai_evaluate_test_{time.time()}")
    mlflow.set_experiment(experiment_id=experiment_id)

    trace_ids = []
    for i in range(2):
        with mlflow.start_span(name=f"test_span_{i}") as span:
            span.set_inputs({"question": f"What is {i}+{i}?"})
            span.set_outputs(f"The answer is {i + i}")
            trace_ids.append(span.trace_id)
    return experiment_id, trace_ids


def _serialized_judge(name: str = "answer_quality") -> str:
    judge = make_judge(
        name=name,
        instructions="Input: {{ inputs }}\nOutput: {{ outputs }}",
        model="gateway:/mock-judge",
        feedback_value_type=Literal["Yes", "No"],
    )
    return json.dumps(judge.model_dump())


def test_invoke_genai_evaluate_basic(client: Client, experiment_with_traces):
    """Happy path: handler returns {job_id, run_id}, run is tagged correctly,
    the job FINISHES, and the run ends FINISHED with traces linked to it.
    """
    experiment_id, trace_ids = experiment_with_traces

    response = client.invoke_genai_evaluate(
        experiment_id=experiment_id,
        trace_ids=trace_ids,
        serialized_scorers=[_serialized_judge()],
    )

    assert "job_id" in response
    assert "run_id" in response
    job_id = response["job_id"]
    run_id = response["run_id"]

    # The run should be visible on /evaluation-runs *immediately* — i.e. before
    # the job even starts work — because the handler creates it synchronously
    # with the right tag.
    run = mlflow.get_run(run_id)
    assert run.data.tags[MLFLOW_RUN_TYPE] == MLFLOW_RUN_TYPE_GENAI_EVALUATE
    assert run.data.tags[MLFLOW_GENAI_EVALUATE_JOB_ID] == job_id

    # Job must succeed; on success the job is responsible for flipping the run
    # from RUNNING to FINISHED.
    job_result = client.wait_job(job_id)
    assert job_result["status"] == "SUCCEEDED", f"job failed: {job_result}"

    # Re-fetch the run to confirm the terminal state transition landed in the
    # store. RunStatus.FINISHED is an int enum; the run.status field is the
    # string form.
    run = mlflow.get_run(run_id)
    assert run.info.status == RunStatus.to_string(RunStatus.FINISHED)


def test_invoke_genai_evaluate_missing_trace_marks_run_failed(
    client: Client, experiment_with_traces
):
    """If any input trace doesn't exist the harness raises; the run must end
    FAILED rather than stuck in RUNNING.
    """
    experiment_id, _ = experiment_with_traces

    response = client.invoke_genai_evaluate(
        experiment_id=experiment_id,
        trace_ids=["tr-does-not-exist-00000000000000"],
        serialized_scorers=[_serialized_judge()],
    )
    job_id = response["job_id"]
    run_id = response["run_id"]

    job_result = client.wait_job(job_id)
    assert job_result["status"] == "FAILED"

    run = mlflow.get_run(run_id)
    assert run.info.status == RunStatus.to_string(RunStatus.FAILED)


def test_invoke_genai_evaluate_multiple_scorers_share_one_run(
    client: Client, experiment_with_traces
):
    """Multiple scorers in one request must produce a SINGLE run (not one per
    scorer). This is the whole point of the new endpoint vs. /scorer/invoke.
    """
    experiment_id, trace_ids = experiment_with_traces

    response = client.invoke_genai_evaluate(
        experiment_id=experiment_id,
        trace_ids=trace_ids,
        serialized_scorers=[
            _serialized_judge("judge_a"),
            _serialized_judge("judge_b"),
            _serialized_judge("judge_c"),
        ],
    )
    job_id = response["job_id"]
    run_id = response["run_id"]

    job_result = client.wait_job(job_id)
    assert job_result["status"] == "SUCCEEDED", f"job failed: {job_result}"
    # The job's return value is the source of truth that *all three* scorers ran
    # inside the same job (vs. e.g. one job per scorer, which is what
    # /scorer/invoke does). Stronger than reading a tag we wrote ourselves.
    assert job_result["result"]["scorer_count"] == 3
    assert job_result["result"]["run_id"] == run_id

    run = mlflow.get_run(run_id)
    assert run.info.status == RunStatus.to_string(RunStatus.FINISHED)


def test_invoke_genai_evaluate_handler_validation_no_traces(client: Client):
    """Empty trace_ids must be rejected before any run is created — otherwise
    we'd litter the UI with empty placeholder runs on every malformed POST.
    """
    response = requests.post(
        f"{client.server_url}/ajax-api/3.0/mlflow/genai/evaluate/invoke",
        json={
            "experiment_id": "0",
            "trace_ids": [],
            "serialized_scorers": [_serialized_judge()],
        },
    )
    assert response.status_code == 400
    assert "trace" in response.text.lower()


def test_invoke_genai_evaluate_handler_validation_no_scorers(client: Client):
    response = requests.post(
        f"{client.server_url}/ajax-api/3.0/mlflow/genai/evaluate/invoke",
        json={
            "experiment_id": "0",
            "trace_ids": ["any"],
            "serialized_scorers": [],
        },
    )
    assert response.status_code == 400
    assert "judge" in response.text.lower()
